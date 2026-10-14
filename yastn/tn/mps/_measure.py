# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from __future__ import annotations
from itertools import groupby, accumulate
from numbers import Number
from typing import Sequence

import numpy as np

from ._env import Env, Env2
from ._mps_obc import MpsMpoOBC
from ...operators._multiplets import _is_multiplet, _multiplet_pair
from ._generate_mpo import generate_mpo, Hterm
from ._initialize import product_mpo
from ...initialize import eye
from ...tensor import YastnError, Tensor, qr, swap_charges, sign_canonical_order, tensordot


def vdot(*args) -> Number:
    r"""
    Calculate the overlap :math:`\langle \textrm{bra}|\textrm{ket}\rangle`,
    or :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle` depending on the number of provided arguments.

    Parameters
    -----------
    *args: yastn.tn.mps.MpsMpoOBC
    """
    if len(args) == 2:
        return measure_overlap(*args)
    return measure_mpo(*args)


def measure_overlap(bra, ket) -> Number:
    r"""
    Calculate overlap :math:`\langle \textrm{bra}|\textrm{ket} \rangle`.
    Conjugate of MPS :code:`bra` is computed internally.

    MPSs :code:`bra` and :code:`ket` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    ket: yastn.tn.mps.MpsMpoOBC
    """
    env = Env(bra, ket)
    return env.measure(bd=(-1, env.N))


def measure_mpo(bra, op: MpsMpoOBC | Sequence[tuple[MpsMpoOBC, Number]], ket) -> Number:
    r"""
    Calculate expectation value :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle`.

    Conjugate of MPS :code:`bra` is computed internally.
    MPSs :code:`bra`, :code:`ket`, and MPO :code:`op` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    op: yastn.tn.mps.MpsMpoOBC or Sequence[tuple(MpsMpoOBC,number)]
        Operator written as (sums of) MPO.

    ket: yastn.tn.mps.MpsMpoOBC
    """
    env = Env(bra, [op, ket])
    return env.measure(bd=(-1, env.N))


def measure_1site(bra, O, ket, sites=None) -> dict[int, Number]:
    r"""
    Calculate expectation values :math:`\langle \textrm{bra}|\textrm{O}_i|\textrm{ket} \rangle` for local operator :code:`O` at sites `i`.

    ``O`` can be provided as a dictionary {site: operator}, limiting the calculation to provided sites.
    ``sites`` can also be provided in the form of a list.

    Conjugate of MPS :code:`bra` is computed internally.
    For fermionic operators, a Jordan-Wigner string related to the operator charge is included in the contraction.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    O: yastn.Tensor | dict[int, yastn.Tensor]
        A rank-2 operator, or a dictionary of such operators {site: operator}.
        In the second case, all operators need to have the same charge.

    ket: yastn.tn.mps.MpsMpoOBC

    sites: int | Sequence[int] | None
        It controls which 1-sites observables to calculate.
        If *int* is provided here, compute the expectation value
        for this single site and return a float.
        In other cases, a dictionary {site: float} is returned.
        The default is None, in which case the calculation is done for all sites.
    """
    return_float = False
    if sites is None:
        sites = list(range(ket.N))
    elif isinstance(sites, int):  # single site
        sites = [sites]
        return_float = True
    else:
        sites = sorted(set(sites) & set(range(ket.N)))

    if isinstance(O, dict):
        op = {k: O[k] for k in sites if k in O}
        if len(op) == 0:
            return {}
        O0 = next(iter(op.values()))
        if any(O0.n != x.n for x in op.values()):
            raise YastnError("In mps.measure_1site, all operators in O should have the same charge.")
    else:
        op = {k: O for k in sites}
        O0 = O

    n_left = O0.config.sym.add_charges(O0.n, new_signature=-1)
    env = Env2(bra, ket, n_left=n_left)
    env.setup_(to='first').setup_(to='last')

    results = {}
    for n, o in op.items():
        env.update_env_op_(n, o, to='first')
        results[n] = env.measure(bd=(n - 1, n))

    return results.popitem()[1] if return_float else results


def measure_2site(bra, O, P, ket, bonds='<') -> dict[tuple[int, int], float] | float:
    r"""
    Calculate expectation values :math:`\langle \textrm{bra}|\textrm{O}_i \textrm{P}_j|\textrm{ket} \rangle`
    of local operators :code:`O` and :code:`P` for pairs of lattice sites :math:`i, j`.

    Conjugate of MPS :code:`bra` is computed internally.
    Fermionic strings are incorporated for fermionic operators by employing :meth:`yastn.swap_gate`.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    O, P: yastn.Tensor or dict
        Rank-two operators with signature (1, -1), or a compatible pair of
        rank-three multiplets whose auxiliary legs are contracted to a singlet.
        Each can also be a dictionary {site: operator} with all operators of the same charge.

    ket: yastn.tn.mps.MpsMpoOBC

    bonds: tuple[int, int] | Sequence[tuple[int, int]] | str
        Which 2-site correlators to calculate.
        For a single bond, tuple[int, int], return float. Otherwise, return dict[bond, float].
        It is possible to provide a string to build a list of bonds as:

        * '<' for all i < j.
        * '=' for all i == j.
        * '>' for all i > j.
        * 'a' for all i, j; equivalent to "<=>".
        * 'rx' for all i, i+x with OBC, e.g. "r1" for nearest-neighbours; x can be negative.
        * 'p' to include PBC terms in 'rx'.

        The default is '<'.
    """
    return_float = False
    if isinstance(bonds, str):
        pairs = _parse_2site_bonds(bonds, ket.N)
    elif isinstance(bonds[0], int):  # single bond
        pairs = [bonds]
        return_float = True
    else:
        pairs = bonds

    if isinstance(O, dict):
        O0 = next(iter(O.values()))
        if any(O0.n != x.n for x in O.values()):
            raise YastnError("In mps.measure_2site, all operators in O should have the same charge.")
    else:  # is a tensor
        O0 = O
        O = {k: O for k in range(ket.N)}

    if isinstance(P, dict):
        P0 = next(iter(P.values()))
        if any(P0.n != x.n for x in P.values()):
            raise YastnError("In mps.measure_2site, all operators in P should have the same charge.")
    else: # is a tensor
        P0 = P
        P = {k: P for k in range(ket.N)}

    # Group only by the representation carried between endpoints and scan
    # direction. Site-dependent operator values share the same environments.
    # Rank-two operators retain the cheaper Env2 contractions; multiplets use
    # an MPO bond carrying the irrep. Both follow exactly the same sweep below.
    groups = {}
    for i, j in pairs:
        if i not in O or j not in P:
            continue
        legs = tuple(op.get_legs(2) if _is_multiplet(op) else None for op in (O[i], P[j]))
        groups.setdefault((*legs, i > j), []).append((i, j))

    envs, results = {}, {}
    for (leg0, leg1, reverse), group in groups.items():
        multiplet = leg0 is not None or leg1 is not None
        left, right = (P, O) if reverse else (O, P)
        ordered = [(j, i) if reverse else (i, j) for i, j in group]
        left = {i: left[i] for i, _ in ordered}
        right = {j: right[j] for _, j in ordered}
        L, R = next(iter(left.values())), next(iter(right.values()))
        if multiplet:
            A, _ = _multiplet_pair(L, R)
            charge = A.get_legs(2).t[0]
            sign = swap_charges([charge], [charge], ket.config.fermionic) if reverse else 1
            left = {i: _multiplet_pair(op, R)[0] for i, op in left.items()}
            right = {j: _multiplet_pair(L, op)[1] for j, op in right.items()}
            onsite = {i: tensordot(left[i], right[i], axes=((1, 2), (0, 2)))
                      for i, j in ordered if i == j}
            if True not in envs:
                identities = [eye(ket.config, legs=ket[n].get_legs(1), isdiag=False) for n in ket.sweep()]
                identity = product_mpo(identities)
                envs[True] = Env(bra, [identity, ket]).setup_(to='first').setup_(to='last')
            env = envs[True].shallow_copy()
            env.op = identity.shallow_copy()
            # An open irrep crosses physical inputs between the endpoints.
            # This is the same MPO convention used by generate_mpo: the left
            # endpoint includes the parity swap, the right endpoint does not.
            wire = eye(ket.config, legs=(A.get_legs(2).conj(), A.get_legs(2)), isdiag=False)
            for n, local in enumerate(identities):
                env.op[n] = tensordot(local, wire, axes=((), ())).transpose((2, 0, 3, 1)).swap_gate(axes=3, charge=charge)
            left = {i: op.add_leg(axis=0, s=-1).transpose((0, 1, 3, 2)).swap_gate(axes=3, charge=charge)
                    for i, op in left.items()}
            right = {j: op.add_leg(axis=2, s=1).transpose((3, 0, 2, 1)) for j, op in right.items()}
            onsite = {i: op.add_leg(axis=0, s=-1).add_leg(axis=2, s=1) for i, op in onsite.items()}
        else:
            if False not in envs:
                n_left = O0.config.sym.add_charges(O0.n, P0.n, new_signature=-1)
                envs[False] = Env2(bra, ket, n_left=n_left).setup_(to='first').setup_(to='last')
            env = envs[False].shallow_copy()
            sign = swap_charges([L.n], [R.n], ket.config.fermionic) if reverse else 1
            onsite = {i: left[i] @ right[i] for i, j in ordered if i == j}
        values = _measure_2site_sweep(env, left, right, onsite, ordered)
        for (i, j), value in values.items():
            results[(j, i) if reverse else (i, j)] = sign * value
    return results.popitem()[1] if return_float and results else results


def _measure_2site_sweep(env, left, right, onsite, pairs):
    """Cache each right endpoint, then propagate each left endpoint once.

    Descending source order preserves the bare left environments needed by
    the next source. Right endpoints are prepared in ascending order so that
    each starts from the bare environment on its right. In-between updates
    transport either the Env2 charge or the explicit MPO irrep, using the
    environment's own contraction kernels. No MPO is rebuilt per pair.
    """
    results = {}
    # On-site products must use the bare environments, before either side
    # acquires a separated endpoint.
    local = env.shallow_copy()
    for i in sorted(onsite):
        local.update_env_op_(i, onsite[i], to='first')
        results[i, i] = local.measure(bd=(i - 1, i))
    separated = [(i, j) for i, j in pairs if i < j]
    for j in sorted({j for _, j in separated}):
        env.update_env_op_(j, right[j], to='first')
    for i, targets in groupby(sorted(separated, key=lambda pair: (-pair[0], pair[1])), key=lambda pair: pair[0]):
        env.update_env_op_(i, left[i], to='last')
        n = i
        for _, j in targets:
            while n + 1 < j:
                n += 1
                env.update_env_(n, to='last')
            results[i, j] = env.measure(bd=(n, j))
    return results


def _parse_2site_bonds(bonds, N):
    if 'a' in bonds:
        return [(i, j) for i in range(N) for j in range(N)]
    pairs = []
    if '<' in bonds:
        pairs += [(i, j) for i in range(N) for j in range(i + 1, N)]
        bonds = bonds.replace('<', '')
    if '=' in bonds:
        pairs += [(i, i) for i in range(N)]
        bonds = bonds.replace('=', '')
    if '>' in bonds:
        pairs += [(i, j) for i in range(N) for j in range(i)]
        bonds = bonds.replace('>', '')
    pbc = False
    if 'p' in bonds:
        pbc = True
        bonds = bonds.replace('p', '')
    if 'r' in bonds:  # only "r" are left
        for r in bonds.split('r')[1:]:
            r = int(r)
            if pbc:
                pairs += [(i, (i + r) % N) for i in range(N)]
            else:
                pairs += [(i, (i + r)) for i in range(N) if 0 <= i + r < N]
    return sorted(set(pairs))


def measure_nsite(bra, *operators, ket, sites=None) -> float:
    r"""
    Calculate expectation value of a product of local operators.

    Conjugate of MPS ``bra`` is computed internally.
    Fermionic strings are incorporated for fermionic operators by employing :meth:`yastn.swap_gate`.

    Parameters
    ----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    operators: Sequence[yastn.Tensor]
        List of local operators to calculate <O0_s0 O1_s1 ...>.

    ket: yastn.tn.mps.MpsMpoOBC
        Should be provided as \*\*kwargs, as operators are given as \*args.

    sites: Sequence[int]
        A list of sites [s0, s1, ...] matching corresponding operators.
    """
    if sites is None or len(operators) != len(sites):
        raise YastnError("Number of operators and sites should match.")
    if any(_is_multiplet(op) for op in operators):
        identities = [eye(ket.config, legs=ket[n].get_legs(1), isdiag=False) for n in ket.sweep()]
        # generate_mpo includes endpoint-exchange signs and Jordan-Wigner
        # swaps for each multiplet pair, and preserves the order of products
        # of pairs (including overlapping sites). Their invariant total n=0
        # is NOT their fermionic parity. Applying sign_canonical_order here
        # would either miss that parity or count the exchange sign twice.
        mpo = generate_mpo(identities, [Hterm(1, sites, operators)])
        env = Env(bra, [mpo, ket])
        for n in bra.sweep(to='last'):
            env.update_env_(n, to='last')
        return env.measure(bd=(bra.N - 1, bra.N))

    sign = sign_canonical_order(*operators, sites=sites, f_ordered=lambda s0, s1: s0 <= s1)
    n_left = bra.config.sym.add_charges(*(op.n for op in operators), new_signature=-1)

    ops = {}
    for n, op in zip(sites, operators):
        ops[n] = ops[n] @ op if n in ops else op

    env = Env2(bra, ket, n_left=n_left)

    for n in bra.sweep(to='last'):
        if n in ops:
            env.update_env_op_(n, ops[n], to='last')
        else:
            env.update_env_(n, to='last')
    return sign * env.measure(bd=(bra.N - 1, bra.N))


def sample(psi, projectors, number=1, return_probabilities=False) -> np.ndarray[int] | tuple[np.ndarray[int], Sequence[float]]:
    r"""
    Sample random configurations from an MPS psi.

    Probabilities follow from :math:`|psi \rangle\langle psi|`. Works also for purification.
    Output samples as numpy array of integers, where samples[k, n] give a projector's index in the k-th sample on the n-th MPS site.

    It does not check whether projectors sum up to identity; calculated probabilities of provided projectors are normalized to one.

    Parameters
    ----------
    projectors: Dict[Any, yast.Tensor] | Sequence[yast.Tensor] | Dict[Site, Dict[Any, yast.Tensor]]
        Local vector states (or projectors) to sample from.
        Their orthogonality or local basis completeness is not checked (normalization is checked).
        We can provide a dict(key: projector), where the same set of projectors is used at each site.
        The keys should be integers, to fit into the output samples array.
        Projectors can also be provided as a list, and then the keys follow from enumeration.
        Finally, we can provide a dictionary between each site and sets of projectors (to have different projections at various sites).

    number: int
        Number of drawn samples.

    return_probabilities: bool
        Whether to also return probability to find each sample in the state.
        If ``True``, return: samples, probabilities.
        If ``False`` (the default), return: samples.

    Note
    ----
    Depending on the basis, sampling might break the symmetry of the state psi.
    In this case, psi and local states/projectors should first be cast down to dense representation.
    It is important to make sure that the local basis ordering between state sites and projectors/vectors is maintained

    Example
    -------

    ::

        ops = yastn.operators.SpinlessFermions(sym='U1')
        I = mps.product_mpo(ops.I(), N=8)
        psi = mps.random_mps(I, n=4, D_total=8)
        # random state with 8 sites and 4 particles.

        projectors = [ops.vec_n(0), ops.vec_n(1)]  # empty and full local states
        samples = mps.sample(psi, projectors, number=15)

    """
    if not psi.is_canonical(to='first'):
        psi = psi.shallow_copy()
        psi.canonize_(to='first')

    sites = list(range(psi.N))
    if not isinstance(projectors, dict) or all(isinstance(x, Tensor) for x in projectors.values()):
        projectors = {site: projectors for site in sites}  # spread projectors over sites
    if set(sites) != set(projectors.keys()):
        raise YastnError(f"Projectors not defined for some sites.")

    # change each list of projectors into keys and projectors
    projs_sites = {}
    for k, v in projectors.items():
        if isinstance(v, dict):
            projs_sites[k, 'k'] = list(v.keys())
            if not all(isinstance(k, int) for k in projs_sites[k, 'k']):
                raise YastnError("Use integer numbers for projector keys.")
            projs_sites[k, 'p'] = list(v.values())
        else:
            projs_sites[k, 'k'] = list(range(len(v)))
            projs_sites[k, 'p'] = list(v)

        for j, pr in enumerate(projs_sites[k, 'p']):
            if pr.ndim == 1:  # vectors need conjugation
                if abs(pr.norm() - 1) > 1e-10:
                    raise YastnError("Local states to project on should be normalized.")
                projs_sites[k, 'p'][j] = pr.conj()
            elif pr.ndim == 2:
                if (pr.n != pr.config.sym.zero()) or abs(pr @ pr - pr).norm() > 1e-10:
                    raise YastnError("Matrix projectors should be projectors, P @ P == P.")
            else:
                raise YastnError("Projectors should consist of vectors with ndim=1 or matrices with ndim=2.")

    samples = np.zeros((number, psi.N), dtype=np.int64)
    probabilities = np.ones(number, dtype=np.float64)

    leg = psi.virtual_leg('first')
    tmp = eye(psi.config, legs=[leg, leg.conj()])
    bdrs = [tmp for _ in range(number)]

    for n in sites:
        rands = psi.config.backend.rand(number)  # in [0, 1]
        An = psi[n] if psi.nr_phys == 1 else psi[n].fuse_legs(axes=(0, 1, (2, 3)))
        for k, (bdr, cut) in enumerate(zip(bdrs, rands)):
            state = bdr @ An
            state = state.fuse_legs(axes=(1, (0, 2)))
            prob, pstates = [], []
            for proj in projs_sites[n, 'p']:
                pst = proj @ state
                pstates.append(pst)
                prob.append(pst.vdot(pst).item())
            norm_prob = sum(prob)
            prob = [x / norm_prob for x in prob]
            ind = sum(apr < cut for apr in accumulate(prob))
            proj = projs_sites[n, 'p'][ind]
            samples[k, n] = projs_sites[n, 'k'][ind]
            probabilities[k] *= prob[ind]
            tmp = pstates[ind] / pstates[ind].norm()
            tmp = tmp.unfuse_legs(axes=tmp.ndim-1)
            if psi.nr_phys == 1:
                axes = (0, 1) if tmp.ndim == 2 else ((0, 1), 2)
            else:  # psi.nr_phys == 1
                tmp = tmp.unfuse_legs(axes=tmp.ndim-1)
                axes = ((0, 2), 1) if tmp.ndim == 3 else ((0, 1, 3), 2)
            _, bdrs[k] = qr(tmp, axes=axes)
    if return_probabilities:
        return samples, probabilities
    return samples


def rdm(psi, *sites):
    """
    Reduced density matrix supported on selected sites for MPS psi.
    """
    ni, nf = min(sites), max(sites)
    #
    if len(sites) != len(set(sites)) or ni < psi.first or nf > psi.last:
        raise YastnError("Repeated site or some sites outside of psi.")
    #
    env = Env2(psi, psi)
    #
    env.setup_(to=(ni - 1, nf + 1))
    FL = env.F[ni - 1, ni]
    FR = env.F[nf + 1, nf]
    ii = 0
    for n in range(ni, nf + 1):
        An = psi[n]
        if psi.nr_phys == 2:
            An = An.swap_gate(axes=(0, 3))

        FL = tensordot(FL, An.conj(), axes=(ii, 0))
        if n in sites:
            axes = (ii, 0) if psi.nr_phys == 1 else ((ii, ii + 3), (0, 3))
            FL = tensordot(FL, An, axes=axes)
            FL = FL.swap_gate(axes=((ii, ii + 2), ii + 3))
            FL = FL.swap_gate(axes=(tuple(range(ii)), ii + 2))
            FL = FL.moveaxis(source=(ii + 2, ii), destination=(ii, ii + 1))
            ii += 2
        else:
            axes = ((ii, ii + 1), (0, 1)) if psi.nr_phys == 1 else ((ii, ii + 1, ii + 3), (0, 1, 3))
            FL = tensordot(FL, An, axes=axes)
    rho = tensordot(FL, FR, axes=((ii, ii + 1), (1, 0)))

    for ii, st in enumerate(sites):
        nd = sum(st > i for i in sites[ii+1:])
        axes_sw = (2 * (ii + nd), tuple(2 * i for i in range(ii, ii + nd)),
                   2 * (ii + nd) + 1, tuple(2 * i + 1 for i in range(ii, ii + nd)))
        rho = rho.swap_gate(axes=axes_sw)
        rho = rho.moveaxis(source=(2 * (ii + nd), 2 * (ii + nd) + 1), destination=(2 * ii, 2 * ii + 1))

    return rho
