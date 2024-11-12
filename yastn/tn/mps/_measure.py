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
from itertools import groupby
from typing import Sequence
from ... import YastnError
from . import MpsMpoOBC
from ._env import Env, Env2
from ...operators import swap_charges


def vdot(*args) -> number:
    r"""
    Calculate the overlap :math:`\langle \textrm{bra}|\textrm{ket}\rangle`,
    or :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle` depending on the number of provided agruments.

    Parameters
    -----------
    *args: yastn.tn.mps.MpsMpoOBC
    """
    if len(args) == 2:
        return measure_overlap(*args)
    return measure_mpo(*args)


def measure_overlap(bra, ket) -> number:
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
    env.setup_(to='first')
    return env.measure(bd=(-1, 0))


def measure_mpo(bra, op: MpsMpoOBC | Sequence[tuple(MpsMpoOBC, number)], ket) -> number:
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
    env.setup_(to='first')
    return env.measure(bd=(-1, 0))


def measure_1site(bra, O, ket, sites=None) -> dict[int, number]:
    r"""
    Calculate expectation values :math:`\langle \textrm{bra}|\textrm{O}_i|\textrm{ket} \rangle` for local operator :code:`O` at sites `i`.

    Local operators can be provided as dictionary {site: operator}, limiting the calculation to provided sites.
    A list of sites can also be provided.

    Conjugate of MPS :code:`bra` is computed internally.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    O: yastn.Tensor or dict
        An operator with signature (1, -1).
        It is possible to provide a dictionary {site: operator} with all operators of the same charge.

    ket: yastn.tn.mps.MpsMpoOBC

    sites: int | Sequence[int] | None
        Which 1-sites observables to calculate.
        For a single site, int, return float; otherwise return dict[site, float]
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
    Includes fermionic strings via swap_gate for fermionic operators.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    O, P: yastn.Tensor or dict
        Operators with signature (1, -1).
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

    n_left = O0.config.sym.add_charges(O0.n, P0.n, new_signature=-1)

    pairs = [(n0, n1) for n0, n1 in pairs if (n0 in O and n1 in P)]
    s0s1 = [pair for pair in pairs if pair[0] < pair[1]]
    s1s0 = [pair[::-1] for pair in pairs if pair[0] > pair[1]]
    s0s0 = sorted(pair[0] for pair in pairs if pair[0] == pair[1])

    s1s = sorted(set(pair[1] for pair in s0s1))
    s0s1 = sorted(s0s1, key=lambda x: (-x[0], x[1]))

    s0s = sorted(set(pair[1] for pair in s1s0))
    s1s0 = sorted(s1s0, key=lambda x: (-x[0], x[1]))

    env0 = Env2(bra, ket, n_left=n_left)
    env0.setup_(to='first').setup_(to='last')
    results = {}

    # here <O P> in desired order
    env = env0.shallow_copy()
    for n1 in s1s:
        env.update_env_op_(n1, P[n1], to='first')
    for n0, n01s in groupby(s0s1, key=lambda x: x[0]):
        env.update_env_op_(n0, O[n0], to='last')
        n = n0
        for _, n1 in n01s:
            while n + 1 < n1:
                n += 1
                env.update_env_(n, to='last')
            results[(n0, n1)] = env.measure(bd=(n, n1))

    # here <P O>, and we need to correct by the sign derived from swapping the operators.
    sign = swap_charges([O0.n], [P0.n], O0.config.fermionic)
    env = env0.shallow_copy()
    for n1 in s0s:
        env.update_env_op_(n1, O[n1], to='first')
    for n0, n01s in groupby(s1s0, key=lambda x: x[0]):
        env.update_env_op_(n0, P[n0], to='last')
        n = n0
        for _, n1 in n01s:
            while n + 1 < n1:
                n += 1
                env.update_env_(n, to='last')
            results[(n1, n0)] = sign * env.measure(bd=(n, n1))

    env = env0.shallow_copy()
    for n0 in s0s0:
        env.update_env_op_(n0, O[n0] @ P[n0], to='first')
        results[(n0, n0)] = env.measure(bd=(n0 - 1, n0))

    return results.popitem()[1] if return_float and len(results) > 0 else results


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
