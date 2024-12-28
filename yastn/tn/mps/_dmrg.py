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
""" Various variants of the DMRG algorithm for Mps."""
from __future__ import annotations
from typing import NamedTuple, Sequence
import logging
from ... import eigs, YastnError
from ._measure import Env
from ._env import Env_sum, Env_project
from . import MpsMpoOBC


logger = logging.Logger('dmrg')


#################################
#           dmrg                #
#################################

class DMRG_out(NamedTuple):
    sweeps: int = 0
    method: str = ''
    energy: float = None
    denergy: float = None
    max_dSchmidt: float = None
    max_discarded_weight: float = None


def dmrg_(psi, H, project=None, method='1site',
        energy_tol=None, Schmidt_tol=None, max_sweeps=1, iterator_step=None,
        opts_eigs=None, opts_svd=None, precompute=False):
    r"""
    Perform DMRG sweeps until convergence, starting from MPS ``psi``.

    The inner loop sweeps over MPS updating sites from the first site to the last and back, constituting a single iteration.
    Convergence can be controlled based on energy and/or Schmidt values (which is a more sensitive measure of convergence).
    The DMRG algorithm sweeps through the lattice at most ``max_sweeps`` times
    or until all convergence measures (with provided tolerance other than the default ``None``) change by less than the provided tolerance during a single sweep.

    Outputs iterator if ``iterator_step`` is given, which allows
    inspecting ``psi`` outside of ``dmrg_`` function after every ``iterator_step`` sweeps.

    Parameters
    ----------
    psi: yastn.tn.mps.MpsMpoOBC
        Initial state. It is updated during execution.
        If ``psi`` is not already canonized to the first site, it will be canonized at the start of the algorithm.
        The output state from ``dmrg_`` is canonized to the first site.

    H: yastn.tn.mps.MpsMpoOBC | Sequence
        MPO (or a sum of MPOs) to minimize against, see :meth:`Env()<yastn.tn.mps.Env>`.

    project: Sequence[yastn.tn.mps.MpsMpoOBC | tuple[float, yastn.tn.mps.MpsMpoOBC]]
        Add a penalty to the directions spanned by MPSs in the list.
        In practice, the penalty is a multiplicative factor that adds a penalty term to the Hamiltonian.
        As a result, the energy of said MPS rizes and another lowest-energy is targeted by energy minimization.
        It can be used to find a few low-energy states of the Hamiltonian
        if the penalty is larger than the energy gap from the ground state.
        Use ``[(penalty, MPS), ...]`` to provide individual penalty for
        each MPS by hand as a list of tuples ``(penalty, MPS)``, where ``penalty`` is a number and ``MPS`` is an MPS object.
        If input is a list of MPSs, i.e., ``[mps, ...]``, the option uses default ``penalty=100``.

    method: str | yastn.Method
        DMRG variant to use; options are ``'1site'`` or ``'2site'``.
        Auxlliary class :class:`yastn.Method` can be used to change the method in between sweeps while the yield gets called after every ``iterator_step`` sweeps.

    energy_tol: float
        Convergence tolerance for the change of energy in a single sweep.
        By default is None, in which case energy convergence is not checked.

    Schmidt_tol: float
        Convergence tolerance for the change of Schmidt values on the worst cut/bond in a single sweep.
        By default is None, in which case Schmidt values convergence is not checked.

    max_sweeps: int
        Maximal number of sweeps.

    iterator_step: int
        If int, ``dmrg_`` returns a generator that would yield output after every iterator_step sweeps.
        The default is None, in which case  ``dmrg_`` sweeps are performed immediately.

    opts_eigs: dict
        options passed to :meth:`yastn.eigs`.
        If None, use default {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    opts_svd: dict
        Options passed to :meth:`yastn.linalg.svd_with_truncation` used to truncate virtual spaces (virtual bond dimension of tensors) in ``method='2site'``.

    precompute: bool
        Controls MPS-MPO-MPS contraction order.
        If ``False``, use an approach optimal for a single matrix-vector product calculation during iterative Krylov-space building,
        scaling as O(D^3 M d + D^2 M^2 d^2).
        If ``True``, uses less optimal contraction order, scaling as O(D^3 M d^2 + D^2 M^2 d^2).
        However, the latter allows precomputing and reusing part of the diagram during consecutive iterations.
        Which one is more efficient depends on the parameters. The default is ``False``.

    Returns
    -------
    Generator if iterator_step is not None.

    DMRG_out(NamedTuple)
        NamedTuple including fields:

            * ``sweeps`` number of performed dmrg sweeps.
            * ``method`` method used in the last sweep.
            * ``energy`` energy after the last sweep.
            * ``denergy`` absolut value of energy change in the last sweep.
            * ``max_dSchmidt`` norm of Schmidt values change on the worst cut in the last sweep.
            * ``max_discarded_weight`` norm of discarded_weights on the worst cut in '2site' procedure.
    """
    tmp = _dmrg_(psi, H, project, method,
                energy_tol, Schmidt_tol, max_sweeps, iterator_step,
                opts_eigs, opts_svd, precompute)
    return tmp if iterator_step else next(tmp)


def _dmrg_(psi, H : MpsMpoOBC | Sequence[tuple[MpsMpoOBC, float]], project, method,
        energy_tol, Schmidt_tol, max_sweeps, iterator_step,
        opts_eigs, opts_svd, precompute):
    """ Generator for dmrg_(). """

    if not psi.is_canonical(to='first'):
        psi.canonize_(to='first')

    env = Env(psi, [H, psi], precompute=precompute)
    if project:
        if not isinstance(env, Env_sum):
            env = Env_sum([env])
        for pr in project:
            penalty, st = (100, pr) if isinstance(pr, MpsMpoOBC) else pr
            env.envs.append(Env_project(psi, st, penalty))
    env.setup_(to='first')

    E_old = env.measure().item().real

    if opts_eigs is None:
        opts_eigs = {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    if Schmidt_tol is not None:
        if not Schmidt_tol > 0:
            raise YastnError('DMRG: Schmidt_tol has to be positive or None.')
        Schmidt_old = psi.get_Schmidt_values()
        Schmidt_old = {(n-1, n): sv for n, sv in enumerate(Schmidt_old)}

    max_dS = None
    Schmidt = None if Schmidt_tol is None else {}

    if energy_tol is not None and not energy_tol > 0:
        raise YastnError('DMRG: energy_tol has to be positive or None.')

    if method not in ('1site', '2site'):
        raise YastnError('DMRG: dmrg method %s not recognized.' % method)

    if opts_svd is None and method == '2site':
        raise YastnError("DMRG: provide opts_svd for %s method." % method)

    for sweep in range(1, max_sweeps + 1):
        if method == '1site':
            _dmrg_sweep_1site_(env, opts_eigs=opts_eigs, Schmidt=Schmidt, precompute=precompute)
            max_dw = None
        else: # method == '2site':
            max_dw = _dmrg_sweep_2site_(env, opts_eigs=opts_eigs,
                                        opts_svd=opts_svd, Schmidt=Schmidt, precompute=precompute)

        E = env.measure().item().real
        dE, E_old = abs(E_old - E), E
        converged = []

        if energy_tol is not None:
            converged.append(abs(dE) < energy_tol)

        if Schmidt_tol is not None:
            max_dS = max((Schmidt[k] - Schmidt_old[k]).norm().item() for k in Schmidt.keys())
            Schmidt_old = Schmidt.copy()
            converged.append(max_dS < Schmidt_tol)

        logger.info(f'Sweep = {sweep:03d}  energy = {E:0.14f}  dE = {dE:0.4f}  dSchmidt = {max_dS}')

        if len(converged) > 0 and all(converged):
            break
        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield DMRG_out(sweep, str(method), E, dE, max_dS, max_dw)
    yield DMRG_out(sweep, str(method), E, dE, max_dS, max_dw)


def _dmrg_sweep_1site_(env, opts_eigs=None, Schmidt=None, precompute=False):
    r"""
    Perform sweep with 1-site DMRG, see :meth:`dmrg` for description.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """
    psi = env.bra
    for to in ('last', 'first'):
        for n in psi.sweep(to=to):
            A = psi.A[n]
            if precompute and env.nr_phys == 1:
                A = A.fuse_legs(axes=(0, (1, 2)))
            _, (A,) = eigs(lambda x: env.Heff1(x, n), A, k=1, **opts_eigs)
            if precompute and env.nr_phys == 1:
                A = A.unfuse_legs(axes=1)
            psi.A[n] = A
            psi.orthogonalize_site_(n, to=to, normalize=True)
            if Schmidt is not None and to == 'first' and n != psi.first:
                Schmidt[psi.pC] = psi[psi.pC].svd(sU=1, compute_uv=False)
            psi.absorb_central_(to=to)
            env.clear_site_(n)
            env.update_env_(n, to=to)


def _dmrg_sweep_2site_(env, opts_eigs=None, opts_svd=None, Schmidt=None, precompute=False):
    r"""
    Perform sweep with 2-site DMRG, see :meth:`dmrg` for description.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """
    psi = env.bra

    max_disc_weight = -1.
    for to, dn in (('last', 0), ('first', 1)):
        for n in psi.sweep(to=to, dl=1):
            bd = (n, n + 1)
            AA = psi.pre_2site(bd)
            if precompute and env.nr_phys == 1:
                AA = AA.fuse_legs(axes=((0, 1), (2, 3)))
            _, (AA,) = eigs(lambda v: env.Heff2(v, bd), AA, k=1, **opts_eigs)
            _disc_weight_bd = psi.post_2site_(AA, bd, opts_svd)
            max_disc_weight = max(max_disc_weight, _disc_weight_bd)
            if Schmidt is not None and to == 'first':
                Schmidt[psi.pC] = psi[psi.pC]
            psi.absorb_central_(to=to)
            env.clear_site_(n, n + 1)
            env.update_env_(n + dn, to=to)
    env.update_env_(psi.first, to='first')
    return max_disc_weight
