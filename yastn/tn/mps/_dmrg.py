""" Various variants of the DMRG algorithm for mps."""
from __future__ import annotations
from typing import NamedTuple, Sequence
import logging
from ... import eigs, YastnError
from ._env import Env
from . import MpsMpoOBC


logger = logging.Logger('dmrg')


#################################
#           dmrg                #
#################################

class DMRG_out(NamedTuple):
    sweeps: int = 0
    energy: float = None
    denergy: float = None
    max_dSchmidt: float = None
    max_discarded_weight: float = None


def dmrg_(psi, H : MpsMpoOBC | Sequence[tuple[MpsMpoOBC, float]], project=None, method='1site',
        energy_tol=None, Schmidt_tol=None, max_sweeps=1, iterator_step=None,
        opts_eigs=None, opts_svd=None):
    r"""
    Perform DMRG sweeps until convergence, starting from MPS :code:`psi`.

    The inner loop sweeps over MPS updating sites from the first site to the last and back, constituting a single iteration.
    Convergence can be controlled based on energy and/or Schmidt values (which is a more sensitive measure of convergence).
    The DMRG algorithm sweeps through the lattice at most :code:`max_sweeps` times
    or until all convergence measures (with provided tolerance other than None) change by less than the provided tolerance during a single sweep.

    Outputs iterator if :code:`iterator_step` is given, which allows
    inspecting :code:`psi` outside of :code:`dmrg_` function after every :code:`iterator_step` sweeps.

    Parameters
    ----------
    psi: yastn.tn.mps.MpsMpoOBC
        Initial state. It is updated during execution.
        It is first canonized to the first site, if not provided in such a form.
        State resulting from :code:`dmrg_` is canonized to the first site.

    H: yastn.tn.mps.MpsMpoOBC or Sequence[tuple(MpsMpoOBC,number)]
        MPO to minimize against.

    project: list(yastn.tn.mps.MpsMpoOBC)
        Optimizes MPS in the subspace orthogonal to MPS's in the list.

    method: str
        Which DMRG variant to use from '1site', '2site'

    energy_tol: float
        Convergence tolerance for the change of energy in a single sweep.
        By default is None, in which case energy convergence is not checked.

    Schmidt_tol: float
        Convergence tolerance for the change of Schmidt values on the worst cut/bond in a single sweep.
        By default is None, in which case Schmidt values convergence is not checked.

    max_sweeps: int
        Maximal number of sweeps.

    iterator_step: int
        If int, :code:`dmrg_` returns a generator that would yield output after every iterator_step sweeps.
        Default is None, in which case  :code:`dmrg_` sweeps are performed immediately.

    opts_eigs: dict
        options passed to :meth:`yastn.eigs`.
        If None, use default {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    opts_svd: dict
        Options passed to :meth:`yastn.linalg.svd_with_truncation` used to truncate virtual spaces in method='2site'.
        If None, use default {'tol': 1e-13}

    Returns
    -------
    DMRG_out(NamedTuple)
        NamedTuple including fields:

            * :code:`sweeps` number of performed dmrg sweeps.
            * :code:`energy` energy after the last sweep.
            * :code:`denergy` absolut value of energy change in the last sweep.
            * :code:`max_dSchmidt` norm of Schmidt values change on the worst cut in the last sweep.
            * :code:`max_discarded_weight` norm of discarded_weights on the worst cut in '2site' procedure.
    """
    tmp = _dmrg_(psi, H, project, method,
                energy_tol, Schmidt_tol, max_sweeps, iterator_step,
                opts_eigs, opts_svd)
    return tmp if iterator_step else next(tmp)


def _dmrg_(psi, H : MpsMpoOBC | Sequence[tuple(MpsMpoOBC,number)], project, method,
        energy_tol, Schmidt_tol, max_sweeps, iterator_step,
        opts_eigs, opts_svd):
    """ Generator for dmrg_(). """

    if not psi.is_canonical(to='first'):
        psi.canonize_(to='first')

    env = Env(bra=psi, op=H, ket=psi, project=project).setup_(to='first')
    E_old = env.measure()

    if opts_eigs is None:
        opts_eigs = {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    if Schmidt_tol is not None:
        if not Schmidt_tol > 0:
            raise YastnError('DMRG: Schmidt_tol has to be positive or None.')
        Schmidt_old = psi.get_Schmidt_values()
        Schmidt_old = {(n-1, n): sv for n, sv in enumerate(Schmidt_old)}

    max_dS, max_dw = None, None
    Schmidt = None if Schmidt_tol is None else {}

    if energy_tol is not None and not energy_tol > 0:
        raise YastnError('DMRG: energy_tol has to be positive or None.')

    if method not in ('1site', '2site'):
        raise YastnError('DMRG: dmrg method %s not recognized.' % method)

    for sweep in range(1, max_sweeps + 1):
        if method == '1site':
            _dmrg_sweep_1site_(env, opts_eigs=opts_eigs, Schmidt=Schmidt)
        else: # method == '2site':
            max_dw = _dmrg_sweep_2site_(env, opts_eigs=opts_eigs,
                                        opts_svd=opts_svd, Schmidt=Schmidt)

        E = env.measure()
        dE, E_old = E_old - E, E
        converged = []

        if energy_tol is not None:
            converged.append(abs(dE) < energy_tol)

        if Schmidt_tol is not None:
            max_dS = max((Schmidt[k] - Schmidt_old[k]).norm() for k in Schmidt.keys())
            Schmidt_old = Schmidt.copy()
            converged.append(max_dS < Schmidt_tol)

        logger.info('Sweep = %03d  energy = %0.14f  dE = %0.4f  dSchmidt = %0.4f', sweep, E, dE, max_dS)

        if len(converged) > 0 and all(converged):
            break
        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield DMRG_out(sweep, E, dE, max_dS, max_dw)
    yield DMRG_out(sweep, E, dE, max_dS, max_dw)


def _dmrg_sweep_1site_(env, opts_eigs=None, Schmidt=None):
    r"""
    Perform sweep with 1-site DMRG, see :meth:`dmrg` for description.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """
    psi = env.ket
    for to in ('last', 'first'):
        for n in psi.sweep(to=to):
            env.update_Aort_(n)
            _, (psi.A[n],) = eigs(lambda x: env.Heff1(x, n), psi.A[n], k=1, **opts_eigs)
            psi.orthogonalize_site_(n, to=to, normalize=True)
            if Schmidt is not None and to == 'first' and n != psi.first:
                Schmidt[psi.pC] = psi[psi.pC].svd(sU=1, compute_uv=False)
            psi.absorb_central_(to=to)
            env.clear_site_(n)
            env.update_env_(n, to=to)


def _dmrg_sweep_2site_(env, opts_eigs=None, opts_svd=None, Schmidt=None):
    r"""
    Perform sweep with 2-site DMRG, see :meth:`dmrg` for description.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """
    psi = env.ket

    if opts_svd is None:
        opts_svd = {'tol': 1e-13}

    max_disc_weight = -1.
    for to, dn in (('last', 0), ('first', 1)):
        for n in psi.sweep(to=to, dl=1):
            bd = (n, n + 1)
            env.update_AAort_(bd)
            AA = psi.merge_two_sites(bd)
            _, (AA,) = eigs(lambda v: env.Heff2(v, bd), AA, k=1, **opts_eigs)
            _disc_weight_bd = psi.unmerge_two_sites_(AA, bd, opts_svd)
            max_disc_weight = max(max_disc_weight, _disc_weight_bd)
            if Schmidt is not None and to == 'first':
                Schmidt[psi.pC] = psi[psi.pC]
            psi.absorb_central_(to=to)
            env.clear_site_(n, n + 1)
            env.update_env_(n + dn, to=to)
    env.update_env_(psi.first, to='first')
    return max_disc_weight
