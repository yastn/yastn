""" Various variants of the DMRG algorithm for mps."""
from typing import NamedTuple
import logging
from ... import tensor, YastError
from ._env import Env3
from ._mps import MpsMpo


logger = logging.Logger('dmrg')


#################################
#           dmrg                #
#################################

class DMRGout(NamedTuple):
    sweeps : int = 0
    energy : float = None
    denergy : float = None
    max_dSchmidt : float = None
    max_discarded_weight : float = None


def _init_dmrg(psi, H, env, project, opts_eigs):
    """ tests and initializations for all dmrg methods. """
    if opts_eigs is None:
        opts_eigs = {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi, project=project).setup(to='first')

    if not (env.bra is psi and env.ket is psi):
        raise YastError('MPS: Require environment env where ket == bra == psi')
    return env, opts_eigs


def dmrg_(psi, H, project=None, method='1site',
        energy_tol=None, Schmidt_tol=None, max_sweeps=1, iterator_step=None,
        opts_eigs=None, opts_svd=None):
    r"""
    Perform DMRG sweeps until convergence, starting from MPS :code:`psi`.

    The outer loop sweeps over MPS updating sites from the first site to last and back.
    Convergence can be controlled based on energy and/or Schmidt values (which is a more sensitive measure).
    The DMRG algorithm sweeps through the lattice at most :code:`max_sweeps` times
    or until all convergence measures with provided tolerance change by less then the tolerance.

    Outputs generator if :code:`iterator_step` is given.
    It allows inspecting :code:`psi` outside of :code:`dmrg_` function after every :code:`iterator_step` sweeps.

    Parameters
    ----------
    psi: yamps.MpsMpo
        Initial MPS, that gets modified in place inside :code:`dmrg_`.
        psi is initialy canonized to to the first site, if not provided in such a form.
        In particular, psi resulting from :code:`dmrg_` is canonized to the first site.

    H: yamps.MpsMpo
        MPO to minimize against.

    project: list(yamps.MpsMpo)
        optimizes MPS in the subspace orthogonal to MPS's in the list.

    method: str
        which DMRG variant to use from :code:`'1site'`, :code:`'2site'`

    energy_tol: float
        convergence tolerance for the change of energy in a single sweep.
        By default is None, in which case energy convergence is not checked.

    energy_tol: float
        convergence tolerance for the change of Schmidt values on the worst cut/bond in a single sweep.
        By default is None, in which case Schmidt values convergence is not checked.

    max_sweeps: int
        maximal number of sweeps

    iterator_step: int
        If int, :code:`dmrg_` returns a generator that would yield output after every iterator_step sweeps.
        Default is None, in which case  :code:`dmrg_` sweeps are performed immidiatly.

    opts_eigs: dict
        options passed to :meth:`yast.eigs`.
        If None, use default {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    opts_svd: dict
        options passed to :meth:`yast.svd` used to truncate virtual spaces in :code:`verions='2site'`.
        If None, use default 

    Returns
    -------
    step: DMRGout(NamedTuple)
        Includes fields:
            :code:`sweeps` number of performed dmrg sweeps.
            :code:`energy` energy after the last sweep.
            :code:`denergy` absolut value of energy change in the last sweep.
            :code:`max_dSchmidt` norm of Schmidt values change on the worst cut in the last sweep
            :code:`max_discarded_weight` norm of discarded_weights on the worst cut in '2site' procedure.
    """
    tmp = _dmrg_(psi, H, project, method, 
                energy_tol, Schmidt_tol, max_sweeps, iterator_step,
                opts_eigs, opts_svd)
    return tmp if iterator_step else next(tmp)


def _dmrg_(psi, H, project, method,
        energy_tol, Schmidt_tol, max_sweeps, iterator_step,
        opts_eigs, opts_svd):
    """ Generator for dmrg_() """

    if not psi.is_canonical(to='first'):
        psi.canonize_sweep(to='first')

    env = None
    env, opts_eigs = _init_dmrg(psi, H, env, project, opts_eigs)
    E_old = env.measure()

    if Schmidt_tol is not None:
        if not Schmidt_tol > 0:
            raise YastError('DMRG: Schmidt_tol has to be positive or None.')
        Schmidt_old = psi.get_Schmidt_values()
    max_dS, max_dw = None, None
    Schmidt = None if Schmidt_tol is None else {}

    if energy_tol is not None and not energy_tol > 0:
        raise YastError('DMRG: energy_tol has to be positive or None.')

    if method not in ('1site', '2site'):
        raise YastError('DMRG: dmrg method %s not recognized.' % method)

    for sweep in range(1, max_sweeps + 1):
        if method == '1site':
            env = dmrg_sweep_1site(psi, H=H, env=env, project=project, 
                opts_eigs=opts_eigs, Schmidt=Schmidt)
        else: # method == '2site':
            env, max_dw = dmrg_sweep_2site(psi, H=H, env=env, project=project,
                opts_eigs=opts_eigs, opts_svd=opts_svd, Schmidt=Schmidt)

        E = env.measure()
        dE, E_old = E_old - E, E
        converged = []

        if energy_tol is not None:
            converged.append(abs(dE) < energy_tol)

        if Schmidt_tol is not None:
            max_dS = max((Schmidt[k] - Schmidt_old[k]).norm() for k in Schmidt.keys())
            Schmidt_old = Schmidt
            converged.append(max_dS < Schmidt_tol)

        logger.info('Sweep = %03d  energy = %0.14f  dE = %0.4f  dSchmidt = %0.4f', sweep, E, dE, max_dS)

        if all(converged) and any(converged):
            break
        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield DMRGout(sweep, E, dE, max_dS, max_dw)
    yield DMRGout(sweep, E, dE, max_dS, max_dw)


def dmrg(psi, H, env=None, project=None, version='1site',
        converge='energy', measure=None,
        atol=-1, max_sweeps=1, opts_eigs=None, opts_svd=None, return_info=False):

    Schmidt_old = psi.get_Schmidt_values() if converge == 'Schmidt' else None

    env, opts_eigs = _init_dmrg(psi, H, env, project, opts_eigs)
    if opts_svd is None:
        opts_svd = {'tol': 1e-12}
    E_old = env.measure()

    for sweep in range(max_sweeps):
        max_disc_weight = None
        Schmidt = {} if converge == 'Schmidt' else None
        if version == '1site':
            env = dmrg_sweep_1site(psi, H=H, env=env, project=project, opts_eigs=opts_eigs, Schmidt=Schmidt)
        elif version == '2site':
            env, max_disc_weight = dmrg_sweep_2site(psi, H=H, env=env, project=project, \
                opts_eigs=opts_eigs, opts_svd=opts_svd, Schmidt=Schmidt)
        else:
            raise YastError('MPS: dmrg version %s not recognized' % version)
        E = env.measure()
        dE, E_old = E_old - E, E
        if not (measure is None):
            measure(sweep, psi, env, E, max_disc_weight)
        if converge == 'energy':
            logger.info('Iteration = %03d  Energy = %0.14f dE = %0.14f', sweep, E, dE)
            if abs(dE) < atol:
                break
        if converge == 'Schmidt':
            dS = max((Schmidt[k] - Schmidt_old[k]).norm() for k in Schmidt.keys())
            Schmidt_old = Schmidt
            logger.info('Iteration = %03d  Energy = %0.14f dE = %0.14f dS = %0.14f', sweep, E, dE, dS)
            if dS < atol:
                break
    if return_info:
        return env, {'sweeps': sweep + 1, 'dEng': dE}
    return env


def dmrg_sweep_1site(psi, H, env=None, project=None, opts_eigs=None, Schmidt=None):
    r"""
    Perform sweep with 1-site DMRG, see :meth:`dmrg` for description.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """

    env, opts_eigs = _init_dmrg(psi, H, env, project, opts_eigs)

    for to in ('last', 'first'):
        for n in psi.sweep(to=to):
            env.update_Aort(n)
            _, (psi.A[n],) = tensor.eigs(lambda x: env.Heff1(x, n), psi.A[n], k=1, **opts_eigs)
            psi.orthogonalize_site(n, to=to)
            if Schmidt is not None and to == 'first' and n != psi.first:
                _, S, _ = psi[psi.pC].svd(sU=1)
                Schmidt[psi.pC] = S
            psi.absorb_central(to=to)
            env.clear_site(n)
            env.update_env(n, to=to)
    return env


def dmrg_sweep_2site(psi, H, env=None, project=None, opts_eigs=None, opts_svd=None, Schmidt=None):
    r"""
    Perform sweep with 2-site DMRG, see :meth:`dmrg` for description.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """

    env, opts_eigs = _init_dmrg(psi, H, env, project, opts_eigs)

    if opts_svd is None:
        raise YastError('DMRG: provide opts_svd for truncation of 2site dmrg.')

    max_disc_weight = -1.
    for to, dn in (('last', 0), ('first', 1)):
        for n in psi.sweep(to=to, dl=1):
            bd = (n, n + 1)
            env.update_AAort(bd)
            AA = psi.merge_two_sites(bd)
            _, (AA,) = tensor.eigs(lambda v: env.Heff2(v, bd), AA, k=1, **opts_eigs)
            _disc_weight_bd = psi.unmerge_two_sites(AA, bd, opts_svd)
            max_disc_weight = max(max_disc_weight, _disc_weight_bd)
            if Schmidt is not None and to == 'first':
                Schmidt[psi.pC] = psi[psi.pC]
            psi.absorb_central(to=to)
            env.clear_site(n, n + 1)
            env.update_env(n + dn, to=to)
    env.update_env(0, to='first')
    return env, max_disc_weight
