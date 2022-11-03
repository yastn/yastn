""" Various variants of the DMRG algorithm for mps."""
import logging
from ... import eigs
from ._env import Env3
from ._mps import YampsError


#logger = logging.Logger('dmrg')


#################################
#           dmrg                #
#################################


def _init_dmrg(psi, H, env, project, opts_eigs):
    """ tests and initializations for all dmrg methods. """
    if opts_eigs is None:
        opts_eigs = {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi, project=project).setup(to='first')

    if not (env.bra is psi and env.ket is psi):
        raise YampsError('Require environment env where ket == bra == psi')
    return env, opts_eigs


def dmrg(psi, H, env=None, project=None, version='1site', \
        converge='energy', measure=None, \
        atol=-1, max_sweeps=1, opts_eigs=None, opts_svd=None, return_info=False):
    r"""
    Perform DMRG sweeps until convergence, starting from MPS :code:`psi`
    in right canonical form. The outer loop sweeps over MPS updating sites
    from the first site to last and back.

    The convergence is controlled either by selected expectation value, i.e., :code:`converge='energy'`
    or by the Schmidt values :code:`converge='schmidt'` which is more sensitive measure.
    The DMRG algorithm then sweeps through the lattice at most :code:`max_sweeps` times
    or until selected convergence measure changes by less then :code:`atol` from sweep to sweep.

    Computational cost of DMRG can be lowered by providing environment :code:`env`
    obtained in previous run.

    Parameters
    ----------
    psi: yamps.MpsMpo
        initial MPS in right canonical form.

    H: yamps.MpsMpo
        MPO to minimize against.

    env: yamps.Env3
        optional environment of tensor network :math:`\langle \psi|H|\psi \rangle` 
        from the previous DMRG run.
    
    project: list(yamps.MpsMpo)
        optimizes MPS in the subspace orthogonal to MPS's in the list

    version: str
        which DMRG variant to use from :code:`'1site'`, :code:`'2site'`

    converge: str
        defines convergence measure. Available options are

            * :code:`'energy'` uses the expectation value of H

            * :code:`'schmidt'` uses Schmidt values on the worst cut

    measure: func(int, yamps.MpsMpo, yamps.Env3, scalar, scalar)->None
        callback allowing measurement/manipulation of MPS after each DMRG sweep.
        The arguments passed are current sweep, current state :math:`|\psi\rangle`,
        current environment corresponding to :math:`\langle\psi|H|\psi\rangle` network,
        current energy, and current truncation error.

    atol: float
        defines converged criterion. DMRG stop once the change in convergence measure
        is less than :code:`atol` between sweeps.

    max_sweeps: int
        maximal number of sweeps

    opts_eigs: dict
        options passed to :meth:`yast.eigs`

    opts_svd: dict
        options passed to :meth:`yast.svd` used to truncate virtual spaces in :code:`verions='2site'`.

    return_info: bool
        if True, return additional information regarding convergence

    Returns
    -------
    env: yamps.Env3
        Environment of the :math:`\langle \psi|H|\psi \rangle` ready for the next iteration.

    info: dict
        if :code:`return_info` is ``True``, return additional information about convergence.
    """

    schmidtold = psi.get_Schmidt_values() if converge == 'schmidt' else None

    env, opts_eigs = _init_dmrg(psi, H, env, project, opts_eigs)
    if opts_svd is None:
        opts_svd = {'tol': 1e-12}
    Eold = env.measure()

    for sweep in range(max_sweeps):
        max_disc_weight = None
        schmidt = {} if converge == 'schmidt' else None
        if version == '1site':
            env = dmrg_sweep_1site(psi, H=H, env=env, project=project, opts_eigs=opts_eigs, schmidt=schmidt)
        elif version == '2site':
            env, max_disc_weight = dmrg_sweep_2site(psi, H=H, env=env, project=project, \
                opts_eigs=opts_eigs, opts_svd=opts_svd, schmidt=schmidt)
        else:
            raise YampsError('dmrg version %s not recognized' % version)
        E = env.measure()
        dE, Eold = Eold - E, E
        if not (measure is None):
            measure(sweep, psi, env, E, max_disc_weight)
        if converge == 'energy':
            logging.info('Iteration = %03d  Energy = %0.14f dE = %0.14f', sweep, E, dE)
            if abs(dE) < atol:
                break
        if converge == 'schmidt':
            dS = max((schmidt[k] - schmidtold[k]).norm() for k in schmidt.keys())
            schmidtold = schmidt
            logging.info('Iteration = %03d  Energy = %0.14f dE = %0.14f dS = %0.14f', sweep, E, dE, dS)
            if dS < atol:
                break
    if return_info:
        return env, {'sweeps': sweep + 1, 'dEng': dE}
    return env


def dmrg_sweep_1site(psi, H, env=None, project=None, opts_eigs=None, schmidt=None):
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
            _, (psi.A[n],) = eigs(lambda x: env.Heff1(x, n), psi.A[n], k=1, **opts_eigs)
            psi.orthogonalize_site(n, to=to)
            if schmidt is not None and to == 'first' and n != psi.first:
                _, S, _ = psi[psi.pC].svd(sU=-1)
                schmidt[psi.pC] = S
            psi.absorb_central(to=to)
            env.clear_site(n)
            env.update_env(n, to=to)
    return env


def dmrg_sweep_2site(psi, H, env=None, project=None, opts_eigs=None, opts_svd=None, schmidt=None):
    r"""
    Perform sweep with 2-site DMRG, see :meth:`dmrg` for description.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """

    env, opts_eigs = _init_dmrg(psi, H, env, project, opts_eigs)
    if opts_svd is None:
        opts_svd = {'tol': 1e-12}

    max_disc_weight = -1.
    for to, dn in (('last', 0), ('first', 1)):
        for n in psi.sweep(to=to, dl=1):
            bd = (n, n + 1)
            env.update_AAort(bd)
            AA = psi.merge_two_sites(bd)
            _, (AA,) = eigs(lambda v: env.Heff2(v, bd), AA, k=1, **opts_eigs)
            _disc_weigth_bd = psi.unmerge_two_sites(AA, bd, opts_svd)
            max_disc_weight = max(max_disc_weight, _disc_weigth_bd)
            if schmidt is not None and to == 'first':
                schmidt[psi.pC] = psi[psi.pC]
            psi.absorb_central(to=to)
            env.clear_site(n, n + 1)
            env.update_env(n + dn, to=to)
    env.update_env(0, to='first')
    return env, max_disc_weight
