""" Various variants of the DMRG algorithm for mps."""
import logging
from yast import eigs
from ._env import Env3
from ._mps import YampsError


logger = logging.Logger('dmrg')


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


def dmrg(psi, H, env=None, project=None, version='1site', converge='energy', atol=-1, max_sweeps=1,
            opts_eigs=None, opts_svd=None, return_info=False):
    r"""
    Perform dmrg sweeps until convergence.

    Assume that psi is canonized to first site.
    Sweeps consists of iterative updates from last site to first and back to the first one.
    Updates psi, returning it in canonical form to the first site.

    Parameters
    ----------
    psi: Mps
        initial state.

    H: Mps, nr_phys=2
        operator to minimize given in the form of mpo.

    env: Env3
        can provide environment <psi|H|psi> from the previous sweep.
        It is initialized if None

    project: list
        optimizes psi in the subspace orthogonal to Mps's in the list

    version: str
        which tdvp procedure to use from ('1site', '2site')

    converge: str
        defines convergence criteria from ('energy', 'schmidt')
        'energy' uses the expectation value of H
        'schmidt' uses the schmidt values on the worst cut

    atol: float
        stop sweeping if converged quantity changes by less than atol in a single sweep

    max_sweeps: int
        maximal number of sweeps

    opts_eigs: dict
        options passed to :meth:`yast.eigs`

    opts_svd: dict
        options passed to :meth:`yast.svd` to truncate virtual bond dimensions when unmerging two merged sites.

    return_info: bool
        if True, return additional information regarding convergence

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
        Can contain temporary objects to reuse from previous sweeps.

    info: dict
        if return_info is True, return some information about reached convergence.
    """
    env, opts_eigs = _init_dmrg(psi, H, env, project, opts_eigs)
    if opts_svd is None:
        opts_svd = {'tol': 1e-12}

    Eold = env.measure()
    for sweep in range(max_sweeps):
        if version == '1site':
            env = dmrg_sweep_1site(psi, H=H, env=env, project=project, opts_eigs=opts_eigs)
        elif version == '2site':
            env = dmrg_sweep_2site(psi, H=H, env=env, project=project, opts_eigs=opts_eigs, opts_svd=opts_svd)
        else:
            raise YampsError('dmrg version %s not recognized' % version)
        E = env.measure()
        dE, Eold = Eold - E, E
        logger.info('Iteration = %03d  Energy = %0.14f dE = %0.14f', sweep, E, dE)
        if converge == 'energy' and abs(dE) < atol:
            break
    if return_info:
        return env, {'sweeps': sweep + 1, 'dEng': dE}
    return env


def dmrg_sweep_1site(psi, H, env=None, project=None, opts_eigs=None):
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
            psi.absorb_central(to=to)
            env.clear_site(n)
            env.update_env(n, to=to)

    return env


def dmrg_sweep_2site(psi, H, env=None, project=None, opts_eigs=None, opts_svd=None):
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

    for to, dn in (('last', 0), ('first', 1)):
        for n in psi.sweep(to=to, dl=1):
            bd = (n, n + 1)
            env.update_AAort(bd)
            AA = psi.merge_two_sites(bd)
            _, (AA,) = eigs(lambda v: env.Heff2(v, bd), AA, k=1, **opts_eigs)
            psi.unmerge_two_sites(AA, bd, opts_svd)
            psi.absorb_central(to=to)
            env.clear_site(n, n + 1)
            env.update_env(n + dn, to=to)

    env.update_env(0, to='first')
    return env
