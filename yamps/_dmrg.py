""" Various variants of the DMRG algorithm for mps."""
import logging
from yast import eigs
from ._env import Env3
from ._mps import YampsError


logger = logging.Logger('dmrg')


#################################
#           dmrg                #
#################################


def dmrg(psi, H, env=None, version='1site', max_sweeps=1, tol_dE=-1, opts_eigs=None, opts_svd=None):
    r"""
    Perform dmrg sweeps until convergence.

    Assume that psi is cannonized to first site.
    Sweep consists of iterative updates from last site to first and back to the first one.
    Updates the state psi.

    Parameters
    ----------
    psi: Mps
        Initial state.
    H: Mps, nr_phys=2
        Operator given in MPO decomposition.
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
    version: str
        Version of dmrg to use. Options: 1site, 2site
    max_sweeps: int
        Maximal number of dmrg sweeps.
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
        Overlap <psi| H |psi> as Env3.
    """
    if opts_eigs is None:
        opts_eigs={'hermitian': True, 'ncv': 3}
    if opts_svd is None:
        opts_svd = {'tol': 1e-12}

    Eold = 100
    sweep = 0
    for sweep in range(max_sweeps):
        env = dmrg_sweep_1site(psi, H=H, env=env, opts_eigs=opts_eigs) if version == '1site' else\
                dmrg_sweep_2site(psi, H=H, env=env, opts_eigs=opts_eigs, opts_svd=opts_svd)
        E = env.measure()
        dE, Eold = Eold - E, E
        logger.info('Iteration = %03d  Energy = %0.14f dE = %0.14f', sweep, E, dE)
        if abs(dE) < tol_dE:
            break
    return env


def dmrg_sweep_1site(psi, H, env=None, opts_eigs=None):
    r"""
    Perform sweep with 1-site DMRG.

    Assume that psi is cannonized to first site.
    Sweep consists of iterative updates from last site to first and back to the first one.
    Updates psi.

    Parameters
    ----------
    psi: Mps
        Initial state.

    H: Mps, nr_phys=2
        Operator to minimize given in the form of mpo.
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]

    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set-up with respect to the last site.

    opts_eigs: dict
        options passed to eigs

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """
    if opts_eigs is None:
        opts_eigs={'hermitian': True, 'ncv': 3, 'which': 'SR'}

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')
    if not (env.bra is psi and env.ket is psi):
        raise YampsError('Require environment env where ket is bra is psi')

    for n in psi.sweep(to='last'):
        f = lambda v: env.Heff1(v, n)
        _, (psi.A[n],) = eigs(f, psi.A[n], k=1, **opts_eigs)
        psi.orthogonalize_site(n, to='last')
        psi.absorb_central(to='last')
        env.clear_site(n)
        env.update(n, to='last')

    for n in psi.sweep(to='first'):
        f = lambda v: env.Heff1(v, n)
        _, (psi.A[n],) = eigs(f, psi.A[n], k=1, **opts_eigs)
        psi.orthogonalize_site(n, to='first')
        psi.absorb_central(to='first')
        env.clear_site(n)
        env.update(n, to='first')
    return env


def dmrg_sweep_2site(psi, H, env=None, opts_eigs=None, opts_svd=None):
    r"""
    Perform sweep with 2-site DMRG.
    Assume input psi is right canonical.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps, nr_phys=1
        Initial state.
    H: Mps, nr_phys=2
        Operator given in MPO decomposition.
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
    hermitian: bool
        default=True
        Is MPO hermitian
    k: int
        default=4
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    """

    if opts_svd is None:
        opts_svd = {'tol': 1e-12}
    if opts_eigs is None:
        opts_eigs={'hermitian': True, 'ncv': 3, 'which': 'SR'}

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')
    if not (env.bra is psi and env.ket is psi):
        raise YampsError('Require environment env where ket is bra is psi')

    for n in psi.sweep(to='last', dl=1):
        bd = (n, n + 1)
        AA = psi.merge_two_sites(bd)
        f = lambda v: env.Heff2(v, bd)
        _, (AA,) = eigs(f, AA, k=1, **opts_eigs)
        psi.unmerge_two_sites(AA, bd, opts_svd)
        psi.absorb_central(to='last')
        env.clear_site(n, n + 1)
        env.update(n, to='last')

    for n in psi.sweep(to='first', dl=1):
        bd = (n, n + 1)
        AA = psi.merge_two_sites(bd)
        f = lambda v: env.Heff2(v, bd)
        _, (AA,) = eigs(f, AA, k=1, **opts_eigs)
        psi.unmerge_two_sites(AA, bd, opts_svd)
        psi.absorb_central(to='first')
        env.clear_site(n, n + 1)
        env.update(n + 1, to='first')

    env.update(0, to='first')
    return env
