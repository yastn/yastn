""" Various variants of the DMRG algorithm for mps."""

from yast import linalg
from ._env import Env3


#################################
#           dmrg                #
#################################


def dmrg_OBC(psi, H, env=None, version='1site', cutoff_sweep=1, cutoff_dE=-1, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=None, SV_min=None, D_totals=None, tol_svds=None, versions=('1site', '2site'), algorithm='arnoldi'):
    r"""
    Perform dmrg on system with open boundary conditions, updating initial state psi.  Assume input psi is right canonical.

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
    nor: Env2
        default = None
        nor - dummy - is not used in DMRG
        Initial overlap <psi|psi>
        Initial environments must be set up with respect to the last site.
    measure_O: list of lists, each of format [n, operator, list_of_ids]
        default = None
        n: int
            1(for single site operator) or 2(for two-site operator)
        operator: Tensor or list of Tensors
            if n==1: Tensor
            if n==2: list of format [Tensor, Tensor]
        list_of_ids: list
            List  of sites which you want to measure.
        e.g.  [2, [OL, OR], [1, 2, 3]], measure expectation value of 2-site operator OL-OR on sites (1, 2), (2, 3), (3, 4)
    version: string
        default = '1site'
        Version of dmrg to use. Options: 1site, 2site
    cutoff_sweep: int
        default=20
        Upper bound for number of sweeps.
    cutoff_dE: double
        default=1e-9
        Target convergence on the H expectation value.
    dtype: str
        default='complex128'
        Type of Tensor.
    hermitian: bool
        default=True
        Is MPO hermitian
    k: int
        default=4
        Dimension of Krylov subspace for linalg.eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for linalg.eigs(.)
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
        Overlap <psi| H |psi> as Env3.
    E: double
        Final expectation value of H
    dE: double
        Final variation on <H>
    out: list
        list of measured expectation values. Set of operators provided by measure_O
    """
    E0 = 0
    dE = cutoff_dE + 1
    sweep = 0
    while sweep < cutoff_sweep and dE > cutoff_dE:
        if version == '1site':
            env = dmrg_sweep_1site(psi, H=H, env=env, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol)
        elif version == '2site':
            env = dmrg_sweep_2site(psi, H=H, env=env, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd)
        E = env.measure()
        dE = abs(E - E0)
        print('Iteration: ', sweep, ' energy: ', E,
              ' dE: ', dE, ' D: ', max(psi.get_bond_dimensions()))
        E0 = E
        sweep += 1
    return env, E, dE


def dmrg_sweep_1site(psi, H, env=None, hermitian=True, k=4, eigs_tol=1e-14):
    r"""
    Perform sweep with 1-site DMRG.

    Assume input psi is cannonized to first site.
    Sweep consists of iterative updates from last site to first and back to the first one,
    updating the state psi.

    Parameters
    ----------
    psi: Mps
        Initial state.
    H: Mps, nr_phys=2
        Operator given as an MPO.
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set-up with respect to the last site.
    hermitian: bool
        default=True
        Is MPO hermitian
    k: int
        default=4
        Dimension of Krylov subspace for linalg.eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for linalg.eigs(.)

    Returns
    -------
    env: Env3
        Overlap <psi| H |psi> as Env3.
    """
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    for n in psi.sweep(to='last'):
        f = lambda v: env.Heff1(v, n)
        _, (psi.A[n],) = linalg.eigs(f, psi.A[n], k=1, which='SR', ncv=k, tol=eigs_tol, hermitian=hermitian)
        psi.orthogonalize_site(n, to='last')
        psi.absorb_central(to='last')
        env.clear_site(n)
        env.update(n, to='last')

    for n in psi.sweep(to='first'):
        f = lambda v: env.Heff1(v, n)
        _, (psi.A[n],) = linalg.eigs(f, psi.A[n], k=1, which='SR', ncv=k, tol=eigs_tol, hermitian=hermitian)
        psi.orthogonalize_site(n, to='first')
        psi.absorb_central(to='first')
        env.clear_site(n)
        env.update(n, to='first')
    return env


def dmrg_sweep_2site(psi, H, env=None, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=None):
    r"""
    Perform sweep with 2site-DMRG.
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
        Dimension of Krylov subspace for linalg.eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for linalg.eigs(.)
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    """

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    for n in psi.sweep(to='last', dl=1):
        bd = (n, n + 1)
        AA = psi.merge_two_sites(bd)
        f = lambda v: env.Heff2(v, bd)
        _, (AA,) = linalg.eigs(f, AA, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
        psi.unmerge_two_sites(AA, bd, opts_svd)
        psi.absorb_central(to='last')
        env.clear_site(n, n + 1)
        env.update(n, to='last')

    for n in psi.sweep(to='first', dl=1):
        bd = (n, n + 1)
        AA = psi.merge_two_sites(bd)
        f = lambda v: env.Heff2(v, bd)
        _, (AA,) = linalg.eigs(f, AA, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
        psi.unmerge_two_sites(AA, bd, opts_svd)
        psi.absorb_central(to='first')
        env.clear_site(n, n + 1)
        env.update(n + 1, to='first')

    env.update(0, to='first')
    return env
