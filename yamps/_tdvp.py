""" Various variants of the TDVP algorithm for mps."""
from yast import expmv
from ._env import Env3


#################################
#           tdvp                #
#################################


def tdvp_sweep_1site(psi, H=False, dt=1., env=None, hermitian=True, k=4, exp_tol=1e-12, optsK_svd=None, **kwargs):
    r"""
    Perform sweep with 1site-TDVP by applying exp(-i*dt*H) on initial vector. Note the convention for time step sign.
    Procedure performs exponantiation: psi(dt) = exp( dt * H )*psi(0). For Hamiltonian real time evolution forward in time: sign(dt)= -1j, for Hamiltonian imaginary time evolution forward in time: sign(dt)= -1.. For Hamiltonian real time evolution forward in time: sign(dt)= -1j, for Hamiltonian imaginary time evolution forward in time: sign(dt)= -1.
    Assume input psi is right canonical.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps, nr_phys=1
        initial state.
    H: Mps, nr_phys=2
        operator given in MPO decomposition.
        legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    M: Mps, nr_phys=1
        Kraus operators.
        legs are [Kraus dimension, ket-physical, bra-physical]
    env: Env3
        default = None
        initial overlap <psi| H |psi>
        initial environments must be set up with respect to the last site.
    dt: double
        default = 1
        time interval for matrix expontiation. May be divided into smaller intervals according to the cost function.
    hermitian: bool
        default =True
        is MPO hermitian
    k: int
        default=4
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
    optsK_svd: dict
        default=None
        options for truncation on auxilliary d.o.f.

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.

    Note
    ----
    psi is updated.
    """

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    for n in psi.sweep(to='last'):
        f = lambda v: env.Heff1(v, n)
        psi.A[n] = expmv(f, psi.A[n], 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        psi.orthogonalize_site(n, to='last')
        env.clear_site(n)
        if n != psi.last:
            env.update(n, to='last')
            f = lambda v: env.Heff0(v, psi.pC)
            psi.A[psi.pC] = expmv(f, psi.A[psi.pC], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        psi.absorb_central(to='last')

    for n in psi.sweep(to='first'):
        f = lambda v: env.Heff1(v, n)
        psi.A[n] = expmv(f, psi.A[n], 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        psi.orthogonalize_site(n, to='first')
        env.clear_site(n)
        if n != psi.first:
            env.update(n, to='first')
            f = lambda v: env.Heff0(v, psi.pC)
            psi.A[psi.pC] = expmv(f, psi.A[psi.pC], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        psi.absorb_central(to='first')
    env.update(0, to='first')
    return env


def tdvp_sweep_2site(psi, H=False, dt=1., env=None, hermitian=True, k=4, exp_tol=1e-12, opts_svd=None, **kwargs):
    r"""
    Perform sweep with 2-site TDVP, calculating the update psi(dt) = exp( dt * H ) @ psi(0).

    Assume input psi is canonized to first site.

    Parameters
    ----------
    psi: Mps, nr_phys=1
        initial state.
    H: Mps, nr_phys=2
        operator given in MPO decomposition.
        legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    M: Mps, nr_phys=1
        Kraus operators.
        legs are [Kraus dimension, ket-physical, bra-physical]
    env: Env3
        default = None
        initial overlap <psi| H |psi>
        initial environments must be set up with respect to the last site.
    dt: double
        default = 1
        time interval for matrix expontiation. May be divided into smaller intervals according to the cost function.
    hermitian: bool
        default = True
        is MPO hermitian
    k: int
        default = 4
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default = 1e-14
        Cutoff for krylov subspace for eigs(.)
    opts_svd: dict
        default=None
        options for truncation on virtual d.o.f.
    optsK_svd: dict
        default=None
        options for truncation on auxilliary d.o.f.

    Returns
    -------
    env: Env3
        Overlap <psi| H |psi> as Env3.

    Note
    ----
    psi is updated.
    """

    if env is None: 
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    for n in psi.sweep(to='last', dl=1):
        bd = (n, n + 1)
        AA = psi.merge_two_sites(bd)
        f = lambda v: env.Heff2(v, bd)
        AA = expmv(f, AA, 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        psi.unmerge_two_sites(AA, bd, opts_svd)
        psi.absorb_central(to='last')
        env.clear_site(n, n + 1)
        env.update(n, to='last')
        if n + 1 != psi.last:
            f = lambda v: env.Heff1(v, n + 1)
            psi.A[n + 1] = expmv(f, psi.A[n + 1], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)

    for n in psi.sweep(to='first', dl=1):
        bd = (n, n + 1)
        AA = psi.merge_two_sites(bd)
        f = lambda v: env.Heff2(v, bd)
        AA = expmv(f, AA, 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        psi.unmerge_two_sites(AA, bd, opts_svd)
        psi.absorb_central(to='first')
        env.clear_site(n, n + 1)
        env.update(n + 1, to='first')
        if n != psi.first:
            f = lambda v: env.Heff1(v, n)
            psi.A[n] = expmv(f, psi.A[n], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)

    env.clear_site(0)
    env.update(0, to='first')
    return env


def tdvp_sweep_mix(psi, H=False, dt=1., env=None, hermitian=True, k=4, exp_tol=1e-12, D_totals=None, tol_svds=None, opts_svd=None, **kwargs):
    r"""
    Perform mixed 1site-2site sweep of TDVP basing on SV_min (smallest Schmidt value on the bond).
    Procedure performs exponantiation: psi(dt) = exp( dt * H )*psi(0).
    Assume input psi is right canonical.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps
        initial state.

    Returns
    -------
    env: Env3
        Overlap <psi| H |psi> as Env3.
    """
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    update_two = False
    for n in psi.sweep(to='last'):
        if not update_two:
            if env.enlarge_bond[(n, n + 1)]:
                update_two = True
            else:
                f = lambda v: env.Heff1(v, n)
                psi.A[n] = expmv(f, psi.A[n], 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                psi.orthogonalize_site(n, to='last')
                env.clear_site(n)
                env.update(n, to='last')
                if n != psi.last:
                    f = lambda v: env.Heff0(v, psi.pC)
                    psi.A[psi.pC] = expmv(f, psi.A[psi.pC], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                psi.absorb_central(to='last')
        else:
            bd = (n - 1, n)
            AA = psi.merge_two_sites(bd)
            f = lambda v: env.Heff2(v, bd)
            AA = expmv(f, AA, 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            psi.unmerge_two_sites(AA, bd, opts_svd)
            psi.absorb_central(to='last')
            env.clear_site(n - 1, n)
            env.update(n - 1, to='last')
            if env.enlarge_bond[(n, n + 1)]:
                if n + 1 != psi.last:
                    f = lambda v: env.Heff1(v, n + 1)
                    psi.A[n + 1] = expmv(f, psi.A[n + 1], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            else:
                psi.ortogonalize_site(n, to='last')
                env.update(n, to='last')
                if n != psi.last:
                    f = lambda v: env.Heff0(v, psi.pC)
                    psi.A[psi.pC] = expmv(f, psi.A[psi.pC], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                psi.absorb_central(to='last')

    for n in psi.sweep(to='first'):
        if not update_two:
            if env.enlarge_bond[(n - 1, n)]:
                update_two = True
            else:
                f = lambda v: env.Heff1(v, n)
                psi.A[n] = expmv(f, psi.A[n], 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                psi.orthogonalize_site(n, to='first')
                env.clear_site(n)
                if n != psi.last:
                    env.update(n, to='first')
                    f = lambda v: env.Heff0(v, psi.pC)
                    psi.A[psi.pC] = expmv(f, psi.A[psi.pC], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                psi.absorb_central(to='last')
        else:
            bd = (n, n + 1)
            AA = psi.merge_two_sites(bd)
            f = lambda v: env.Heff2(v, bd)
            AA = expmv(f, AA, 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            psi.unmerge_two_sites(AA, bd, opts_svd)
            psi.absorb_central(to='first')
            env.clear_site(n, n + 1)
            env.update(n + 1, to='first')
            if env.enlarge_bond[(n - 1, n)]:
                if n != psi.first:
                    f = lambda v: env.Heff1(v, n + 1)
                    psi.A[n + 1] = expmv(f, psi.A[n + 1], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            else:
                psi.ortogonalize_site(n, to='first')
                if n != psi.first:
                    env.update(n, to='first')
                    f = lambda v: env.Heff0(v, psi.pC)
                    psi.A[psi.pC] = expmv(f, psi.A[psi.pC], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                psi.absorb_central(to='last')
    env.clear_site(0)
    env.update(0, to='first')
    return env
