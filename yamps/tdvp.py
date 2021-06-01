from yast import expmv, tensordot, svd
import logging
import numpy as np
from ._env3 import Env3


class FatalError(Exception):
    pass


logger = logging.getLogger('yast.tensor.tdvp')


#################################
#           tdvp                #
#################################

def tdvp_OBC(psi, tmax, dt=1, H=False, M=False, env=None, D_totals=None, tol_svds=None, SV_min=None,
             versions=('1site', '2site'), cutoff_dE=1e-9, hermitian=True, fermionic=False,
             k=4, eigs_tol=1e-12, exp_tol=1e-12, bi_orth=False, NA=None, version='1site',
             opts_svd=None, optsK_svd=None, algorithm='arnoldi'):
    # evolve with TDVP method, up to tmax and initial guess of the time step dt
    # opts - optional info for MPS truncation
    curr_t = 0
    if not env and H:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')
    if H:
        E0 = env.measure()
        dE = cutoff_dE + 1
    else:
        E0, dE = 0, 0
    while abs(curr_t) < abs(tmax):
        dt = min([abs(tmax - curr_t) / abs(dt), 1.]) * dt
        if not H and not M:
            logger.error('yast.tdvp: Neither Hamiltonian nor Kraus operators defined.')
        else:
            if version == '1site':
                env = tdvp_sweep_1site(psi=psi, H=H, M=M, dt=dt, env=env, hermitian=hermitian,
                                       k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd, algorithm=algorithm)
            elif version == '2site':
                env = tdvp_sweep_2site(psi=psi, H=H, M=M, dt=dt, env=env, hermitian=hermitian,
                                       k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd, algorithm=algorithm)
            elif version == '2site_group':
                env = tdvp_sweep_2site_group(psi=psi, H=H, M=M, dt=dt, env=env, hermitian=hermitian,
                                             k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd, algorithm=algorithm)
            else:
                env = tdvp_sweep_mix(psi=psi, H=H, M=M, dt=dt, env=env, hermitian=hermitian, versions=versions, D_totals=D_totals, tol_svds=tol_svds, SV_min=SV_min,
                                     k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd, algorithm=algorithm)
        E = env.measure()
        dE = abs(E - E0)
        # print('Iteration: ', sweep, ' energy: ', E, ' dE: ', dE, ' D: ', max(psi.get_D()))
        E0 = E
        curr_t += dt
    return env, E, dE


def tdvp_sweep_1site(psi, H=False, M=False, dt=1., env=None, hermitian=True, k=4, exp_tol=1e-12, optsK_svd=None, **kwargs):
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

    for n in psi.sweep(to='last'):  # sweep from first to last
        if M:  # apply the Kraus operator
            tmp = tensordot(M.A[n], psi.A[n], axes=(2, 1))
            tmp.swap_gate(axes=(0, 2), inplace=True)  # for fermions
            u, s, _ = svd(tmp, axes=((2, 1, 4), (0, 3)), **optsK_svd)  # discard V
            psi.A[n] = tensordot(u, s, axes=(3, 0)).transpose(axes=(0, 1, 3, 2))
        # forward in time evolution of a single site: T(+dt/2)
        if H:
            f = lambda v: env.Heff1(v, n)
            psi.A[n] = expmv(f, psi.A[n], 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        psi.orthogonalize_site(n, to='last')
        env.clear_site(n)
        env.update(n, to='last')
        # backward in time evolution of a central site: T(-dt/2)
        if H and n != psi.sweep(to='last')[-1]:
            f = lambda v: env.Heff0(v, psi.pC)
            psi.A[psi.pC] = expmv(f, psi.A[psi.pC], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        psi.absorb_central(to='last')

    for n in psi.sweep(to='first'):
        if H:  # forward in time evolution of a single site: dt / 2
            f = lambda v: env.Heff1(v, n)
            psi.A[n] = expmv(f, psi.A[n], 0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        if M:  # apply the Kraus operator
            tmp = tensordot(M.A[n], psi.A[n], axes=(2, 1))
            tmp.swap_gate(axes=(0, 2), inplace=True)
            u, s, _ = svd(tmp, axes=((2, 1, 4), (0, 3)), **optsK_svd)  # discard V
            psi.A[n] = tensordot(u, s, axes=(3, 0)).transpose(axes=(0, 1, 3, 2))
        psi.orthogonalize_site(n, to='first')
        env.clear_site(n)
        env.update(n, to='first')
        # backward in time evolution of a central site: -dt / 2
        if H and n != psi.sweep(to='first')[-1]:
            f = lambda v: env.Heff0(v, psi.pC)
            psi.A[psi.pC] = expmv(f, psi.A[psi.pC], -0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        psi.absorb_central(to='first')
    return env


def tdvp_sweep_2site(psi, H=False, M=False, dt=1., env=None, hermitian=True, k=4, exp_tol=1e-12, opts_svd=None, optsK_svd=None, **kwargs):
    r"""
    Perform sweep with 2site-TDVP.
    Procedure performs exponantiation: psi(dt) = exp( dt * H )*psi(0). For Hamiltonian real time evolution forward in time: sign(dt)= -1j, for Hamiltonian imaginary time evolution forward in time: sign(dt)= -1.
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
        if M:  # Apply the Kraus operator on n
            tmp = tensordot(M.A[n], psi.A[n], axes=(2, 1))
            tmp.swap_gate(axes=(0, 2), inplace=True)
            u, s, _ = svd(tmp, axes=((2, 1, 4), (0, 3)), **optsK_svd)  # discard V
            psi.A[n] = tensordot(u, s, axes=(3, 0)).transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, to='last')
                psi.absorb_central(to='last')

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, to='last')
            init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
            f = lambda v: env.Heff2(v, n)
            init = expmv(f, v=init, t=+0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            # split and save
            A1, S, A2 = svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n] = A1
            psi.A[n1] = S.tensordot(A2, axes=(1, psi.left))
        env.clear_site(n)
        env.update(n, to='last')

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.sweep(to='last', dl=1)[-1]:
            f = lambda v: env.Heff1(v, n1)
            psi.A[n1] = expmv(f, psi.A[n1], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)


    for n in psi.sweep(to='first', df=1):
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = tensordot(M.A[n], init, axes=(2, 1))
            tmp.swap_gate(axes=(0, 2), inplace=True)
            u, s, _ = svd(tmp, axes=((2, 1, 4), (0, 3)), **optsK_svd)  # discard V
            psi.A[n] = tensordot(u, s, axes=(3, 0)).transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, to='first')
                psi.absorb_central(to='first')

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, to='first')
            init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
            f = lambda v: env.Heff2(v, n1)
            init = expmv(f, v=init, t=0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            # split and save
            A1, S, A2 = svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n1] = A1.tensordot(S, axes=(psi.right, 0))
            psi.A[n] = A2
        env.clear_site(n)
        env.update(n, to='first')

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.sweep(to='first', df=1)[-1]:
            f = lambda v: env.Heff1(v, n1)
            psi.A[n1] = expmv(f, psi.A[n1], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
    env.clear_site(n1)
    env.update(n1, to='first')

    return env


def tdvp_sweep_2site_group(psi, H=False, M=False, dt=1, env=None, hermitian=True, fermionic=False, k=4, exp_tol=1e-12, opts_svd=None, optsK_svd=None, **kwargs):
    r"""
    Perform sweep with 2site-TDVP with grouping neigbouring sites.
    Procedure performs exponantiation: psi(dt) = exp( dt * H )*psi(0). For Hamiltonian real time evolution forward in time: sign(dt)= -1j, for Hamiltonian imaginary time evolution forward in time: sign(dt)= -1.
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
    dtype: str
        default='complex128'
        Type of Tensor.
    hermitian: bool
        default=True
        is MPO hermitian
    fermionic: bool
        default = False
        use while pllying a SWAP gate. True for fermionic systems.
    k: int
        default=4
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
    bi_orth: bool
        default=True
        Option for exponentiation = exp(). For True and non-Hermitian cases will bi-orthogonalize set of generated vectors.
    NA: bool
        default=None
        The cost of matrix-vector multiplication used to optimize Krylov subspace and time intervals.
        Option for exponentiation = exp().
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
    psi: Mps
        Is self updated.
    """

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    for n in psi.sweep(to='last', dl=1):
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = tensordot(M.A[n], init, axes=(2, 1))
            tmp.swap_gate(axes=(0, 2), inplace=True)
            u, s, _ = svd(tmp, axes=((2, 1, 4), (0, 3)), **optsK_svd)  # discard V
            init = tensordot(u, s, axes=(3, 0))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, to='last')
                psi.absorb_central(to='last')

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, to='last')
            init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
            init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
            f = lambda v: env.Heff2_group(v, n)
            init = expmv(f, v=init, t=0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            # split and save
            init.unfuse_legs(axes=1, inplace=True)
            A1, S, A2 = svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n] = A1
            psi.A[n1] = S.tensordot(A2, axes=(1, psi.left))
        env.clear_site(n)
        env.update(n, to='last')

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.sweep(to='last', dl=1)[-1]:
            f = lambda v: env.Heff1(v, n1)
            psi.A[n1] = expmv(f, psi.A[n1], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)

    for n in psi.sweep(to='first', df=1):
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = tensordot(M.A[n], init, axes=(2, 1))
            tmp.swap_gate(axes=(0, 2), inplace=True)
            u, s, _ = svd(tmp, axes=((2, 1, 4), (0, 3)), **optsK_svd)  # discard V
            init = tensordot(u, s, axes=(3, 0))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, to='first')
                psi.absorb_central(to='first')

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, to='first')
            init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
            init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
            f = lambda v: env.Heff2_group(v, n1)
            init = expmv(f, v=init, t=0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            # split and save
            init.unfuse_legs(axes=1, inplace=True)
            A1, S, A2 = svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n1] = A1.tensordot(S, axes=(psi.right, 0))
            psi.A[n] = A2
        env.clear_site(n)
        env.update(n, to='first')

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.sweep(to='first', df=1)[-1]:
            f = lambda v: env.Heff1(v, n1)
            psi.A[n1] = expmv(f, psi.A[n1], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
    env.clear_site(n1)
    env.update(n1, to='first')

    return env


def tdvp_sweep_mix(psi, SV_min, versions, H=False, M=False, dt=1., env=None, hermitian=True, fermionic=False, k=4, exp_tol=1e-12, D_totals=None, tol_svds=None, opts_svd=None, optsK_svd=None, **kwargs):
    r"""
    Perform mixed 1site-2site sweep of TDVP basing on SV_min (smallest Schmidt value on the bond).
    Procedure performs exponantiation: psi(dt) = exp( dt * H )*psi(0). For Hamiltonian real time evolution forward in time: sign(dt)= -1j, for Hamiltonian imaginary time evolution forward in time: sign(dt)= -1.
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
    SV_min: list
        list of minimal Schmidt values on each bond
    D_totals: list
        list of upper bound on bond dimension
    tol_svds: list
        list of lower bound on Schmidt values.
    versions: 2-element tuple
        version = (else, version to increase bond_dimension)
        tuple with what algorithm to use during a sweep. Algorithm is chosen basing on Schmidt values.
    dt: double
        default = 1
        time interval for matrix expontiation. May be divided into smaller intervals according to the cost function.
    dtype: str
        default='complex128'
        Type of Tensor.
    hermitian: bool
        default = True
        is MPO hermitian
    fermionic: bool
        default = False
        use while pllying a SWAP gate. True for fermionic systems.
    k: int
        default = 4
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default = 1e-14
        Cutoff for krylov subspace for eigs(.)
    bi_orth: bool
        default = True
        Option for exponentiation = exp(). For True and non-Hermitian cases will bi-orthogonalize set of generated vectors.
    NA: bool
        default = None
        The cost of matrix-vector multiplication used to optimize Krylov subspace and time intervals.
        Option for exponentiation = exp().
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
    psi: Mps
        Is self updated.
    """
    Ds = psi.get_D()
    if not D_totals:
        D_totals = [None] * (psi.N + 1)
        max_vdim = 1
        for n in range(psi.N):
            D_totals[n] = min([max_vdim, opts_svd['D_total']])
            max_vdim = D_totals[n] * np.prod(psi.A[n].get_shape(psi.phys))
        max_vdim = 1
        for n in range(psi.N - 1, -1, -1):
            max_vdim *= np.prod(psi.A[n].get_shape(psi.phys))
            D_totals[n] = min([D_totals[n], max_vdim, opts_svd['D_total']])
            max_vdim = D_totals[n]
        D_totals[-1] = 1
    if not tol_svds:
        tol_svds = [opts_svd['tol'] for n in Ds]
    #
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')
    for n in psi.sweep(to='last'):
        opts_svd['D_total'] = D_totals[n]
        opts_svd['tol'] = tol_svds[n]
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = tensordot(M.A[n], init, axes=(2, 1))
            tmp.swap_gate(axes=(0, 2), inplace=True)
            u, s, _ = svd(tmp, axes=((2, 1, 4), (0, 3)), **optsK_svd)  # discard V
            init = tensordot(u, s, axes=(3, 0))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, to='last')
                psi.absorb_central(to='last')
        #
        if (SV_min[n] > tol_svds[n] and Ds[n] < D_totals[n]) or (Ds[n] > D_totals[n]):  # choose 2site
            version = versions[1]
        else:  # choose 1site
            version = versions[0]
        #
        # print(n, version, SV_min[n] , tol_svds[n] , Ds[n] , D_totals[n])
        if version == '1site':
            # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
            if H:
                f = lambda v: env.Heff1(v, n)
                psi.A[n] = expmv(f, psi.A[n], t=0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            # canonize and save
            psi.orthogonalize_site(n, to='last')
            env.clear_site(n)
            env.update(n, to='last')

            # backward in time evolution of a central site: T(-dt*.5)
            if H and n != psi.sweep(to='last')[-1]:
                f = lambda v: env.Heff0(v, psi.pC)
                psi.A[psi.pC] = expmv(f, psi.A[psi.pC], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            psi.absorb_central(to='last')
        elif version == '2site':
            if n == psi.sweep(to='last')[-1]:
                env.clear_site(n)
                env.update(n, to='first')
            else:
                # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
                if H:
                    n1, _, _ = psi.g.from_site(n, to='last')
                    init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
                    f = lambda v: env.Heff2(v, n)
                    init = expmv(f, v=init, t=0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                    # split and save
                    A1, S, A2 = svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                    psi.A[n] = A1
                    psi.A[n1] = S.tensordot(A2, axes=(1, psi.left))
                env.clear_site(n)
                env.update(n, to='last')

                # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
                if H and n != psi.sweep(to='last', dl=1)[-1]:
                    f = lambda v: env.Heff1(v, n1)
                    psi.A[n1] = expmv(f, psi.A[n1], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        elif version == '2site_group':
            if n == psi.sweep(to='last')[-1]:
                env.clear_site(n)
                env.update(n, to='first')
            else:
                # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
                if H:
                    n1, _, _ = psi.g.from_site(n, to='last')
                    init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
                    init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
                    f = lambda v: env.Heff2_group(v, n)
                    init = expmv(f, v=init, t=0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                    # split and save
                    init.unfuse_legs(axes=1, inplace=True)
                    A1, S, A2 = svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                    psi.A[n] = A1
                    psi.A[n1] = S.tensordot(A2, axes=(1, psi.left))
                env.clear_site(n)
                env.update(n, to='last')

                # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
                if H and n != psi.sweep(to='last', dl=1)[-1]:
                    f = lambda v: env.Heff1(v, n1)
                    psi.A = expmv(f, psi.A[n1], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
    Ds = psi.get_D()
    for n in psi.sweep(to='first'):
        opts_svd['D_total'] = D_totals[n]
        opts_svd['tol'] = tol_svds[n]
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = tensordot(M.A[n], init, axes=(2, 1))
            tmp.swap_gate(axes=(0, 2), inplace=True)
            u, s, _ = svd(tmp, axes=((2, 1, 4), (0, 3)), **optsK_svd)  # discard V
            init = tensordot(u, s, axes=(3, 0))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, to='first')
                psi.absorb_central(to='first')
        #
        if (SV_min[n] > tol_svds[n] and Ds[n] < D_totals[n]) or (Ds[n] > D_totals[n]):  # choose 2site
            version = versions[1]
        else:  # choose 1site
            version = versions[0]
        #print(n, version, SV_min[n] , tol_svds[n] , Ds[n] , D_totals[n])
        #
        if version == '1site':
            if H:  # forward in time evolution of a central site: T(+dt*.5)
                f = lambda v: env.Heff1(v, n)
                psi.A[n] = expmv(f, psi.A[n], t=0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)

            psi.orthogonalize_site(n, to='first')
            env.clear_site(n)
            env.update(n, to='first')

            # backward in time evolution of a central site: T(-dt*.5)
            if H and n != psi.sweep(to='first')[-1]:
                f = lambda v: env.Heff0(v, psi.pC)
                psi.A[psi.pC] = expmv(f, psi.A[psi.pC], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
            psi.absorb_central(to='first')
        elif version == '2site':
            if n == psi.sweep(to='first')[-1]:
                env.clear_site(n)
                env.update(n, to='first')
            else:
                # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
                if H:
                    n1, _, _ = psi.g.from_site(n, to='first')
                    init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
                    f = lambda v: env.Heff2(v, n1)
                    init = expmv(f, v=init, t=0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                    # split and save
                    A1, S, A2 = svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                    psi.A[n1] = A1.tensordot(S, axes=(psi.right, 0))
                    psi.A[n] = A2
                env.clear_site(n)
                env.update(n, to='first')

                # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
                if H and n != psi.sweep(to='first', df=1)[-1]:
                    f = lambda v: env.Heff1(v, n1)
                    psi.A[n1] = expmv(f, psi.A[n1], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
        elif version == '2site_group':
            if n == psi.sweep(to='first')[-1]:
                env.clear_site(n)
                env.update(n, to='first')
            else:
                # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
                if H:
                    n1, _, _ = psi.g.from_site(n, to='first')
                    init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
                    init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
                    f = lambda v: env.Heff2_group(v, n1)
                    init = expmv(format, v=init, t=0.5 * dt, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
                    # split and save
                    init.unfuse_legs(axes=1, inplace=True)
                    A1, S, A2 = svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                    psi.A[n1] = A1.tensordot(S, axes=(psi.right, 0))
                    psi.A[n] = A2
                env.clear_site(n)
                env.update(n, to='first')

                # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
                if H and n != psi.sweep(to='first', df=1)[-1]:
                    f = lambda v: env.Heff1(v, n1)
                    psi.A[n1] = expmv(f, psi.A[n1], t=-dt * .5, tol=exp_tol, ncv=k, hermitian=hermitian, normalize=True)
    return env
