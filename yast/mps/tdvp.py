from yast import linalg
import logging
import numpy as np
from .env3 import Env3


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
        env.setup_to_first()
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
                env = tdvp_sweep_1site(psi=psi, H=H, M=M, dt=dt, env=env, hermitian=hermitian, fermionic=fermionic,
                                       k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd, algorithm=algorithm)
            elif version == '2site':
                env = tdvp_sweep_2site(psi=psi, H=H, M=M, dt=dt, env=env, hermitian=hermitian, fermionic=fermionic,
                                       k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd, algorithm=algorithm)
            elif version == '2site_group':
                env = tdvp_sweep_2site_group(psi=psi, H=H, M=M, dt=dt, env=env, hermitian=hermitian, fermionic=fermionic,
                                             k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd, algorithm=algorithm)
            else:
                env = tdvp_sweep_mix(psi=psi, H=H, M=M, dt=dt, env=env, hermitian=hermitian, fermionic=fermionic, versions=versions, D_totals=D_totals, tol_svds=tol_svds, SV_min=SV_min,
                                     k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd, algorithm=algorithm)
        E = env.measure()
        dE = abs(E - E0)
        # print('Iteration: ', sweep, ' energy: ', E, ' dE: ', dE, ' D: ', max(psi.get_D()))
        E0 = E
        curr_t += dt
    return env, E, dE


def tdvp_sweep_1site(psi, H=False, M=False, dt=1., env=None, hermitian=True, fermionic=False, k=4,  eigs_tol=1e-12, exp_tol=1e-12, bi_orth=True, NA=None, opts_svd=None, optsK_svd=None, algorithm='arnoldi'):
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
        env.setup_to_first()

    for n in psi.g.sweep(to='last'):  # sweep from fist to last
        if M:  # apply the Kraus operator
            tmp = M.A[n].tensordot(psi.A[n], axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = linalg.svd(tmp, axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            psi.A[n] = u.tensordot(s, axes=((3,), (0,))).transpose(axes=(0, 1, 3, 2))

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            init = psi.A[n]
            psi.A[n] = linalg.expmv(f=lambda v: env.Heff1(v, n), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
        # canonize and save
        psi.orthogonalize_site(n, towards=psi.g.last)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

        # backward in time evolution of a central site: T(-dt*.5)
        if H and n != psi.g.sweep(to='last')[-1]:
            init = psi.A[psi.pC]
            psi.A[psi.pC] = linalg.expmv(f=lambda v: env.Heff0(v, psi.pC), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
        psi.absorb_central(towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        init = psi.A[n]
        if H:  # forward in time evolution of a central site: T(+dt*.5)
            init = linalg.expmv(f=lambda v: env.Heff1(v, n), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)

        if M:  # apply the Kraus operator
            tmp = M.A[n].tensordot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = linalg.svd(tmp, axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.tensordot(s, axes=((3,), (0,)))
            init = init.transpose(axes=(0, 1, 3, 2))

        # canonize and save
        psi.A[n] = init
        psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)

        # backward in time evolution of a central site: T(-dt*.5)
        if H and n != psi.g.sweep(to='first')[-1]:
            init = psi.A[psi.pC]
            psi.A[psi.pC] = linalg.expmv(f=lambda v: env.Heff0(v, psi.pC), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
        psi.absorb_central(towards=psi.g.first)
    return env


def tdvp_sweep_2site(psi, H=False, M=False, dt=1., env=None, hermitian=True, fermionic=False, k=4, eigs_tol=1e-12, exp_tol=1e-12, bi_orth=True, NA=None, opts_svd=None, optsK_svd=None, algorithm='arnoldi'):
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

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].tensordot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = linalg.svd(tmp, axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.tensordot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.last)
                psi.absorb_central(towards=psi.g.last)

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
            init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
            init = linalg.expmv(f=lambda v: env.Heff2(v, n), v=init, t=+0.5 * dt, tol=eigs_tol, ncv=k, hermitian=hermitian)
            # split and save
            A1, S, A2 = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n] = A1
            psi.A[n1] = S.tensordot(A2, axes=(1, psi.left))
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.g.sweep(to='last', dl=1)[-1]:
            init = psi.A[n1]
            psi.A[n1] = linalg.expmv(f=lambda v: env.Heff1(v, n1), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)


    for n in psi.g.sweep(to='first', df=1):
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].tensordot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = linalg.svd(tmp, axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.tensordot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.first)
                psi.absorb_central(towards=psi.g.first)

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
            init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
            init = linalg.expmv(f=lambda v: env.Heff2(v, n1), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
            # split and save
            A1, S, A2 = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n1] = A1.tensordot(S, axes=(psi.right, 0))
            psi.A[n] = A2
        env.clear_site(n)
        env.update(n, towards=psi.g.first)

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.g.sweep(to='first', df=1)[-1]:
            init = psi.A[n1]
            psi.A[n1] = linalg.expmv(f=lambda v: env.Heff1(v, n1), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
    env.clear_site(n1)
    env.update(n1, towards=psi.g.first)

    return env


def tdvp_sweep_2site_group(psi, H=False, M=False, dt=1, env=None, hermitian=True, fermionic=False, k=4, eigs_tol=1e-12, exp_tol=1e-12, bi_orth=True, NA=None, opts_svd=None, optsK_svd=None, algorithm='arnoldi'):
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
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].tensordot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = linalg.svd(tmp, axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.tensordot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.last)
                psi.absorb_central(towards=psi.g.last)

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
            init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
            init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
            init = linalg.expmv(f=lambda v: env.Heff2_group(v, n), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
            # split and save
            init.unfuse_legs(axes=1, inplace=True)
            A1, S, A2 = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n] = A1
            psi.A[n1] = S.tensordot(A2, axes=(1, psi.left))
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.g.sweep(to='last', dl=1)[-1]:
            init = psi.A[n1]
            psi.A[n1] = linalg.expmv(f=lambda v: env.Heff1(v, n1), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)

    for n in psi.g.sweep(to='first', df=1):
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].tensordot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = linalg.svd(tmp, axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.tensordot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.first)
                psi.absorb_central(towards=psi.g.first)

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
            init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
            init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
            init = linalg.expmv(f=lambda v: env.Heff2_group(v, n1), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
            # split and save
            init.unfuse_legs(axes=1, inplace=True)
            A1, S, A2 = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n1] = A1.tensordot(S, axes=(psi.right, 0))
            psi.A[n] = A2
        env.clear_site(n)
        env.update(n, towards=psi.g.first)

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.g.sweep(to='first', df=1)[-1]:
            init = psi.A[n1]
            psi.A[n1] = linalg.expmv(f=lambda v: env.Heff1(v, n1), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
    env.clear_site(n1)
    env.update(n1, towards=psi.g.first)

    return env


def tdvp_sweep_mix(psi, SV_min, versions, H=False, M=False, dt=1., env=None, hermitian=True, fermionic=False, k=4, eigs_tol=1e-12, exp_tol=1e-12, bi_orth=True, NA=None, D_totals=None, tol_svds=None, opts_svd=None, optsK_svd=None, algorithm='arnoldi'):
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
        env.setup_to_first()
    for n in psi.g.sweep(to='last'):
        opts_svd['D_total'] = D_totals[n]
        opts_svd['tol'] = tol_svds[n]
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].tensordot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = linalg.svd(tmp, axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.tensordot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.last)
                psi.absorb_central(towards=psi.g.last)
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
                init = psi.A[n]
                psi.A[n] = linalg.expmv(f=lambda v: env.Heff1(v, n), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
            # canonize and save
            psi.orthogonalize_site(n, towards=psi.g.last)
            env.clear_site(n)
            env.update(n, towards=psi.g.last)

            # backward in time evolution of a central site: T(-dt*.5)
            if H and n != psi.g.sweep(to='last')[-1]:
                init = psi.A[psi.pC]
                psi.A[psi.pC] = linalg.expmv(f=lambda v: env.Heff0(v, psi.pC), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
            psi.absorb_central(towards=psi.g.last)
        elif version == '2site':
            if n == psi.g.sweep(to='last')[-1]:
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
            else:
                # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
                if H:
                    n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
                    init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
                    init = linalg.expmv(f=lambda v: env.Heff2(v, n), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
                    # split and save
                    A1, S, A2 = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                    psi.A[n] = A1
                    psi.A[n1] = S.tensordot(A2, axes=(1, psi.left))
                env.clear_site(n)
                env.update(n, towards=psi.g.last)

                # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
                if H and n != psi.g.sweep(to='last', dl=1)[-1]:
                    init = psi.A[n1]
                    psi.A[n1] = linalg.expmv(f=lambda v: env.Heff1(v, n1), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
        elif version == '2site_group':
            if n == psi.g.sweep(to='last')[-1]:
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
            else:
                # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
                if H:
                    n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
                    init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
                    init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
                    init = linalg.expmv(f=lambda v: env.Heff2_group(v, n), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
                    # split and save
                    init.unfuse_legs(axes=1, inplace=True)
                    A1, S, A2 = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                    psi.A[n] = A1
                    psi.A[n1] = S.tensordot(A2, axes=(1, psi.left))
                env.clear_site(n)
                env.update(n, towards=psi.g.last)

                # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
                if H and n != psi.g.sweep(to='last', dl=1)[-1]:
                    init = psi.A[n1]
                    psi.A = linalg.expmv(f=lambda v: env.Heff1(v, n1), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
    Ds = psi.get_D()
    for n in psi.g.sweep(to='first'):
        opts_svd['D_total'] = D_totals[n]
        opts_svd['tol'] = tol_svds[n]
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].tensordot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = linalg.svd(tmp, axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.tensordot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.first)
                psi.absorb_central(towards=psi.g.first)
        #
        if (SV_min[n] > tol_svds[n] and Ds[n] < D_totals[n]) or (Ds[n] > D_totals[n]):  # choose 2site
            version = versions[1]
        else:  # choose 1site
            version = versions[0]
        #print(n, version, SV_min[n] , tol_svds[n] , Ds[n] , D_totals[n])
        #
        if version == '1site':
            init = psi.A[n]
            if H:  # forward in time evolution of a central site: T(+dt*.5)
                init = linalg.expmv(f=lambda v: env.Heff1(v, n), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)

            # canonize and save
            psi.A[n] = init
            psi.orthogonalize_site(n, towards=psi.g.first)
            env.clear_site(n)
            env.update(n, towards=psi.g.first)

            # backward in time evolution of a central site: T(-dt*.5)
            if H and n != psi.g.sweep(to='first')[-1]:
                init = psi.A[psi.pC]
                psi.A[psi.pC] = linalg.expmv(f=lambda v: env.Heff0(v, psi.pC), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
            psi.absorb_central(towards=psi.g.first)
        elif version == '2site':
            if n == psi.g.sweep(to='first')[-1]:
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
            else:
                # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
                if H:
                    n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
                    init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
                    init = linalg.expmv(f=lambda v: env.Heff2(v, n1), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
                    # split and save
                    A1, S, A2 = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                    psi.A[n1] = A1.tensordot(S, axes=(psi.right, 0))
                    psi.A[n] = A2
                env.clear_site(n)
                env.update(n, towards=psi.g.first)

                # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
                if H and n != psi.g.sweep(to='first', df=1)[-1]:
                    init = psi.A[n1]
                    psi.A[n1] = linalg.expmv(f=lambda v: env.Heff1(v, n1), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
        elif version == '2site_group':
            if n == psi.g.sweep(to='first')[-1]:
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
            else:
                # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
                if H:
                    n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
                    init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
                    init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
                    init = linalg.expmv(f=lambda v: env.Heff2_group(v, n1), v=init, t=+dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
                    # split and save
                    init.unfuse_legs(axes=1, inplace=True)
                    A1, S, A2 = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                    psi.A[n1] = A1.tensordot(S, axes=(psi.right, 0))
                    psi.A[n] = A2
                env.clear_site(n)
                env.update(n, towards=psi.g.first)

                # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
                if H and n != psi.g.sweep(to='first', df=1)[-1]:
                    init = psi.A[n1]
                    psi.A[n1] = linalg.expmv(f=lambda v: env.Heff1(v, n1), v=init, t=-dt * .5, tol=eigs_tol, ncv=k, hermitian=hermitian)
    return env
