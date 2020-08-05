import numpy as np
from yamps.mps import Env3
from yamps.tensor.eigs import expmw
import warnings
#################################
#           tdvp                #
#################################
# TO PUSH: aux d.o.f are removed hot to obtain nice evolution on purificatin ?


class TDVPWarning(UserWarning):
    pass


def tdvp_OBC(psi, tmax, dt=1, H=False, M=False, env=None, measure_O=None, cutoff_sweep=20, cutoff_dE=1e-9, hermitian=True, fermionic=False, k=4, eigs_tol=1e-14, exp_tol=1e-14, dtype='complex128', bi_orth=True, NA=None, version='1site', opts_svd=None, optsK_svd=None):
    # evolve with TDVP method, up to tmax and initial guess of the time step dt
    # meaure_O - list of things to measure e.g.  [2,[OL, OR], [1,2,3]] -
    # measure exp.  val of 2-site operator OL, OR on sites (1,2), (2,3), (3,4)
    # opts - optional info for MPS truncation
    sweep = 0
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
        dt = min([abs(tmax - curr_t), abs(dt)]) * \
            (np.sign(dt.real)+np.sign(dt.imag)*1j)
        if not H and not M:
            print('yamps.tdvp: Neither Hamiltonian nor Kraus operators defined.')
        else:
            if version == '2site':
                env = tdvp_sweep_2site(psi=psi, H=H, M=M, dt=dt, env=env, dtype=dtype, hermitian=hermitian, fermionic=fermionic,
                                       k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd)
            elif version == '2site_group':
                env = tdvp_sweep_2site(psi=psi, H=H, M=M, dt=dt, env=env, dtype=dtype, hermitian=hermitian, fermionic=fermionic,
                                       k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd)
            else:
                env = tdvp_sweep_1site(psi=psi, H=H, M=M, dt=dt, env=env, dtype=dtype, hermitian=hermitian, fermionic=fermionic,
                                       k=k, eigs_tol=eigs_tol, exp_tol=exp_tol, bi_orth=bi_orth, NA=NA, opts_svd=opts_svd, optsK_svd=optsK_svd)

        E = env.measure()
        dE = abs(E - E0)
        print('Iteration: ', sweep, ' energy: ', E, ' dE: ', dE, ' time: ',
              curr_t, ' norma:', psi.norma, ' D: ', max(psi.get_D()))
        E0 = E
        sweep += 1
        curr_t += dt
        #
        out = (env, E, dE,)
    return out


def tdvp_sweep_1site(psi, H=False, M=False, dt=1., env=None, dtype='complex128', hermitian=True, fermionic=False, k=4, eigs_tol=1e-14, exp_tol=1e-14, bi_orth=True, NA=None, opts_svd=None, optsK_svd=None):
    r"""
    Perform sweep with 1site-TDVP by applying exp(-i*dt*H) on initial vector. Note the convention for time step sign.
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
        default = 1. (real time evolution, forward in time)
        time interval for matrix expontiation. May be divided into smaller intervals according to the cost function.
        sign(dt) = +1j for imaginary time evolution.
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

    if opts_svd:
        warnings.warn("tdvp_sweep_1site: Truncation not implemeted.",  TDVPWarning)

    # change. adjust the sign according to the convention
    sgn = 1j * (np.sign(dt.real) + 1j * np.sign(dt.imag))
    dt = sgn * abs(dt)

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last'):  # sweep from fist to last
        if M:  # apply the Kraus operator
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = tmp.split_svd(
                axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.dot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            init = psi.A[n]
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff1(v, n), init=[init], Bv=lambda v: env.Heff1(
                    v, n, conj=True), dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol, k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff1(v, n), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
        # canonize and save
        psi.A[n] = init
        psi.orthogonalize_site(n, towards=psi.g.last)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

        # backward in time evolution of a central site: T(-dt*.5)
        if H and n != psi.N - 1:
            init = psi.A[psi.pC]
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff0(v, psi.pC), init=[init], Bv=lambda v: env.Heff0(
                    v, psi.pC, conj=True), dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff0(v, psi.pC), init=[
                             init], dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[psi.pC] = init[0]
        psi.absorb_central(towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        init = psi.A[n]
        if H:  # forward in time evolution of a central site: T(+dt*.5)
            init = psi.A[n]
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff1(v, n), init=[init], Bv=lambda v: env.Heff1(
                    v, n, conj=True), dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff1(v, n), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]

        if M:  # apply the Kraus operator
            tmp = M.A[n].dot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = tmp.split_svd(
                axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.dot(s, axes=((3,), (0,)))
            init = init.transpose(axes=(0, 1, 3, 2))

        # canonize and save
        psi.A[n] = init
        psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)

        # backward in time evolution of a central site: T(-dt*.5)
        if H and n != 0:
            init = psi.A[psi.pC]
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff0(v, psi.pC), init=[init], Bv=lambda v: env.Heff0(
                    v, psi.pC, conj=True), dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff0(v, psi.pC), init=[
                             init], dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[psi.pC] = init[0]
        psi.absorb_central(towards=psi.g.first)

    return env


def tdvp_sweep_2site(psi, H=False, M=False, dt=1., env=None, dtype='complex128', hermitian=True, fermionic=False, k=4, eigs_tol=1e-14, exp_tol=1e-14, bi_orth=True, NA=None, opts_svd=None, optsK_svd=None):
    r"""
    Perform sweep with 2site-TDVP.
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
        default = 1. (real time evolution, forward in time)
        time interval for matrix expontiation. May be divided into smaller intervals according to the cost function.
        sign(dt) = +1j for imaginary time evolution
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
    # change. adjust the sign according to the convention
    sgn = 1j * (np.sign(dt.real) + 1j * np.sign(dt.imag))
    dt = sgn * abs(dt)

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = tmp.split_svd(
                axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.dot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.last)
                psi.absorb_central(towards=psi.g.last)

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
            init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff2(v, n), Bv=lambda v: env.Heff2(v, n, conj=True), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff2(v, n), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            # split and save
            A1, S, A2 = init[0].split_svd(axes=(psi.left + psi.phys, tuple(
                a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n] = A1
            psi.A[n1] = A2.dot_diag(S, axis=psi.left)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.g.sweep(to='last', dl=1)[-1]:
            init = psi.A[n1]
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff1(v, n1), init=[init], Bv=lambda v: env.Heff1(
                    v, n1, conj=True), dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff1(v, n1), init=[
                             init], dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n1] = init[0]

    for n in psi.g.sweep(to='first', df=1):
        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.g.sweep(to='first', df=1)[-1]:
            init = psi.A[n]
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff1(v, n), init=[init], Bv=lambda v: env.Heff1(
                    v, n, conj=True), dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff1(v, n), init=[
                             init], dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n] = init[0]

        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = tmp.split_svd(
                axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.dot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.first)
                psi.absorb_central(towards=psi.g.first)

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
            init = psi.A[n1].dot(psi.A[n], axes=(psi.right, psi.left))
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff2(v, n1), Bv=lambda v: env.Heff2(v, n1, conj=True), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff2(v, n1), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            # split and save
            A1, S, A2 = init[0].split_svd(axes=(psi.left + psi.phys, tuple(
                a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n1] = A1.dot_diag(S, axis=psi.right)
            psi.A[n] = A2
        env.clear_site(n)
        env.update(n, towards=psi.g.first)
    env.clear_site(n1)
    env.update(n1, towards=psi.g.first)

    return env


def tdvp_sweep_2site_group(psi, H=False, M=False, dt=1., env=None, dtype='complex128', hermitian=True, fermionic=False, k=4, eigs_tol=1e-14, exp_tol=1e-14, bi_orth=True, NA=None, opts_svd=None, optsK_svd=None):
    r"""
    Perform sweep with 2site-TDVP with grouping neigbouring sites.
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
        default = 1. (real time evolution, forward in time)
        time interval for matrix expontiation. May be divided into smaller intervals according to the cost function.
        sign(dt) = +1j for imaginary time evolution.
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
    # change. adjust the sign according to the convention
    sgn = 1j * (np.sign(dt.real) + 1j * np.sign(dt.imag))
    dt = sgn * abs(dt)

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = tmp.split_svd(
                axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.dot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.last)
                psi.absorb_central(towards=psi.g.last)

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
            init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
            init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff2_group(v, n), Bv=lambda v: env.Heff2_group(v, n, conj=True), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff2_group(v, n), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            # split and save
            init = init[0].ungroup_leg(axis=1, leg_order=leg_order)
            A1, S, A2 = init.split_svd(axes=(psi.left + psi.phys, tuple(
                a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n] = A1
            psi.A[n1] = A2.dot_diag(S, axis=psi.left)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.g.sweep(to='last', dl=1)[-1]:
            init = psi.A[n1]
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff1(v, n1), init=[init], Bv=lambda v: env.Heff1(
                    v, n1, conj=True), dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff1(v, n1), init=[
                             init], dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n1] = init[0]

    for n in psi.g.sweep(to='first', df=1):
        # matrix exponentiation, backward in time evolution of a single site: T(-dt*.5)
        if H and n != psi.g.sweep(to='first', df=1)[-1]:
            init = psi.A[n]
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff1(v, n), init=[init], Bv=lambda v: env.Heff1(
                    v, n, conj=True), dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff1(v, n), init=[
                             init], dt=-dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n] = init[0]

        if M:  # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,), (1,)))
            tmp.swap_gate(axes=(0, 2), fermionic=fermionic)
            u, s, _ = tmp.split_svd(
                axes=((2, 1, 4), (0, 3)), opts=optsK_svd)  # discard V
            init = u.dot(s, axes=((3,), (0,)))
            psi.A[n] = init.transpose(axes=(0, 1, 3, 2))
            if not H:
                psi.orthogonalize_site(n, towards=psi.g.first)
                psi.absorb_central(towards=psi.g.first)

        # matrix exponentiation, forward in time evolution of a single site: T(+dt*.5)
        if H:
            n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
            init = psi.A[n1].dot(psi.A[n], axes=(psi.right, psi.left))
            init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
            if not hermitian:
                init = expmw(Av=lambda v: env.Heff2_group(v, n1), Bv=lambda v: env.Heff2_group(v, n1, conj=True), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=False, bi_orth=bi_orth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=lambda v: env.Heff2_group(v, n1), init=[
                             init], dt=+dt * .5, eigs_tol=eigs_tol, exp_tol=exp_tol,  k=k, hermitian=True, dtype=dtype, NA=NA)
            # split and save
            init = init[0].ungroup_leg(axis=1, leg_order=leg_order)
            A1, S, A2 = init.split_svd(axes=(psi.left + psi.phys, tuple(
                a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
            psi.A[n1] = A1.dot_diag(S, axis=psi.right)
            psi.A[n] = A2
        env.clear_site(n)
        env.update(n, towards=psi.g.first)
    env.clear_site(n1)
    env.update(n1, towards=psi.g.first)

    return env
