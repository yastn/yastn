""" Linalg methods for yast tensor. """

import numpy as np
from scipy.linalg import expm as expm
from ._auxliary import _clear_axes, _unpack_axes, _common_keys
from ._auxliary import YastError, _check, _test_tensors_match, _test_all_axes
from ._merging import _merge_to_matrix, _unmerge_matrix, _unmerge_diagonal
from ._merging import _leg_struct_trivial, _leg_struct_truncation
from ._contractions import vdot

__all__ = ['svd', 'svd_lowrank', 'qr', 'eigh', 'norm', 'norm_diff', 'entropy', 'expmv', 'eigs']


def norm(a, p='fro'):
    r"""
    Norm of the rensor.

    Parameters
    ----------
    p: str
        'fro' = Frobenious; 'inf' = max(abs())

    Returns
    -------
    norm : float64
    """
    if len(a.A) == 0:
        return a.zero_of_dtype()
    return a.config.backend.norm(a.A, p)


def norm_diff(a, b, p='fro'):
    """
    Norm of the difference of two tensors.

    Parameters
    ----------
    other: Tensor

    ord: str
        'fro' = Frobenious; 'inf' = max(abs())

    Returns
    -------
    norm : float64
    """
    _test_tensors_match(a, b)
    if (len(a.A) == 0) and (len(b.A) == 0):
        return a.zero_of_dtype()
    meta = _common_keys(a.A, b.A)
    return a.config.backend.norm_diff(a.A, b.A, meta, p)


def svd_lowrank(a, axes=(0, 1), sU=1, nU=True, Uaxis=-1, Vaxis=0,
                tol=0, D_block=6, D_total=np.inf,
                keep_multiplets=False, eps_multiplet=1e-14,
                n_iter=60, k_fac=6, **kwargs):
    r"""
    Split tensor into U @ S @ V using svd. Can truncate smallest singular values.

    Truncate based on relative tolerance, bond dimension of each block,
    and total bond dimension from all blocks (whichever gives smaller bond dimension).
    By default, do not truncate.

    Charge of tensor a is attached to U if nU and to V otherwise.

    Parameters
    ----------
    axes: tuple
        Specify two groups of legs between which to perform svd, as well as
        their final order.

    sU: int
        signature of the new leg in U; equal 1 or -1. Default is 1.

    Uaxis, Vaxis: int
        specify which leg of U and V tensors are connecting with S. By default
        it is the last leg of U and the first of V.

    tol: float
        relative tolerance of singular values below which to truncate.

    D_block: int
        largest number of singular values to keep in a single block.
        also used in lowrank svd in the backend

    D_total: int
        largest total number of singular values to keep.

    n_iter, k_fac: ints
        number of iterations and multiplicative factor of stored singular values in lowrank svd procedure
        (relevant options might depend on backend)

    Returns
    -------
    U, S, V: Tensor
        U and V are unitary projectors. S is diagonal.
    """
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a, lout_l, lout_r)
    _test_all_axes(a, axes)

    s_eff = (-sU, sU)
    Am, ls_l, ls_r, ul, ur = _merge_to_matrix(a, axes, s_eff)

    if nU:
        meta = tuple((il + ir, il + ir, ir, ir + ir) for il, ir in zip(ul, ur))
        n_l, n_r = a.struct.n, None
    else:
        meta = tuple((il + ir, il + il, il, il + ir) for il, ir in zip(ul, ur))
        n_l, n_r = None, a.struct.n
    U = a.__class__(config=a.config, s=ls_l.s + (sU,), n=n_l,
                    meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])
    S = a.__class__(config=a.config, s=s_eff, isdiag=True)
    V = a.__class__(config=a.config, s=(-sU,) + ls_r.s, n=n_r,
                    meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r])

    U.A, S.A, V.A = a.config.backend.svd_lowrank(Am, meta, D_block, n_iter, k_fac)

    ls_s = _leg_struct_truncation(S, tol, D_block, D_total, keep_multiplets, eps_multiplet, 'svd')
    _unmerge_matrix(U, ls_l, ls_s)
    _unmerge_diagonal(S, ls_s)
    _unmerge_matrix(V, ls_s, ls_r)
    U.moveaxis(source=-1, destination=Uaxis, inplace=True)
    V.moveaxis(source=0, destination=Vaxis, inplace=True)
    return U, S, V


def svd(a, axes=(0, 1), sU=1, nU=True, Uaxis=-1, Vaxis=0,
        tol=0, D_block=np.inf, D_total=np.inf,
        keep_multiplets=False, eps_multiplet=1e-14, **kwargs):
    r"""
    Split tensor into U @ S @ V using svd. Can truncate smallest singular values.

    Truncate based on relative tolerance, bond dimension of each block,
    and total bond dimension from all blocks (whichever gives smaller bond dimension).
    By default, do not truncate.

    Charge of tensor a is attached to U if nU and to V otherwise.

    Parameters
    ----------
    axes: tuple
        Specify two groups of legs between which to perform svd, as well as
        their final order.

    sU: int
        signature of the new leg in U; equal 1 or -1. Default is 1.

    Uaxis, Vaxis: int
        specify which leg of U and V tensors are connecting with S. By default
        it is the last leg of U and the first of V.

    tol: float
        relative tolerance of singular values below which to truncate.

    D_block: int
        largest number of singular values to keep in a single block.

    D_total: int
        largest total number of singular values to keep.

    Returns
    -------
    U, S, V: Tensor
        U and V are unitary projectors. S is diagonal.
    """
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a, lout_l, lout_r)
    _test_all_axes(a, axes)

    s_eff = (-sU, sU)
    Am, ls_l, ls_r, ul, ur = _merge_to_matrix(a, axes, s_eff)

    if nU:
        meta = tuple((il + ir, il + ir, ir, ir + ir) for il, ir in zip(ul, ur))
        n_l, n_r = a.struct.n, None
    else:
        meta = tuple((il+ir, il+il, il, il+ir) for il, ir in zip(ul, ur))
        n_l, n_r = None, a.struct.n
    U = a.__class__(config=a.config, s=ls_l.s + (sU,), n=n_l,
                    meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])
    S = a.__class__(config=a.config, s=s_eff, isdiag=True)
    V = a.__class__(config=a.config, s=(-sU,) + ls_r.s, n=n_r,
                    meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r])
    U.A, S.A, V.A = a.config.backend.svd(Am, meta)

    ls_s = _leg_struct_truncation(
        S, tol, D_block, D_total, keep_multiplets, eps_multiplet, 'svd')

    _unmerge_matrix(U, ls_l, ls_s)
    _unmerge_diagonal(S, ls_s)
    _unmerge_matrix(V, ls_s, ls_r)
    U.moveaxis(source=-1, destination=Uaxis, inplace=True)
    V.moveaxis(source=0, destination=Vaxis, inplace=True)
    return U, S, V


def qr(a, axes=(0, 1), sQ=1, Qaxis=-1, Raxis=0):
    r"""
    Split tensor using reduced qr decomposition.
    Charge of R is zero.

    Parameters
    ----------
    axes: tuple
        Specify two groups of legs between which to perform svd, as well as their final order.

    sQ: int
        signature of connecting leg in Q; equal 1 or -1. Default is 1.

    Qaxis, Raxis: int
        specify which leg of Q and R tensors are connecting to the other tensor.
        By delault it is the last leg of Q and the first leg of R.

    Returns
    -------
        Q, R: Tensor
    """
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a, lout_l, lout_r)
    _test_all_axes(a, axes)

    s_eff = (-sQ, sQ)
    Am, ls_l, ls_r, ul, ur = _merge_to_matrix(a, axes, s_eff)

    Qs = tuple(a.struct.s[lg] for lg in axes[0]) + (sQ,)
    Rs = (-sQ,) + tuple(a.struct.s[lg] for lg in axes[1])
    Q = a.__class__(config=a.config, s=Qs, n=a.struct.n, meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])
    R = a.__class__(config=a.config, s=Rs, meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r])

    meta = tuple((il + ir, il + ir, ir + ir) for il, ir in zip(ul, ur))
    Q.A, R.A = a.config.backend.qr(Am, meta)

    ls = _leg_struct_trivial(R, axis=0)
    _unmerge_matrix(Q, ls_l, ls)
    _unmerge_matrix(R, ls, ls_r)
    Q.moveaxis(source=-1, destination=Qaxis, inplace=True)
    R.moveaxis(source=0, destination=Raxis, inplace=True)
    return Q, R


def eigh(a, axes, sU=1, Uaxis=-1, tol=0, D_block=np.inf, D_total=np.inf,
         keep_multiplets=False, eps_multiplet=1e-14):
    r"""
    Split tensor using eig, tensor = U * S * U^dag. Truncate smallest eigenvalues if neccesary.

    Tensor should be hermitian and has charge 0.
    Truncate using -- whichever gives smaller bond dimension:
    relative tolerance, bond dimension of each block, and total bond dimension from all blocks.
    By default do not truncate.
    Truncate based on tolerance only if some eigenvalues are positive -- than all negative ones are discarded.
    Function primarly intended to be used for positively defined tensors.

    Parameters
    ----------
    axes: tuple
        Specify two groups of legs between which to perform svd, as well as their final order.

    sU: int
        signature of connecting leg in U equall 1 or -1. Default is 1.

    Uaxis: int
        specify which leg of U is the new connecting leg. By delault it is the last leg.

    tol: float
        relative tolerance of singular values below which to truncate.

    D_block: int
        largest number of singular values to keep in a single block.

    D_total: int
        largest total number of singular values to keep.

    Returns
    -------
        S, U: Tensor
            U is unitary projector. S is diagonal.
    """
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a, lout_l, lout_r)
    _test_all_axes(a, axes)

    if any(x != 0 for x in a.struct.n):
        raise YastError('Charge should be zero')

    s_eff = (-sU, sU)
    Am, ls_l, ls_r, ul, ur = _merge_to_matrix(a, axes, s_eff)

    if _check["consistency"] and not (ul == ur and ls_l.match(ls_r)):
        raise YastError(
            'Something went wrong in matching the indices of the two tensors')

    Us = tuple(a.struct.s[lg] for lg in axes[0]) + (sU,)
    S = a.__class__(config=a.config, s=(-sU, sU), isdiag=True)
    U = a.__class__(config=a.config, s=Us, meta_fusion=[
                    a.meta_fusion[ii] for ii in lout_l] + [(1,)])

    # meta = (indA, indS, indU)
    meta = tuple((il + ir, il, il + ir) for il, ir in zip(ul, ur))
    S.A, U.A = a.config.backend.eigh(Am, meta)

    ls_s = _leg_struct_truncation(
        S, tol, D_block, D_total, keep_multiplets, eps_multiplet, 'eigh')
    _unmerge_matrix(U, ls_l, ls_s)
    _unmerge_diagonal(S, ls_s)
    U.moveaxis(source=-1, destination=Uaxis, inplace=True)
    return S, U


def entropy(a, axes=(0, 1), alpha=1):
    r"""
    Calculate entropy from spliting the tensor using svd.

    If diagonal, calculates entropy treating S^2 as probabilities. Normalizes S^2 if neccesary.
    If not diagonal, calculates svd first to get the diagonal S.
    Use base-2 log.

    Parameters
    ----------
    axes: tuple
        Specify two groups of legs between which to perform svd

    alpha: float
        Order of Renyi entropy.
        alpha=1 is von Neuman entropy -Tr(S^2 log2(S^2))
        otherwise: 1/(1-alpha) log2(Tr(S^(2*alpha)))

    Returns
    -------
    entropy, minimal singular value, normalization : float64
    """
    if len(a.A) == 0:
        return a.zero_of_dtype(), a.zero_of_dtype(), a.zero_of_dtype()

    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a, lout_l, lout_r)
    _test_all_axes(a, axes)

    if not a.isdiag:
        Am, *_ = _merge_to_matrix(a, axes, (-1, 1))
        Sm = a.config.backend.svd_S(Am)
    else:
        Sm = {t: a.config.backend.diag_get(x) for t, x in a.A.items()}
    # entropy, Smin, normalization
    return a.config.backend.entropy(Sm, alpha=alpha)


# Krylov based methods, handled by anonymous function decribing action of matrix on a vector

def expmv(f, v, t=1., tol=1e-13, ncv=5, hermitian=False):
    r"""
    Calculate exp(t*A)*v, where v is a yast tensor, and A is linear operator acting on v.
    """
    backend = v.config.backend

    # Krylov parameters
    ncv_max = min([20, v.get_size()])
    ncv = max(1, ncv)

    # Initialize variables
    reject = False
    happy = False

    t_now, t_out = 0, abs(t)
    sgn = t / t_out if t_out > 0 else 0
    tau = t_out  # first quess for a time-step

    gamma = 0.6  # Safety factors
    delta = 1.4

    j = 0  # size of krylov space
    w = v
    ncv_old, tau_old, omega = None, None, None
    order_old, ncv_est_old = True, True

    while t_now < t_out:
        if j == 0:
            H = {}
            beta = w.norm()
            if beta == 0:  # multiply with a zero vector; result is zero # TODO handle this
                tau = t_out - t_now
                break
            V = [w / beta]  # first vector in Krylov basis
        while j < ncv:
            w = f(V[-1])
            for i in range(j + 1):
                H[(i, j)] = vdot(V[i], w)
                w = w.apxb(V[i], x=-H[(i, j)])
            ss = w.norm()
            if ss < tol:
                happy = True
                tau = t_out - t_now
                break
            H[(j + 1, j)] = ss
            V.append(w / ss)
            j += 1

        H[(0, j)] = backend.dtype_scalar(1, device=v.config.device)
        h = H.pop((j, j - 1)) if (j, j - 1) in H else 0
        T = backend.square_matrix_from_dict(H, j + 1, device=v.config.device)
        F = backend.expm((sgn * tau) * T)
        err = abs(beta * h * F[j - 1, j]).item()

        # Error per unit step
        omega_old = omega
        omega = (t_out / tau) * (err / tol)

        # Estimate order
        if ncv == ncv_old and tau != tau_old and reject:
            order = max([1., np.log(omega / omega_old) / np.log(tau / tau_old)])
            order_old = False
        elif order_old or not reject:
            order_old = True
            order = 0.25 * j
        else:
            order_old = True

        # Estimate k
        if ncv != ncv_old and tau == tau_old and reject:
            ncv_est = max([1.1, (omega / omega_old) ** (1. / (ncv_old - ncv))]) if omega > 0 else 1.1
            ncv_est_old = False
        elif ncv_est_old or not reject:
            ncv_est_old = True
            ncv_est = 2
        else:
            ncv_est_old = True

        tau_old, ncv_old = tau, ncv
        if happy:
            omega = 0
            tau_new = tau
            ncv_new = ncv
        elif j == ncv_max and omega > delta:  # Krylov subspace to small and stepsize to large
            tau_new, tau * (omega / gamma) ** (-1. / order)
            ncv_new = j
        else:  # Determine optimal tau and m
            tau_opt = tau * (omega / gamma) ** (-1. / order)
            ncv_opt = max([1., np.ceil(j + np.log(omega / gamma) / np.log(ncv_est))])
            Ctau = ncv * np.ceil((t_out - t_now) / tau_opt)
            Ck = ncv_opt * np.ceil((t_out - t_now) / tau)
            tau_new, ncv_new = (tau_opt, j) if Ctau < Ck else (tau, ncv_opt)

        if omega <= delta:  # Check error against target
            w = (beta * F[0, 0]) * V[0]
            for it in range(1, j + 1):
                w = w.apxb(V[it], x=(beta * F[it, 0]))
            t_now += tau
            j = 0
            reject = False
        else:
            reject = True
            H[(j, j - 1)] = h
            H.pop((0, j))

        # Another safety factors
        tau = min(t_out - t_now, max(0.2 * tau, min(tau_new, 2. * tau)))
        ncv = max(1, min(ncv_max, max(np.floor(.75 * ncv), min(ncv_new, np.ceil(1.3333 * ncv)))))
    return w / w.norm()


def eigs(f, v0, k=1, which=None, ncv=5, maxiter=None, tol=1e-13, hermitian=True):
    r"""
    f: function handlers.
        Action of a matrix on a vector. For non-symmetric lanczos both have to be defined.
    v0: Tensor
        Initial vector for iteration.
    k: int
        default = 1, number of eigenvalues equals number of non-zero Krylov vectors
        The number of eigenvalues and eigenvectors desired. It is not possible to compute all eigenvectors of a matrix.
    sigma: float
        default = None (search for smallest)
        Find eigenvalues near sigma.
    ncv: int
        default = 5
        The number of Lanczos vectors generated ncv must be greater than k; it is recommended that ncv > 2*k.
    which: str, [‘LM’ | ‘SM’ | ‘LR’ | ‘SR’ | ‘LI’ | ‘SI’]
        default = None (search for closest to sigma - if defined and for 'SR' else)
        Which k eigenvectors and eigenvalues to find:
            ‘LM’ : largest magnitude
            ‘SM’ : smallest magnitude
            ‘LR’ : largest real part
            ‘SR’ : smallest real part
            ‘LI’ : largest imaginary part
            ‘SI’ : smallest imaginary part
    tol: float
        defoult = 1e-14
        Relative accuracy for eigenvalues (stopping criterion) for Krylov subspace.
    return_eigenvectors: bool
        default = True
        Return eigenvectors (True) in addition to eigenvalues.
    bi_orth: bool
        default = True
        Option for non-symmetric Lanczos method. Whether to bi-orthonomalize Krylov-subspace vectors.
    algorithm: str
        default = 'arnoldi'
        What method to use. Possible options: arnoldi, lanczos
    """
    
    backend = v0.config.backend
    beta = None
    V = [v0]
    H = {}

    for j in range(ncv):
        w = f(V[-1])
        for i in range(j + 1):
            H[(i, j)] = vdot(V[i], w)
            w = w.apxb(V[i], x=-H[(i, j)])
        H[(j + 1, j)] = w.norm()
        if H[(j + 1, j)] < tol:
            beta, happy = 0, True
            break
        V.append(w /H[(j + 1, j)])
    if beta == None:
        beta, happy = H[(j + 1, j)], False

    T = backend.square_matrix_from_dict(H, j + 1, device=v0.config.device)

    if hermitian:
        val, vr = backend.eigh(T)
    else:
        val, vr = backend.eig(T)

    ind = backend.eigs_which(val, which)

    val, vr = val[ind], vr[:, ind]
    Y = []
    for it in range(k):
        sit = vr[:, it]
        Y.append(sit[0] * V[0])
        for jt in range(1, len(ind)):
            Y[it] = Y[it].apxb(V[jt], x=sit[jt])
    return val[:len(Y)], Y