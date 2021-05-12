""" Linalg methods for yast tensor. """

import numpy as np
from scipy.linalg import expm as expm
import time
import tracemalloc
from ._auxliary import _clear_axes, _unpack_axes, _common_keys
from ._auxliary import YastError, _check, _test_tensors_match, _test_all_axes
from ._merging import _merge_to_matrix, _unmerge_matrix, _unmerge_diagonal
from ._merging import _leg_struct_trivial, _leg_struct_truncation
from ._krylov import krylov


__all__ = ['svd', 'svd_lowrank', 'qr', 'eigh', 'norm', 'norm_diff', 'entropy', 'expmv', 'eigsh']


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

def expmv(Av, init, Bv=None, dt=1, eigs_tol=1e-14, exp_tol=1e-14, k=5, hermitian=False, bi_orth=True, NA=None, cost_estim=0, algorithm='arnoldi'):
    # return expA(Av=Av, Bv=Bv, init=init, dt=dt, eigs_tol=eigs_tol, exp_tol=exp_tol, k=k, hermitian=hermitian, bi_orth=bi_orth, NA=NA, cost_estim=cost_estim, algorithm=algorithm)
    # def expA(Av, init, Bv, dt, eigs_tol, exp_tol, k, hermitian, bi_orth, NA, cost_estim, algorithm):
    backend = init[0].config.backend
    if not hermitian and not Bv:
        raise YastError(
            'expA: For non-hermitian case provide Av and Bv. In addition you can start with two')
    k_max = min([20, init[0].get_size()])
    if not NA:
        # n - cost of vector init
        # NA - cost of matrix Av
        # v_i - cost of exponatiation of size m(m-krylov dim)
        T = backend.expm(
            backend.diag_create( backend.randR((k_max-1)), -1) + \
            backend.diag_create( backend.randR((k_max-1)), +1) + \
            backend.diag_create( backend.randR((k_max)), 0)
            )
        if cost_estim == 0:  # approach 0: defaoult based on matrix size
            # in units of n
            n = init[0].get_size()
            NA = round(4.*n**(0.5), 2)
            v_i = k_max**2/n
            n = 1.
        elif cost_estim == 1:  # approach 1: based on time usage. normalized to time of saving copy of the initial vector. Don't make measures to small
            # in units of v_i
            start_time = time.time()
            init[0].copy()
            n = time.time() - start_time

            start_time = time.time()
            backend.expm(T)
            v_i = time.time() - start_time

            start_time = time.time()
            Av(init[0]).copy()
            NA = (time.time() - start_time)/v_i

            n *= (1./v_i)
            NA *= (1./v_i)
            v_i = 1.
        elif cost_estim == 2:  # approach 2: based on memory usage
            # in units of v_i
            tracemalloc.start()
            init[0].copy()
            n, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            backend.expm(T)
            v_i, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            Av(init[0])
            NA, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            n *= (1./v_i)
            NA *= (1./v_i)
            v_i = 1.
    # Krylov parameters
    mmax = k_max
    m = max([1, k])

    # Initialize variables
    step = 0
    ireject = 0
    reject = 0
    happy = 0
    sgn = (dt/abs(dt)).real*1j if dt.real == 0. else (dt/abs(dt)).real*1
    tnow = 0
    tout = abs(dt)
    j = 0
    tau = abs(dt)

    # Set safety factors
    gamma = 0.8
    delta = 1.2

    # Initial condition
    w = init
    oldm, oldtau, omega = None, None, None
    orderold, kestold = True, True

    # Iterate until we reach the final t
    while tnow < tout:
        # Compute the exponential of the augmented matrix
        err, evec, good = krylov(
            w, Av, Bv, eigs_tol, None, algorithm, hermitian, m, bi_orth, tau=(sgn, tau))

        happy = good[1]
        j = good[2]

        # Error per unit step
        oldomega = omega
        omega = (tout/tau)*(err/exp_tol)

        # Estimate order
        if m == oldm and tau != oldtau and ireject > 0:
            order = max([1., backend.log(omega/oldomega)/backend.log(tau/oldtau)])
            orderold = False
        elif orderold or ireject == 0:
            orderold = True
            order = j*.25
        else:
            orderold = True

        # Estimate k
        if m != oldm and tau == oldtau and ireject > 0:
            kest = max([1.1, (omega/oldomega)**(1./(oldm-m))]
                       ) if omega > 0 else 1.1
            kestold = False
        elif kestold or ireject == 0:
            kestold = True
            kest = 2
        else:
            kestold = True

        # This if statement is the main difference between fixed and variable m
        oldtau, oldm = tau, m
        if happy == 1:
            # Happy breakdown; wrap up
            omega = 0
            taunew, mnew = tau, m
        elif j == mmax and omega > delta:
            # Krylov subspace to small and stepsize to large
            taunew, mnew = tau*(omega/gamma)**(-1./order), j
        else:
            # Determine optimal tau and m
            tauopt = tau*(omega/gamma)**(-1./order)
            mopt = max([1., backend.ceil(j+backend.log(omega/gamma)/backend.log(kest))])

            # evaluate Cost functions
            Ctau = (m * NA + 3. * m * n + (m/mmax)**2*v_i * (10. + 3. * (m - 1.))
                    * (m + 1.) ** 3.) * backend.ceil((tout - tnow) / tauopt)

            Ck = (mopt * NA + 3. * mopt * n + (mopt/mmax)**2*v_i * (10. + 3. * (mopt - 1.))
                  * (mopt + 1.) ** 3) * backend.ceil((tout - tnow) / tau)
            if Ctau < Ck:
                taunew, mnew = tauopt, m
            else:
                taunew, mnew = tau, mopt

        # Check error against target
        if omega <= delta:  # use this one
            reject += ireject
            step += 1
            w = evec
            # update time
            tnow += tau
            ireject = 0
        else:  # try again
            ireject += 1

        # Another safety factors
        tau = min([tout - tnow, max([.2 * tau, min([taunew, 2. * tau])])])
        m = int(
            max([1, min([mmax, max([backend.floor(.75 * m), min([mnew, backend.ceil(1.3333 * m)])])])]))

    if abs(tnow/tout) < 1.:
        raise YastError('eigs/expA: Failed to approximate matrix exponent with given parameters.\nLast update of omega/delta = '+omega /
                        delta+'\nRemaining time = '+abs(1.-tnow/tout)+'\nChceck: max_iter - number of iteractions,\nk - Krylov dimension,\ndt - time step.')
    return (w[0], step, j, tnow,)


def eigs(Av, init, Bv=None, hermitian=True, k='all', sigma=None, ncv=5, which=None, tol=1e-14, bi_orth=True, return_eigenvectors=True, algorithm='arnoldi'):
    r"""
    Av, Bv: function handlers
        Bv: default = None
        Action of a matrix on a vector. For non-symmetric lanczos both have to be defined.
    init: Tensor
        Initial vector for iteration.
    k: int
        default = 'all', number of eigenvalues eqauls number of non-zero Krylov vectors
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
    norm, _ = init[0].norm(), None
    init = [(1. / it.norm())*it for it in init]
    val, Y, good = krylov(init, Av, Bv, tol, k, algorithm,
                          hermitian, ncv, bi_orth, sigma, which, return_eigenvectors)
    if return_eigenvectors:
        return val, [norm*Y[it] for it in range(len(Y))], good
    else:
        return val, Y, good


def eigsh(Av, init, tol=1e-14, k=5, algorithm='arnoldi'):
    norm = init[0].norm()
    init = [(1. / it.norm())*it for it in init]
    val, Y, good = krylov(init, Av, None, tol, 'all', algorithm, True, k)
    return val, [norm*Y[it] for it in range(len(Y))], good

