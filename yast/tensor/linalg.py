# Linalg methods for yast tensor.
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes
from ._tests import YastError, _test_axes_all
from ._merging import _merge_to_matrix, _unmerge_matrix, _unmerge_diagonal
from ._merging import _leg_struct_trivial, _leg_struct_truncation, _Fusion
from ._krylov import _expand_krylov_space

__all__ = ['svd', 'svd_lowrank', 'qr', 'eigh', 'norm', 'entropy', 'expmv', 'eigs']


def norm(a, p='fro'):
    r"""
    Norm of the tensor.

    Parameters
    ----------
    p: str
        ``'fro'`` for Frobenius norm;  ``'inf'`` for :math:`l^\infty` (or supremum) norm

    Returns
    -------
    norm : float64
    """
    if p not in ('fro', 'inf'):
        raise YastError("Error in norm: p not in ('fro', 'inf'). ")
    if len(a.A) == 0:
        return a.zero_of_dtype()
    return a.config.backend.norm(a.A, p)


def svd_lowrank(a, axes=(0, 1), n_iter=60, k_fac=6, **kwargs):
    r"""
    Split tensor into :math:`a \approx USV` using approximate singular value decomposition (SVD),
    where `U` and `V` are orthonormal and `S` is positive and diagonal matrix.
    The approximate SVD is computed using stochastic method (TODO add ref).

    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).

    Charge of input tensor `a` is attached to `U` if `nU` and to `V` otherwise.

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
        relative tolerance of singular values below which to truncate across all blocks.

    tol_block: float
        relative tolerance of singular values below which to truncate within individual blocks

    D_block: int
        largest number of singular values to keep in a single block.
        also used in lowrank svd in the backend

    D_total: int
        largest total number of singular values to keep.

    n_iter, k_fac: ints
        number of iterations and multiplicative factor of stored singular values in lowrank svd procedure
        (relevant options might depend on backend)

    untruncated_S: bool
        returns U, S, V, uS  with dict uS with a copy of untruncated singular values and truncated bond dimensions.

    Returns
    -------
    U, S, V: Tensor
        U and V are unitary projectors. S is diagonal.
    """
    return svd(a, axes=axes, policy='lowrank', n_iter=n_iter, k_fac=k_fac, **kwargs)


def svd(a, axes=(0, 1), sU=1, nU=True, Uaxis=-1, Vaxis=0,
        tol=0, tol_block=0, D_block=np.inf, D_total=np.inf,
        keep_multiplets=False, eps_multiplet=1e-14, untruncated_S=False,
        policy='fullrank', **kwargs):
    r"""
    Split tensor into :math:`a=USV` using exact singular value decomposition (SVD), 
    where `U` and `V` are orthonormal bases and `S` is positive and diagonal matrix.
    Optionally, truncate the result.

    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).

    Charge of input tensor `a` is attached to `U` if `nU` and to `V` otherwise.

    Parameters
    ----------
    axes: tuple
        Specify two groups of legs between which to perform SVD, as well as
        their final order.

    sU: int
        signature of the new leg in U; equal 1 or -1. Default is 1.

    Uaxis, Vaxis: int
        specify which leg of U and V tensors are connecting with S. By default
        it is the last leg of U and the first of V.

    tol: float
        relative tolerance of singular values below which to truncate across all blocks

    tol_block: float
        relative tolerance of singular values below which to truncate within individual blocks

    D_block: int
        largest number of singular values to keep in a single block.

    D_total: int
        largest total number of singular values to keep.

    untruncated_S: bool
        returns U, S, V, uS  with dict uS with a copy of untruncated singular values and truncated bond dimensions.

    Returns
    -------
    U, S, V: Tensor
        U and V are unitary projectors. S is a diagonal tensor.
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.meta_fusion, lout_l, lout_r)

    s_eff = (-sU, sU)
    Am, ls_l, ls_r, ul, ur = _merge_to_matrix(a, axes, s_eff)

    if nU:
        meta = tuple((il + ir, il + ir, ir, ir + ir) for il, ir in zip(ul, ur))
        n_l, n_r = a.struct.n, None
    else:
        meta = tuple((il+ir, il+il, il, il+ir) for il, ir in zip(ul, ur))
        n_l, n_r = None, a.struct.n

    Us = tuple(a.struct.s[ii] for ii in axes[0]) + (sU,)
    Vs = (-sU,) + tuple(a.struct.s[ii] for ii in axes[1])

    U = a.__class__(config=a.config, s=Us, n=n_l,
                    meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)],
                    hard_fusion=[a.hard_fusion[ii] for ii in axes[0]] + [_Fusion(s=(sU,))])
    S = a.__class__(config=a.config, s=s_eff, isdiag=True)
    V = a.__class__(config=a.config, s=Vs, n=n_r,
                    meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r],
                    hard_fusion=[_Fusion(s=(-sU,))] + [a.hard_fusion[ii] for ii in axes[1]])

    if policy == 'fullrank':
        U.A, S.A, V.A = a.config.backend.svd(Am, meta)
    elif policy == 'lowrank':
        U.A, S.A, V.A = a.config.backend.svd_lowrank(Am, meta, D_block, **kwargs)
    else:
        raise YastError('svd policy should be one of (`lowrank`, `fullrank`)')

    ls_s = _leg_struct_truncation(
        S, tol=tol, tol_block=tol_block, D_block=D_block, D_total=D_total,
        keep_multiplets=keep_multiplets, eps_multiplet=eps_multiplet, ordering='svd')

    if untruncated_S:
        uS = {k: a.config.backend.copy(v) for k, v in S.A.items()}
        uS['D'] = ls_s.Dtot.copy()

    _unmerge_matrix(U, ls_l, ls_s)
    _unmerge_diagonal(S, ls_s)
    _unmerge_matrix(V, ls_s, ls_r)
    U.moveaxis(source=-1, destination=Uaxis, inplace=True)
    V.moveaxis(source=0, destination=Vaxis, inplace=True)
    return (U, S, V, uS) if untruncated_S else (U, S, V)


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
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.meta_fusion, lout_l, lout_r)

    s_eff = (-sQ, sQ)
    Am, ls_l, ls_r, ul, ur = _merge_to_matrix(a, axes, s_eff)

    Qs = tuple(a.struct.s[lg] for lg in axes[0]) + (sQ,)
    Rs = (-sQ,) + tuple(a.struct.s[lg] for lg in axes[1])
    Q = a.__class__(config=a.config, s=Qs, n=a.struct.n,
                    meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)],
                    hard_fusion=[a.hard_fusion[ii] for ii in axes[0]] + [_Fusion(s=(sQ,))])
    R = a.__class__(config=a.config, s=Rs,
                    meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r],
                    hard_fusion=[_Fusion(s=(-sQ,))] + [a.hard_fusion[ii] for ii in axes[1]])

    meta = tuple((il + ir, il + ir, ir + ir) for il, ir in zip(ul, ur))
    Q.A, R.A = a.config.backend.qr(Am, meta)

    ls = _leg_struct_trivial(R, axis=0)
    _unmerge_matrix(Q, ls_l, ls)
    _unmerge_matrix(R, ls, ls_r)
    Q.moveaxis(source=-1, destination=Qaxis, inplace=True)
    R.moveaxis(source=0, destination=Raxis, inplace=True)
    return Q, R


def eigh(a, axes, sU=1, Uaxis=-1, tol=0, tol_block=0, D_block=np.inf, D_total=np.inf,
         keep_multiplets=False, eps_multiplet=1e-14, untruncated_S=False):
    r"""
    Split symmetric tensor using exact eigenvalue decomposition, :math:`a= USU^{\dagger}`. 
    Optionally, truncate the resulting decomposition.

    Tensor is expected to be symmetric (hermitian) with total charge 0.
    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).
    Truncate based on tolerance only if some eigenvalues are positive -- than all negative ones are discarded.

    Parameters
    ----------
    axes: tuple
        Specify two groups of legs between which to perform svd, as well as their final order.

    sU: int
        signature of connecting leg in U equall 1 or -1. Default is 1.

    Uaxis: int
        specify which leg of U is the new connecting leg. By delault it is the last leg.

    tol: float
        relative tolerance of singular values below which to truncate across all blocks.

    tol_block: float
        relative tolerance of singular values below which to truncate within individual blocks

    D_block: int
        largest number of singular values to keep in a single block.

    D_total: int
        largest total number of singular values to keep.

    untruncated_S: bool
        returns S, U, uS  with dict uS with a copy of untruncated eigenvalues and truncated bond dimensions.

    Returns
    -------
        S, U: Tensor
            U is unitary projector. S is a diagonal tensor.
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.meta_fusion, lout_l, lout_r)

    if any(x != 0 for x in a.struct.n):
        raise YastError('Charge should be zero')

    s_eff = (-sU, sU)
    Am, ls_l, ls_r, ul, ur = _merge_to_matrix(a, axes, s_eff)

    if ul != ur or ls_l != ls_r:
        raise YastError(
            'Something went wrong in matching the indices of the two tensors')

    Us = tuple(a.struct.s[lg] for lg in axes[0]) + (sU,)
    S = a.__class__(config=a.config, s=(-sU, sU), isdiag=True)
    U = a.__class__(config=a.config, s=Us,
                    meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)],
                    hard_fusion=[a.hard_fusion[ii] for ii in axes[0]] + [_Fusion(s=(sU,))])

    # meta = (indA, indS, indU)
    meta = tuple((il + ir, il, il + ir) for il, ir in zip(ul, ur))
    S.A, U.A = a.config.backend.eigh(Am, meta)

    ls_s = _leg_struct_truncation(
        S, tol=tol, tol_block=tol_block, D_block=D_block, D_total=D_total,
        keep_multiplets=keep_multiplets, eps_multiplet=eps_multiplet, ordering='eigh')

    if untruncated_S:
        uS = {k: a.config.backend.copy(v) for k, v in S.A.items()}
        uS['D'] = ls_s.Dtot.copy()

    _unmerge_matrix(U, ls_l, ls_s)
    _unmerge_diagonal(S, ls_s)
    U.moveaxis(source=-1, destination=Uaxis, inplace=True)
    return (S, U, uS) if untruncated_S else (S, U)


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

    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.meta_fusion, lout_l, lout_r)

    if not a.isdiag:
        Am, *_ = _merge_to_matrix(a, axes, (-1, 1))
        Sm = a.config.backend.svd_S(Am)
    else:
        Sm = {t: a.config.backend.diag_get(x) for t, x in a.A.items()}
    # entropy, Smin, normalization
    return a.config.backend.entropy(Sm, alpha=alpha)


# Krylov based methods, handled by anonymous function decribing action of matrix on a vector
def expmv(f, v, t=1., tol=1e-12, ncv=10, hermitian=False, normalize=False, return_info=False):
    r"""
    Calculate exp(t*f)*v, where v is a yast tensor, and f(v) is linear operator acting on v.

    Parameters
    ----------
        f: function
            define an action of a 'square matrix' on the vector x.
            f(x) should preserve the signature of x.

        v: Tensor

        t: number

        tol: number
            targeted tolerance; it is used to update the time-step and size of Krylov space.
           The result should have better tolerance, as corrected result is outputed.

        ncv: int
            Initial guess for the size of the Krylov space

        hermitian: bool
            Assume that f is a hermitian operator, in which case Lanczos iterations are used.
            Otherwise Arnoldi iterations are used to span the Krylov space.

        normalize: bool
            The result is normalized to unity using 2-norm.

        return_info: bool
            info.ncv : guess of the Krylov-space size,
            info.error : estimate of error (likely over-estimate)
            info.krylov_steps : number of execution of f(x),
            info.steps : number of steps to reach t,

        Returns
        -------
        out : Tensor if not return_info else (out, info)

        Note
        ----
        Employ the algorithm of:
        J. Niesen, W. M. Wright, ACM Trans. Math. Softw. 38, 22 (2012),
        Algorithm 919: A Krylov subspace algorithm for evaluating
        the phi-functions appearing in exponential integrators.
    """
    backend = v.config.backend
    ncv, ncv_max = max(1, ncv), min([30, v.size])  # Krylov space parameters
    t_now, t_out = 0, abs(t)
    sgn = t / t_out if t_out > 0 else 0
    tau = t_out  # initial quess for a time-step
    gamma, delta = 0.8, 1.2  # Safety factors
    V, H = None, None  # reset Krylov space
    ncv_old, tau_old, omega = None, None, None
    reject, order_computed, ncv_computed = False, False, False
    info = {'ncv': ncv, 'error': 0., 'krylov_steps': 0, 'steps': 0}

    normv = v.norm()
    if normv == 0:
        if normalize:
            raise YastError('expmv got zero vector that cannot be normalized')
        t_out = 0
    else:
        v = v / normv

    while t_now < t_out:
        if V is None:
            V = [v]
        V, H, happy = _expand_krylov_space(f, tol, ncv, hermitian, V, H, info)
        if happy:
            tau = t_out - t_now
            m = len(V)
            h = 0
        else:
            m = len(V) - 1
            h = H.pop((m, m - 1))
        H[(0, m)] = backend.dtype_scalar(1, device=v.config.device)
        T = backend.square_matrix_from_dict(H, m + 1, device=v.config.device)
        F = backend.expm((sgn * tau) * T)
        err = abs(h * F[m - 1, m]).item()

        # renormalized error per unit step
        omega_old, omega = omega, (t_out / tau) * (err / tol)

        # Estimate order
        if ncv == ncv_old and tau != tau_old and reject:
            order = max([1., np.log(omega / omega_old) / np.log(tau / tau_old)])
            order_computed = True
        elif reject and order_computed:
            order_computed = False
        else:
            order_computed = False
            order = 0.25 * m

        # Estimate ncv
        if ncv != ncv_old and tau == tau_old and reject:
            ncv_est = max([1.1, (omega / omega_old) ** (1. / (ncv_old - ncv))]) if omega > 0 else 1.1
            ncv_computed = True
        elif reject and ncv_computed:
            ncv_computed = False
        else:
            ncv_computed = False
            ncv_est = 2

        tau_old, ncv_old = tau, ncv
        if happy:
            omega = 0
            tau_new, ncv_new = tau, ncv
        elif m == ncv_max and omega > delta:
            tau_new, ncv_new = tau * (omega / gamma) ** (-1. / order), ncv_max
        else:
            tau_opt = tau * (omega / gamma) ** (-1. / order) if omega > 0 else t_out - t_now
            ncv_opt = int(max([1, np.ceil(m + np.log(omega / gamma) / np.log(ncv_est))])) if omega > 0 else 1
            C1 = ncv * int(np.ceil((t_out - t_now) / tau_opt))
            C2 = ncv_opt * int(np.ceil((t_out - t_now) / tau))
            tau_new, ncv_new = (tau_opt, m) if C1 < C2 else (tau, ncv_opt)

        if omega <= delta:  # Check error against target
            F[m, 0] = F[m - 1, m] * h
            F = F[:, 0]
            normF = backend.norm_matrix(F)
            normv = normv * normF
            F = F / normF
            v = F[0] * V[0]
            for it in range(1, len(V)):
                v = v.apxb(V[it], x= F[it])
            t_now += tau
            info['steps'] += 1
            info['error'] += err
            V, H, reject = None, None, False
        else:
            reject = True
            H[(m, m - 1)] = h
            H.pop((0, m))
        tau = min(max(0.2 * tau, tau_new), t_out - t_now, 2 * tau)
        ncv = int(max(1, min(ncv_max, np.ceil(1.3333 * m), max(np.floor(0.75 * m), ncv_new))))
    info['ncv'] = ncv
    if not normalize:
        v = normv * v
    return (v, info) if return_info else v


def eigs(f, v0, k=1, which='SR', ncv=10, maxiter=None, tol=1e-13, hermitian=True):
    r"""
    Search for dominant eigenvalues of linear operator f using Arnoldi algorithm.
    ONLY A SINGLE ITERATION FOR NOW

    f: function
        define an action of a 'square matrix' on the 'vector' x.
        f(x) should preserve the signature of x.

    v0: Tensor
        Initial guess, 'vector' to span the Krylov space.

    k: int
        Number of desired eigenvalues and eigenvectors. default is 1.

    which: str in [‘LM’, ‘SM’, ‘LR’, ‘SR’]
        Which k eigenvectors and eigenvalues to find:
            ‘LM’ : largest magnitude
            ‘SM’ : smallest magnitude
            ‘LR’ : largest real part
            ‘SR’ : smallest real part

    ncv: int
        Dimension of the employed Krylov space. Default is 10.
        Must be greated than k.

    maxiter: int
        Maximal number of restarts; NOT IMPLEMENTED FOR NOW.

    tol: float
        Relative accuracy for eigenvalues and th stopping criterion for Krylov subspace.
        Default is 1e-13.

    hermitian: bool
        Assume that f is a hermitian operator, in which case Lanczos iterations are used.
        Otherwise Arnoldi iterations are used to span the Krylov space.
    """
    backend = v0.config.backend
    normv = v0.norm()
    if normv == 0:
        raise YastError('Initial vector v0 of eigs should be nonzero.')
    V = [v0 / normv]
    V, H, happy = _expand_krylov_space(f, 1e-13, ncv, hermitian, V)  # tol=1e-13
    m = len(V) if happy else len(V) - 1

    T = backend.square_matrix_from_dict(H, m, device=v0.config.device)
    val, vr = backend.eigh(T) if hermitian else backend.eig(T)
    ind = backend.eigs_which(val, which)

    val, vr = val[ind], vr[:, ind]
    Y = []
    for it in range(k):
        sit = vr[:, it]
        Y.append(sit[0] * V[0])
        for jt in range(1, len(ind)):
            Y[it] = Y[it].apxb(V[jt], x=sit[jt])
    return val[:len(Y)], Y
