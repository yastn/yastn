""" Linalg methods for yast tensor. """

import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _common_keys
from ._auxliary import YastError, _check, _test_tensors_match, _test_all_axes
from ._merging import _merge_to_matrix, _unmerge_from_matrix, _unmerge_from_diagonal
from ._merging import _leg_struct_trivial, _leg_struct_truncation

__all__ = ['svd', 'svd_lowrank', 'qr', 'eigh', 'norm', 'norm_diff', 'entropy']


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
        meta = tuple((il+ir, il+ir, ir, ir+ir) for il, ir in zip(ul, ur))
        n_l, n_r = a.n, 0*a.n
    else:
        meta = tuple((il+ir, il+il, il, il+ir) for il, ir in zip(ul, ur))
        n_l, n_r = 0*a.n, a.n
    U = a.__class__(config=a.config, s=ls_l.s + (sU,), n=n_l, meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])
    S = a.__class__(config=a.config, s=s_eff, isdiag=True)
    V = a.__class__(config=a.config, s=(-sU,) + ls_r.s, n=n_r, meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r])
    
    U.A, S.A, V.A = a.config.backend.svd_lowrank(Am, meta, D_block, n_iter, k_fac)

    ls_s = _leg_struct_truncation(S, tol, D_block, D_total, keep_multiplets, eps_multiplet, 'svd')
    U.A = _unmerge_from_matrix(U.A, ls_l, ls_s)
    S.A = _unmerge_from_diagonal(S.A, ls_s)
    V.A = _unmerge_from_matrix(V.A, ls_s, ls_r)
    U.update_struct()
    S.update_struct()
    V.update_struct()
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
        meta = tuple((il+ir, il+ir, ir, ir+ir) for il, ir in zip(ul, ur))
        n_l, n_r = a.n, 0*a.n
    else:
        meta = tuple((il+ir, il+il, il, il+ir) for il, ir in zip(ul, ur))
        n_l, n_r = 0*a.n, a.n
    U = a.__class__(config=a.config, s=ls_l.s + (sU,), n=n_l, meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])
    S = a.__class__(config=a.config, s=s_eff, isdiag=True)
    V = a.__class__(config=a.config, s=(-sU,) + ls_r.s, n=n_r, meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r])
    U.A, S.A, V.A = a.config.backend.svd(Am, meta)

    ls_s = _leg_struct_truncation(S, tol, D_block, D_total, keep_multiplets, eps_multiplet, 'svd')
    U.A = _unmerge_from_matrix(U.A, ls_l, ls_s)
    S.A = _unmerge_from_diagonal(S.A, ls_s)
    V.A = _unmerge_from_matrix(V.A, ls_s, ls_r)
    U.update_struct()
    S.update_struct()
    V.update_struct()
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

    Qs = tuple(a.s[lg] for lg in axes[0]) + (sQ,)
    Rs = (-sQ,) + tuple(a.s[lg] for lg in axes[1])
    Q = a.__class__(config=a.config, s=Qs, n=a.n, meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])
    R = a.__class__(config=a.config, s=Rs, meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r])

    meta = tuple((il+ir, il+ir, ir+ir) for il, ir in zip(ul, ur))
    Q.A, R.A = a.config.backend.qr(Am, meta)

    ls = _leg_struct_trivial(R, axis=0)
    Q.A = _unmerge_from_matrix(Q.A, ls_l, ls)
    R.A = _unmerge_from_matrix(R.A, ls, ls_r)
    Q.update_struct()
    R.update_struct()

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

    if np.any(a.n != 0):
        raise YastError('Charge should be zero')

    s_eff = (-sU, sU)
    Am, ls_l, ls_r, ul, ur = _merge_to_matrix(a, axes, s_eff)

    if _check["consistency"] and not (ul == ur and ls_l.match(ls_r)):
        raise YastError('Something went wrong in matching the indices of the two tensors')

    Us = tuple(a.s[lg] for lg in axes[0]) + (sU,)
    S = a.__class__(config=a.config, s=(-sU, sU), isdiag=True)
    U = a.__class__(config=a.config, s=Us, meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])

    # meta = (indA, indS, indU)
    meta = tuple((il+ir, il, il+ir) for il, ir in zip(ul, ur))
    S.A, U.A = a.config.backend.eigh(Am, meta)

    ls_s = _leg_struct_truncation(S, tol, D_block, D_total, keep_multiplets, eps_multiplet, 'eigh')
    U.A = _unmerge_from_matrix(U.A, ls_l, ls_s)
    S.A = _unmerge_from_diagonal(S.A, ls_s)

    U.update_struct()
    S.update_struct()

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
    return a.config.backend.entropy(Sm, alpha=alpha)  # entropy, Smin, normalization
