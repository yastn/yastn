""" Linalg decompositions of yast tensor. """
import numpy as np
from .core import Tensor, _LegDecomposition, YastError, _check
from ._auxliary import _clear_axes, _unpack_axes

def svd_lowrank(a, truncated_nbit=60, truncated_kfac=6):
    pass

# linalg.svd

###########################
#     spliting tensor     #
###########################
def svd(a, axes=(0, 1), sU=1, nU=True, Uaxis=-1, Vaxis=0, \
    tol=0, D_block=np.inf, D_total=np.inf, \
    keep_multiplets=False, eps_multiplet=1.0e-14, **kwargs):
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
    out_l, out_r = _unpack_axes(a, lout_l, lout_r)
    a._test_axes_split(out_l, out_r)

    Am, ls_l, ls_r, ul, ur = a.merge_to_matrix(out_l, out_r, news_l=-sU, news_r=sU)

    if nU:
        meta = tuple((il+ir, il+ir, ir, ir+ir) for il, ir in zip(ul, ur))
        n_l, n_r = a.n, 0*a.n
    else:
        meta = tuple((il+ir, il+il, il, il+ir) for il, ir in zip(ul, ur))
        n_l, n_r = 0*a.n, a.n

    opts = {'tol': tol, 'D_total': D_total, 'D_block': D_block,
            'keep_multiplets': keep_multiplets, 'eps_multiplet': eps_multiplet}

    Um, Sm, Vm = a.config.backend.svd(Am, meta, opts)

    U = Tensor(config=a.config, s=ls_l.s + (sU,), n=n_l, meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])
    S = Tensor(config=a.config, s=(-sU, sU), isdiag=True)
    V = Tensor(config=a.config, s=(-sU,) + ls_r.s, n=n_r, meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r])

    ls_s = _LegDecomposition(a.config)
    ls_s.leg_struct_for_truncation(Sm, opts, 'svd')

    U.A = a.unmerge_from_matrix(Um, ls_l, ls_s)
    S.A = a.unmerge_from_diagonal(Sm, ls_s)
    V.A = a.unmerge_from_matrix(Vm, ls_s, ls_r)

    U.update_tD_arrays()
    S.update_tD_arrays()
    V.update_tD_arrays()
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
        specify which leg of Q and R tensors are connecting to the other tensor. By delault it is the last leg of Q and the first of R.

    Returns
    -------
        Q, R: Tensor
    """
    lout_l, lout_r = _clear_axes(*axes)
    out_l, out_r = _unpack_axes(a, lout_l, lout_r)
    a._test_axes_split(out_l, out_r)

    Am, ls_l, ls_r, ul, ur = a.merge_to_matrix(out_l, out_r, news_l=-sQ, news_r=sQ)

    meta = tuple((l+r, l+r, r+r) for l, r in zip(ul, ur))
    Qm, Rm = a.config.backend.qr(Am, meta)

    Qs = tuple(a.s[lg] for lg in out_l) + (sQ,)
    Rs = (-sQ,) + tuple(a.s[lg] for lg in out_r)
    Q = Tensor(config=a.config, s=Qs, n=a.n, meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])
    R = Tensor(config=a.config, s=Rs, meta_fusion=[(1,)] + [a.meta_fusion[ii] for ii in lout_r])

    ls = _LegDecomposition(a.config, -sQ, -sQ)
    ls.leg_struct_trivial(Rm, 0)

    Q.A = a.unmerge_from_matrix(Qm, ls_l, ls)
    R.A = a.unmerge_from_matrix(Rm, ls, ls_r)

    Q.update_tD_arrays()
    R.update_tD_arrays()

    Q.moveaxis(source=-1, destination=Qaxis, inplace=True)
    R.moveaxis(source=0, destination=Raxis, inplace=True)
    return Q, R

def eigh(a, axes, sU=1, Uaxis=-1, tol=0, D_block=np.inf, D_total=np.inf):
    r"""
    Split tensor using eig, tensor = U * S * U^dag. Truncate smallest eigenvalues if neccesary.

    Tensor should be hermitian and has charge 0.
    Truncate using (whichever gives smaller bond dimension) relative tolerance, bond dimension of each block, and total bond dimension from all blocks.
    By default do not truncate. Truncate based on tolerance only if some eigenvalue is positive -- than all negative ones are discarded.
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
    out_l, out_r = _unpack_axes(a, lout_l, lout_r)
    a._test_axes_split(out_l, out_r)

    if np.any(a.n != 0):
        raise YastError('Charge should be zero')

    Am, ls_l, ls_r, ul, ur = a.merge_to_matrix(out_l, out_r, news_l=-sU, news_r=sU)

    if _check["consistency"] and not (ul == ur and ls_l.match(ls_r)):
        raise YastError('Something went wrong in matching the indices of the two tensors')

    # meta = (indA, indS, indU)
    meta = tuple((l+r, l, l+r) for l, r in zip(ul, ur))
    Sm, Um = a.config.backend.eigh(Am, meta)

    opts = {'D_block': D_block, 'tol': tol, 'D_total': D_total}
    ls_s = _LegDecomposition(a.config, -sU, -sU)
    ls_s.leg_struct_for_truncation(Sm, opts, 'eigh')

    Us = tuple(a.s[lg] for lg in out_l) + (sU,)

    S = Tensor(config=a.config, s=(-sU, sU), isdiag=True)
    U = Tensor(config=a.config, s=Us, meta_fusion=[a.meta_fusion[ii] for ii in lout_l] + [(1,)])

    U.A = a.unmerge_from_matrix(Um, ls_l, ls_s)
    S.A = a.unmerge_from_diagonal(Sm, ls_s)

    U.update_tD_arrays()
    S.update_tD_arrays()

    U.moveaxis(source=-1, destination=Uaxis, inplace=True)
    return S, U
