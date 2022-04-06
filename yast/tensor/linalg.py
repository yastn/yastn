# Linalg methods for yast tensor.
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _struct
from ._tests import YastError, _test_axes_all
from ._merging import _merge_to_matrix, _meta_unfuse_legdec
from ._merging import _leg_struct_trivial, _leg_struct_truncation, _Fusion
from ._krylov import _expand_krylov_space

__all__ = ['svd', 'svd_pure', 'svd_old', 'svd_lowrank', 'qr', 'eigh', 'norm', 'entropy', 'expmv', 'eigs']


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
    return a.config.backend.norm(a._data, p)


def svd_lowrank(a, axes=(0, 1), n_iter=60, k_fac=6, **kwargs):
    r"""
    Split tensor into :math:`a \approx USV^\dag` using approximate singular value decomposition (SVD),
    where `U` and `V` are orthonormal and `S` is positive and diagonal matrix.
    The approximate SVD is computed using stochastic method (TODO add ref).

    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).

    Charge of input tensor `a` is attached to `U` if `nU` and to `Vh` otherwise.

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
        returns U, S, Vh, uS  with dict uS with a copy of untruncated singular values and truncated bond dimensions.

    Returns
    -------
    U, S, Vh: Tensor
        U and Vh are unitary projectors. S is real diagonal.
    """
    return svd(a, axes=axes, policy='lowrank', n_iter=n_iter, k_fac=k_fac, **kwargs)


def svd_old(a, axes=(0, 1), sU=1, nU=True, Uaxis=-1, Vaxis=0,
        tol=0, tol_block=0, D_block=2 ** 32, D_total=2 ** 32,
        keep_multiplets=False, eps_multiplet=1e-14, untruncated_S=False,
        policy='fullrank', **kwargs):
    r"""
    Split tensor into :math:`a=USV^\dag` using exact singular value decomposition (SVD),
    where `U` and `V` are orthonormal bases and `S` is positive and diagonal matrix.
    Optionally, truncate the result.

    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).

    Charge of input tensor `a` is attached to `U` if `nU` and to `Vh` otherwise.

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
        returns U, S, Vh, uS  with dict uS with a copy of untruncated singular values and truncated bond dimensions.

    Returns
    -------
    U, S, Vh: Tensor
        U and Vh are unitary projectors. S is a real diagonal tensor.
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    s_eff = (-sU, sU)
    data, struct, ls_l, ls_r = _merge_to_matrix(a, axes, s_eff)
    minD = tuple(min(ds) for ds in struct.D)
    if policy == 'lowrank':
        minD = tuple(min(D_block, d) for d in minD)

    nsym = len(a.struct.n)
    n0 = (0,) * nsym
    if nU:
        Ut = struct.t
        St = tuple(x[nsym:] for x in struct.t)
        Vt = tuple(x[nsym:] * 2 for x in struct.t)
        Un, Vn = a.struct.n, n0
    else:
        Ut = tuple(x[:nsym] * 2 for x in struct.t)
        St = tuple(x[:nsym] for x in struct.t)
        Vt = struct.t
        Un, Vn = n0, a.struct.n

    UD = tuple((ds[0], dm) for ds, dm in zip(struct.D, minD))
    SD = tuple((dm, dm) for dm in minD)
    VD = tuple((dm, ds[1]) for dm, ds in zip(minD, struct.D))

    UDp = tuple(np.prod(UD, axis=1))
    SDp = tuple(dd[0] for dd in SD)
    VDp = tuple(np.prod(VD, axis=1))

    Usl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(UDp), UDp))
    Ssl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(SDp), SDp))
    Vsl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(VDp), VDp))

    Ustruct = _struct(s=s_eff, n=Un, diag=False, t=Ut, D=UD, Dp=UDp, sl=Usl)
    Sstruct = _struct(s=s_eff, n=n0, diag=True, t=St, D=SD, Dp=SDp, sl=Ssl)
    Vstruct = _struct(s=s_eff, n=Vn, diag=False, t=Vt, D=VD, Dp=VDp, sl=Vsl)

    Usize = Ustruct.sl[-1][1] if len(Ustruct.sl) > 0 else 0
    Ssize = Sstruct.sl[-1][1] if len(Sstruct.sl) > 0 else 0
    Vsize = Vstruct.sl[-1][1] if len(Vstruct.sl) > 0 else 0

    meta = tuple(zip(struct.sl, struct.D, Ustruct.sl, Sstruct.sl, Vstruct.sl))
    if policy == 'fullrank':
        Udata, Sdata, Vdata = a.config.backend.svd(data, meta, Usize, Ssize, Vsize,\
            diagnostics=kwargs['diagonostics'] if 'diagonostics' in kwargs else None)
    elif policy == 'lowrank':
        Udata, Sdata, Vdata = a.config.backend.svd_lowrank(data, meta, Usize, Ssize, Vsize, D_block, **kwargs)
    else:
        raise YastError('svd policy should be one of (`lowrank`, `fullrank`)')

    ls_s = _leg_struct_truncation(a.config, Sdata, Sstruct.t, Sstruct.sl,
        tol=tol, tol_block=tol_block, D_block=D_block, D_total=D_total,
        keep_multiplets=keep_multiplets, eps_multiplet=eps_multiplet, ordering='svd')

    if untruncated_S:
        uS = {t: a.config.backend.copy(Sdata[slice(*sl)]) for t, sl in zip(Sstruct.t, Sstruct.sl)}
        uS['D'] = ls_s.Dtot.copy()

    Us = tuple(a.struct.s[ii] for ii in axes[0]) + (sU,)
    Umeta_unmerge, Ustruct = _meta_unfuse_legdec(a.config, Ustruct, [ls_l, ls_s], Us)
    Udata = a.config.backend.unmerge_from_1d(Udata, Umeta_unmerge)
    Umfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=Ustruct, data=Udata, mfs=Umfs, hfs=Uhfs)

    Smeta_unmerge, Sstruct = _meta_unfuse_legdec(a.config, Sstruct, [ls_s], s_eff)
    Sdata = a.config.backend.unmerge_from_1d(Sdata, Smeta_unmerge)
    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=Sstruct, data=Sdata, mfs=Smfs, hfs=Shfs)

    Vs = (-sU,) + tuple(a.struct.s[ii] for ii in axes[1])
    Vmeta_unmerge, Vstruct = _meta_unfuse_legdec(a.config, Vstruct, [ls_s, ls_r], Vs)
    Vdata = a.config.backend.unmerge_from_1d(Vdata, Vmeta_unmerge)
    Vmfs = ((1,),) + tuple(a.mfs[ii] for ii in lout_r)
    Vhfs = (_Fusion(s=(-sU,)),) + tuple(a.hfs[ii] for ii in axes[1])
    V = a._replace(struct=Vstruct, data=Vdata, mfs=Vmfs, hfs=Vhfs)

    U = U.move_leg(source=-1, destination=Uaxis)
    V = V.move_leg(source=0, destination=Vaxis)
    return (U, S, V, uS) if untruncated_S else (U, S, V)


def svd(a, axes=(0, 1), sU=1, nU=True, Uaxis=-1, Vaxis=0,
        tol=0, tol_block=0, D_block=2 ** 32, D_total=2 ** 32,
        keep_multiplets=False, eps_multiplet=1e-14, untruncated_S=False,
        policy='fullrank', **kwargs):
    r"""
    Split tensor into :math:`a=USV^\dag` using exact singular value decomposition (SVD),
    where `U` and `V` are orthonormal bases and `S` is positive and diagonal matrix.
    Optionally, truncate the result.

    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).

    Charge of input tensor `a` is attached to `U` if `nU` and to `Vh` otherwise.

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
        returns U, S, Vh, uS  with dict uS with a copy of untruncated singular values and truncated bond dimensions.

    Returns
    -------
    U, S, Vh: Tensor
        U and Vh are unitary projectors. S is a real diagonal tensor.
    """
    U, S, V = svd_pure(a, axes=axes, sU=sU, nU=nU, diagnostics=kwargs['diagonostics']\
        if 'diagonostics' in kwargs else None)

    Smask = truncation_mask(S, tol=tol, tol_block=tol_block, D_block=D_block, D_total=D_total,
                            keep_multiplets= keep_multiplets, eps_multiplet=eps_multiplet)

    nsym = len(S.n)
    if untruncated_S:
        uS = {t[:nsym]: a.config.backend.copy(S.data[slice(*sl)]) for t, sl in zip(S.struct.t, S.struct.sl)}
        uS['D'] = Smask.trace(axes=(0, 1))

    U = Smask.mask_apply(U, axis=-1)
    S = Smask.mask_apply(S, axis=0)
    V = Smask.mask_apply(V, axis=0)

    U = U.move_leg(source=-1, destination=Uaxis)
    V = V.move_leg(source=0, destination=Vaxis)
    return (U, S, V, uS) if untruncated_S else (U, S, V)


def svd_pure(a, axes=(0, 1), sU=1, nU=True, Uaxis=-1, Vaxis=0, policy='fullrank', **kwargs):
    r"""
    Split tensor into :math:`a=U @ S @ V` using exact singular value decomposition (SVD),
    where `U` and `V` are orthonormal bases and `S` is positive and diagonal matrix.

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

    Returns
    -------
    U, S, V: Tensor
        U and V are unitary projectors. S is a real diagonal tensor.
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    s_eff = (-sU, sU)
    data, struct, ls_l, ls_r = _merge_to_matrix(a, axes, s_eff)
    minD = tuple(min(ds) for ds in struct.D)
    if policy == 'lowrank':
        minD = tuple(min(D_block, d) for d in minD)

    nsym = len(a.struct.n)
    n0 = (0,) * nsym
    if nU:
        Ut = struct.t
        St = tuple(x[nsym:] * 2 for x in struct.t)
        Vt = tuple(x[nsym:] * 2 for x in struct.t)
        Un, Vn = a.struct.n, n0
    else:
        Ut = tuple(x[:nsym] * 2 for x in struct.t)
        St = tuple(x[:nsym] * 2 for x in struct.t)
        Vt = struct.t
        Un, Vn = n0, a.struct.n

    UD = tuple((ds[0], dm) for ds, dm in zip(struct.D, minD))
    SD = tuple((dm, dm) for dm in minD)
    VD = tuple((dm, ds[1]) for dm, ds in zip(minD, struct.D))

    UDp = tuple(np.prod(UD, axis=1))
    SDp = tuple(dd[0] for dd in SD)
    VDp = tuple(np.prod(VD, axis=1))

    Usl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(UDp), UDp))
    Ssl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(SDp), SDp))
    Vsl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(VDp), VDp))

    Ustruct = _struct(s=s_eff, n=Un, diag=False, t=Ut, D=UD, Dp=UDp, sl=Usl)
    Sstruct = _struct(s=s_eff, n=n0, diag=True, t=St, D=SD, Dp=SDp, sl=Ssl)
    Vstruct = _struct(s=s_eff, n=Vn, diag=False, t=Vt, D=VD, Dp=VDp, sl=Vsl)

    Usize = Ustruct.sl[-1][1] if len(Ustruct.sl) > 0 else 0
    Ssize = Sstruct.sl[-1][1] if len(Sstruct.sl) > 0 else 0
    Vsize = Vstruct.sl[-1][1] if len(Vstruct.sl) > 0 else 0

    meta = tuple(zip(struct.sl, struct.D, Ustruct.sl, Sstruct.sl, Vstruct.sl))
    if policy == 'fullrank':
        Udata, Sdata, Vdata = a.config.backend.svd(data, meta, Usize, Ssize, Vsize,\
            diagnostics=kwargs['diagnostics'] if 'diagnostics' in kwargs else None)
    elif policy == 'lowrank':
        Udata, Sdata, Vdata = a.config.backend.svd_lowrank(data, meta, Usize, Ssize, Vsize, D_block, **kwargs)
    else:
        raise YastError('svd policy should be one of (`lowrank`, `fullrank`)')

    ls_s = _leg_struct_trivial(Sstruct, axis=0)

    Us = tuple(a.struct.s[ii] for ii in axes[0]) + (sU,)
    Umeta_unmerge, Ustruct = _meta_unfuse_legdec(a.config, Ustruct, [ls_l, ls_s], Us)
    Udata = a.config.backend.unmerge_from_1d(Udata, Umeta_unmerge)
    Umfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=Ustruct, data=Udata, mfs=Umfs, hfs=Uhfs)

    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=Sstruct, data=Sdata, mfs=Smfs, hfs=Shfs)

    Vs = (-sU,) + tuple(a.struct.s[ii] for ii in axes[1])
    Vmeta_unmerge, Vstruct = _meta_unfuse_legdec(a.config, Vstruct, [ls_s, ls_r], Vs)
    Vdata = a.config.backend.unmerge_from_1d(Vdata, Vmeta_unmerge)
    Vmfs = ((1,),) + tuple(a.mfs[ii] for ii in lout_r)
    Vhfs = (_Fusion(s=(-sU,)),) + tuple(a.hfs[ii] for ii in axes[1])
    V = a._replace(struct=Vstruct, data=Vdata, mfs=Vmfs, hfs=Vhfs)

    U = U.move_leg(source=-1, destination=Uaxis)
    V = V.move_leg(source=0, destination=Vaxis)
    return U, S, V


def truncation_mask(S, tol=0, tol_block=0, D_block=2 ** 32, D_total=2 ** 32,
             keep_multiplets=False, eps_multiplet=1e-14):
    """
    Generate mask tensor based on diagonal and real tensor S.
    It can be then used for truncation.
    """
    if not (S.isdiag and S.yast_dtype == "float64"):
        raise YastError("Truncation_mask requires S to be real and diagonal")

    # makes a copy for partial truncations; also detaches from autograd computation graph
    S = S.copy()
    Smask = S.copy()

    nsym = S.config.sym.NSYM
    tol_null = 0. if isinstance(tol_block, dict) else tol_block
    D_null = 2 ** 32 if isinstance(D_block, dict) else D_block
    for t, Db, sl in zip(S.struct.t, S.struct.Dp, S.struct.sl):
        t = t[:nsym]
        tol_rel = tol_block[t] if (isinstance(tol_block, dict) and t in tol_block) else tol_null
        tol_abs = tol_rel * S.config.backend.max_abs(S.data[slice(*sl)])
        D_bl = D_block[t] if (isinstance(D_block, dict) and t in D_block) else D_null
        D_bl = min(D_bl, Db)
        tol_D = S.config.backend.nth_largest(S.data[slice(*sl)], D_bl)
        tol_tru = max(tol_D, tol_abs) * (1 - 1e-15)
        if keep_multiplets and eps_multiplet > 0:
            while sum(S.data[slice(*sl)] > tol_tru) != sum(S.data[slice(*sl)] > (tol_tru - eps_multiplet)):
                tol_tru = tol_tru + eps_multiplet
        Smask.data[slice(*sl)] = S.data[slice(*sl)] > tol_tru

    S._data = S.data * Smask.data
    tol_abs = tol * S.config.backend.max_abs(S.data)
    D_total = min(D_total, sum(Smask.data > 0))
    tol_D = S.config.backend.nth_largest(S._data, D_total)
    tol_tru = max(tol_D, tol_abs) * (1 - 1e-15)
    if keep_multiplets and eps_multiplet > 0:
            while sum(S.data > tol_tru) != sum(S.data > (tol_tru - eps_multiplet)):
                tol_tru = tol_tru + eps_multiplet
    Smask._data = S.data > tol_tru
    return Smask


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
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    s_eff = (-sQ, sQ)

    data, struct, ls_l, ls_r = _merge_to_matrix(a, axes, s_eff)
    minD = tuple(min(ds) for ds in struct.D)

    Qt = struct.t
    QD = tuple((ds[0], dm) for ds, dm in zip(struct.D, minD))
    QDp = tuple(np.prod(QD, axis=1))
    Qsl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(QDp), QDp))
    Qstruct = a.struct._replace(t=Qt, D=QD, Dp=QDp, sl=Qsl, s=s_eff)
    Qsize = Qstruct.sl[-1][1] if len(Qstruct.sl) > 0 else 0

    nsym = len(struct.n)
    Rt = tuple(x[nsym:] * 2 for x in struct.t)
    RD = tuple((dm, ds[1]) for dm, ds in zip(minD, struct.D))
    RDp = tuple(np.prod(RD, axis=1))
    Rsl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(RDp), RDp))
    Rstruct = a.struct._replace(t=Rt, D=RD, Dp=RDp, sl=Rsl, s=s_eff, n=(0,) * nsym)
    Rsize = Rstruct.sl[-1][1] if len(Rstruct.sl) > 0 else 0

    meta = tuple(zip(struct.sl, struct.D, Qstruct.sl, Rstruct.sl))
    Qdata, Rdata = a.config.backend.qr(data, meta, Qsize, Rsize)

    ls = _leg_struct_trivial(Rstruct, axis=0)

    Qs = tuple(a.struct.s[lg] for lg in axes[0]) + (sQ,)
    Qmeta_unmerge, Qstruct = _meta_unfuse_legdec(a.config, Qstruct, [ls_l, ls], Qs)
    Qdata = a.config.backend.unmerge_from_1d(Qdata, Qmeta_unmerge)
    Qmfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Qhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sQ,)),)
    Q = a._replace(struct=Qstruct, data=Qdata, mfs=Qmfs, hfs=Qhfs)

    Rs = (-sQ,) + tuple(a.struct.s[lg] for lg in axes[1])
    Rmeta_unmerge, Rstruct = _meta_unfuse_legdec(a.config, Rstruct, [ls, ls_r], Rs)
    Rdata = a.config.backend.unmerge_from_1d(Rdata, Rmeta_unmerge)
    Rmfs = ((1,),) + tuple(a.mfs[ii] for ii in lout_r)
    Rhfs = (_Fusion(s=(-sQ,)),) + tuple(a.hfs[ii] for ii in axes[1])
    R = a._replace(struct=Rstruct, data=Rdata, mfs=Rmfs, hfs=Rhfs)

    Q = Q.move_leg(source=-1, destination=Qaxis)
    R = R.move_leg(source=0, destination=Raxis)
    return Q, R


def eigh_old(a, axes, sU=1, Uaxis=-1, tol=0, tol_block=0, D_block=2 ** 32, D_total=2 ** 32,
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
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    nsym = len(a.struct.n)
    n0 = (0,) * nsym

    if a.struct.n != n0:
        raise YastError('Charge should be zero')

    s_eff = (-sU, sU)
    data, struct, ls_l, ls_r = _merge_to_matrix(a, axes, s_eff)

    if ls_l != ls_r:
        raise YastError('Something went wrong in matching the indices of the two tensors')

    SDp = tuple(dd[0] for dd in struct.D)
    Ssl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(SDp), SDp))
    # Ustruct = struct
    Sstruct = struct._replace(Dp=SDp, sl=Ssl, diag=True)

    Usize = struct.sl[-1][1] if len(struct.sl) > 0 else 0
    Ssize = Sstruct.sl[-1][1] if len(Sstruct.sl) > 0 else 0

    # meta = (indA, indS, indU)
    meta = tuple(zip(struct.sl, struct.D, Sstruct.sl))
    Sdata, Udata = a.config.backend.eigh(data, meta, Ssize, Usize)

    ls_s = _leg_struct_truncation(a.config, Sdata, Sstruct.t, Sstruct.sl,
        tol=tol, tol_block=tol_block, D_block=D_block, D_total=D_total,
        keep_multiplets=keep_multiplets, eps_multiplet=eps_multiplet, ordering='eigh')

    if untruncated_S:
        uS = {t: a.config.backend.copy(Sdata[slice(*sl)]) for t, sl in zip(Sstruct.t, Sstruct.sl)}
        uS['D'] = ls_s.Dtot.copy()

    Us = tuple(a.struct.s[lg] for lg in axes[0]) + (sU,)
    Umeta_unmerge, Ustruct = _meta_unfuse_legdec(a.config, struct, [ls_l, ls_s], Us)
    Udata = a.config.backend.unmerge_from_1d(Udata, Umeta_unmerge)
    Umfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=Ustruct, data=Udata, mfs=Umfs, hfs=Uhfs)

    Smeta_unmerge, Sstruct = _meta_unfuse_legdec(a.config, Sstruct, [ls_s], s_eff)
    Sdata = a.config.backend.unmerge_from_1d(Sdata, Smeta_unmerge)
    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=Sstruct, data=Sdata, mfs=Smfs, hfs=Shfs)

    U = U.move_leg(source=-1, destination=Uaxis)
    return (S, U, uS) if untruncated_S else (S, U)


def eigh_pure(a, axes, sU=1, Uaxis=-1):
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

    Returns
    -------
        S, U: Tensor
            U is unitary projector. S is a diagonal tensor.
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    nsym = len(a.struct.n)
    n0 = (0,) * nsym

    if a.struct.n != n0:
        raise YastError('Charge should be zero')

    s_eff = (-sU, sU)
    data, struct, ls_l, ls_r = _merge_to_matrix(a, axes, s_eff)

    if ls_l != ls_r:
        raise YastError('Something went wrong in matching the indices of the two tensors')

    SDp = tuple(dd[0] for dd in struct.D)
    Ssl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(SDp), SDp))
    # Ustruct = struct
    Sstruct = struct._replace(Dp=SDp, sl=Ssl, diag=True)

    Usize = struct.sl[-1][1] if len(struct.sl) > 0 else 0
    Ssize = Sstruct.sl[-1][1] if len(Sstruct.sl) > 0 else 0

    # meta = (indA, indS, indU)
    meta = tuple(zip(struct.sl, struct.D, Sstruct.sl))
    Sdata, Udata = a.config.backend.eigh(data, meta, Ssize, Usize)

    ls_s = _leg_struct_trivial(Sstruct, axis=0)

    Us = tuple(a.struct.s[lg] for lg in axes[0]) + (sU,)
    Umeta_unmerge, Ustruct = _meta_unfuse_legdec(a.config, struct, [ls_l, ls_s], Us)
    Udata = a.config.backend.unmerge_from_1d(Udata, Umeta_unmerge)
    Umfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=Ustruct, data=Udata, mfs=Umfs, hfs=Uhfs)

    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=Sstruct, data=Sdata, mfs=Smfs, hfs=Shfs)

    U = U.move_leg(source=-1, destination=Uaxis)
    return S, U


def eigh(a, axes, sU=1, Uaxis=-1, tol=0, tol_block=0, D_block=2 ** 32, D_total=2 ** 32,
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
    S, U = eigh_pure(a, axes=axes, sU=sU)

    Smask = truncation_mask(S, tol=tol, tol_block=tol_block, D_block=D_block, D_total=D_total,
                            keep_multiplets= keep_multiplets, eps_multiplet=eps_multiplet)

    nsym = len(S.n)
    if untruncated_S:
        uS = {t[:nsym]: a.config.backend.copy(S.data[slice(*sl)]) for t, sl in zip(S.struct.t, S.struct.sl)}
        uS['D'] = Smask.trace(axes=(0, 1))

    U = Smask.mask_apply(U, axis=-1)
    S = Smask.mask_apply(S, axis=0)

    U = U.move_leg(source=-1, destination=Uaxis)
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
    if len(a._data) == 0:
        return a.zero_of_dtype(), a.zero_of_dtype(), a.zero_of_dtype()
    if not a.isdiag:
        _, a, _ = svd(a, axes=axes)
    # entropy, Smin, normalization
    return a.config.backend.entropy(a._data, alpha=alpha)


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
        H[(0, m)] = backend.ones((), dtype=v.yast_dtype, device=v.device)
        T = backend.square_matrix_from_dict(H, m + 1, device=v.device)
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
                v = v.apxb(V[it], x=F[it])
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

    T = backend.square_matrix_from_dict(H, m, device=v0.device)
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
