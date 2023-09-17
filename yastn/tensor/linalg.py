""" Linalg methods for yastn tensor. """
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _struct
from ._tests import YastnError, _test_axes_all
from ._merging import _merge_to_matrix, _meta_unmerge_matrix, _unmerge
from ._merging import _leg_struct_trivial, _Fusion

__all__ = ['svd', 'svd_with_truncation', 'qr', 'eigh', 'eigh_with_truncation', 'norm', 'entropy', 'truncation_mask', 'truncation_mask_multiplets']


def norm(a, p='fro'):
    r"""
    Norm of the tensor.

    Parameters
    ----------
    p: str
        ``'fro'`` for Frobenius norm;  ``'inf'`` for :math:`l^\infty` (or supremum) norm

    Returns
    -------
    real scalar
    """
    if p not in ('fro', 'inf'):
        raise YastnError("Error in norm: p not in ('fro', 'inf'). ")
    return a.config.backend.norm(a._data, p)


def svd_with_truncation(a, axes=(0, 1), sU=1, nU=True, Uaxis=-1, Vaxis=0, policy='fullrank', fix_signs=False,
        tol=0, tol_block=0, D_block=2 ** 32, D_total=2 ** 32,
        mask_f=None, **kwargs):
    r"""
    Split tensor into :math:`a = U S V` using exact singular value decomposition (SVD),
    where the columns of `U` and the rows of `V` form orthonormal bases
    and `S` is positive and diagonal matrix. Optionally, truncate the result.

    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).

    Charge of input tensor `a` is attached to `U` if `nU` and to `V` otherwise.

    Parameters
    ----------
    axes: Sequence(int) or Sequence(Sequence(int),Sequence(int))
        Specify two groups of legs between which to perform SVD, as well as
        their final order.

    sU: int
        signature of the new leg in U; equal 1 or -1. Default is 1.
        V is going to have opposite signature on connecting leg.

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

    mask_f: function(yastn.Tensor) -> yastn.Tensor
        custom truncation mask

    Returns
    -------
    U, S, V: yastn.Tensor
        U and V are unitary projectors. S is a real diagonal tensor.
    """
    diagnostics = kwargs['diagonostics'] if 'diagonostics' in kwargs else None
    U, S, V = svd(a, axes=axes, sU=sU, nU=nU, policy=policy, D_block=D_block, diagnostics=diagnostics, fix_signs=fix_signs)

    if mask_f:
        Smask = mask_f(S)
    else:
        Smask = truncation_mask(S, tol=tol, tol_block=tol_block, D_block=D_block, D_total=D_total)

    U, S, V = Smask.apply_mask(U, S, V, axes=(-1, 0, 0))

    U = U.move_leg(source=-1, destination=Uaxis)
    V = V.move_leg(source=0, destination=Vaxis)
    return U, S, V


def svd(a, axes=(0, 1), sU=1, nU=True, Uaxis=-1, Vaxis=0, policy='fullrank', fix_signs=False, **kwargs):
    r"""
    Split tensor into :math:`a = U S V` using exact singular value decomposition (SVD),
    where the columns of `U` and the rows of `V` form orthonormal bases
    and `S` is a positive and diagonal matrix.

    Charge of input tensor `a` is attached to `U` if `nU` and to `V` otherwise.

    Parameters
    ----------
    axes: Sequence(int) or Sequence(Sequence(int),Sequence(int))
        Specify two groups of legs between which to perform SVD, as well as
        their final order.

    sU: int
        signature of the new leg in U; equal 1 or -1. Default is 1.
        V is going to have opposite signature on connecting leg.

    Uaxis, Vaxis: int
        specify which leg of U and V tensors are connecting with S. By default
        it is the last leg of U and the first of V, in which case a = U @ S @ V.

    Returns
    -------
    U, S, V: yastn.Tensor
        U and V are unitary projectors. S is a real diagonal tensor.
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    data, struct, ls_l, ls_r = _merge_to_matrix(a, axes)

    minD = tuple(min(ds) for ds in struct.D)
    if policy == 'lowrank':
        if 'D_block' not in kwargs:
            raise YastnError("lowrank policy in svd requires passing argument D_block.")
        D_block = kwargs['D_block']
        if not isinstance(D_block, dict):
            minD = tuple(min(D_block, d) for d in minD)
        else:
            nsym = a.config.sym.NSYM
            st = [x[nsym:] for x in struct.t] if nU else [x[:nsym] for x in struct.t]
            minD = tuple(min(D_block.get(t, 0), d) for t, d in zip(st, minD))

    meta, Ustruct, Sstruct, Vstruct = _meta_svd(a.config, struct, minD, sU, nU)
    sizes = tuple(x.sl[-1][1] if len(x.sl) > 0 else 0 for x in (Ustruct, Sstruct, Vstruct))

    if policy == 'fullrank':
        Udata, Sdata, Vdata = a.config.backend.svd(data, meta, sizes, \
            diagnostics=kwargs['diagnostics'] if 'diagnostics' in kwargs else None)
    elif policy == 'lowrank':
        Udata, Sdata, Vdata = a.config.backend.svd_lowrank(data, meta, sizes, **kwargs)
    else:
        raise YastnError('svd policy should be one of (`lowrank`, `fullrank`)')

    if fix_signs:
        Udata, Vdata = a.config.backend.fix_svd_signs(Udata, Vdata, meta)

    ls_s = _leg_struct_trivial(Sstruct, axis=0)

    Us = tuple(a.struct.s[ii] for ii in axes[0]) + (sU,)
    Umeta_unmerge, Ustruct = _meta_unmerge_matrix(a.config, Ustruct, ls_l, ls_s, Us)

    Udata = _unmerge(a.config, Udata, Umeta_unmerge)
    Umfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=Ustruct, data=Udata, mfs=Umfs, hfs=Uhfs)

    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=Sstruct, data=Sdata, mfs=Smfs, hfs=Shfs)

    Vs = (-sU,) + tuple(a.struct.s[ii] for ii in axes[1])
    Vmeta_unmerge, Vstruct = _meta_unmerge_matrix(a.config, Vstruct, ls_s, ls_r, Vs)
    Vdata = _unmerge(a.config, Vdata, Vmeta_unmerge)
    Vmfs = ((1,),) + tuple(a.mfs[ii] for ii in lout_r)
    Vhfs = (_Fusion(s=(-sU,)),) + tuple(a.hfs[ii] for ii in axes[1])
    V = a._replace(struct=Vstruct, data=Vdata, mfs=Vmfs, hfs=Vhfs)

    U = U.move_leg(source=-1, destination=Uaxis)
    V = V.move_leg(source=0, destination=Vaxis)
    return U, S, V


def _meta_svd(config, struct, minD, sU, nU):
    """
    meta and struct for svd
    U has signature = (struct.s[0], sU)
    S has signature = (-sU, sU)
    V has signature = (-sU, struct.s[1])
    if nU than U carries struct.n, otherwise V
    """
    nsym = len(struct.n)
    n0 = (0,) * nsym

    if any(D == 0 for D in minD):
        temp = [(at, aD, asl, mD) for at, aD, asl, mD in zip(struct.t, struct.D, struct.sl, minD) if mD > 0]
        at, aD, asl, minD = zip(*temp) if len(temp) > 0 else ((), (), (), ())
        struct = struct._replace(t=at, D=aD, sl=asl)

    if nU and sU == struct.s[1]:
        t_con = tuple(x[nsym:] for x in struct.t)
    elif nU: # and -sQ == struct.s[1]
        t_con = np.array(struct.t, dtype=int).reshape((len(struct.t), 2, nsym))
        t_con = tuple(tuple(x.flat) for x in config.sym.fuse(t_con[:, 1:, :], (1,), -1))
    elif sU == -struct.s[0]: # and nV (not nU)
        t_con = tuple(x[:nsym] for x in struct.t)
    else: # not nU and sU == struct.s[0]
        t_con = np.array(struct.t, dtype=int).reshape((len(struct.t), 2, nsym))
        t_con = tuple(tuple(x.flat) for x in config.sym.fuse(t_con[:, :1, :], (1,), -1))
    Un, Vn = (struct.n, n0) if nU else (n0, struct.n)

    Ut = tuple(x[:nsym] + y for x, y in zip(struct.t, t_con))
    St = tuple(y + y for y in t_con)
    Vt = tuple(y + x[nsym:] for y, x in zip(t_con, struct.t))
    UD = tuple((ds[0], dm) for ds, dm in zip(struct.D, minD))
    SD = tuple((dm, dm) for dm in minD)
    VD = tuple((dm, ds[1]) for dm, ds in zip(minD, struct.D))
    UDp = tuple(np.prod(UD, axis=1))
    Usl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(UDp), UDp))

    meta = tuple(zip(struct.sl, struct.D, Usl, UD, St, Vt, VD))

    St, Vt, SD, VD = zip(*sorted(zip(St, Vt, SD, VD))) if len(St) > 0 else ((), (), (), ())
    SDp = tuple(dd[0] for dd in SD)
    VDp = tuple(np.prod(VD, axis=1))
    Ssl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(SDp), SDp))
    Vsl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(VDp), VDp))
    Sdict = dict(zip(St, Ssl))
    Vdict = dict(zip(Vt, Vsl))

    meta = tuple((sl, d, slu, du, Sdict[ts], Vdict[tv], dv) for sl, d, slu, du, ts, tv, dv in meta)

    Ustruct = _struct(s=(struct.s[0], sU), n=Un, diag=False, t=Ut, D=UD, Dp=UDp, sl=Usl)
    Sstruct = _struct(s=(-sU, sU), n=n0, diag=True, t=St, D=SD, Dp=SDp, sl=Ssl)
    Vstruct = _struct(s=(-sU, struct.s[1]), n=Vn, diag=False, t=Vt, D=VD, Dp=VDp, sl=Vsl)
    return meta, Ustruct, Sstruct, Vstruct


def truncation_mask_multiplets(S, tol=0, D_total=2 ** 32, eps_multiplet=1e-14, **kwargs):
    """
    Generate a mask tensor from real positive spectrum S, while preserving
    degenerate multiplets. This is achieved by truncating the spectrum
    at the boundary between multiplets.

    Parameters
    ----------
    S: yastn.Tensor
        Diagonal tensor with spectrum.
    tol: float
        relative tolerance
    D_total: int
        maximum number of elements kept
    eps_multiplet:
        relative tolerance on multiplet splitting. If relative difference between
        two consecutive elements of S is larger than ``eps_multiplet``, these
        elements are not considered as part of the same multiplet.

    Returns
    -------
    yastn.Tensor
    """
    if not (S.isdiag and S.yast_dtype == "float64"):
        raise YastnError("Truncation_mask requires S to be real and diagonal")

    # makes a copy for partial truncations; also detaches from autograd computation graph
    Smask = S.copy()
    Smask._data = Smask.data > float('inf') # all False ?
    S_global_max = None

    # find all multiplets in the spectrum
    # 0) convert to plain dense numpy vector and sort in descending order
    s = S.config.backend.to_numpy(S.data)
    inds = np.argsort(s)[::-1].copy() # make descending
    s = s[inds]

    S_global_max = s[0]
    D_trunc = min(sum(s > (S_global_max * tol)), D_total)
    if D_trunc >= len(s):
        # no truncation
        Smask._data = S.data > -float('inf') # all True ?
        return Smask

    # compute gaps and normalize by magnitude of (abs) larger value.
    # value of gaps[i] gives gap between i-th and i+1 the element of s
    maxgap = np.maximum(np.abs(s[:len(s) - 1]), np.abs(s[1:len(s)])) + 1.0e-16
    gaps = np.abs(s[:len(s) - 1] - s[1:len(s)]) / maxgap

    # find nearest multiplet boundary, keeping at most D_trunc elements
    # i-th element of gaps gives gap between i-th and (i+1)-th element of s
    # Note, s[:D_trunc] selects D_trunc values: from 0th to (D_trunc-1)-th element
    for i in range(D_trunc - 1, -1, -1):
        if gaps[i] > eps_multiplet:
            D_trunc = i+1
            break

    Smask._data[inds[:D_trunc]] = True

    # check symmetry related blocks and truncate to equal length
    active_sectors = filter(lambda x: any(Smask[x]), Smask.struct.t)
    for t in active_sectors:
        tn = np.array(t, dtype=int).reshape((1, 1, -1))
        tn = tuple(S.config.sym.fuse(tn, (1,), -1).flat)
        if t == tn:
            continue

        common_size = min(len(Smask[t]), len(Smask[tn]))
        # if related blocks do not have equal length
        if common_size > len(Smask[t]):
            # assert sum(Smask[t][common_size:]) <= 0 ,\
            #     "Symmetry-related blocks do not match"
            Smask[t][common_size:] = False
        if common_size > len(Smask[tn]):
            # assert sum(Smask[tn][common_size:])<=0,\
            #     "Symmetry-related blocks do not match"
            Smask[tn][common_size:] = False

        if not all(Smask[t][:common_size] == Smask[tn][:common_size]):
            Smask[t][:common_size] = Smask[tn][:common_size] = Smask[t][:common_size] & Smask[tn][:common_size]
    return Smask


def truncation_mask(S, tol=0, tol_block=0, D_block=2 ** 32, D_total=2 ** 32, **kwargs):
    """
    Generate mask tensor based on diagonal and real tensor S.
    It can be then used for truncation.

    Per block options ``D_block`` and ``tol_block`` govern truncation within individual blocks,
    keeping at most ``D_block`` values which are larger than relative cutoff ``tol_block``.

    Parameters
    ----------
    S: yastn.Tensor
        Diagonal tensor with spectrum.
    tol: float
        relative tolerance
    tol_block: float
        relative tolerance per block
    D_total: int
        maximum number of elements kept
    D_block: int
        maximum number of elements kept per block

    Returns
    -------
    yastn.Tensor
    """
    if not (S.isdiag and S.yast_dtype == "float64"):
        raise YastnError("Truncation_mask requires S to be real and diagonal")

    # makes a copy for partial truncations; also detaches from autograd computation graph
    S = S.copy()
    Smask = S.copy()
    Smask._data = Smask._data > -float('inf') # all True

    nsym = S.config.sym.NSYM
    tol_null = 0. if isinstance(tol_block, dict) else tol_block
    D_null = 0 if isinstance(D_block, dict) else D_block
    for t, Db, sl in zip(S.struct.t, S.struct.Dp, S.struct.sl):
        t = t[:nsym]
        tol_rel = tol_block[t] if (isinstance(tol_block, dict) and t in tol_block) else tol_null
        D_tol = sum(S.data[slice(*sl)] > tol_rel * S.config.backend.max_abs(S.data[slice(*sl)])).item()
        D_bl = D_block[t] if (isinstance(D_block, dict) and t in D_block) else D_null
        D_bl = min(D_bl, D_tol)
        if 0 < D_bl < Db:  # no block truncation
            inds = S.config.backend.argsort(S.data[slice(*sl)])
            Smask._data[slice(*sl)][inds[:-D_bl]] = False
        elif D_bl == 0:
            Smask._data[slice(*sl)] = False

    temp_data = S._data * Smask.data
    D_tol = sum(temp_data > tol * S.config.backend.max_abs(temp_data)).item()
    D_total = min(D_total, D_tol)
    if 0 < D_total < sum(Smask.data):
        inds = S.config.backend.argsort(temp_data)
        Smask._data[inds[:-D_total]] = False
    elif D_total == 0:
        Smask._data[:] = False
    return Smask


def qr(a, axes=(0, 1), sQ=1, Qaxis=-1, Raxis=0):
    r"""
    Split tensor using reduced QR decomposition, such that :math:`a = Q R`,
    with :math:`QQ^\dagger=I`. The charge of R is zero.

    Parameters
    ----------
    axes: Sequence(int) or Sequence(Sequence(int),Sequence(int))
        Specify two groups of legs between which to perform QR, as well as their final order.

    sQ: int
        signature of connecting leg in Q; equal 1 or -1. Default is 1.
        R is going to have opposite signature on connecting leg.

    Qaxis, Raxis: int
        specify which leg of Q and R tensors are connecting to the other tensor.
        By delault it is the last leg of Q and the first leg of R.

    Returns
    -------
    Q, R: yastn.Tensor
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    data, struct, ls_l, ls_r = _merge_to_matrix(a, axes)
    meta, Qstruct, Rstruct = _meta_qr(a.config, struct, sQ)

    sizes = tuple(x.sl[-1][1] if len(x.sl) > 0 else 0 for x in (Qstruct, Rstruct))
    Qdata, Rdata = a.config.backend.qr(data, meta, sizes)

    ls = _leg_struct_trivial(Rstruct, axis=0)

    Qs = tuple(a.struct.s[lg] for lg in axes[0]) + (sQ,)
    Qmeta_unmerge, Qstruct = _meta_unmerge_matrix(a.config, Qstruct, ls_l, ls, Qs)
    Qdata = _unmerge(a.config, Qdata, Qmeta_unmerge)
    Qmfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Qhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sQ,)),)
    Q = a._replace(struct=Qstruct, data=Qdata, mfs=Qmfs, hfs=Qhfs)

    Rs = (-sQ,) + tuple(a.struct.s[lg] for lg in axes[1])
    Rmeta_unmerge, Rstruct = _meta_unmerge_matrix(a.config, Rstruct, ls, ls_r, Rs)
    Rdata = _unmerge(a.config, Rdata, Rmeta_unmerge)
    Rmfs = ((1,),) + tuple(a.mfs[ii] for ii in lout_r)
    Rhfs = (_Fusion(s=(-sQ,)),) + tuple(a.hfs[ii] for ii in axes[1])
    R = a._replace(struct=Rstruct, data=Rdata, mfs=Rmfs, hfs=Rhfs)

    Q = Q.move_leg(source=-1, destination=Qaxis)
    R = R.move_leg(source=0, destination=Raxis)
    return Q, R


def _meta_qr(config, struct, sQ):
    """
    meta and struct for qr.
    Q has signature = (struct.s[0], sQ)
    R has signature = (-sQ, struct.s[1])
    """
    minD = tuple(min(ds) for ds in struct.D)
    nsym = len(struct.n)
    n0 = (0,) * nsym

    if sQ == struct.s[1]:
        t_con = tuple(x[nsym:] for x in struct.t)
    else: # -sQ == struct.s[1]
        t_con = np.array(struct.t, dtype=int).reshape((len(struct.t), 2, nsym))
        t_con = tuple(tuple(x.flat) for x in config.sym.fuse(t_con[:, 1:, :], (1,), -1))

    Qt = tuple(x[:nsym] + y for x, y in zip(struct.t, t_con))
    Rt = tuple(y + x[nsym:] for y, x in zip(t_con, struct.t))
    QD = tuple((ds[0], dm) for ds, dm in zip(struct.D, minD))
    RD = tuple((dm, ds[1]) for dm, ds in zip(minD, struct.D))
    QDp = tuple(np.prod(QD, axis=1))
    Qsl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(QDp), QDp))

    meta = tuple(zip(struct.sl, struct.D, Qsl, QD, Rt, RD))

    Rt, RD = zip(*sorted(zip(Rt, RD))) if len(Rt) > 0 else ((), ())
    RDp = tuple(np.prod(RD, axis=1))
    Rsl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(RDp), RDp))
    Rdict = dict(zip(Rt, Rsl))

    meta = tuple((sl, d, slq, dq, Rdict[tr], dr) for sl, d, slq, dq, tr, dr in meta)
    Qstruct = struct._replace(t=Qt, D=QD, Dp=QDp, sl=Qsl, s=(struct.s[0], sQ))
    Rstruct = struct._replace(t=Rt, D=RD, Dp=RDp, sl=Rsl, s=(-sQ, struct.s[1]), n=n0)
    return meta, Qstruct, Rstruct


def eigh(a, axes, sU=1, Uaxis=-1):
    r"""
    Split symmetric tensor using exact eigenvalue decomposition, :math:`a= USU^{\dagger}`.

    Tensor is expected to be symmetric (hermitian) with total charge 0.

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
    S, U: yastn.Tensor
        U is unitary projector. S is a diagonal tensor.
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    if not all(x == 0 for x in a.struct.n):
        raise YastnError('eigh requires tensor charge to be zero')

    data, struct, ls_l, ls_r = _merge_to_matrix(a, axes)

    if ls_l != ls_r:
        raise YastnError("Tensor likely not hermitian. Legs of effective square blocks not match.")

    meta, Sstruct, Ustruct = _meta_eigh(a.config, struct, sU)
    sizes = tuple(x.sl[-1][1] if len(x.sl) > 0 else 0 for x in (Sstruct, Ustruct))

    Sdata, Udata = a.config.backend.eigh(data, meta, sizes)

    ls_s = _leg_struct_trivial(Sstruct, axis=1)

    Us = tuple(a.struct.s[lg] for lg in axes[0]) + (sU,)
    Umeta_unmerge, Ustruct = _meta_unmerge_matrix(a.config, Ustruct, ls_l, ls_s, Us)
    Udata = _unmerge(a.config, Udata, Umeta_unmerge)
    Umfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=Ustruct, data=Udata, mfs=Umfs, hfs=Uhfs)

    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=Sstruct, data=Sdata, mfs=Smfs, hfs=Shfs)

    U = U.move_leg(source=-1, destination=Uaxis)
    return S, U


def _meta_eigh(config, struct, sU):
    """
    meta and struct for eigh
    U has signature = (struct.s[0], sU)
    S has signature = (-sU, sU)
    """
    nsym = len(struct.n)
    n0 = (0,) * nsym

    if sU == -struct.s[0]:
        t_con = tuple(x[:nsym] for x in struct.t)
    else: # and sU == struct.s[0]
        t_con = np.array(struct.t, dtype=int).reshape((len(struct.t), 2, nsym))
        t_con = tuple(tuple(x.flat) for x in config.sym.fuse(t_con[:, :1, :], (1,), -1))

    Ut = tuple(x[:nsym] + y for x, y in zip(struct.t, t_con))
    Ustruct = struct._replace(t=Ut, s=(struct.s[0], sU))

    St = tuple(y + y for y in t_con)
    SD = struct.D

    meta = tuple(zip(struct.sl, struct.D, Ustruct.sl, Ustruct.D, St))

    St, SD = zip(*sorted(zip(St, SD))) if len(St) > 0 else ((), ())
    SDp = tuple(dd[0] for dd in SD)
    Ssl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(SDp), SDp))
    Sdict = dict(zip(St, Ssl))

    meta = tuple((sl, d, slu, du, Sdict[ts]) for sl, d, slu, du, ts in meta)

    Sstruct = _struct(s=(-sU, sU), n=n0, diag=True, t=St, D=SD, Dp=SDp, sl=Ssl)
    return meta, Sstruct, Ustruct


def eigh_with_truncation(a, axes, sU=1, Uaxis=-1, tol=0, tol_block=0,
    D_block=2 ** 32, D_total=2 ** 32):
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
    S, U: yastn.Tensor
        U is unitary projector. S is a diagonal tensor.
    """
    S, U = eigh(a, axes=axes, sU=sU)

    Smask = truncation_mask(S, tol=tol, tol_block=tol_block, D_block=D_block, D_total=D_total)

    S, U = Smask.apply_mask(S, U, axes=(0, -1))
    U = U.move_leg(source=-1, destination=Uaxis)
    return S, U


def entropy(a, axes=(0, 1), alpha=1):
    r"""
    Calculate entropy from spliting the tensor using svd.

    If diagonal, calculates entropy treating S^2 as probabilities. Normalizes S^2 if neccesary.
    If not diagonal, starts with svd to get the diagonal S.
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
    entropy, minimal singular value, normalization : number
    """
    if len(a._data) == 0:
        return a.zero_of_dtype(), a.zero_of_dtype(), a.zero_of_dtype()
    if not a.isdiag:
        _, a, _ = svd(a, axes=axes)
    # entropy, Smin, normalization
    return a.config.backend.entropy(a._data, alpha=alpha)
