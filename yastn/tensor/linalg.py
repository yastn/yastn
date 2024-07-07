# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Linalg methods for yastn.Tensor. """
from __future__ import annotations
from itertools import accumulate
import numpy as np
from ._auxliary import _struct, _slc, _clear_axes, _unpack_axes
from ._tests import YastnError, _test_axes_all
from ._merging import _merge_to_matrix, _meta_unmerge_matrix, _unmerge
from ._merging import _Fusion, _leg_struct_trivial

__all__ = ['svd', 'svd_with_truncation', 'qr', 'eigh', 'eigh_with_truncation', 'norm', 'entropy', 'truncation_mask', 'truncation_mask_multiplets']


def norm(a, p='fro') -> number:
    r"""
    Norm of the tensor.

    Parameters
    ----------
    p: str
        ``'fro'`` for Frobenius norm;  ``'inf'`` for :math:`l^\infty` (or supremum) norm.
    """
    if p not in ('fro', 'inf'):
        raise YastnError("Error in norm: p not in ('fro', 'inf'). ")
    return a.config.backend.norm(a._data, p)


def svd_with_truncation(a, axes=(0, 1), sU=1, nU=True,
        Uaxis=-1, Vaxis=0, policy='fullrank', fix_signs=False,
        tol=0, tol_block=0, D_block=float('inf'), D_total=float('inf'),
        mask_f=None, **kwargs) -> tuple[yastn.Tensor, yastn.Tensor, yastn.Tensor]:
    r"""
    Split tensor into :math:`a = U S V` using exact singular value decomposition (SVD),
    where the columns of `U` and the rows of `V` form orthonormal bases
    and `S` is positive and diagonal matrix. Optionally, truncate the result.

    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).

    Charge of input tensor `a` is attached to `U` if `nU` and to `V` otherwise.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform SVD, as well as
        their final order.

    sU: int
        signature of the new leg in U; equal 1 or -1. Default is 1.
        V is going to have opposite signature on connecting leg.

    nU: bool
        Whether or not to attach the charge of  `a` to `U`.
        Otherwise it is attached to `V`. By default is True.

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

    mask_f: function[yastn.Tensor] -> yastn.Tensor
        custom truncation mask

    Returns
    -------
    U, S, V
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


def svd(a, axes=(0, 1), sU=1, nU=True, compute_uv=True,
        Uaxis=-1, Vaxis=0, policy='fullrank',
        fix_signs=False, **kwargs) -> tuple[yastn.Tensor, yastn.Tensor, yastn.Tensor] | yastn.Tensor:
    r"""
    Split tensor into :math:`a = U S V` using exact singular value decomposition (SVD),
    where the columns of `U` and the rows of `V` form orthonormal bases
    and `S` is a positive and diagonal matrix.



    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform SVD, as well as
        their final order.

    sU: int
        Signature of the new leg in U; equal to 1 or -1. By default is 1.
        V is going to have the opposite signature on the connecting leg.

    nU: bool
        Whether or not to attach the charge of  `a` to `U`.
        Otherwise it is attached to `V`. By default is True.

    compute_uv: bool
        If True, compute U and V in addition to S. Default is True.

    Uaxis, Vaxis: int
        Specify which leg of U and V tensors are connecting with S. By default
        it is the last leg of U and the first of V, in which case a = U @ S @ V.

    policy: str
        "fullrank" or "lowrank". Use standard full (but reduced) SVD for "fullrank".
        For "lowrank", uses randomized/truncated SVD and requires providing `D_block` in `kwargs`.

    fix_signs: bool
        Whether or not to fix phases in `U` and `V`,
        so that the largest element in each column of `U` is positive.
        Provide uniqueness of decomposition for non-degenerate cases.
        By default is False.

    Returns
    -------
    U, S, V or S
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    data, struct, slices, ls_l, ls_r = _merge_to_matrix(a, axes)

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

    meta, Ustruct, Uslices, Sstruct, Sslices, Vstruct, Vslices = _meta_svd(a.config, struct, slices, minD, sU, nU)
    sizes = tuple(x.size for x in (Ustruct, Sstruct, Vstruct))

    if compute_uv and policy == 'fullrank':
        Udata, Sdata, Vdata = a.config.backend.svd(data, meta, sizes, \
            diagnostics=kwargs['diagnostics'] if 'diagnostics' in kwargs else None)
    elif not compute_uv and policy == 'fullrank':
        Sdata = a.config.backend.svdvals(data, meta, sizes[1])
    elif compute_uv and policy == 'lowrank':
        Udata, Sdata, Vdata = a.config.backend.svd_lowrank(data, meta, sizes, **kwargs)
    else:
        raise YastnError('svd policy should in (`lowrank`, `fullrank`). compute_uv == False only works with `fullrank`')

    if compute_uv and fix_signs:
        Udata, Vdata = a.config.backend.fix_svd_signs(Udata, Vdata, meta)

    ls_s = _leg_struct_trivial(Sstruct, axis=0)

    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=Sstruct, slices=Sslices, data=Sdata, mfs=Smfs, hfs=Shfs)

    if not compute_uv:
        return S

    Us = tuple(a.struct.s[ii] for ii in axes[0]) + (sU,)
    Umeta_unmerge, Ustruct, Uslices = _meta_unmerge_matrix(a.config, Ustruct, Uslices, ls_l, ls_s, Us)
    Udata = _unmerge(a.config, Udata, Umeta_unmerge)
    Umfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=Ustruct, slices=Uslices, data=Udata, mfs=Umfs, hfs=Uhfs)

    Vs = (-sU,) + tuple(a.struct.s[ii] for ii in axes[1])
    Vmeta_unmerge, Vstruct, Vslices = _meta_unmerge_matrix(a.config, Vstruct, Vslices, ls_s, ls_r, Vs)
    Vdata = _unmerge(a.config, Vdata, Vmeta_unmerge)
    Vmfs = ((1,),) + tuple(a.mfs[ii] for ii in lout_r)
    Vhfs = (_Fusion(s=(-sU,)),) + tuple(a.hfs[ii] for ii in axes[1])
    V = a._replace(struct=Vstruct, slices=Vslices, data=Vdata, mfs=Vmfs, hfs=Vhfs)

    U = U.move_leg(source=-1, destination=Uaxis)
    V = V.move_leg(source=0, destination=Vaxis)
    return U, S, V


def _meta_svd(config, struct, slices, minD, sU, nU):
    """
    meta and struct for svd
    U has signature = (struct.s[0], sU)
    S has signature = (-sU, sU)
    V has signature = (-sU, struct.s[1])
    if nU than U carries struct.n, otherwise V
    """
    n0 = config.sym.zero()
    nsym = config.sym.NSYM

    if any(D == 0 for D in minD):
        at = tuple(x for x, mD in zip(struct.t, minD) if mD > 0)
        aD = tuple(x for x, mD in zip(struct.D, minD) if mD > 0)
        slices = tuple(x for x, mD in zip(slices, minD) if mD > 0)
        minD = tuple(mD for mD in minD if mD > 0)
        struct = struct._replace(t=at, D=aD)

    if nU and sU == struct.s[1]:
        t_con = tuple(x[nsym:] for x in struct.t)
    elif nU: # and -sQ == struct.s[1]
        t_con = np.array(struct.t, dtype=np.int64).reshape((len(struct.t), 2, nsym))
        t_con = tuple(map(tuple, config.sym.fuse(t_con[:, 1:, :], (1,), -1).tolist()))
    elif sU == -struct.s[0]: # and nV (not nU)
        t_con = tuple(x[:nsym] for x in struct.t)
    else: # not nU and sU == struct.s[0]
        t_con = np.array(struct.t, dtype=np.int64).reshape((len(struct.t), 2, nsym))
        t_con = tuple(map(tuple, config.sym.fuse(t_con[:, :1, :], (1,), -1).tolist()))
    Un, Vn = (struct.n, n0) if nU else (n0, struct.n)

    Ut = tuple(x[:nsym] + y for x, y in zip(struct.t, t_con))
    St = tuple(y + y for y in t_con)
    Vt = tuple(y + x[nsym:] for y, x in zip(t_con, struct.t))
    UD = tuple((ds[0], dm) for ds, dm in zip(struct.D, minD))
    SD = tuple((dm, dm) for dm in minD)
    VD = tuple((dm, ds[1]) for dm, ds in zip(minD, struct.D))
    UDp = tuple(np.prod(UD, axis=1))
    Usl = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(UDp), UDp, UD))

    meta = tuple(zip(slices, struct.D, Usl, UD, St, Vt, VD))
    St, Vt, SD, VD = zip(*sorted(zip(St, Vt, SD, VD))) if len(St) > 0 else ((), (), (), ())
    SDp = tuple(dd[0] for dd in SD)
    VDp = tuple(np.prod(VD, axis=1))
    Ssl = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(SDp), SDp, SD))
    Vsl = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(VDp), VDp, VD))
    Sdict = {x: y.slcs[0] for x, y in zip(St, Ssl)}
    Vdict = {x: y.slcs[0] for x, y in zip(Vt, Vsl)}

    meta = tuple((sl.slcs[0], d, slu.slcs[0], du, Sdict[ts], Vdict[tv], dv) for sl, d, slu, du, ts, tv, dv in meta)

    Ustruct = _struct(s=(struct.s[0], sU), n=Un, diag=False, t=Ut, D=UD, size=sum(UDp))
    Sstruct = _struct(s=(-sU, sU), n=n0, diag=True, t=St, D=SD, size=sum(SDp))
    Vstruct = _struct(s=(-sU, struct.s[1]), n=Vn, diag=False, t=Vt, D=VD, size=sum(VDp))
    return meta, Ustruct, Usl, Sstruct, Ssl, Vstruct, Vsl


def truncation_mask_multiplets(S, tol=0, D_total=float('inf'),
                               eps_multiplet=1e-13, **kwargs) -> yastn.Tensor[bool]:
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
    eps_multiplet: float
        relative tolerance on multiplet splitting. If relative difference between
        two consecutive elements of S is larger than ``eps_multiplet``, these
        elements are not considered as part of the same multiplet.
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
        tn = np.array(t, dtype=np.int64).reshape((1, 1, -1))
        tn = tuple(S.config.sym.fuse(tn, (1,), -1).ravel().tolist())
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


def truncation_mask(S, tol=0, tol_block=0, D_block=float('inf'),
                    D_total=float('inf'), **kwargs) -> yastn.Tensor[bool]:
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
    for t, sl in zip(S.struct.t, S.slices):
        t = t[:nsym]
        tol_rel = tol_block[t] if (isinstance(tol_block, dict) and t in tol_block) else tol_null
        D_tol = sum(S.data[slice(*sl.slcs[0])] > tol_rel * S.config.backend.max_abs(S.data[slice(*sl.slcs[0])])).item()
        D_bl = D_block[t] if (isinstance(D_block, dict) and t in D_block) else D_null
        D_bl = min(D_bl, D_tol)
        if 0 < D_bl < sl.Dp:  # no block truncation
            inds = S.config.backend.argsort(S.data[slice(*sl.slcs[0])])
            Smask._data[slice(*sl.slcs[0])][inds[:-D_bl]] = False
        elif D_bl == 0:
            Smask._data[slice(*sl.slcs[0])] = False

    temp_data = S._data * Smask.data
    D_tol = sum(temp_data > tol * S.config.backend.max_abs(temp_data)).item()
    D_total = min(D_total, D_tol)
    if 0 < D_total < sum(Smask.data):
        inds = S.config.backend.argsort(temp_data)
        Smask._data[inds[:-D_total]] = False
    elif D_total == 0:
        Smask._data[:] = False
    return Smask


def qr(a, axes=(0, 1), sQ=1, Qaxis=-1, Raxis=0) -> tuple[yastn.Tensor, yastn.Tensor]:
    r"""
    Split tensor using reduced QR decomposition, such that :math:`a = Q R`,
    with :math:`QQ^\dagger=I`. The charge of R is zero.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform QR, as well as their final order.

    sQ: int
        signature of connecting leg in Q; equal 1 or -1. Default is 1.
        R is going to have opposite signature on connecting leg.

    Qaxis, Raxis: int
        specify which leg of Q and R tensors are connecting to the other tensor.
        By default it is the last leg of Q and the first leg of R.

    Returns
    -------
    Q, R
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    data, struct, slices, ls_l, ls_r = _merge_to_matrix(a, axes)
    meta, Qstruct, Qslices, Rstruct, Rslices = _meta_qr(a.config, struct, slices, sQ)

    sizes = tuple(x.size for x in (Qstruct, Rstruct))
    Qdata, Rdata = a.config.backend.qr(data, meta, sizes)

    ls = _leg_struct_trivial(Rstruct, axis=0)

    Qs = tuple(a.struct.s[lg] for lg in axes[0]) + (sQ,)
    Qmeta_unmerge, Qstruct, Qslices = _meta_unmerge_matrix(a.config, Qstruct, Qslices, ls_l, ls, Qs)
    Qdata = _unmerge(a.config, Qdata, Qmeta_unmerge)
    Qmfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Qhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sQ,)),)
    Q = a._replace(struct=Qstruct, slices=Qslices, data=Qdata, mfs=Qmfs, hfs=Qhfs)

    Rs = (-sQ,) + tuple(a.struct.s[lg] for lg in axes[1])
    Rmeta_unmerge, Rstruct, Rslices = _meta_unmerge_matrix(a.config, Rstruct, Rslices, ls, ls_r, Rs)
    Rdata = _unmerge(a.config, Rdata, Rmeta_unmerge)
    Rmfs = ((1,),) + tuple(a.mfs[ii] for ii in lout_r)
    Rhfs = (_Fusion(s=(-sQ,)),) + tuple(a.hfs[ii] for ii in axes[1])
    R = a._replace(struct=Rstruct, slices=Rslices, data=Rdata, mfs=Rmfs, hfs=Rhfs)

    Q = Q.move_leg(source=-1, destination=Qaxis)
    R = R.move_leg(source=0, destination=Raxis)
    return Q, R


def _meta_qr(config, struct, slices, sQ):
    """
    meta and struct for qr.
    Q has signature = (struct.s[0], sQ)
    R has signature = (-sQ, struct.s[1])
    """
    minD = tuple(min(ds) for ds in struct.D)
    n0 = config.sym.zero()
    nsym = config.sym.NSYM

    if sQ == struct.s[1]:
        t_con = tuple(x[nsym:] for x in struct.t)
    else: # -sQ == struct.s[1]
        t_con = np.array(struct.t, dtype=np.int64).reshape((len(struct.t), 2, nsym))
        t_con = tuple(map(tuple, config.sym.fuse(t_con[:, 1:, :], (1,), -1).tolist()))

    Qt = tuple(x[:nsym] + y for x, y in zip(struct.t, t_con))
    Rt = tuple(y + x[nsym:] for y, x in zip(t_con, struct.t))
    QD = tuple((ds[0], dm) for ds, dm in zip(struct.D, minD))
    RD = tuple((dm, ds[1]) for dm, ds in zip(minD, struct.D))
    QDp = tuple(np.prod(QD, axis=1))
    Qsl = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(QDp), QDp, QD))

    meta = tuple(zip(slices, struct.D, Qsl, QD, Rt, RD))

    Rt, RD = zip(*sorted(zip(Rt, RD))) if len(Rt) > 0 else ((), ())
    RDp = tuple(np.prod(RD, axis=1))
    Rsl = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(RDp), RDp, RD))
    Rdict = {x: y.slcs[0] for x, y in zip(Rt, Rsl)}

    meta = tuple((sl.slcs[0], d, slq.slcs[0], dq, Rdict[tr], dr) for sl, d, slq, dq, tr, dr in meta)
    Qstruct = struct._replace(t=Qt, D=QD, size=sum(QDp), s=(struct.s[0], sQ))
    Rstruct = struct._replace(t=Rt, D=RD, size=sum(RDp), s=(-sQ, struct.s[1]), n=n0)
    return meta, Qstruct, Qsl, Rstruct, Rsl


def eigh(a, axes, sU=1, Uaxis=-1) -> tuple[yastn.Tensor, yastn.Tensor]:
    r"""
    Split symmetric tensor using exact eigenvalue decomposition, :math:`a= USU^{\dagger}`.

    Tensor is expected to be symmetric (hermitian) with total charge 0.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform svd, as well as their final order.

    sU: int
        signature of connecting leg in U equall 1 or -1. Default is 1.

    Uaxis: int
        specify which leg of U is the new connecting leg. By default it is the last leg.

    Returns
    -------
    S, U
    """
    _test_axes_all(a, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(a.mfs, lout_l, lout_r)

    if not all(x == 0 for x in a.struct.n):
        raise YastnError('eigh requires tensor charge to be zero')

    data, struct, slices, ls_l, ls_r = _merge_to_matrix(a, axes)

    if ls_l != ls_r:
        raise YastnError("Tensor likely not hermitian. Legs of effective square blocks not match.")

    meta, Sstruct, Sslices, Ustruct, Uslices = _meta_eigh(a.config, struct, slices, sU)
    sizes = tuple(x.size for x in (Sstruct, Ustruct))

    Sdata, Udata = a.config.backend.eigh(data, meta, sizes)

    ls_s = _leg_struct_trivial(Sstruct, axis=1)

    Us = tuple(a.struct.s[lg] for lg in axes[0]) + (sU,)
    Umeta_unmerge, Ustruct, Uslices = _meta_unmerge_matrix(a.config, Ustruct, Uslices, ls_l, ls_s, Us)
    Udata = _unmerge(a.config, Udata, Umeta_unmerge)
    Umfs = tuple(a.mfs[ii] for ii in lout_l) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in axes[0]) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=Ustruct, slices=Uslices, data=Udata, mfs=Umfs, hfs=Uhfs)

    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=Sstruct, slices=Sslices, data=Sdata, mfs=Smfs, hfs=Shfs)

    U = U.move_leg(source=-1, destination=Uaxis)
    return S, U


def _meta_eigh(config, struct, slices, sU):
    """
    meta and struct for eigh
    U has signature = (struct.s[0], sU)
    S has signature = (-sU, sU)
    """
    n0 = config.sym.zero()
    nsym = config.sym.NSYM

    if sU == -struct.s[0]:
        t_con = tuple(x[:nsym] for x in struct.t)
    else: # and sU == struct.s[0]
        t_con = np.array(struct.t, dtype=np.int64).reshape((len(struct.t), 2, nsym))
        t_con = tuple(map(tuple, config.sym.fuse(t_con[:, :1, :], (1,), -1).tolist()))

    Ut = tuple(x[:nsym] + y for x, y in zip(struct.t, t_con))
    Ustruct = struct._replace(t=Ut, s=(struct.s[0], sU))

    St = tuple(y + y for y in t_con)
    SD = struct.D

    meta = tuple(zip(slices, struct.D, St))

    St, SD = zip(*sorted(zip(St, SD))) if len(St) > 0 else ((), ())
    SDp = tuple(dd[0] for dd in SD)

    Ssl = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(SDp), SDp, SD))
    Sdict = {x: y.slcs[0] for x, y in zip(St, Ssl)}

    meta = tuple((sl.slcs[0], d, sl.slcs[0], d, Sdict[ts]) for sl, d, ts in meta)

    Sstruct = _struct(s=(-sU, sU), n=n0, diag=True, t=St, D=SD, size=sum(SDp))
    return meta, Sstruct, Ssl, Ustruct, slices


def eigh_with_truncation(a, axes, sU=1, Uaxis=-1, tol=0, tol_block=0,
    D_block=float('inf'), D_total=float('inf')) -> tuple[yastn.Tensor, yastn.Tensor]:
    r"""
    Split symmetric tensor using exact eigenvalue decomposition, :math:`a= USU^{\dagger}`.
    Optionally, truncate the resulting decomposition.

    Tensor is expected to be symmetric (hermitian) with total charge 0.
    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).
    Truncate based on tolerance only if some eigenvalues are positive -- than all negative ones are discarded.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform svd, as well as their final order.

    sU: int
        signature of connecting leg in U equall 1 or -1. Default is 1.

    Uaxis: int
        specify which leg of U is the new connecting leg. By default it is the last leg.

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
    S, U
    """
    S, U = eigh(a, axes=axes, sU=sU)

    Smask = truncation_mask(S, tol=tol, tol_block=tol_block, D_block=D_block, D_total=D_total)

    S, U = Smask.apply_mask(S, U, axes=(0, -1))
    U = U.move_leg(source=-1, destination=Uaxis)
    return S, U


def entropy(a, alpha=1, tol=1e-12) -> number:
    r"""
    Calculate entropy from probabilities encoded in diagonal tensor `a`.

    Normalizes (sum of) `a` to 1, but do not check correctness otherwise.
    Use base-2 log. For empty or zero tensor, return 0.

    Parameters
    ----------
    alpha: float
        Order of Renyi entropy.
        alpha == 1 is von Neuman entropy: -Tr(a log2(a))
        otherwise: 1/(1-alpha) log2(Tr(a ** alpha))

    tol: float
        Discard all probabilities smaller than `tol` during calculation.
    """
    if not a.isdiag:
        raise YastnError("yastn.linalg.entropy requires diagonal tensor.")
    if not alpha > 0:
        raise YastnError("yastn.linalg.entropy requires positive order alpha.")
    return a.config.backend.entropy(a._data, alpha=alpha, tol=tol)
