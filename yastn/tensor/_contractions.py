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
""" Contractions of yastn tensors """
from __future__ import annotations

import abc
from functools import lru_cache
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

from .._profile import nsys_profile
from ._auxiliary import _struct, _clear_axes, _unpack_axes, sign_canonical_order, _compress_slices
from ._auxiliary import find_matching_indices, argsort_t, get_blocks, get_trimmed_struct
from ._merging import _merge_to_matrix, _unmerge, _meta_unmerge_matrix, _meta_fuse_hard
from ._merging import _transpose_and_merge, _mask_tensors_leg_intersection, _meta_mask
from ._tests import YastnError, _test_can_be_combined, _unpack_trans_test_axes_pair

if TYPE_CHECKING:
    from . import Tensor

__all__ = ['tensordot', 'vdot', 'trace', 'swap_gate', 'broadcast', 'apply_mask', 'SpecialTensor', 'fkron']


class SpecialTensor(metaclass=abc.ABCMeta):
    """
    A parent class to create a special tensor-like object.

    ``yastn.tensordot(a, b, axes)`` check if ``a`` or ``b`` is an instance of SpecialTensor
    and calls ``a.tensordo(b, axes)`` or ``b.tensordo(a, axes, reverse=True)``
    """

    @abc.abstractmethod
    def tensordot(self, b, axes, reverse=False):
        pass  # pragma: no cover


def __matmul__(a, b) -> 'Tensor':
    r"""
    The operation ``A @ B`` uses ``@`` operator to compute tensor dot product.
    The operation contracts the last axis of ``self``, i.e., ``a``,
    with the first axis of ``b``.

    It is equivalent to ``yastn.tensordot(a, b, axes=(a.ndim - 1, 0))``.
    """
    return tensordot(a, b, axes=(a.ndim - 1, 0))


def tensordot(a, b, axes, conj=(0, 0)) -> 'Tensor':
    r"""
    Compute tensor dot product of two tensors along specified axes.

    Outgoing legs are ordered such that first ones are the remaining legs
    of the first tensor in the original order, followed by the remaining legs
    of the second tensor in the original order.

    Parameters
    ----------
    a, b: yastn.Tensor
        Tensors to contract.

    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        legs of both tensors to be contracted (for each, they are specified by int or tuple of ints)
        e.g. ``axes=(0, 3)`` to contract 0th leg of ``a`` with 3rd leg of ``b``;
        ``axes=((0, 3), (1, 2))`` to contract legs 0 and 3 of ``a`` with 1 and 2 of ``b``, respectively.

    conj: tuple[int, int]
        specify tensors to conjugate by: ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, or ``(1, 1)``.
        The default is ``(0, 0)``, i.e., neither tensor is conjugated.
    """
    if conj[0]:
        a = a.conj()
    if conj[1]:
        b = b.conj()

    if isinstance(a, SpecialTensor):
        return a.tensordot(b, axes=axes)
    if isinstance(b, SpecialTensor):
        return b.tensordot(a, axes=axes, reverse=True)

    in_a, in_b = _clear_axes(*axes)  # contracted meta legs
    mask_needed, (nin_a, nin_b) = _unpack_trans_test_axes_pair(a, b, sgn=-1, axes=(in_a, in_b))
    # nin_a and nin_b take into account a.trans and b.trans, respectively

    if a.isdiag:
        return _tensordot_diag(a, b, in_b, destination=(0,))
    if b.isdiag:
        return _tensordot_diag(b, a, in_a, destination=(-1,))

    _test_can_be_combined(a, b)
    nout_a = tuple(ii for ii in a.trans if ii not in nin_a)  # outgoing native legs
    nout_b = tuple(ii for ii in b.trans if ii not in nin_b)  # outgoing native legs

    mfs_c = tuple(a.mfs[ii] for ii in range(a.ndim) if ii not in in_a) + \
            tuple(b.mfs[ii] for ii in range(b.ndim) if ii not in in_b)
    hfs_c = tuple(a.hfs[ii] for ii in nout_a) + tuple(b.hfs[ii] for ii in nout_b)

    if mask_needed:
        msk_a, msk_b, a_hfs, b_hfs = _mask_tensors_leg_intersection(a, b, nin_a, nin_b)
        a = _apply_mask_axes(a, nin_a, msk_a)
        b = _apply_mask_axes(b, nin_b, msk_b)
        a = a._replace(hfs=a_hfs)
        b = b._replace(hfs=b_hfs)

    if a.config.tensordot_policy not in ['fuse_to_matrix', 'fuse_contracted', 'no_fusion']:
        raise YastnError("Tensordot policy not recognized. It should be 'fuse_to_matrix', 'fuse_contracted', or 'no_fusion'.")

    if a.config.backend.BACKEND_ID == 'torch_cutensor' and all(0 < x < 33 for x in (a.ndim_n, b.ndim_n)) and len(hfs_c)<33:
        data, struct_out = _tensordot_cutensor(a, b, nout_a, nin_a, nin_b, nout_b)
    elif a.config.tensordot_policy == 'fuse_to_matrix':
        data, struct_out = _tensordot_f2m(a, b, nout_a, nin_a, nin_b, nout_b)
    elif a.config.tensordot_policy == 'fuse_contracted':
        data, struct_out = _tensordot_fc(a, b, nout_a, nin_a, nin_b, nout_b)
    elif a.config.tensordot_policy == 'no_fusion':
        data, struct_out = _tensordot_nf(a, b, nout_a, nin_a, nin_b, nout_b)

    out = a._replace(data=data, struct=struct_out, mfs=mfs_c, hfs=hfs_c, trans=None)
    return out


def _tensordot_diag(a, b, in_b, destination):
    r""" Executes broadcast and then transpose into order expected by tensordot. """
    if len(in_b) == 1:
        c = a.broadcast(b, axes=in_b[0])
        return c.moveaxis(source=in_b, destination=destination)
    if len(in_b) == 2:
        c = a.broadcast(b, axes=in_b[0])
        return c.trace(axes=in_b)
    raise YastnError('Outer product with diagonal tensor not supported. Use yastn.diag() first.')  # if len(in_a) == 0


def _tensordot_f2m(a, b, nout_a, nin_a, nin_b, nout_b):
    r"""
    Perform tensordot by fuse_to_matrix:
    merging tensors to matrices, executing dot, and unmerging outgoing legs.
    """
    legs_a, legs_b = list(a.struct.legs), list(b.struct.legs)
    for ia, ib in zip(nin_a, nin_b):
        try:
            leg = legs_a[ia].intersection(legs_b[ib].conj())
        except ValueError:
            raise YastnError('Bond dimensions of some charges do not match.')
        legs_a[ia] = leg
        legs_b[ib] = leg.conj()

    data_a, struct_am, ls_l, _, legs_group_a = _merge_to_matrix(a, (nout_a, nin_a), tuple(legs_a))
    data_b, struct_bm, _, ls_r, legs_group_b = _merge_to_matrix(b, (nin_b, nout_b), tuple(legs_b))

    meta_dot, size_c, struct_c = _meta_tensordot_f2m(a.config.sym, struct_am, struct_bm)
    data = a.config.backend.dot(data_a, data_b, meta_dot, size_c)

    struct_out = struct_c._replace(legs=(*legs_group_a[0], *legs_group_b[1]))
    meta_unmerge, size_out, struct_out = _meta_unmerge_matrix(a.config.sym, struct_c, ls_l, ls_r, struct_out)
    data = _unmerge(a.config, data, meta_unmerge, size=size_out)
    return data, struct_out


def _tensordot_fc(a, b, nout_a, nin_a, nin_b, nout_b):
    r"""
    Perform tensordot by fuse_contracted: merging contracted legs, and executing dot.
    Outgoing legs are not merged so unmerge is not needed.
    """
    legs_a, legs_b = list(a.struct.legs), list(b.struct.legs)
    for ia, ib in zip(nin_a, nin_b):
        try:
            leg = legs_a[ia].intersection(legs_b[ib].conj())
        except ValueError:
            raise YastnError('Bond dimensions of some charges do not match.')
        legs_a[ia] = leg
        legs_b[ib] = leg.conj()
    legs_a, legs_b = tuple(legs_a), tuple(legs_b)
    #
    axes_a = tuple((x,) for x in nout_a) + (nin_a,)
    order_a = nout_a + nin_a
    meta_mrg_a, size_a, struct_a_new, legs_a_old = _meta_fuse_hard(a.config.sym, a.struct, axes_a, legs_a, empty_first_axis_s_conj=False)
    data_a = _transpose_and_merge(a.config, a._data, order_a, meta_mrg_a, size_a)

    axes_b = (nin_b,) + tuple((x,) for x in nout_b)
    order_b = nin_b + nout_b
    meta_mrg_b, size_b, struct_b_new, legs_b_old = _meta_fuse_hard(b.config.sym, b.struct, axes_b, legs_b, empty_first_axis_s_conj=True)
    data_b = _transpose_and_merge(b.config, b._data, order_b, meta_mrg_b, size_b)

    meta_dot, size_c, struct_c = _meta_tensordot_fc(a.config.sym, struct_a_new, struct_b_new)
    data = a.config.backend.dot(data_a, data_b, meta_dot, size_c)
    return data, struct_c


@nsys_profile
def _tensordot_nf(a, b, nout_a, nin_a, nin_b, nout_b):
    r"""
    Perform tensordot directly: permute blocks and execute dot accumulating results into result blocks.
    """
    meta_dot, reshape_a, reshape_b, size_c, struct_c = _meta_tensordot_nf(a.config.sym, a.struct, b.struct, nout_a, nin_a, nin_b, nout_b)
    order_a = nout_a + nin_a
    order_b = nin_b + nout_b
    data = a.config.backend.transpose_dot_sum(a.data, b.data, meta_dot,
                                              reshape_a, reshape_b, order_a, order_b, size_c)
    return data, struct_c


def _remap_nout_(nout, offset):
    r_nout = [None] * len(nout)
    for n, i in enumerate(np.argsort(nout)):
        r_nout[i] = n + offset
    return r_nout


@nsys_profile
def _tensordot_cutensor(a, b, nout_a, nin_a, nin_b, nout_b):
    struct_c, size_c, *metas = _meta_tensordot_cutensor(a.config.sym, a.struct, b.struct, nout_a, nin_a, nin_b, nout_b)
    if size_c == 0:
        data = a.config.backend.zeros((0,), dtype=a.yastn_dtype, device=a.data.device)
        return data, struct_c

    dot_product= len(nout_a)+len(nout_b) == 0
    if dot_product:
        assert all(len(x)==1 for x in metas[-5:-1]), "Unexpected output tensor meta (dot product)."
        # infer if extra mode was attached to a or b
        if len(nin_a)+len(nout_a)+1 == len(metas[0]): # a_numSectionsPerMode + 1
            nout_a=modes_out= [len(metas[0])-1]
        elif len(nin_b)+len(nout_b)+1 == len(metas[5]):
            nout_b=modes_out= [len(metas[5])-1]
        else:
            raise YastnError("Unexpected input tensor meta (dot product).")

    modes_out = _remap_nout_(nout_a, 0) + _remap_nout_(nout_b, len(nout_a))

    if a.config.sym.NSYM == 0:
        data = a.config.backend.tensordot_dense(a.data, b.data, metas[1], metas[6], nin_a, nin_b, modes_out)
    else:
        data = a.config.backend.tensordot_bs_v2(a.data, b.data, list(nin_a), list(nin_b), *metas, modes_out)
    return data, struct_c


def _indices_from_counts(count_a, count_b):
    nn = np.sum(count_a * count_b, dtype=np.int64)
    ind_a = np.empty(nn, dtype=np.int64)
    ind_b = np.empty(nn, dtype=np.int64)
    ia, ib, ic = 0, 0, 0
    for ca, cb in zip(count_a, count_b):
        cab = ca * cb
        ind_a[ic: ic + cab].reshape(ca, cb)[:] = np.arange(ia, ia + ca).reshape(ca, 1)
        ind_b[ic: ic + cab].reshape(ca, cb)[:] = np.arange(ib, ib + cb).reshape(1, cb)
        ia += ca
        ib += cb
        ic += cab
    return ind_a, ind_b


@lru_cache(maxsize=1024)
def _meta_tensordot_f2m(sym, struct_a, struct_b):
    #
    nout_a, nin_a = [0], [1]
    nin_b, nout_b = [0], [1]
    struct_a_sub, struct_b_sub, struct_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    bl_a, slc_a = get_blocks_and_subslices(sym, struct_a_sub, struct_a)
    bl_b, slc_b = get_blocks_and_subslices(sym, struct_b_sub, struct_b)
    bl_c = get_blocks(sym, struct_c)
    #
    inds = argsort_t(bl_a.t[:, 1, :])
    #
    meta_dt = np.dtype([
        ('slc', np.int64, (2,)),
        ('Dc',  np.int64, (2,)),
        ('sla', np.int64, (2,)),
        ('Da',  np.int64, (2,)),
        ('slb', np.int64, (2,)),
        ('Db',  np.int64, (2,))])
    meta = np.hstack([bl_c.slc[inds], bl_c.D[inds], slc_a[inds], bl_a.D[inds], slc_b, bl_b.D], dtype=np.int64)
    meta = meta.view(meta_dt).reshape(-1)
    return meta, bl_c.size, struct_c


@lru_cache(maxsize=1024)
def _meta_tensordot_fc(sym, struct_a, struct_b):
    #
    ndima, ndimb = len(struct_a.legs), len(struct_b.legs)
    nout_a, nin_a = list(range(ndima - 1)), [ndima - 1]
    nin_b, nout_b = [0], list(range(1, ndimb))
    struct_a_sub, struct_b_sub, struct_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    bl_a, slc_a = get_blocks_and_subslices(sym, struct_a_sub, struct_a)
    bl_b, slc_b = get_blocks_and_subslices(sym, struct_b_sub, struct_b)
    bl_c = get_blocks(sym, struct_c)
    #
    Do_a = np.prod(bl_a.D[:, :-1], axis=1, dtype=np.int64)
    Dc_a = bl_a.D[:, -1]
    Dc_b = bl_b.D[:, 0]
    Do_b = np.prod(bl_b.D[:, 1:], axis=1, dtype=np.int64)
    #
    unique_a, inv_a, count_a = np.unique(bl_a.t[:, -1, :], return_inverse=True, return_counts=True, axis=0)
    unique_b, inv_b, count_b = np.unique(bl_b.t[:,  0, :], return_inverse=True, return_counts=True, axis=0)
    assert np.array_equal(unique_a, unique_b), "Sanity check. Contact developers."
    arg_a = np.argsort(inv_a, kind='stable')
    arg_b = np.argsort(inv_b, kind='stable')
    #
    ind_a, ind_b = _indices_from_counts(count_a, count_b)
    ind_a = arg_a[ind_a]
    ind_b = arg_b[ind_b]
    #
    tn = np.column_stack([bl_a.t[ind_a, :-1, :], bl_b.t[ind_b, 1:, :]])
    ind_tn = find_matching_indices(bl_c.t, tn, both=False)
    slc_c = bl_c.slc[ind_tn]
    #
    slc_a = slc_a[ind_a]
    Do_a = Do_a[ind_a]
    Dc_a = Dc_a[ind_a]
    slc_b = slc_b[ind_b]
    Dc_b = Dc_b[ind_b]
    Do_b = Do_b[ind_b]
    #
    meta = np.column_stack([slc_c, Do_a, Do_b, slc_a, Do_a, Dc_a, slc_b, Dc_b, Do_b])
    meta_dt = np.dtype([
        ('slc', np.int64, (2,)),
        ('Dc',  np.int64, (2,)),
        ('sla', np.int64, (2,)),
        ('Da',  np.int64, (2,)),
        ('slb', np.int64, (2,)),
        ('Db',  np.int64, (2,))])
    meta = meta.view(meta_dt).reshape(-1)
    return meta, bl_c.size, struct_c


@lru_cache(maxsize=1024)
@nsys_profile
def _meta_tensordot_nf(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b):
    #
    struct_a_sub, struct_b_sub, struct_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    bl_a, slc_a = get_blocks_and_subslices(sym, struct_a_sub, struct_a)
    bl_b, slc_b = get_blocks_and_subslices(sym, struct_b_sub, struct_b)
    bl_c = get_blocks(sym, struct_c)
    #
    Daop = np.prod(bl_a.D[:, nout_a], axis=1, dtype=np.int64)
    Dacp = np.prod(bl_a.D[:, nin_a], axis=1, dtype=np.int64)
    unique_a, inv_a, count_a = np.unique(bl_a.t[:, nin_a, :], return_inverse=True, return_counts=True, axis=0) # if more blocks of a contribute to given contracted sector (in b)
    arg_a = np.argsort(inv_a)
    #
    Dbop = np.prod(bl_b.D[:, nout_b], axis=1, dtype=np.int64)
    Dbcp = np.prod(bl_b.D[:, nin_b], axis=1, dtype=np.int64)
    unique_b, inv_b, count_b = np.unique(bl_b.t[:, nin_b, :], return_inverse=True, return_counts=True, axis=0)
    arg_b = np.argsort(inv_b)
    #
    # padding count_a and count_b with zero to allign matching charges
    unique_ab = np.unique(np.vstack([unique_a, unique_b]), axis=0)
    in_a = find_matching_indices(unique_ab, unique_a, both=False)
    in_b = find_matching_indices(unique_ab, unique_b, both=False)
    count_a2 = np.zeros(len(unique_ab), dtype=np.int64)
    count_a2[in_a] = count_a
    count_b2 = np.zeros(len(unique_ab), dtype=np.int64)
    count_b2[in_b] = count_b
    #
    ind_a, ind_b = _indices_from_counts(count_a2, count_b2)
    ind_a = arg_a[ind_a]
    ind_b = arg_b[ind_b]
    #
    tao = bl_a.t[:, nout_a, :]
    tbo = bl_b.t[:, nout_b, :]
    tn = np.column_stack([tao[ind_a], tbo[ind_b]])
    unique_c, inv_c = np.unique(tn, return_inverse=True, axis=0)
    #
    ind_c = find_matching_indices(bl_c.t, unique_c, both=False)
    #
    mask = np.zeros(bl_c.nblocks, dtype=bool)
    mask[ind_c] = True
    struct_c = struct_c.replace(mask=mask)
    bl_c = get_blocks(sym, struct_c)
    slc_c = bl_c.slc[inv_c]
    #
    meta = np.column_stack([slc_c, Daop[ind_a], Dbop[ind_b], ind_a, ind_b])
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('Dn',  np.int64, (2,)),
        ('ta', np.int64),
        ('tb', np.int64)])
    meta = meta.view(meta_dt).reshape(-1)
    #
    ra_dt = np.dtype([
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (len(struct_a.legs),)),
        ('Dl', np.int64),
        ('Dr', np.int64)])
    reshape_a = np.column_stack([slc_a, bl_a.D, Daop, Dacp])
    reshape_a = reshape_a.view(ra_dt).reshape(-1)
    #
    rb_dt = np.dtype([
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (len(struct_b.legs),)),
        ('Dl', np.int64),
        ('Dr', np.int64)])
    reshape_b = np.column_stack([slc_b, bl_b.D, Dbcp, Dbop])
    reshape_b = reshape_b.view(rb_dt).reshape(-1)
    return meta, reshape_a, reshape_b, bl_c.size, struct_c

@nsys_profile
def _convert_bl_for_cutensor(struct, bl, slc=None, dot_product=False):
    """
    Params
    ------
    dot_product: if True, add a dummy mode of size 1,
                 which is needed for cutensor API to perform dot product.
    """
    # extents
    numSectionsPerMode = [len(leg.D) for leg in struct.legs] + ([1] if dot_product else [])
    sectionExtents = [DD for leg in struct.legs for DD in leg.D] + ([1] if dot_product else [])
    #
    # block_coordinates
    iblocks = bl.coords
    if dot_product:
        iblocks = np.column_stack([iblocks, np.zeros((len(iblocks), 1), dtype=np.int64)])
    coords = np.ascontiguousarray(iblocks.reshape(-1))
    #
    # strides
    s0, s1 = bl.D.shape
    strides = np.ones((s0, s1), dtype=np.int64)
    for i in range(s1 - 1, 0, -1):
        strides[:, i - 1] =  strides[:, i] * bl.D[:, i]
    if dot_product:
        strides = np.column_stack([strides, np.ones((s0, 1), dtype=np.int64)])
    strides = np.ascontiguousarray(strides.reshape(-1))
    #
    # offsets
    if slc is None:
        slc = bl.slc
    offsets = np.ascontiguousarray(slc[:, 0])
    #
    return numSectionsPerMode, sectionExtents, coords, strides, offsets


@lru_cache(maxsize=1024)
@nsys_profile
def _meta_tensordot_cutensor(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b):
    #
    struct_a_sub, struct_b_sub, struct_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    bl_a, slc_a = get_blocks_and_subslices(sym, struct_a_sub, struct_a)
    bl_b, slc_b = get_blocks_and_subslices(sym, struct_b_sub, struct_b)
    bl_c = get_blocks(sym, struct_c)
    #
    unique_a, inv_a, count_a = np.unique(bl_a.t[:, nin_a, :], return_inverse=True, return_counts=True, axis=0) # if more blocks of a contribute to given contracted sector (in b)
    arg_a = np.argsort(inv_a)
    #
    unique_b, inv_b, count_b = np.unique(bl_b.t[:, nin_b, :], return_inverse=True, return_counts=True, axis=0)
    arg_b = np.argsort(inv_b)
    #
    # padding count_a and count_b with zero to allign matching charges
    unique_ab = np.unique(np.vstack([unique_a, unique_b]), axis=0)
    in_a = find_matching_indices(unique_ab, unique_a, both=False)
    in_b = find_matching_indices(unique_ab, unique_b, both=False)
    count_a2 = np.zeros(len(unique_ab), dtype=np.int64)
    count_a2[in_a] = count_a
    count_b2 = np.zeros(len(unique_ab), dtype=np.int64)
    count_b2[in_b] = count_b
    #
    ind_a, ind_b = _indices_from_counts(count_a2, count_b2)
    ind_a = arg_a[ind_a]
    ind_b = arg_b[ind_b]
    #
    tao = bl_a.t[:, nout_a, :]
    tbo = bl_b.t[:, nout_b, :]
    tn = np.column_stack([tao[ind_a], tbo[ind_b]])
    unique_c, inv_c = np.unique(tn, return_inverse=True, axis=0)
    #
    ind_c = find_matching_indices(bl_c.t, unique_c, both=False)
    #
    mask = np.zeros(bl_c.nblocks, dtype=bool)
    mask[ind_c] = True
    struct_c = struct_c.replace(mask=mask)
    bl_c = get_blocks(sym, struct_c)
    #
    # dot product, which cannot be simply dispatched to vdot
    #              and either one of operands is (effectively) zero
    if not (len(slc_a) > 0 and len(slc_b) > 0):
        return struct_c, 0, *([None] * 15)
    else:
        dot_product = len(nout_a) + len(nout_b) == 0
        a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets = \
            _convert_bl_for_cutensor(struct_a_sub, bl_a, slc_a, dot_product=(dot_product and len(slc_a) < len(slc_b)))
        b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets = \
            _convert_bl_for_cutensor(struct_b_sub, bl_b, slc_b, dot_product=(dot_product and len(slc_a) >= len(slc_b)))
        c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets = \
            _convert_bl_for_cutensor(struct_c, bl_c, dot_product=dot_product)

    return (struct_c, bl_c.size,
            a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets,
            b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets,
            c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets)


@nsys_profile
def _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b):
    assert not struct_a.isdiag, "Sanity check. Contact developers."
    assert not struct_b.isdiag, "Sanity check. Contact developers."
    n_c = sym.add_charges(struct_a.n, struct_b.n)
    #
    struct_a_0, struct_b_0 = struct_a, struct_b
    legs_a_new, legs_b_new = list(struct_a.legs), list(struct_b.legs)
    while True:
        for ia, ib in zip(nin_a, nin_b):
            try:
                leg = struct_a_0.legs[ia].intersection(struct_b_0.legs[ib].conj())
            except ValueError:
                raise YastnError('Bond dimensions of some charges do not match.')
            legs_a_new[ia] = leg
            legs_b_new[ib] = leg.conj()

        struct_a_1 = get_trimmed_struct(sym, struct_a_0, legs_a_new)
        struct_b_1 = get_trimmed_struct(sym, struct_b_0, legs_b_new)

        legs_c = (*(struct_a_1.legs[ax] for ax in nout_a), *(struct_b_1.legs[ax] for ax in nout_b))
        struct_c_0 = _struct(legs=legs_c, n=n_c, isdiag=False)
        struct_c_1 = get_trimmed_struct(sym, struct_c_0)

        if struct_a_0 == struct_a_1 and struct_b_0 == struct_b_1 and struct_c_0 == struct_c_1:
            break

        struct_a_0 = struct_a_1
        struct_b_0 = struct_b_1
        for ii, ax in enumerate(nout_a):
            legs_a_new[ax] = struct_c_1.legs[ii].intersection(struct_a_1.legs[ax])
        for ii, ax in enumerate(nout_b, start=len(nout_a)):
            legs_b_new[ax] = struct_c_1.legs[ii].intersection(struct_b_1.legs[ax])

    return struct_a_1, struct_b_1, struct_c_1


def get_blocks_and_subslices(sym, struct_sub, struct_full):
    bl = get_blocks(sym, struct_sub)
    if struct_sub == struct_full:
        slc = bl.slc
    else:
        bl_full = get_blocks(sym, struct_full)
        slc = bl_full.slc[find_matching_indices(bl_full.t, bl.t, both=False)]
    return bl, slc


def broadcast(a, *args, axes=0) -> 'Tensor' | tuple['Tensor']:
    r"""
    Compute tensordot product of diagonal tensor ``a`` with tensors in ``args``.

    Produce diagonal tensor if both are diagonal.
    Legs of the resulting tensors are ordered in the same way as those of tensors in ``args``.
    It is used (in combination with :meth:`yastn.transpose`) as a subroutine of
    :meth:`yastn.tensordot` for contractions involving diagonal tensor.

    Parameters
    ----------
    a, args: yastn.Tensor
        ``a`` is diagonal tensor to be broadcasted.

    axes: int | Sequence[int]
        legs of tensors in ``args`` to be multiplied by diagonal tensor ``a``.
        Number of tensors provided in ``args`` should match the length of ``axes``.
    """
    if not a.isdiag:
        raise YastnError('First tensor should be diagonal.')

    multiple_axes = hasattr(axes, '__iter__')
    axes = (axes,) if not multiple_axes else axes
    if len(axes) != len(args):
        raise YastnError("There should be exactly one axis for each tensor to be projected.")
    results = []
    for b, ax in zip(args, axes):
        _test_can_be_combined(a, b)
        ax = ax % len(b.mfs)
        if b.mfs[ax] != (1,):
            raise YastnError('Second tensor`s leg specified by axis cannot be fused.')
        ax = sum(b.mfs[ii][0] for ii in range(ax))  # unpack mfs
        ax = b.trans[ax]  # transpose
        if b.hfs[ax].tree != (1,):
            raise YastnError('Second tensor`s leg specified in axes cannot be fused.')

        meta, size_c, struct_c, ax, ndim = _meta_broadcast(a.config.sym, a.struct, b.struct, ax)
        data = b.config.backend.dot_diag(a._data, b._data, meta, size_c, ax, ndim)
        results.append(b._replace(struct=struct_c, data=data))
    return results if multiple_axes else results.pop()


@lru_cache(maxsize=1024)
def _meta_broadcast(sym, struct_a, struct_b, axis):
    r""" meta information for backend, and new tensor structure for brodcast. """
    bl_a_full = get_blocks(sym, struct_a)
    bl_b_full = get_blocks(sym, struct_b)

    leg_b = struct_b.legs[axis]
    leg_a = struct_a.legs[0] if struct_a.legs[0].s == leg_b.s else struct_a.legs[0].conj()  # .conj() to match signature
    try:
        leg_c = leg_b.intersection(leg_a)
    except ValueError:
        raise YastnError("Bond dimensions of some charges do not match.")
    legs_c = struct_b.legs[:axis] + (leg_c,) + struct_b.legs[axis + 1:]

    struct_c = get_trimmed_struct(sym, struct_b, legs_c)
    struct_aa = get_trimmed_struct(sym, struct_a, (leg_c, leg_c.conj()))
    bl_c = get_blocks(sym, struct_c)
    bl_a = get_blocks(sym, struct_aa)
    #
    inds_c = find_matching_indices(bl_b_full.t, bl_c.t, both=False)
    inds_a = find_matching_indices(bl_a_full.t, bl_a.t, both=False)
    slc_a = bl_a_full.slc[inds_a]
    #
    if struct_b.isdiag:
        D_b = bl_b_full.D[:, 0]
        axis = 0
        ndimb = 1
    else:
        D_b = bl_b_full.D
        ndimb = len(struct_b.legs)
    #
    unique_tax, inv_tax = np.unique(bl_c.t[:, axis, :], return_inverse=True, axis=0)
    assert np.array_equal(bl_a.t[:, 0, :], unique_tax), "Sanity check. Contact developers."
    #
    meta = np.column_stack([bl_c.slc, bl_b_full.slc[inds_c], D_b[inds_c], slc_a[inv_tax]])
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('slb',  np.int64, (2,)),
        ('Db', np.int64, (ndimb,)),
        ('sla',  np.int64, (2,))])
    meta = meta.view(meta_dt).reshape(-1)
    return meta, bl_c.size, struct_c, axis, ndimb


def apply_mask(a, *args, axes=0) -> 'Tensor' | tuple['Tensor']:
    r"""
    Apply mask given by nonzero elements of diagonal tensor ``a`` on specified axes of tensors in args.
    Number of tensors in ``args`` is not restricted.
    The length of the list ``axes`` has to be matching with ``args``.

    Legs of resulting tensor are ordered in the same way as those of tensors in ``args``.
    Bond dimensions of specified ``axes`` of ``args`` are truncated according to the mask ``a``.
    Produce diagonal tensor if both are diagonal.

    Parameters
    ----------
    a, args: yastn.Tensor
        ``a`` is a diagonal tensor

    axes: int | Sequence[int]
        leg of tensors in ``args`` where the mask is applied.
    """
    if not a.isdiag:
        raise YastnError('First tensor should be diagonal.')

    multiple_axes = hasattr(axes, '__iter__')
    axes = (axes,) if not multiple_axes else axes
    if len(axes) != len(args):
        raise YastnError("There should be exactly one axis for each tensor to be projected.")
    results = []

    bl_a = get_blocks(a.config.sym, a.struct)
    mask = {tuple(t[0].tolist()): a.config.backend.to_mask(a._data[slice(*sl)]) for t, sl in zip(bl_a.t, bl_a.slc)}
    mask_t = tuple(mask.keys())
    mask_D = tuple(len(v) for v in mask.values())

    for b, ax in zip(args, axes):
        _test_can_be_combined(a, b)
        ax = ax % len(b.mfs)
        if b.mfs[ax] != (1,):
            raise YastnError('Second tensor`s leg specified by axis cannot be fused.')
        ax = sum(b.mfs[ii][0] for ii in range(ax))  # unpack mfs
        ax = b.trans[ax]  # transpose
        if b.hfs[ax].tree != (1,):
            raise YastnError('Second tensor`s leg specified by axes cannot be fused.')

        meta, size_c, struct_c, ax, ndim = _meta_mask(b.config.sym, b.struct, mask_t, mask_D, ax)
        data = a.config.backend.apply_mask(b._data, mask, meta, size_c, ax, ndim)
        results.append(b._replace(struct=struct_c, data=data))
    return results.pop() if len(results) == 1 else results


def _apply_mask_axes(a, naxes, masks):
    r""" Auxlliary function applying mask tensors to native legs. """
    for axis, mask in zip(naxes, masks):
        if mask is not None:
            mask_tD = {k: len(v) for k, v in mask.items() if len(v) > 0}
            mask_t = tuple(mask_tD.keys())
            mask_D = tuple(mask_tD.values())
            meta, size_c, struct_c, axis, ndim = _meta_mask(a.config.sym, a.struct, mask_t, mask_D, axis)
            data = a.config.backend.apply_mask(a._data, mask, meta, size_c, axis, ndim)
            a = a._replace(struct=struct_c, data=data)
    return a


def vdot(a, b, conj=(1, 0)) -> Number:
    r"""
    Compute scalar product :math:`\langle a|b \rangle` between two tensors.

    Parameters
    ----------
    a, b: yastn.Tensor
        Tensors to contract.

    conj: tuple[int, int]
        indicate which tensors to conjugate: ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, or ``(1, 1)``.
        The default is ``(1, 0)``, i.e., tensor ``a`` is conjugated.
    """
    # axes = (tuple(range(a.ndim)), tuple(range(b.ndim)))
    # return tensordot(a, b, axes=axes, conj=conj).to_number()
    _test_can_be_combined(a, b)
    if conj[0] == 1:
        a = a.conj()
    if conj[1] == 1:
        b = b.conj()

    if a.isdiag and not b.isdiag:
        a = a.diag()
    if b.isdiag and not a.isdiag:
        b = b.diag()

    if a.trans != b.trans:
        a = a.consume_transpose()
        b = b.consume_transpose()

    mask_needed, (nin_a, nin_b) = _unpack_trans_test_axes_pair(a, b, sgn=-1)

    n_c = a.config.sym.add_charges(a.struct.n, b.struct.n)
    if n_c == a.config.sym.zero():
        if mask_needed:
            msk_a, msk_b, a_hfs, b_hfs = _mask_tensors_leg_intersection(a, b, nin_a, nin_b)
            a = _apply_mask_axes(a, nin_a, msk_a)
            b = _apply_mask_axes(b, nin_b, msk_b)
            a = a._replace(hfs=a_hfs)
            b = b._replace(hfs=b_hfs)
        meta = _meta_vdot(a.config.sym, a.struct, b.struct)
    else:
        meta = ()

    return a.config.backend.vdot(a.data, b.data, meta)


@lru_cache(maxsize=1024)
def _meta_vdot(sym, struct_a, struct_b):
    if not all(leg_a.are_consistent(leg_b, sgn=-1) for leg_a, leg_b in zip(struct_a.legs, struct_b.legs)):
        raise YastnError('Bond dimensions of some charges do not match.')
    bl_a = get_blocks(sym, struct_a)
    bl_b = get_blocks(sym, struct_b)
    ind_a, ind_b = find_matching_indices(bl_a.t, bl_b.t)
    meta = np.column_stack([bl_a.slc[ind_a], bl_b.slc[ind_b]])
    meta = _compress_slices(meta)
    meta_dt = np.dtype([
        ('sla', np.int64, (2,)),
        ('slb', np.int64, (2,))])
    meta = meta.view(meta_dt).reshape(-1)
    return meta


def trace(a, axes=(0, 1)) -> 'Tensor':
    r"""
    Compute trace of legs specified by axes.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Legs to be traced out, e.g., ``axes=(0, 1)``; or ``axes=((2, 3, 4), (0, 1, 5))``.
    """
    in_0, in_1 = _clear_axes(*axes)  # contracted legs
    if set(in_0) & set(in_1):
        raise YastnError('The same axis in axes[0] and axes[1].')
    mask_needed, (nin_0, nin_1) = _unpack_trans_test_axes_pair(a, a, sgn=-1, axes=(in_0, in_1))
    # nin_0, nin_1 take into account a.trans

    if len(nin_0) == 0:
        return a

    order = nin_0 + nin_1
    out = tuple(ax for ax in a.trans if ax not in order)
    order = order + out

    mfs = tuple(a.mfs[i] for i in range(a.ndim) if i not in in_0 + in_1)
    hfs = tuple(a.hfs[ax] for ax in out)

    if a.isdiag:
        struct = _struct(legs=(), n=a.n, isdiag=False)
        data = a.config.backend.sum_elements(a._data)
        return a._replace(struct=struct, mfs=mfs, hfs=hfs, isdiag=False, data=data, trans=None)

    if mask_needed:
        msk_0, msk_1, a_hfs, _ = _mask_tensors_leg_intersection(a, a, nin_0, nin_1)
        a = _apply_mask_axes(a, nin_0 + nin_1, msk_0 + msk_1)
        a = a._replace(hfs=a_hfs)

    meta, size, struct = _meta_trace(a.config.sym, a.struct, nin_0, nin_1, out)
    data = a.config.backend.trace(a._data, order, meta, size)

    out = a._replace(mfs=mfs, hfs=hfs, struct=struct, data=data, trans=None)
    return out


@lru_cache(maxsize=1024)
def _meta_trace(sym, struct, nin_0, nin_1, out):
    r""" meta-information for backend and struct of traced tensor. """
    bl = bl_full = get_blocks(sym, struct)
    #
    legs_part = list(struct.legs)
    struct_0 = struct
    while True:
        for ia, ib in zip(nin_0, nin_1):
            try:
                leg = struct_0.legs[ia].intersection(struct_0.legs[ib].conj())
            except ValueError:
                raise YastnError('Bond dimensions of some charges do not match.')
            legs_part[ia] = leg
            legs_part[ib] = leg.conj()

        struct_1 = get_trimmed_struct(sym, struct_0, legs_part)
        bl = get_blocks(sym, struct_1)

        struct_c_0 = _struct(legs=tuple(struct_1.legs[ax] for ax in out), n=struct.n, isdiag=struct.isdiag)
        struct_c_1 = get_trimmed_struct(sym, struct_c_0)

        if struct_0 == struct_1 and struct_c_0 == struct_c_1:
            break

        struct_0 = struct_1
        for ii, ax in enumerate(out):
            legs_part[ax] = struct_c_1.legs[ii].intersection(struct_1.legs[ax])

    bl_c = get_blocks(sym, struct_c_1)

    if struct_1 == struct:
        slo = bl_full.slc
    else:
        bl = get_blocks(sym, struct_1)
        slo = bl_full.slc[find_matching_indices(bl_full.t, bl.t, both=False)]

    t0 = bl.t[:, nin_0, :].reshape(bl.nblocks, len(nin_0) * sym.NSYM)
    t1 = bl.t[:, nin_1, :].reshape(bl.nblocks, len(nin_1) * sym.NSYM)
    ind = (np.all(t0 == t1, axis=1)).nonzero()[0]

    tn = bl.t[ind][:, out, :]
    slo = slo[ind]
    Do = bl.D[ind]
    Dnp = np.prod(Do[:, out], axis=1, dtype=np.int64)
    pD0 = np.prod(Do[:, nin_0], axis=1, dtype=np.int64)
    pD1 = np.prod(Do[:, nin_1], axis=1, dtype=np.int64)

    unique_tn, inv_tn = np.unique(tn, return_inverse=True, axis=0)
    assert len(tn) == 0 or np.array_equal(bl_c.t, unique_tn), "Sanity check. Contact developers."

    meta = np.column_stack([bl_c.slc[inv_tn], slo, Do, pD0, pD1, Dnp])  # sln, slo, Do, Drsh;  Drsh = (pD0, pD1, Dnp)
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (len(struct.legs),)),
        ('Drsh',  np.int64, (3,))])
    meta = meta.view(meta_dt).reshape(-1)
    return meta, bl_c.size, struct_c_1


@nsys_profile
def swap_gate(a, axes, charge=None) -> 'Tensor':
    r"""
    Return tensor after application of a swap gate.

    The function's action is controlled by the ``fermionic`` flag
    in the tensor :ref:`config <tensor/configuration:YASTN configuration>`.
    Multiply blocks with odd charges on swapped legs by :math:`-1`.
    The ``fermionic`` flag selects which individual charges (in case of a direct product of a few symmetries)
    are tested for oddity, where the contributions from each selected charge get multiplied.
    See :class:`yastn.operators.SpinfulFermions` for an example.
    For ``fermionic=True``, all charges are considered.
    For ``fermionic=False``, swap_gate returns ``a``.

    Parameters
    ----------
    axes: Sequence[int | Sequence[int]]
        Tuple with groups of legs. Consecutive pairs of grouped legs that are to be swapped.
        For instance, ``axes = (0, 1)`` apply swap gate between 0th and 1st leg.
        ``axes = ((0, 1), (2, 3), 4, 5)`` swaps ``(0, 1)`` with ``(2, 3)``, and ``4`` with ``5``.

    charge: Optional[Sequence[int] | Sequence[Sequence[int]]]
        If provided, the swap gate is applied between a virtual one-dimensional leg
        of specified charge, e.g., a fermionic string, and tensor legs specified in axes.
        In this case, there is no application of a swap gates between legs specified in axes.
        One can provide list of charges corresponding to each axes, of a single charge to be applied to all axes.
    """
    if not a.config.fermionic:
        return a
    nsym = a.config.sym.NSYM
    fss = (True,) * nsym if a.config.fermionic is True else a.config.fermionic
    if charge is None:
        axes = tuple(_clear_axes(*axes))  # swapped groups of legs
        axes = _unpack_axes(a.mfs, *axes)
        axes = tuple(tuple(a.trans[ax] for ax in axs) for axs in axes)
        negate_slices = _meta_swap_gate(a.config.sym, a.struct, axes, fss)
    else:
        axes, = _clear_axes(axes)  # swapped groups of legs
        if isinstance(charge[0], int):
            charge = (charge,) * len(axes)
        charges = ()
        for t, ax in zip(charge, axes):
            charges += t * a.mfs[ax][0]
        axes, = _unpack_axes(a.mfs, axes)
        axes = tuple(a.trans[ax] for ax in axes)
        negate_slices = _meta_swap_gate_charge(a.config.sym, a.struct, charges, axes, fss)

    newdata = a.config.backend.negate_blocks(a._data, negate_slices)
    return a._replace(data=newdata)


@lru_cache(maxsize=1024)
def _meta_swap_gate(sym, struct, axes, fss):
    r""" Calculate which blocks to negate. """
    bl = get_blocks(sym, struct)
    tp = np.zeros(bl.nblocks, dtype=np.int64)
    if len(axes) % 2 == 1:
        raise YastnError('Odd number of elements in axes. Elements of axes should come in pairs.')
    iaxes = iter(axes)
    for l1, l2 in zip(*(iaxes, iaxes)):
        t1 = np.sum(bl.t[:, l1, :], axis=1, dtype=np.int64) % 2
        t2 = np.sum(bl.t[:, l2, :], axis=1, dtype=np.int64) % 2
        tp += np.sum(t1[:, fss] * t2[:, fss], axis=1, dtype=np.int64)
    tp = tp % 2
    inds = np.where(tp)[0]
    return _compress_slices(bl.slc[inds])


@lru_cache(maxsize=1024)
def _meta_swap_gate_charge(sym, struct, charges, axes, fss):
    #tset, slices, charges, ndim, nsym, axes, fss):
    r""" Calculate which blocks to negate. """
    bl = get_blocks(sym, struct)
    tp = bl.t[:, axes, :]
    try:
        charges = np.array(charges, dtype=np.int64).reshape(1, len(axes), sym.NSYM) % 2
    except ValueError:
        raise YastnError(f'Length or number of charges does not match sym.NSYM or axes.')
    tp = np.sum(tp[:, :, fss] * charges[:, :, fss], axis=(1, 2), dtype=np.int64) % 2
    inds = np.where(tp)[0]
    return _compress_slices(bl.slc[inds])


def fkron(*operators, sites=None, application_order=None):
    """
    Returns a Kronecker product of operators,
    including swap-gate (fermionic string) to handle fermionic operators.

    Parameters
    ----------
    operators: yastn.Tensor
        a sequence of rank-2 tensors

    sites: Sequence[int] | None
        sites corresponding to the provided operators.
        Should be a permutation of 0, 1, ..., len(operators) - 1.
        If None, assume 0, 1, ..., len(operators) - 1.
        Site 0 is the first in the fermionic order.

    application_order: Sequence[int] | None
        Order of applying operators, which might correspond to a sign change for fermionic operators.
        Should be a permutation of 0, 1, ..., len(operators) - 1.
        If None, the last operator is applied first.


    Results
    -------
    Order of outgoing legs, where sites 0, 1, ... go from left to right,
    e.g., fkron(A, B, C, sites=(0, 1, 2)) gives ::

           1     3     5
           |     |     |
        ┌──┴─────┴─────┴──┐
        |  A     B     C  |
        └──┬─────┬─────┬──┘
           |     |     |
           0     2     4

    """
    if sites is None:
        sites = list(range(len(operators)))

    if len(operators) != len(sites) or set(sites) != set(range(len(sites))):
        raise YastnError("sites should be a permutation of 0, 1, ..., len(operators) - 1.")

    if application_order is not None:
        if len(application_order) != len(sites) or set(application_order) != set(range(len(sites))):
            raise YastnError("application_order should be a permutation of 0, 1, ..., len(operators) - 1.")
        sites = [sites[ind] for ind in application_order[::-1]]
        operators = [operators[ind] for ind in application_order[::-1]]

    sym = operators[0].config.sym

    sign = sign_canonical_order(*operators, sites=sites, f_ordered=lambda s1, s2: s1 <= s2)
    operators = dict(zip(sites, operators))
    operators = [operators[n] for n in range(len(operators))]
    n_pattern = [op.n for op in operators]
    acc_n_pattern = [sym.add_charges(*n_pattern[n+1:]) for n in range(len(n_pattern))]
    operators = [op.swap_gate(axes=1, charge=charge) for op, charge in zip(operators, acc_n_pattern)]

    res = sign * operators[0]
    for op in operators[1:]:
        res = tensordot(res, op, axes=((), ()))
    return res
