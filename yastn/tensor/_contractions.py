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
import os
from functools import lru_cache
from numbers import Number
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .._profile import nsys_profile, nvtx_range
from ._auxiliary import _encode_rows_shared, _row_keys_pair, _struct, _clear_axes, _unpack_axes, sign_canonical_order, _compress_slices
from ._auxiliary import find_matching_indices, argsort_t, get_blocks, hash_blocks, get_trimmed_struct, convert_to_tuples_and_slices
from ._merging import _unfuse_blocks, _fuse_blocks, _mask_tensors_leg_intersection, _meta_mask
from ._tests import YastnError, _test_can_be_combined, _unpack_trans_test_axes_pair
from ..backend import import_backend

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


def tensordot(a, b, axes, conj=(0, 0), lazy_threshold=None) -> 'Tensor':
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

    if lazy_threshold is None:
        lazy_threshold = a.config.lazy_threshold

    if a.config.backend.BACKEND_ID == 'torch_cutensor' and all(0 < x < 33 for x in (a.ndim_n, b.ndim_n)) and len(hfs_c)<33:
        data, struct_out = _tensordot_cutensor(a, b, nout_a, nin_a, nin_b, nout_b, lazy_threshold)
    elif a.config.tensordot_policy == 'fuse_to_matrix':
        data, struct_out = _tensordot_f2m(a, b, nout_a, nin_a, nin_b, nout_b, lazy_threshold)
    elif a.config.tensordot_policy == 'fuse_contracted':
        data, struct_out = _tensordot_fc(a, b, nout_a, nin_a, nin_b, nout_b, lazy_threshold)
    elif a.config.tensordot_policy == 'no_fusion':
        data, struct_out = _tensordot_nf(a, b, nout_a, nin_a, nin_b, nout_b, lazy_threshold)

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


def _tensordot_f2m(a, b, nout_a, nin_a, nin_b, nout_b, lazy_threshold):
    r"""
    Perform tensordot by fuse_to_matrix:
    merging tensors to matrices, executing dot, and unmerging outgoing legs.
    """
    struct_a_sub, struct_b_sub, _ = _match_legs_tensordot(a.config.sym, a.struct, b.struct, nout_a, nin_a, nin_b, nout_b)
    #
    data_am, struct_am, hfs_am = _fuse_blocks(a.config, a._data, a.struct, (nout_a, nin_a), struct_a_sub,
                                              connector_first=False, lazy_threshold=lazy_threshold)
    data_bm, struct_bm, hfs_bm = _fuse_blocks(b.config, b._data, b.struct, (nin_b, nout_b), struct_b_sub,
                                              connector_first=True, lazy_threshold=lazy_threshold)
    #
    meta_dot, size_cm, struct_cm = _meta_tensordot_f2m(a.config.sym, struct_am, struct_bm)
    if not nout_a and not nout_b:
        size_cm = 1
    data_cm = a.config.backend.dot(data_am, data_bm, meta_dot, size_cm)
    #
    hfs_cm, axes_cm, legs_cm = [], [], []
    if nout_a:
        hfs_cm.append(hfs_am[0])
        axes_cm.append(0)
        legs_cm.append(struct_cm.legs[0])
    if nout_b:
        hfs_cm.append(hfs_bm[-1])
        axes_cm.append(len(axes_cm))
        legs_cm.append(struct_cm.legs[1])
    struct_cm = struct_cm.replace(legs=legs_cm)
    #
    data_c, struct_c = _unfuse_blocks(a.config, data_cm, struct_cm, tuple(axes_cm), tuple(hfs_cm),
                                      lazy_threshold=lazy_threshold)
    return data_c, struct_c


def _tensordot_fc(a, b, nout_a, nin_a, nin_b, nout_b, lazy_threshold):
    r"""
    Perform tensordot by fuse_contracted: merging contracted legs, and executing dot.
    Outgoing legs are not merged so unmerge is not needed.
    """
    struct_a_sub, struct_b_sub, _ = _match_legs_tensordot(a.config.sym, a.struct, b.struct, nout_a, nin_a, nin_b, nout_b)
    #
    axes_a = tuple((x,) for x in nout_a) + (nin_a,)
    data_am, struct_am, _ = _fuse_blocks(a.config, a._data, a.struct, axes_a, struct_a_sub, connector_first=False,
                                         lazy_threshold=lazy_threshold)
    axes_b = (nin_b,) + tuple((x,) for x in nout_b)
    data_bm, struct_bm, _ = _fuse_blocks(b.config, b._data, b.struct, axes_b, struct_b_sub, connector_first=True,
                                         lazy_threshold=lazy_threshold)
    #
    meta_dot, size_c, struct_c = _meta_tensordot_fc(a.config.sym, struct_am, struct_bm,
                                                    lazy_threshold=lazy_threshold)
    data = a.config.backend.dot(data_am, data_bm, meta_dot, size_c)
    return data, struct_c


@nsys_profile
def _tensordot_nf(a, b, nout_a, nin_a, nin_b, nout_b, lazy_threshold):
    r"""
    Perform tensordot directly: permute blocks and execute dot accumulating results into result blocks.
    """
    meta_dot, reshape_a, reshape_b, size_c, struct_c = _meta_tensordot_nf(a.config.sym, a.struct, b.struct, nout_a, nin_a, nin_b, nout_b,
                                                                          lazy_threshold=lazy_threshold)
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
def _tensordot_cutensor(a, b, nout_a, nin_a, nin_b, nout_b, lazy_threshold):
    struct_c, size_c, hash_a, hash_b, hash_c, *metas = \
        _meta_tensordot_cutensor(a.config.sym, a.struct, b.struct, nout_a, nin_a, nin_b, nout_b,
            lazy_threshold=lazy_threshold, policy=a.config.meta_tensordot_policy,
            device=a.device, backend_id=a.config.backend.BACKEND_ID)
    if size_c == 0:
        data = a.config.backend.zeros((0,), dtype=a.yastn_dtype, device=a.device)
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
        # each 64-byte blake2b digest -> 8 int64; stacked into a (3, 8) array
        hashes_int64 = np.hstack([np.frombuffer(h, dtype=np.int64) for h in (hash_a, hash_b, hash_c)]).tolist()
        data = a.config.backend.tensordot_bs_v2(a.data, b.data, list(nin_a), list(nin_b), *metas, modes_out, hashes_int64)
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
    nout_a, nin_a = (0,), (1,)
    nin_b, nout_b = (0,), (1,)
    struct_a_sub, struct_b_sub, struct_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    bl_a, slc_a = get_blocks_and_subslices(sym, struct_a_sub, struct_a)
    bl_b, slc_b = get_blocks_and_subslices(sym, struct_b_sub, struct_b)
    bl_c = get_blocks(sym, struct_c)
    #
    inds = argsort_t(bl_a.t[:, -1, :])
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
    meta = convert_to_tuples_and_slices(meta)
    return meta, bl_c.size, struct_c


@lru_cache(maxsize=1024)
def _meta_tensordot_fc(sym, struct_a, struct_b, lazy_threshold):
    #
    ndima, ndimb = len(struct_a.legs), len(struct_b.legs)
    nout_a, nin_a = tuple(range(ndima - 1)), (ndima - 1,)
    nin_b, nout_b = (0,), tuple(range(1, ndimb))
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
    #
    if lazy_threshold and bl_c.nblocks and len(ind_tn) / bl_c.nblocks < lazy_threshold:
        arg_tn = np.argsort(ind_tn)  # tn are not ordered
        ind_a = ind_a[arg_tn]
        ind_b = ind_b[arg_tn]
        struct_c = struct_c.mask_from_ind(bl_c.nblocks, ind_tn)
        struct_c = get_trimmed_struct(sym, struct_c)
        bl_c = get_blocks(sym, struct_c)
        slc_c = bl_c.slc
    else:
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
    meta = convert_to_tuples_and_slices(meta)
    return meta, bl_c.size, struct_c


@lru_cache(maxsize=1024)
@nsys_profile
def _meta_tensordot_nf(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b, lazy_threshold):
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
    if lazy_threshold and bl_c.nblocks and len(ind_c) / bl_c.nblocks < lazy_threshold:
        struct_c = struct_c.mask_from_ind(bl_c.nblocks, ind_c)
        struct_c = get_trimmed_struct(sym, struct_c)
        bl_c = get_blocks(sym, struct_c)
        slc_c = bl_c.slc
    else:
        slc_c = bl_c.slc[ind_c]
    #
    meta = np.column_stack([slc_c[inv_c], Daop[ind_a], Dbop[ind_b], ind_a, ind_b])
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('Dn',  np.int64, (2,)),
        ('ta', np.int64),
        ('tb', np.int64)])
    meta = meta.view(meta_dt).reshape(-1)
    meta = convert_to_tuples_and_slices(meta)
    #
    ra_dt = np.dtype([
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (len(struct_a.legs),)),
        ('Dl', np.int64),
        ('Dr', np.int64)])
    reshape_a = np.column_stack([slc_a, bl_a.D, Daop, Dacp])
    reshape_a = reshape_a.view(ra_dt).reshape(-1)
    reshape_a = convert_to_tuples_and_slices(reshape_a)
    #
    rb_dt = np.dtype([
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (len(struct_b.legs),)),
        ('Dl', np.int64),
        ('Dr', np.int64)])
    reshape_b = np.column_stack([slc_b, bl_b.D, Dbcp, Dbop])
    reshape_b = reshape_b.view(rb_dt).reshape(-1)
    reshape_b = convert_to_tuples_and_slices(reshape_b)
    return meta, reshape_a, reshape_b, bl_c.size, struct_c


class _cutensor_meta(NamedTuple):
    numSectionsPerMode: np.ndarray
    sectionExtents: np.ndarray
    coords: np.ndarray
    strides: np.ndarray


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


@nsys_profile
def _matched_pair_indices(unique_a, count_a, arg_a, unique_b, count_b, arg_b):
    """ Block-index lists (ind_a, ind_b) of every matching (a, b) pair, aligned by contracted
    charge sector (counts padded with zeros to line up matching charges). """
    unique_ab = np.unique(np.vstack([unique_a, unique_b]), axis=0)
    in_a = find_matching_indices(unique_ab, unique_a, both=False)
    in_b = find_matching_indices(unique_ab, unique_b, both=False)
    count_a2 = np.zeros(len(unique_ab), dtype=np.int64)
    count_a2[in_a] = count_a
    count_b2 = np.zeros(len(unique_ab), dtype=np.int64)
    count_b2[in_b] = count_b
    #
    ind_a, ind_b = _indices_from_counts(count_a2, count_b2)
    return arg_a[ind_a], arg_b[ind_b]


@lru_cache(maxsize=1024)
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


# A/B toggle for the cutensor tensordot meta builder.
# Both share leg-matching, block enumeration, the "align charge indices" step and the
# backend-meta finalization (kept here); they differ only in "unique out blocks" and "mask c".
# Override default choice "auto" with YASTN_META_CUTENSOR=CPU|GPU. Each version is independently lru_cached.
_META_CUTENSOR_VERSION = os.environ.get("YASTN_META_CUTENSOR", "AUTO").lower()


def _meta_tensordot_cutensor(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b,
                             lazy_threshold:float=None, policy:str=None,
                             device:str=None, backend_id:str=None):
    """ Dispatch to the optimized CPU or GPU _meta_tensordot_cutensor builder.
    ``device`` (the first operand's data device) is used only by GPU. """

    if policy in ["gpu",] or ((policy in ["auto"] and not import_backend(backend_id).is_cpu_device(device)) \
                              or _META_CUTENSOR_VERSION in ["gpu",]):
        from ._contractions_cutensor import _meta_tensordot_cutensor_gpu  # lazy: avoids import cycle
        meta= _meta_tensordot_cutensor_gpu(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b,
                    lazy_threshold=lazy_threshold, device=device, backend_id=backend_id)
        if meta is not None:
            return meta
        # int64 overflow in optimized mask construction; fall back to CPU version
    return _meta_tensordot_cutensor_cpu(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b, lazy_threshold)


@nsys_profile
def locate_rows(tset, query):
    """
    Index of each row of ``query`` within ``tset``, or ``len(tset)`` when the row is absent.

    ``tset`` must have unique rows sorted in the same (per-column, lexicographic) order
    produced by ``np.unique(..., axis=0)``; ``query`` may contain arbitrary/repeated rows.
    Unlike :func:`yastn.tensor._auxiliary.find_matching_indices`, the result has one entry
    per row of ``query`` (misses flagged with the sentinel ``len(tset)``), so it maps rows
    to their class id.
    """
    rs, *cs = tset.shape
    rq, *cq = query.shape
    assert cs == cq, "Sanity check. Contact developers."
    cp = np.prod(cs, dtype=np.int64)
    if rs == 0:  # empty tset: every query row is absent
        return np.full(rq, rs, dtype=np.int64)
    if cp == 0:  # zero-width rows collapse to a single (empty) class present in tset
        return np.zeros(rq, dtype=np.int64)
    v_t, v_q = _row_keys_pair(tset.reshape(rs, cp), query.reshape(rq, cp))
    ids = np.searchsorted(v_t, v_q)
    hit = ids < rs
    safe_ind = np.where(hit, ids, 0)
    hit &= v_t[safe_ind] == v_q
    ids[~hit] = rs  # sentinel for rows absent from tset
    return ids


def unique_rows(t, return_inverse=False, return_counts=False):
    """
    Drop-in accelerator for ``np.unique(t, axis=0, ...)`` on integer charge blocks.

    Encodes rows as order-preserving int64 keys and uniques those in 1-D, which is markedly
    faster than numpy's structured/void ``axis=0`` sort; falls back to ``np.unique(axis=0)``
    when the key space would overflow. Returns rows in the same order as ``np.unique(axis=0)``
    and preserves the input's row shape ``t.shape[1:]``.
    """
    rs = t.shape[0]
    cp = int(np.prod(t.shape[1:], dtype=np.int64))
    enc = _encode_rows_shared([t.reshape(rs, cp)]) if rs > 0 else None
    if enc is None:
        return np.unique(t, axis=0, return_inverse=return_inverse, return_counts=return_counts)
    uk, first, inv, cnt = np.unique(enc[0], return_index=True, return_inverse=True, return_counts=True)
    res = [t[first]]  # representative row per class, in key-sorted (== structured-sorted) order
    if return_inverse:
        res.append(inv.reshape(-1))
    if return_counts:
        res.append(cnt)
    return res[0] if len(res) == 1 else tuple(res)


@lru_cache(maxsize=1024)
@nsys_profile
def _meta_tensordot_cutensor_cpu(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b, lazy_threshold=None):
    struct_a_sub, struct_b_sub, struct_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    bl_a, slc_a = get_blocks_and_subslices(sym, struct_a_sub, struct_a)
    bl_b, slc_b = get_blocks_and_subslices(sym, struct_b_sub, struct_b)
    bl_c = get_blocks(sym, struct_c)

    if lazy_threshold and bl_c.nblocks:
        with nvtx_range("unique in blocks"):
            unique_a, inv_a, count_a = unique_rows(bl_a.t[:, nin_a, :], return_inverse=True, return_counts=True) # if more blocks of a contribute to given contracted sector (in b)
            arg_a = np.argsort(inv_a)
            #
            unique_b, inv_b, count_b = unique_rows(bl_b.t[:, nin_b, :], return_inverse=True, return_counts=True)
            arg_b = np.argsort(inv_b)
        #
        ind_a, ind_b = _matched_pair_indices(unique_a, count_a, arg_a, unique_b, count_b, arg_b)
        #
        with nvtx_range("unique out blocks"):
            # A candidate output block of bl_c is produced iff its (out_a, out_b) charge tuple
            # arises from some matched (a, b) pair. Encode each out_a / out_b tuple as a small
            # integer id (one class per distinct tuple among the blocks) and reduce the per-pair
            # test to 1-D int64 operations. This avoids building and sorting the ~nn wide rows
            # that column_stack + np.unique(axis=0) would otherwise require (nn = matched pairs).
            na, nb, nsym = len(nout_a), len(nout_b), bl_a.t.shape[2]  # explicit widths: bl_*.nblocks may be 0
            ua, id_a = unique_rows(bl_a.t[:, nout_a, :].reshape(bl_a.nblocks, na * nsym), return_inverse=True)
            ub, id_b = unique_rows(bl_b.t[:, nout_b, :].reshape(bl_b.nblocks, nb * nsym), return_inverse=True)
            id_a, id_b, n_b = id_a.reshape(-1), id_b.reshape(-1), len(ub)
            #
            keys = np.unique(id_a[ind_a] * n_b + id_b[ind_b])  # produced (out_a, out_b) classes
            #
            # struct_c legs are ordered (out_a from nout_a, then out_b from nout_b), so bl_c.t
            # splits into its out_a (:na) and out_b (na:) columns aligned with ua / ub above
            cid_a = locate_rows(ua, bl_c.t[:, :na, :].reshape(bl_c.nblocks, na * nsym))
            cid_b = locate_rows(ub, bl_c.t[:, na:, :].reshape(bl_c.nblocks, nb * nsym))
            c_keys = np.where((cid_a < len(ua)) & (cid_b < n_b), cid_a * n_b + cid_b, -1)
            #
            mask = np.isin(c_keys, keys)  # -1 sentinel (tuple absent from a/b blocks) never matches
            struct_c = struct_c.replace(mask=mask)
            bl_c = get_blocks(sym, struct_c)

    # dot product, which cannot be simply dispatched to vdot
    #              and either one of operands is (effectively) zero
    if not (len(slc_a) > 0 and len(slc_b) > 0):
        return struct_c, 0, *([None] * 18)
    else:
        dot_product = len(nout_a) + len(nout_b) == 0
        a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets = \
            _convert_bl_for_cutensor(struct_a_sub, bl_a, slc_a, dot_product=(dot_product and len(slc_a) < len(slc_b)))
        b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets = \
            _convert_bl_for_cutensor(struct_b_sub, bl_b, slc_b, dot_product=(dot_product and len(slc_a) >= len(slc_b)))
        c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets = \
            _convert_bl_for_cutensor(struct_c, bl_c, dot_product=dot_product)

        h_a, h_b, h_c = hash_blocks(_cutensor_meta(a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides), out=bytes),\
                    hash_blocks(_cutensor_meta(b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides), out=bytes),\
                    hash_blocks(_cutensor_meta(c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides), out=bytes)

    return (struct_c, bl_c.size, h_a, h_b, h_c,
            a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets,
            b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets,
            c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets)


def get_blocks_and_subslices(sym, struct_sub, struct_full):
    bl = get_blocks(sym, struct_sub)
    if struct_sub == struct_full:
        slc = bl.slc
    else:
        bl_full = get_blocks(sym, struct_full)
        slc = bl_full.slc[find_matching_indices(bl_full.t, bl.t, both=False)]
    return bl, slc


def broadcast(a, *args, axes=0, lazy_threshold=None) -> 'Tensor' | tuple['Tensor']:
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
    meta = convert_to_tuples_and_slices(meta)
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
    meta = convert_to_tuples_and_slices(meta)
    return meta


def trace(a, axes=(0, 1), lazy_threshold=None) -> 'Tensor':
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

    if lazy_threshold is None:
        lazy_threshold = a.config.lazy_threshold

    meta, size, struct = _meta_trace(a.config.sym, a.struct, nin_0, nin_1, out, lazy_threshold)
    data = a.config.backend.trace(a._data, order, meta, size)

    out = a._replace(mfs=mfs, hfs=hfs, struct=struct, data=data, trans=None)
    return out


@lru_cache(maxsize=1024)
def _meta_trace(sym, struct, nin_0, nin_1, out, lazy_threshold):
    r""" meta-information for backend and struct of traced tensor. """
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

    bl, slo = get_blocks_and_subslices(sym, struct_1, struct)
    bl_c = get_blocks(sym, struct_c_1)

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
    ind = find_matching_indices(bl_c.t, unique_tn, both=False)

    if lazy_threshold and bl_c.nblocks and len(ind) / bl_c.nblocks < lazy_threshold:
        struct_c_1 = struct_c_1.mask_from_ind(bl_c.nblocks, ind)
        bl_c = get_blocks(sym, struct_c_1)
        sln = bl_c.slc
    else:
        sln = bl_c.slc[ind]

    meta = np.column_stack([sln[inv_tn], slo, Do, pD0, pD1, Dnp])  # sln, slo, Do, Drsh;  Drsh = (pD0, pD1, Dnp)
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (len(struct.legs),)),
        ('Drsh',  np.int64, (3,))])
    meta = meta.view(meta_dt).reshape(-1)
    meta = convert_to_tuples_and_slices(meta)
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
