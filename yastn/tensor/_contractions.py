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

from ._auxiliary import _struct, _clear_axes, _unpack_axes, sign_canonical_order, _compress_slices
from ._auxiliary import find_matching_indices, argsort_t, get_blocks, get_blocks_charges
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

    if a.config.backend.BACKEND_ID == 'torch_cpp' and all(0 < x < 32 for x in (a.ndim_n, b.ndim_n, len(hfs_c))):
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
    legs_a, legs_b = list(a.legs), list(b.legs)
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
    legs_a, legs_b = list(a.legs), list(b.legs)
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


def _tensordot_nf(a, b, nout_a, nin_a, nin_b, nout_b):
    r"""
    Perform tensordot directly: permute blocks and execute dot accumulating results into result blocks.
    """
    if a.config.profile: a.config.backend.nvtx.range_push(f"_meta_tensordot_nf")
    meta_dot, reshape_a, reshape_b, size_c, struct_c = _meta_tensordot_nf(a.config.sym, a.struct, b.struct, nout_a, nin_a, nin_b, nout_b)
    if a.config.profile: a.config.backend.nvtx.range_pop()
    order_a = nout_a + nin_a
    order_b = nin_b + nout_b
    data = a.config.backend.transpose_dot_sum(a.data, b.data, meta_dot,
                                              reshape_a, reshape_b, order_a, order_b, size_c)
    if a.config.profile: a.config.backend.nvtx.range_pop()
    return data, struct_c


def _remap_nout_(nout, offset):
    r_nout = [None] * len(nout)
    for n, i in enumerate(np.argsort(nout)):
        r_nout[i] = n + offset
    return r_nout


def _tensordot_cutensor(a, b, nout_a, nin_a, nin_b, nout_b):
    if a.config.profile: a.config.backend.nvtx.range_push(f"_meta_tensordot_cutensor")
    struct_c, size_c, *metas = _meta_tensordot_cutensor(a.config.sym, a.struct, b.struct, nout_a, nin_a, nin_b, nout_b)
    if a.config.profile: a.config.backend.nvtx.range_pop()
    if size_c == 0:
        data = a.config.backend.zeros((0,), dtype=a.yastn_dtype, device=a.data.device)
        return data, struct_c

    modes_out = _remap_nout_(nout_a, 0) + _remap_nout_(nout_b, len(nout_a))

    if a.config.profile: a.config.backend.nvtx.range_push(f"ops.tensordot_bs")
    if a.config.sym.NSYM == 0:
        data = a.config.backend.tensordot_dense(a.data, b.data, metas[1], metas[6], nin_a, nin_b, modes_out)
    else:
        data = a.config.backend.tensordot_bs(a.data, b.data, list(nin_a), list(nin_b), *metas, modes_out)
    if a.config.profile: a.config.backend.nvtx.range_pop()
    return data, struct_c


def _int_block_coordinates(sym, struct):
    saxes = tuple(leg.s for leg in struct.legs)
    taxes = tuple(leg.t for leg in struct.legs)
    tblocks, iblocks, _ = get_blocks_charges(sym, taxes, saxes, struct.n)
    return tblocks, iblocks.reshape(-1).tolist()


def _get_extends(legs):
    numSectionsPerMode = [len(leg.D) for leg in legs]
    sectionExtents = [DD for leg in legs for DD in leg.D]
    return numSectionsPerMode, sectionExtents


def _get_strides(Dset):
    s0, s1 = Dset.shape
    S = np.ones((s0, s1), dtype=np.int64)
    for i in range(s1 - 1, 0, -1):
        S[:, i - 1] =  S[:, i] * Dset[:, i]
    return S


@lru_cache(maxsize=1024)
def _meta_tensordot_f2m(sym, struct_a, struct_b):
    #
    nout_a, nin_a = [0], [1]
    nin_b, nout_b = [0], [1]
    bl_a, slc_a, bl_b, slc_b, bl_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
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
    return meta, bl_c.size, bl_c.struct


@lru_cache(maxsize=1024)
def _meta_tensordot_fc(sym, struct_a, struct_b):
    #
    ndima, ndimb = len(struct_a.legs), len(struct_b.legs)
    nout_a, nin_a = list(range(ndima - 1)), [ndima - 1]
    nin_b, nout_b = [0], list(range(1, ndimb))
    bl_a, slc_a, bl_b, slc_b, bl_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    #
    slDoc_a = np.empty((bl_a.nblocks, 4), dtype=np.int64)
    slDoc_a[:, :2] = slc_a
    slDoc_a[:, 2] = np.prod(bl_a.D[:, :-1], axis=1, dtype=np.int64)
    slDoc_a[:, 3] = bl_a.D[:, -1]
    slDoc_b = np.empty((bl_b.nblocks, 4), dtype=np.int64)
    slDoc_b[:, :2] = slc_b
    slDoc_b[:, 2] = np.prod(bl_b.D[:, 1:], axis=1, dtype=np.int64)
    slDoc_b[:, 3] = bl_b.D[:, 0]
    #
    unique_a, inv_a = np.unique(bl_a.t[:, -1, :], return_inverse=True, axis=0)
    unique_b, inv_b = np.unique(bl_b.t[:,  0, :], return_inverse=True, axis=0)
    uas = {tuple(ta): ia for ia, ta in enumerate(unique_a)}
    ubs = {tuple(tb): ib for ib, tb in enumerate(unique_b)}
    #
    meta = []
    for k in uas.keys() & ubs.keys():
        inda = np.argwhere(inv_a == uas[k]).ravel()
        indb = np.argwhere(inv_b == ubs[k]).ravel()
        t_a = bl_a.t[inda, :-1, :]
        t_b = bl_b.t[indb, 1:, :]
        slD_a = slDoc_a[inda, :]
        slD_b = slDoc_b[indb, :]
        #
        indices = np.indices([len(t_a), len(t_b)]).reshape(2, -1).T
        comb_t = np.empty((len(indices), len(struct_a.legs) + len(struct_b.legs) - 2, sym.NSYM), dtype=np.int64)
        comb_t[:, :len(struct_a.legs)-1, :] = t_a[indices[:, 0], :, :]
        comb_t[:, len(struct_a.legs)-1:, :] = t_b[indices[:, 1], :, :]
        comb_slD = np.empty((len(indices), 2, 4), dtype=np.int64)
        comb_slD[:, 0, :] = slD_a[indices[:, 0], :]
        comb_slD[:, 1, :] = slD_b[indices[:, 1], :]
        #
        ic = 0
        for tt, slD in zip(comb_t, comb_slD):
            while not np.array_equal(tt, bl_c.t[ic]):
                ic += 1
            meta.append((bl_c.slc[ic], slD[:, 2], slD[0, :2], (slD[0, 2], slD[0, 3]), slD[1, :2], (slD[1, 3], slD[1, 2])))

    meta = np.array(meta, dtype=np.int64).reshape(len(meta), 12)
    meta_dt = np.dtype([
        ('slc', np.int64, (2,)),
        ('Dc',  np.int64, (2,)),
        ('sla', np.int64, (2,)),
        ('Da',  np.int64, (2,)),
        ('slb', np.int64, (2,)),
        ('Db',  np.int64, (2,))])
    meta = meta.view(meta_dt).reshape(-1)
    return meta, bl_c.size, bl_c.struct


@lru_cache(maxsize=1024)
def _meta_tensordot_nf(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b):
    #
    nsym = sym.NSYM
    bl_a, slc_a, bl_b, slc_b, bl_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)

    tao = bl_a.t[:, nout_a, :]      # narrowed to contracted modes, and serialize <num-of-ingoing-modes(=legs)> x <order-of-sym-group> to 1D
    tac = bl_a.t[:, nin_a, :].reshape(bl_a.nblocks, len(nin_a) * nsym)         # narrowed to outgoing modes, and serialize <num-of-ingoing-modes(=legs)> x <order-of-sym-group> to 1D
    Dao = bl_a.D[:, nout_a]  # narrow sector sizes
    Dac = bl_a.D[:, nin_a]
    Daop = np.prod(Dao, axis=1, dtype=np.int64) # total size of outgoing sectors (per block)
    Dacp = np.prod(Dac, axis=1, dtype=np.int64) # total size of contracted sectors (per block)
    #
    unique_tac, inv_tac, count_tac = np.unique(tac, return_inverse=True, return_counts=True, axis=0) # if more blocks of a contribute to given contracted sector (in b)
    # arg_tac = np.argsort(inv_tac)

    tbo = bl_b.t[:, nout_b, :]
    tbc = bl_b.t[:, nin_b, :].reshape(bl_b.nblocks, len(nin_b) * nsym)
    Dbo = bl_b.D[:, nout_b]
    Dbc = bl_b.D[:, nin_b]
    Dbop = np.prod(Dbo, axis=1, dtype=np.int64)
    Dbcp = np.prod(Dbc, axis=1, dtype=np.int64)
    #
    unique_tbc, inv_tbc, count_tbc = np.unique(tbc, return_inverse=True, return_counts=True, axis=0)
    # arg_tbc = np.argsort(inv_tbc)
    #
    reshape_a = tuple(zip(slc_a, bl_a.D, Daop, Dacp)) # narrowed to contracted blocks
    reshape_b = tuple(zip(slc_b, bl_b.D, Dbcp, Dbop))
    #
    uas = {tuple(ta): ia for ia, ta in enumerate(unique_tac)}
    ubs = {tuple(tb): ib for ib, tb in enumerate(unique_tbc)}
    #
    D1 = np.prod(bl_c.D[:, :len(nout_a)], axis=1, dtype=np.int64)
    D2 = np.prod(bl_c.D[:, len(nout_a):], axis=1, dtype=np.int64)
    meta_dot = {tuple(slc): ((d1, d2), []) for slc, d1, d2 in zip(bl_c.slc, D1, D2)}

    for k in uas.keys() & ubs.keys():
        inda = np.argwhere(inv_tac == uas[k]).ravel()
        indb = np.argwhere(inv_tbc == ubs[k]).ravel()
        t_a = tao[inda, :]
        t_b = tbo[indb, :]
        arg_ta = argsort_t(t_a)
        t_a = t_a[arg_ta]
        inda = inda[arg_ta]
        arg_tb = argsort_t(t_b)
        t_b = t_b[arg_tb]
        indb = indb[arg_tb]
        #
        indices = np.indices([len(t_a), len(t_b)]).reshape(2, -1).T
        comb_t = np.empty((len(indices), len(nout_a) + len(nout_b), sym.NSYM), dtype=np.int64)
        comb_t[:, :len(nout_a), :] = t_a[indices[:, 0], :, :]
        comb_t[:, len(nout_a):, :] = t_b[indices[:, 1], :, :]
        comb_ii = np.empty((len(indices), 2), dtype=np.int64)
        comb_ii[:, 0] = inda[indices[:, 0]]
        comb_ii[:, 1] = indb[indices[:, 1]]
        #
        ics = find_matching_indices(bl_c.t, comb_t, both=False)
        for ic, ii in zip(ics, comb_ii):
            meta_dot[tuple(bl_c.slc[ic])][1].append(ii)

    meta = []
    for slc, (dds, gr) in meta_dot.items():
        for ta, tb in gr:
            meta.append((*slc, *dds, ta, tb))

    meta = np.array(meta, dtype=np.int64).reshape(len(meta), 6)
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('Dn',  np.int64, (2,)),
        ('ta', np.int64),
        ('tb', np.int64)])
    meta = meta.view(meta_dt).reshape(-1)
    return meta, reshape_a, reshape_b, bl_c.size, bl_c.struct


# @lru_cache(maxsize=1024)
# def _meta_tensordot_nf(sym, struct_a, slices_a, legs_a, n_a, isdiag_a,
#                             struct_b, slices_b, legs_b, n_b, isdiag_b,
#                             ind_a, ind_b, nout_a, nin_a, nin_b, nout_b):

#     nsym = len(struct_a.n)

#     ta = struct_a.t if ind_a is None else [struct_a.t[ii] for ii in ind_a]
#     Da = struct_a.D if ind_a is None else [struct_a.D[ii] for ii in ind_a]
#     slices_a = [sl.slcs[0] for sl in slices_a] if ind_a is None else [slices_a[ii].slcs[0] for ii in ind_a] # narrow struct and slice information to relevant blocks
#                                                                                                             # i.e. ones, which are contracted with existing (non-zero) blocks in b

#     lta, ndima = len(ta), len(struct_a.s)
#     ata = np.array(ta, dtype=np.int64).reshape((lta, ndima, nsym)) # array for block charges as: block x <num-of-(native)modes(=legs)> x <order-of-sym-group>
#     aDa = np.array(Da, dtype=np.int64).reshape((lta, ndima))       # array for block dimensions
#     tao = ata[:, nout_a, :].reshape(lta, len(nout_a) * nsym)       # narrowed to contracted modes, and serialize <num-of-ingoing-modes(=legs)> x <order-of-sym-group> to 1D
#     tac = ata[:, nin_a, :].reshape(lta, len(nin_a) * nsym)         # narrowed to outgoing modes, and serialize <num-of-ingoing-modes(=legs)> x <order-of-sym-group> to 1D
#     Dao = aDa[:, nout_a]  # narrow sector sizes
#     Dac = aDa[:, nin_a]
#     Daop = np.prod(Dao, axis=1, dtype=np.int64) # total size of outgoing sectors (per block)
#     Dacp = np.prod(Dac, axis=1, dtype=np.int64) # total size of contracted sectors (per block)

#     legs_a = []
#     for i in range(ndima):
#         tai = np.hstack([ata[:, i, :], aDa[:, (i,)]])
#         unique_tai = sorted(np.unique(tai, axis=0).tolist())
#         ti = tuple(tuple(x[:-1]) for x in unique_tai)
#         Di = tuple(x[-1] for x in unique_tai)
#         legs_a.append((ti, Di))

#     tDac = np.hstack([tac, Dac])
#     unique_tDac, inv_tDac, count_tDac = np.unique(tDac, return_inverse=True, return_counts=True, axis=0) # if more blocks of a contribute to given contracted sector (in b)
#     arg_tDac = np.argsort(inv_tDac)

#     tb = struct_b.t if ind_b is None else [struct_b.t[ii] for ii in ind_b]
#     Db = struct_b.D if ind_b is None else [struct_b.D[ii] for ii in ind_b]
#     slices_b = [sl.slcs[0] for sl in slices_b] if ind_b is None else [slices_b[ii].slcs[0] for ii in ind_b]

#     ltb, ndimb = len(tb), len(struct_b.s)
#     atb = np.array(tb, dtype=np.int64).reshape((ltb, ndimb, nsym))
#     aDb = np.array(Db, dtype=np.int64).reshape((ltb, ndimb))
#     tbo = atb[:, nout_b, :].reshape(ltb, len(nout_b) * nsym)
#     tbc = atb[:, nin_b, :].reshape(ltb, len(nin_b) * nsym)
#     Dbo = aDb[:, nout_b]
#     Dbc = aDb[:, nin_b]
#     Dbop = np.prod(Dbo, axis=1, dtype=np.int64)
#     Dbcp = np.prod(Dbc, axis=1, dtype=np.int64)

#     legs_b = []
#     for i in range(ndimb):
#         tbi = np.hstack([atb[:, i, :], aDb[:, (i,)]])
#         unique_tbi = sorted(np.unique(tbi, axis=0).tolist())
#         ti = tuple(tuple(x[:-1]) for x in unique_tbi)
#         Di = tuple(x[-1] for x in unique_tbi)
#         legs_b.append((ti, Di))

#     tDbc = np.hstack([tbc, Dbc])
#     unique_tDbc, inv_tDbc, count_tDbc = np.unique(tDbc, return_inverse=True, return_counts=True, axis=0)
#     arg_tDbc = np.argsort(inv_tDbc)

#     # if not np.array_equal(unique_tDac, unique_tDbc):
#     #     raise YastnError('Bond dimensions of some charges do not match.')

#     # blocks are enumerated consistent with slices_a,b
#     reshape_a = tuple(zip(slices_a, Da, Daop, Dacp)) # narrowed to contracted blocks
#     reshape_b = tuple(zip(slices_b, Db, Dbcp, Dbop))

#     count_ab = count_tDac * count_tDbc
#     sum_count_ab = sum(count_ab)
#     ind_a = np.zeros(sum_count_ab, dtype=np.int64)
#     ind_b = np.zeros(sum_count_ab, dtype=np.int64)
#     start_a, start_b, start_ab = 0, 0, 0
#     for da, db, dab in zip(count_tDac, count_tDbc, count_ab):
#         stop_a, stop_b, stop_ab = start_a + da, start_b + db, start_ab + dab
#         ind_a[start_ab: stop_ab].reshape(da, db)[:, :] = arg_tDac[start_a: stop_a].reshape(da, 1)
#         ind_b[start_ab: stop_ab].reshape(da, db)[:, :] = arg_tDbc[start_b: stop_b].reshape(1, db)
#         start_a, start_b, start_ab = stop_a, stop_b, stop_ab

#     tc = np.hstack([tao[ind_a], tbo[ind_b]])
#     utc, uind, invs, cnts = np.unique(tc, return_index=True, return_inverse=True, return_counts=True, axis=0)

#     uind_a, uind_b = ind_a[uind], ind_b[uind]
#     uDc = np.hstack([Dao[uind_a], Dbo[uind_b]])
#     uDcp2 = np.column_stack([Daop[uind_a], Dbop[uind_b]])

#     c_Dp = np.prod(uDcp2, axis=1, dtype=np.int64).tolist()
#     c_t = tuple(map(tuple, utc.tolist()))
#     c_D = tuple(map(tuple, uDc.tolist()))
#     c_Dp2 = tuple(map(tuple, uDcp2.tolist()))

#     acc_Dp = tuple(accumulate(c_Dp, initial=0))
#     slc_c = tuple(zip(acc_Dp, acc_Dp[1:]))
#     slices_c = tuple(_slc((sl,), ds, dp) for sl, dp, ds in zip(slc_c, c_Dp, c_D))

#     ind_ab = np.column_stack([ind_a, ind_b])
#     arg_invs = np.argsort(invs)
#     ind_ab = ind_ab[arg_invs].tolist()
#     # ind_ab: indices of blocks in a and b to multiply; consistent with enumeration of reshape_a,b
#     acc_cnts = tuple(accumulate(cnts, initial=0))
#     groups_tab = (ind_ab[i: f] for i, f in zip(acc_cnts, acc_cnts[1:]))
#     meta_dot = list(zip(slc_c, c_Dp2, groups_tab))

#     s_c = tuple(struct_a.s[i] for i in nout_a) + tuple(struct_b.s[i] for i in nout_b)
#     struct_c = _struct(s=s_c, t=c_t, D=c_D, size=acc_Dp[-1], n=n_c)

#     legs_c2 = legs_from_struct(struct_c)
#     assert legs_c2 == legs_c
#     return meta_dot, reshape_a, reshape_b, struct_c, slices_c, legs_a, legs_b


@lru_cache(maxsize=1024)
def _meta_tensordot_cutensor(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b):
    bl_a, slc_a, bl_b, slc_b, bl_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    _, a_coords = _int_block_coordinates(sym, bl_a.struct)
    _, b_coords = _int_block_coordinates(sym, bl_b.struct)
    _, c_coords = _int_block_coordinates(sym, bl_c.struct)

    a_numSectionsPerMode, a_sectionExtents = _get_extends(bl_a.struct.legs)
    b_numSectionsPerMode, b_sectionExtents = _get_extends(bl_b.struct.legs)
    c_numSectionsPerMode, c_sectionExtents = _get_extends(bl_c.struct.legs)

    a_strides = _get_strides(bl_a.D).reshape(-1).tolist()
    b_strides = _get_strides(bl_b.D).reshape(-1).tolist()
    c_strides = _get_strides(bl_c.D).reshape(-1).tolist()

    a_offsets = slc_a[:, 0].tolist()
    b_offsets = slc_b[:, 0].tolist()
    c_offsets = bl_c.slc[:, 0].tolist()

    return (bl_c.struct, bl_c.size,
            a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets,
            b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets,
            c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets)


def _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b):
    assert not struct_a.isdiag, "Sanity check. Contact developers."
    assert not struct_b.isdiag, "Sanity check. Contact developers."
    n_c = sym.add_charges(struct_a.n, struct_b.n)
    bl_a = bl_a_full = get_blocks(sym, struct_a)
    bl_b = bl_b_full = get_blocks(sym, struct_b)
    #
    legs_a_new, legs_b_new = list(bl_a.struct.legs), list(bl_b.struct.legs)
    while True:
        for ia, ib in zip(nin_a, nin_b):
            try:
                leg = bl_a.struct.legs[ia].intersection(bl_b.struct.legs[ib].conj())
            except ValueError:
                raise YastnError('Bond dimensions of some charges do not match.')
            legs_a_new[ia] = leg
            legs_b_new[ib] = leg.conj()

        struct_a_new = struct_a._replace(legs=tuple(legs_a_new))
        struct_b_new = struct_b._replace(legs=tuple(legs_b_new))
        bl_a = get_blocks(sym, struct_a_new)
        bl_b = get_blocks(sym, struct_b_new)

        legs_c = (*(struct_a_new.legs[ax] for ax in nout_a), *(struct_b_new.legs[ax] for ax in nout_b))
        struct_c = _struct(legs=legs_c, n=n_c, isdiag=False)
        bl_c = get_blocks(sym, struct_c)

        if bl_a.struct == struct_a_new and bl_b.struct == struct_b_new and bl_c.struct == struct_c:
            break

        for ii, ax in enumerate(nout_a):
            legs_a_new[ax] = bl_c.struct.legs[ii].intersection(bl_a.struct.legs[ax])
        for ii, ax in enumerate(nout_b, start=len(nout_a)):
            legs_b_new[ax] = bl_c.struct.legs[ii].intersection(bl_b.struct.legs[ax])

    slc_a = bl_a_full.slc if struct_a_new == struct_a else \
            bl_a_full.slc[find_matching_indices(bl_a_full.t, bl_a.t, both=False)]

    slc_b = bl_b_full.slc if struct_b_new == struct_b else \
            bl_b_full.slc[find_matching_indices(bl_b_full.t, bl_b.t, both=False)]

    return bl_a, slc_a, bl_b, slc_b, bl_c


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
    bl_a = get_blocks(sym, struct_a)
    bl_b = get_blocks(sym, struct_b)

    leg_b = struct_b.legs[axis]
    leg_a = struct_a.legs[0] if struct_a.legs[0].s == leg_b.s else struct_a.legs[0].conj()  # .conj() to match signature
    try:
        leg_c = leg_b.intersection(leg_a)
    except ValueError:
        raise YastnError("Bond dimensions of some charges do not match.")
    legs_c = struct_b.legs[:axis] + (leg_c,) + struct_b.legs[axis + 1:]
    bl_c = get_blocks(sym, struct_b._replace(legs=legs_c))
    bl_aa = get_blocks(sym, struct_a._replace(legs=(leg_c, leg_c.conj())))
    #
    inds_c = find_matching_indices(bl_b.t, bl_c.t, both=False)
    inds_a = find_matching_indices(bl_a.t, bl_aa.t, both=False)
    slc_aa = bl_a.slc[inds_a]
    #
    if struct_b.isdiag:
        D_b = bl_b.D[:, 0]
        axis = 0
        ndimb = 1
    else:
        D_b = bl_b.D
        ndimb = len(struct_b.legs)
    #
    unique_tax, inv_tax = np.unique(bl_c.t[:, axis, :], return_inverse=True, axis=0)
    assert np.array_equal(bl_aa.t[:, 0, :], unique_tax), "Sanity check. Contact developers."
    #
    meta = np.column_stack([bl_c.slc, bl_b.slc[inds_c], D_b[inds_c], slc_aa[inv_tax]])
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('slb',  np.int64, (2,)),
        ('Db', np.int64, (ndimb,)),
        ('sla',  np.int64, (2,))])
    meta = meta.view(meta_dt).reshape(-1)
    return meta, bl_c.size, bl_c.struct, axis, ndimb


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
    legs_part = list(bl.struct.legs)
    while True:
        for ia, ib in zip(nin_0, nin_1):
            try:
                leg = bl.struct.legs[ia].intersection(bl.struct.legs[ib].conj())
            except ValueError:
                raise YastnError('Bond dimensions of some charges do not match.')
            legs_part[ia] = leg
            legs_part[ib] = leg.conj()

        struct_part = struct._replace(legs=tuple(legs_part))
        bl = get_blocks(sym, struct_part)

        struct_c = struct._replace(legs=tuple(bl.struct.legs[ax] for ax in out))
        bl_c = get_blocks(sym, struct_c)

        if bl.struct == struct_part and bl_c.struct == struct_c:
            break

        for ii, ax in enumerate(out):
            legs_part[ax] = bl_c.struct.legs[ii].intersection(bl.struct.legs[ax])

    slo = bl_full.slc if struct_part == struct else \
          bl_full.slc[find_matching_indices(bl_full.t, bl.t, both=False)]

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
    return meta, bl_c.size, struct_c


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
