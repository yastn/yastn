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
from itertools import groupby, accumulate, product
from numbers import Number
from operator import itemgetter

import numpy as np

from ._auxliary import _struct, _slc, _clear_axes, _unpack_axes, _flatten, _join_contiguous_slices
from ._merging import _merge_to_matrix, _unmerge, _meta_unmerge_matrix, _meta_fuse_hard
from ._merging import _transpose_and_merge, _mask_tensors_leg_intersection, _meta_mask
from ._tests import YastnError, _test_can_be_combined, _test_axes_match

__all__ = ['tensordot', 'vdot', 'trace', 'swap_gate', 'ncon', 'einsum', 'broadcast', 'apply_mask', 'SpecialTensor']


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
    mask_needed, (nin_a, nin_b) = _test_axes_match(a, b, sgn=-1, axes=(in_a, in_b))

    if a.isdiag:
        return _tensordot_diag(a, b, in_b, destination=(0,))
    if b.isdiag:
        return _tensordot_diag(b, a, in_a, destination=(-1,))

    _test_can_be_combined(a, b)
    nout_a = tuple(ii for ii in range(a.ndim_n) if ii not in nin_a)  # outgoing native legs
    nout_b = tuple(ii for ii in range(b.ndim_n) if ii not in nin_b)  # outgoing native legs

    n_c = a.config.sym.add_charges(a.struct.n, b.struct.n)
    s_c = tuple(a.struct.s[i1] for i1 in nout_a) + tuple(b.struct.s[i2] for i2 in nout_b)
    mfs_c = tuple(a.mfs[ii] for ii in range(a.ndim) if ii not in in_a)
    mfs_c += tuple(b.mfs[ii] for ii in range(b.ndim) if ii not in in_b)
    hfs_c = tuple(a.hfs[ii] for ii in nout_a) + tuple(b.hfs[ii] for ii in nout_b)

    if mask_needed:
        msk_a, msk_b, a_hfs, b_hfs = _mask_tensors_leg_intersection(a, b, nin_a, nin_b)
        a = _apply_mask_axes(a, nin_a, msk_a)
        b = _apply_mask_axes(b, nin_b, msk_b)
        a = a._replace(hfs=a_hfs)
        b = b._replace(hfs=b_hfs)

    if a.config.tensordot_policy == 'fuse_to_matrix':
        data, struct_c, slices_c = _tensordot_f2m(a, b, nout_a, nin_a, nin_b, nout_b, s_c)
    elif a.config.tensordot_policy == 'fuse_contracted':
        data, struct_c, slices_c = _tensordot_fc(a, b, nout_a, nin_a, nin_b, nout_b)
    elif a.config.tensordot_policy == 'no_fusion':
        data, struct_c, slices_c = _tensordot_nf(a, b, nout_a, nin_a, nin_b, nout_b)
    else:
        raise YastnError("Tensordot policy not recognized. It should be 'fuse_to_matrix', 'fuse_contracted', or 'no_fusion'.")

    struct_c = struct_c._replace(n=n_c)
    return a._replace(data=data, struct=struct_c, slices=slices_c, mfs=mfs_c, hfs=hfs_c)


def _tensordot_diag(a, b, in_b, destination):
    r""" Executes broadcast and then transpose into order expected by tensordot. """
    if len(in_b) == 1:
        c = a.broadcast(b, axes=in_b[0])
        return c.moveaxis(source=in_b, destination=destination)
    if len(in_b) == 2:
        c = a.broadcast(b, axes=in_b[0])
        return c.trace(axes=in_b)
    raise YastnError('Outer product with diagonal tensor not supported. Use yastn.diag() first.')  # if len(in_a) == 0


def _tensordot_f2m(a, b, nout_a, nin_a, nin_b, nout_b, s_c):
    r"""
    Perform tensordot by fuse_to_matrix:
    merging tensors to matrices, executing dot, and unmerging outgoing legs.
    """
    ind_a, ind_b = _common_inds(a.struct.t, b.struct.t, nin_a, nin_b, a.ndim_n, b.ndim_n, a.config.sym.NSYM)
    data_a, struct_a, slices_a, ls_l, ls_ac = _merge_to_matrix(a, (nout_a, nin_a), ind_a)
    data_b, struct_b, slices_b, ls_bc, ls_r = _merge_to_matrix(b, (nin_b, nout_b), ind_b)

    if ls_ac != ls_bc:
        raise YastnError('Bond dimensions do not match.')

    meta_dot, struct_c, slices_c = _meta_tensordot_f2m(struct_a, slices_a, struct_b, slices_b)
    data = a.config.backend.dot(data_a, data_b, meta_dot, struct_c.size)

    meta_unmerge, struct_c, slices_c = _meta_unmerge_matrix(a.config, struct_c, slices_c, ls_l, ls_r, s_c)
    data = _unmerge(a.config, data, meta_unmerge)
    return data, struct_c, slices_c


def _tensordot_fc(a, b, nout_a, nin_a, nin_b, nout_b):
    r"""
    Perform tensordot by fuse_contracted: merging contracted legs, and executing dot.
    Outgoing legs are not merged so unmerge is not needed.
    """
    ind_a, ind_b = _common_inds(a.struct.t, b.struct.t, nin_a, nin_b, a.ndim_n, b.ndim_n, a.config.sym.NSYM)

    axes_a = tuple((x,) for x in nout_a) + (nin_a,)
    order_a = nout_a + nin_a
    struct_a, slices_a, meta_mrg_a, t_a, D_a = _meta_fuse_hard(a.config, a.struct, a.slices, axes_a, ind_a)
    data_a = _transpose_and_merge(a.config, a._data, order_a, struct_a, slices_a, meta_mrg_a)

    axes_b = (nin_b,) + tuple((x,) for x in nout_b)
    order_b = nin_b + nout_b
    struct_b, slices_b, meta_mrg_b, t_b, D_b = _meta_fuse_hard(b.config, b.struct, b.slices, axes_b, ind_b)
    data_b = _transpose_and_merge(b.config, b._data, order_b, struct_b, slices_b, meta_mrg_b)

    if not all(D_a[ia] == D_b[ib] for ia, ib in zip(nin_a, nin_b)):
        raise YastnError('Bond dimensions do not match.')
    assert all(t_a[ia] == t_b[ib] for ia, ib in zip(nin_a, nin_b)), "Sanity check."

    meta_dot, struct_c, slices_c = _meta_tensordot_fc(struct_a, slices_a, struct_b, slices_b)
    data = a.config.backend.dot(data_a, data_b, meta_dot, struct_c.size)
    return data, struct_c, slices_c


def _tensordot_nf(a, b, nout_a, nin_a, nin_b, nout_b):
    r"""
    Perform tensordot directly: permute blocks and execute dot accumulaing results into result blocks.
    """
    ind_a, ind_b = _common_inds(a.struct.t, b.struct.t, nin_a, nin_b, a.ndim_n, b.ndim_n, a.config.sym.NSYM)
    meta_dot, reshape_a, reshape_b, struct_c, slices_c = _meta_tensordot_nf(a.struct, a.slices, b.struct, b.slices,
                                                                            ind_a, ind_b, nout_a, nin_a, nin_b, nout_b)
    order_a = nout_a + nin_a
    order_b = nin_b + nout_b
    data = a.config.backend.transpose_dot_sum(a.data, b.data, meta_dot,
                                              reshape_a, reshape_b, order_a, order_b, struct_c.size)
    return data, struct_c, slices_c


@lru_cache(maxsize=1024)
def _common_inds(t_a, t_b, nin_a, nin_b, ndimn_a, ndimn_b, nsym):
    r""" Return row indices of nparray ``a`` that are in ``b``, and vice versa. Outputs tuples."""
    t_a = np.array(t_a, dtype=np.int64).reshape((len(t_a), ndimn_a, nsym))
    t_b = np.array(t_b, dtype=np.int64).reshape((len(t_b), ndimn_b, nsym))
    t_a = t_a[:, nin_a, :].reshape(len(t_a), len(nin_a) * nsym).tolist()
    t_b = t_b[:, nin_b, :].reshape(len(t_b), len(nin_b) * nsym).tolist()
    la = [tuple(x) for x in t_a]
    lb = [tuple(x) for x in t_b]
    sa = set(la)
    sb = set(lb)
    ia = tuple(ii for ii, el in enumerate(la) if el in sb)
    ib = tuple(ii for ii, el in enumerate(lb) if el in sa)
    if len(ia) == len(la):
        ia = None
    if len(ib) == len(lb):
        ib = None
    return ia, ib


@lru_cache(maxsize=1024)
def _meta_tensordot_f2m(struct_a, slices_a, struct_b, slices_b):
    nsym = len(struct_a.n)
    struct_a_resorted = sorted(((t[nsym:], t, D, sl.slcs[0]) for t, D, sl in zip(struct_a.t, struct_a.D, slices_a)))
    struct_b_resorted = ((t[:nsym], t, D, sl.slcs[0]) for t, D, sl in zip(struct_b.t, struct_b.D, slices_b))
    meta = []
    for (tar, ta, Da, sla), (tbl, tb, Db, slb) in zip(struct_a_resorted, struct_b_resorted):
        assert tar == tbl, "Sanity check."
        meta.append((ta[:nsym] + tb[nsym:], (Da[0], Db[1]), sla, Da, slb, Db))
    meta = sorted(meta)
    t_c = tuple(x[0] for x in meta)
    D_c = tuple(x[1] for x in meta)
    Dp_c = tuple(D[0] * D[1] for D in D_c)
    slices_c = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(Dp_c), Dp_c, D_c))
    meta_dot = tuple((sl.slcs[0], *mt[1:]) for sl, mt in zip(slices_c, meta))
    s_c = (struct_a.s[0], struct_b.s[1])
    struct_c = _struct(s=s_c, t=t_c, D=D_c, size=sum(Dp_c))
    return meta_dot, struct_c, slices_c


@lru_cache(maxsize=1024)
def _meta_tensordot_fc(struct_a, slices_a, struct_b, slices_b):
    nsym = len(struct_a.n)
    lta, ndima = len(struct_a.t), len(struct_a.s)
    ta = np.array(struct_a.t, dtype=np.int64).reshape((lta, ndima, nsym))
    Da = np.array(struct_a.D, dtype=np.int64).reshape((lta, ndima))
    tao = ta[:, :-1, :].reshape(lta, (ndima - 1) * nsym).tolist()
    tac = ta[:, -1, :].tolist()
    Dao = Da[:, :-1]
    Daop = np.prod(Dao, axis=1, dtype=np.int64).tolist()
    Dao = Dao.tolist()
    Dac = Da[:, -1].tolist()
    struct_a_resorted = sorted(((tuple(tc), tuple(to), Dc, Dop, tuple(Do), sl.slcs[0])
                                for tc, to, Dc, Dop, Do, sl in zip(tac, tao, Dac, Daop, Dao, slices_a)))

    ltb, ndimb = len(struct_b.t), len(struct_b.s)
    tb = np.array(struct_b.t, dtype=np.int64).reshape((ltb, ndimb, nsym))
    Db = np.array(struct_b.D, dtype=np.int64).reshape((ltb, ndimb))

    tbo = tb[:, 1:, :].reshape(ltb, (ndimb - 1) * nsym).tolist()
    tbc = tb[:, 0, :].tolist()
    Dbo = Db[:, 1:]
    Dbop = np.prod(Dbo, axis=1, dtype=np.int64).tolist()
    Dbo = Dbo.tolist()
    Dbc = Db[:, 0].tolist()
    struct_b_resorted = [(tuple(tc), tuple(to), Dc, Dop, tuple(Do), sl.slcs[0])
                         for tc, to, Dc, Dop, Do, sl in zip(tbc, tbo, Dbc, Dbop, Dbo, slices_b)]

    struct_a_resorted = groupby(struct_a_resorted, key=itemgetter(0))
    struct_b_resorted = groupby(struct_b_resorted, key=itemgetter(0))

    meta = []
    for (tar, group_ta), (tbl, group_tb) in zip(struct_a_resorted, struct_b_resorted):
        assert tar == tbl, "Sanity check."
        for (_, toa, Dca, Dopa, Doa, sla), (_, tob, Dcb, Dopb, Dob, slb) in product(group_ta, group_tb):
            meta.append((toa + tob, Doa + Dob, Dopa * Dopb, (Dopa, Dopb), sla, (Dopa, Dca), slb, (Dcb, Dopb)))

    meta = sorted(meta)
    t_c = tuple(x[0] for x in meta)
    D_c = tuple(x[1] for x in meta)
    Dp_c = tuple(x[2] for x in meta)
    slices_c = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(Dp_c), Dp_c, D_c))
    meta_dot = tuple((sl.slcs[0], *mt[3:]) for sl, mt in zip(slices_c, meta))
    s_c = struct_a.s[:-1] + struct_b.s[1:]
    struct_c = _struct(s=s_c, t=t_c, D=D_c, size=sum(Dp_c))
    return meta_dot, struct_c, slices_c


@lru_cache(maxsize=1024)
def _meta_tensordot_nf(struct_a, slices_a, struct_b, slices_b, ind_a, ind_b, nout_a, nin_a, nin_b, nout_b):
    nsym = len(struct_a.n)

    ta = struct_a.t if ind_a is None else [struct_a.t[ii] for ii in ind_a]
    Da = struct_a.D if ind_a is None else [struct_a.D[ii] for ii in ind_a]
    slices_a = [sl.slcs[0] for sl in slices_a] if ind_a is None else [slices_a[ii].slcs[0] for ii in ind_a]

    lta, ndima = len(ta), len(struct_a.s)
    ata = np.array(ta, dtype=np.int64).reshape((lta, ndima, nsym))
    aDa = np.array(Da, dtype=np.int64).reshape((lta, ndima))
    tao = ata[:, nout_a, :].reshape(lta, len(nout_a) * nsym)
    tac = ata[:, nin_a, :].reshape(lta, len(nin_a) * nsym)
    Dao = aDa[:, nout_a]
    Dac = aDa[:, nin_a]
    Daop = np.prod(Dao, axis=1, dtype=np.int64)
    Dacp = np.prod(Dac, axis=1, dtype=np.int64)

    tDac = np.hstack([tac, Dac])
    unique_tDac, inv_tDac, count_tDac = np.unique(tDac, return_inverse=True, return_counts=True, axis=0)
    arg_tDac = np.argsort(inv_tDac)

    tb = struct_b.t if ind_b is None else [struct_b.t[ii] for ii in ind_b]
    Db = struct_b.D if ind_b is None else [struct_b.D[ii] for ii in ind_b]
    slices_b = [sl.slcs[0] for sl in slices_b] if ind_b is None else [slices_b[ii].slcs[0] for ii in ind_b]

    ltb, ndimb = len(tb), len(struct_b.s)
    atb = np.array(tb, dtype=np.int64).reshape((ltb, ndimb, nsym))
    aDb = np.array(Db, dtype=np.int64).reshape((ltb, ndimb))
    tbo = atb[:, nout_b, :].reshape(ltb, len(nout_b) * nsym)
    tbc = atb[:, nin_b, :].reshape(ltb, len(nin_b) * nsym)
    Dbo = aDb[:, nout_b]
    Dbc = aDb[:, nin_b]
    Dbop = np.prod(Dbo, axis=1, dtype=np.int64)
    Dbcp = np.prod(Dbc, axis=1, dtype=np.int64)

    tDbc = np.hstack([tbc, Dbc])
    unique_tDbc, inv_tDbc, count_tDbc = np.unique(tDbc, return_inverse=True, return_counts=True, axis=0)
    arg_tDbc = np.argsort(inv_tDbc)

    if not np.array_equal(unique_tDac, unique_tDbc):
        raise YastnError('Bond dimensions do not match.')

    # blocks are enumerated consistent with slices_a,b
    reshape_a = tuple(zip(slices_a, Da, Daop, Dacp))
    reshape_b = tuple(zip(slices_b, Db, Dbcp, Dbop))

    count_ab = count_tDac * count_tDbc
    sum_count_ab = sum(count_ab)
    ind_a = np.zeros(sum_count_ab, dtype=np.int64)
    ind_b = np.zeros(sum_count_ab, dtype=np.int64)
    start_a, start_b, start_ab = 0, 0, 0
    for da, db, dab in zip(count_tDac, count_tDbc, count_ab):
        stop_a, stop_b, stop_ab = start_a + da, start_b + db, start_ab + dab
        ind_a[start_ab: stop_ab].reshape(da, db)[:, :] = arg_tDac[start_a: stop_a].reshape(da, 1)
        ind_b[start_ab: stop_ab].reshape(da, db)[:, :] = arg_tDbc[start_b: stop_b].reshape(1, db)
        start_a, start_b, start_ab = stop_a, stop_b, stop_ab

    tc = np.hstack([tao[ind_a], tbo[ind_b]])
    utc, uind, invs, cnts = np.unique(tc, return_index=True, return_inverse=True, return_counts=True, axis=0)

    uind_a, uind_b = ind_a[uind], ind_b[uind]
    uDc = np.hstack([Dao[uind_a], Dbo[uind_b]])
    uDcp2 = np.column_stack([Daop[uind_a], Dbop[uind_b]])

    c_Dp = np.prod(uDcp2, axis=1, dtype=np.int64).tolist()
    c_t = tuple(map(tuple, utc.tolist()))
    c_D = tuple(map(tuple, uDc.tolist()))
    c_Dp2 = tuple(map(tuple, uDcp2.tolist()))

    acc_Dp = tuple(accumulate(c_Dp, initial=0))
    slc_c = tuple(zip(acc_Dp, acc_Dp[1:]))
    slices_c = tuple(_slc((sl,), ds, dp) for sl, dp, ds in zip(slc_c, c_Dp, c_D))

    ind_ab = np.column_stack([ind_a, ind_b])
    arg_invs = np.argsort(invs)
    ind_ab = ind_ab[arg_invs].tolist()
    # ind_ab: indices of blocks in a and b to multiply; consistent with enumeration of reshape_a,b
    acc_cnts = tuple(accumulate(cnts, initial=0))
    groups_tab = (ind_ab[i: f] for i, f in zip(acc_cnts, acc_cnts[1:]))
    meta_dot = list(zip(slc_c, c_Dp2, groups_tab))

    s_c = tuple(struct_a.s[i] for i in nout_a) + tuple(struct_b.s[i] for i in nout_b)
    struct_c = _struct(s=s_c, t=c_t, D=c_D, size=acc_Dp[-1])
    return meta_dot, reshape_a, reshape_b, struct_c, slices_c


def broadcast(a, *args, axes=0) -> 'Tensor' | tuple['Tensor']:
    r"""
    Compute tensordot product of diagonal tensor ``a`` with tensors in ``args``.

    Produce diagonal tensor if both are diagonal.
    Legs of the resulting tensors are ordered in the same way as those of tensors in ``args``.
    It is used (in combination with :meth:`yastn.transpose`) as a subrutine of
    :meth:`yastn.tensordot` for contractions involving diagonal tensor.

    Parameters
    ----------
    a, args: yastn.Tensor
        ``a`` is diagonal tensor to be broadcasted.

    axes: int | Sequence[int]
        legs of tensors in ``args`` to be multiplied by diagonal tensor ``a``.
        Number of tensors provided in ``args`` should match the length of ``axes``.
    """
    multiple_axes = hasattr(axes, '__iter__')
    axes = (axes,) if not multiple_axes else axes
    if len(axes) != len(args):
        raise YastnError("There should be exactly one axis for each tensor to be projected.")
    results = []
    for b, ax in zip(args, axes):
        _test_can_be_combined(a, b)
        ax = _broadcast_input(ax, b.mfs, a.isdiag)
        if b.hfs[ax].tree != (1,):
            raise YastnError('Second tensor`s leg specified in axes cannot be fused.')

        meta, struct, slices = _meta_broadcast(b.struct, b.slices, a.struct, a.slices, ax)

        if b.isdiag:
            b_ndim, ax = (1, 0)
            meta = tuple((sln, slb, Db[0], sla) for sln, slb, Db, sla in meta)
        else:
            b_ndim = b.ndim_n
        data = b.config.backend.dot_diag(a._data, b._data, meta, struct.size, ax, b_ndim)
        results.append(b._replace(struct=struct, slices=slices, data=data))
    return results if multiple_axes else results.pop()


def _broadcast_input(axis, mf, isdiag):
    if not isdiag:
        raise YastnError('First tensor should be diagonal.')
    axis = axis % len(mf)
    if mf[axis] != (1,):
        raise YastnError('Second tensor`s leg specified by axis cannot be fused.')
    axis = sum(mf[ii][0] for ii in range(axis))  # unpack
    return axis


@lru_cache(maxsize=1024)
def _meta_broadcast(b_struct, b_slices, a_struct, a_slices, axis):
    r""" meta information for backend, and new tensor structure for brodcast. """
    nsym = len(a_struct.n)
    ind_tb = tuple(x[axis * nsym: (axis + 1) * nsym] for x in b_struct.t)
    ind_ta = tuple(x[:nsym] for x in a_struct.t)
    sl_a = dict(zip(ind_ta, a_slices))

    meta = tuple((tb, slb.slcs[0], Db, slb.Dp, sl_a[ib].slcs[0])
                 for tb, slb, Db, ib in zip(b_struct.t, b_slices, b_struct.D, ind_tb) if ib in ind_ta)

    if any(Db[axis] != sla[1] - sla[0] for _, _, Db, _, sla in meta):
        raise YastnError("Bond dimensions do not match.")

    if len(meta) < len(b_struct.t):
        c_t = tuple(mt[0] for mt in meta)
        c_D = tuple(mt[2] for mt in meta)
        c_Dp = tuple(mt[3] for mt in meta)
        c_slices = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(c_Dp), c_Dp, c_D))
        c_struct = b_struct._replace(t=c_t, D=c_D, size=sum(c_Dp))
    else:
        c_struct = b_struct
        c_slices = b_slices

    meta = tuple((sln.slcs[0], slb, Db, sla) for (_, slb, Db, _, sla), sln in zip(meta, c_slices))
    return meta, c_struct, c_slices


def apply_mask(a, *args, axes=0) -> 'Tensor' | tuple['Tensor']:
    r"""
    Apply mask given by nonzero elements of diagonal tensor ``a`` on specified axes of tensors in args.
    Number of tensors in ``args`` is not restricted.
    The length of the list ``axes`` has to be mathing with ``args``.

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
    multiple_axes = hasattr(axes, '__iter__')
    axes = (axes,) if not multiple_axes else axes
    if len(axes) != len(args):
        raise YastnError("There should be exactly one axis for each tensor to be projected.")
    results = []

    nsym = a.config.sym.NSYM
    mask = {t[:nsym]: a.config.backend.to_mask(a._data[slice(*sl.slcs[0])]) for t, sl in zip(a.struct.t, a.slices)}
    mask_t = tuple(mask.keys())
    mask_D = tuple(len(v) for v in mask.values())

    for b, ax in zip(args, axes):
        _test_can_be_combined(a, b)
        ax = _broadcast_input(ax, b.mfs, a.isdiag)
        if b.hfs[ax].tree != (1,):
            raise YastnError('Second tensor`s leg specified by axes cannot be fused.')

        meta, struct, slices, ax, ndim = _meta_mask(b.struct, b.slices, b.isdiag, mask_t, mask_D, ax)
        data = a.config.backend.apply_mask(b._data, mask, meta, struct.size, ax, ndim)
        results.append(b._replace(struct=struct, slices=slices, data=data))
    return results.pop() if len(results) == 1 else results


def _apply_mask_axes(a, naxes, masks):
    r""" Auxlliary function applying mask tensors to native legs. """
    for axis, mask in zip(naxes, masks):
        if mask is not None:
            mask_tD = {k: len(v) for k, v in mask.items() if len(v) > 0}
            mask_t = tuple(mask_tD.keys())
            mask_D = tuple(mask_tD.values())
            meta, struct, slices, axis, ndim = _meta_mask(a.struct, a.slices, a.isdiag, mask_t, mask_D, axis)
            data = a.config.backend.apply_mask(a._data, mask, meta, struct.size, axis, ndim)
            a = a._replace(struct=struct, slices=slices, data=data)
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
    _test_can_be_combined(a, b)
    if conj[0] == 1:
        a = a.conj()
    if conj[1] == 1:
        b = b.conj()

    mask_needed, (nin_a, nin_b) = _test_axes_match(a, b, sgn=-1)

    n_c = a.config.sym.add_charges(a.struct.n, b.struct.n)
    if n_c == a.config.sym.zero():
        if mask_needed:
            msk_a, msk_b, a_hfs, b_hfs = _mask_tensors_leg_intersection(a, b, nin_a, nin_b)
            a = _apply_mask_axes(a, nin_a, msk_a)
            b = _apply_mask_axes(b, nin_b, msk_b)
            a = a._replace(hfs=a_hfs)
            b = b._replace(hfs=b_hfs)
        meta_vdot = _meta_vdot(a.struct, a.slices, b.struct, b.slices)
    else:
        meta_vdot = ()

    return a.config.backend.vdot(a.data, b.data, meta_vdot)


@lru_cache(maxsize=1024)
def _meta_vdot(struct_a, slices_a, struct_b, slices_b):
    ia, ib, slcs_a, slcs_b = 0, 0, [], []
    while ia < len(struct_a.t) and ib < len(struct_b.t):
        if struct_a.t[ia] == struct_b.t[ib]:
            if struct_a.D[ia] != struct_b.D[ib]:
                raise YastnError('Bond dimensions do not match.')
            slcs_a.append(slices_a[ia].slcs[0])
            slcs_b.append(slices_b[ib].slcs[0])
            ia += 1
            ib += 1
        elif struct_a.t[ia] < struct_b.t[ib]:
            ia += 1
        else:
            ib += 1
    meta_vdot = _join_contiguous_slices(slcs_a, slcs_b)
    return meta_vdot


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
    mask_needed, (nin_0, nin_1) = _test_axes_match(a, a, sgn=-1, axes=(in_0, in_1))

    if len(nin_0) == 0:
        return a

    order = nin_0 + nin_1
    out = tuple(i for i in range(a.ndim_n) if i not in order)
    order = order + out
    mfs = tuple(a.mfs[i] for i in range(a.ndim) if i not in in_0 + in_1)
    hfs = tuple(a.hfs[ii] for ii in out)

    if a.isdiag:
        struct = a.struct._replace(s=(), diag=False, t=((),), D=((),), size=1)
        data = a.config.backend.sum_elements(a._data)
        return a._replace(struct=struct, slices=(_slc(((0, 1),), (), 1),), mfs=mfs, hfs=hfs, isdiag=False, data=data)

    if mask_needed:
        msk_0, msk_1, a_hfs, _ = _mask_tensors_leg_intersection(a, a, nin_0, nin_1)
        a = _apply_mask_axes(a, nin_0 + nin_1, msk_0 + msk_1)
        a = a._replace(hfs=a_hfs)

    meta, struct, slices = _meta_trace(a.struct, a.slices, nin_0, nin_1, out)
    data = a.config.backend.trace(a._data, order, meta, struct.size)
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, slices=slices, data=data)


@lru_cache(maxsize=1024)
def _meta_trace(struct, slices, nin_0, nin_1, out):
    r""" meta-information for backend and struct of traced tensor. """
    lt, nsym = len(struct.t), len(struct.n)
    tset = np.array(struct.t, dtype=np.int64).reshape((lt, len(struct.s), nsym))
    Dset = np.array(struct.D, dtype=np.int64).reshape((lt, len(struct.s)))
    t0 = tset[:, nin_0, :].reshape(lt, len(nin_0) * nsym)
    t1 = tset[:, nin_1, :].reshape(lt, len(nin_1) * nsym)
    tn = tset[:, out, :].reshape(lt, len(out) * nsym)
    D0 = Dset[:, nin_0]
    D1 = Dset[:, nin_1]
    Dn = Dset[:, out]
    Dnp = np.prod(Dn, axis=1, dtype=np.int64)
    pD0 = np.prod(D0, axis=1, dtype=np.int64)
    pD1 = np.prod(D1, axis=1, dtype=np.int64)
    Drsh = np.column_stack([pD0, pD1, Dnp])

    ind = (np.all(t0 == t1, axis=1)).nonzero()[0]
    if not np.all(D0[ind] == D1[ind]):
        raise YastnError('Bond dimensions do not match.')
    tn = tuple(map(tuple, tn[ind].tolist()))
    Dn = tuple(map(tuple, Dn[ind].tolist()))
    Dnp = Dnp[ind].tolist()
    slo = tuple(slices[n].slcs[0] for n in ind)
    Do = tuple(struct.D[n] for n in ind)
    Drsh = tuple(map(tuple, Drsh[ind].tolist()))

    pre_meta = sorted(zip(tn, Dn, Dnp, slo, Do, Drsh), key=itemgetter(0))

    start, c_t, c_D, c_slices, meta_trace = 0, [], [], [], []
    for (tn, Dn, Dnp), group in groupby(pre_meta, key=itemgetter(0, 1, 2)):
        c_t.append(tn)
        c_D.append(Dn)
        stop = start + Dnp
        c_slices.append(_slc(((start, stop),), Dn, Dnp))
        meta_trace.append(((start, stop), tuple(mt[3:] for mt in group)))
        start = stop
    c_s = tuple(struct.s[i] for i in out)
    c_struct = _struct(s=c_s, n=struct.n, t=tuple(c_t), D=tuple(c_D), size=start)
    return tuple(meta_trace), c_struct, tuple(c_slices)


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
    For ``fermionic=False``,  swap_gate returns ``a``.

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
        negate_slices = _meta_swap_gate(a.struct.t, a.slices, a.mfs, a.ndim_n, nsym, axes, fss)
    else:
        axes, = _clear_axes(axes)  # swapped groups of legs
        negate_slices = _meta_swap_gate_charge(a.struct.t, a.slices, tuple(charge), a.mfs, a.ndim_n, nsym, axes, fss)

    newdata = a.config.backend.negate_blocks(a._data, negate_slices)
    return a._replace(data=newdata)


@lru_cache(maxsize=1024)
def _meta_swap_gate(tset, slices, mf, ndim, nsym, axes, fss):
    r""" Calculate which blocks to negate. """
    axes = _unpack_axes(mf, *axes)
    lt = len(tset)
    tset = np.array(tset, dtype=np.int64).reshape((lt, ndim, nsym))
    iaxes = iter(axes)
    tp = np.zeros(lt, dtype=np.int64)

    if len(axes) % 2 == 1:
        raise YastnError('Odd number of elements in axes. Elements of axes should come in pairs.')
    for l1, l2 in zip(*(iaxes, iaxes)):
        t1 = np.sum(tset[:, l1, :], axis=1, dtype=np.int64) % 2
        t2 = np.sum(tset[:, l2, :], axis=1, dtype=np.int64) % 2
        tp += np.sum(t1[:, fss] * t2[:, fss], axis=1, dtype=np.int64)
    tp = tp % 2
    return _slices_to_negate(tp, slices)


@lru_cache(maxsize=1024)
def _meta_swap_gate_charge(tset, slices, charge, mf, ndim, nsym, axes, fss):
    r""" Calculate which blocks to negate. """
    if isinstance(charge[0], int):
        charge = (charge,) * len(axes)

    charges = ()
    for t, ax in zip(charge, axes):
        charges += t * mf[ax][0]

    axes, = _unpack_axes(mf, axes)
    tset = np.array(tset, dtype=np.int64).reshape((len(tset), ndim, nsym))
    tp = tset[:, axes, :]

    try:
        charges = np.array(charges, dtype=np.int64).reshape(1, len(axes), nsym) % 2
    except ValueError:
        raise YastnError(f'Length or number of charges does not match sym.NSYM or axes.')

    tp = np.sum(tp[:, :, fss] * charges[:, :, fss], axis=(1, 2), dtype=np.int64) % 2
    return _slices_to_negate(tp, slices)


def _slices_to_negate(tp, slices):
    negate = tuple(slc.slcs[0] for slc, negate in zip(slices, tp) if negate)
    if not negate:
        return negate

    joined_negate = []
    start, stop = negate[0]
    for next_start, next_stop in negate[1:]:
        if stop == next_start:
            stop = next_stop
        else:
            joined_negate.append((start, stop))
            start, stop = next_start, next_stop
    joined_negate.append((start, stop))
    return tuple(joined_negate)


def einsum(subscripts, *operands, order=None) -> 'Tensor':
    r"""
    Execute series of tensor contractions.

    Covering trace, tensordot (including outer products), and transpose.
    Follows notation of :meth:`np.einsum` as close as possible.

    Parameters
    ----------
    subscripts: str

    operands: Sequence[yastn.Tensor]

    order: str
        Specify order in which repeated indices from subscipt are contracted.
        By default it follows alphabetic order.

    Example
    -------

    ::

        yastn.einsum('*ij,jh->ih', t1, t2)

        # matrix-matrix multiplication, where the first matrix is conjugated.
        # Equivalent to

        t1.conj() @ t2

        yastn.einsum('ab,al,bm->lm', t1, t2, t3, order='ba')

        # Contract along `b` first, and `a` second.
    """
    if not isinstance(subscripts, str):
        raise YastnError('The first argument should be a string.')

    subscripts = subscripts.replace(' ', '')

    tmp = subscripts.split('->')
    if len(tmp) == 1:
        sin, sout = tmp[0], ''
    elif len(tmp) == 2:
        sin, sout = tmp
    else:
        raise YastnError('Subscript should have at most one separator ->')

    alphabet1 = 'ABCDEFGHIJKLMNOPQRSTUWXYZabcdefghijklmnopqrstuvwxyz'
    alphabet2 = alphabet1 + ',*'
    if any(v not in alphabet1 for v in sout) or \
       any(v not in alphabet2 for v in sin):
        raise YastnError('Only alphabetic characters can be used to index legs.')

    conjs = [1 if '*' in ss else 0 for ss in sin.split(',')]
    sin = sin.replace('*', '')

    if sout == '':
        for v in sin.replace(',', ''):
            if sin.count(v) == 1:
                sout += v
    elif len(sout) != len(set(sout)):
        raise YastnError('Repeated index after ->')

    if order is None:
        order = []
        for v in sin.replace(',', ''):
            if sin.count(v) > 1:
                order.append(v)
        order = ''.join(sorted(order))
    din = {v: i + 1 for i, v in enumerate(order)}
    dout = {v: -i for i, v in enumerate(sout)}
    d = {**din, **dout}
    d[','] = 0

    if any(v not in d for v in sin):
        raise YastnError('Order does not cover all contracted indices.')
    inds = [tuple(d[v] for v in ss) for ss in sin.split(',')]
    ts = list(operands)
    return ncon(ts, inds, conjs=conjs)


def ncon(ts, inds, conjs=None, order=None) -> 'Tensor':
    r"""
    Execute series of tensor contractions.

    Parameters
    ----------
    ts: Sequence[yastn.Tensor]
        list of tensors to be contracted.

    inds: Sequence[Sequence[int]]
        each inner tuple labels legs of respective tensor with integers.
        Positive values label legs to be contracted,
        with pairs of legs to be contracted denoted by the same integer label.
        Non-positive numbers label legs of the resulting tensor, in reversed order.

    conjs: Sequence[int]
        For each tensor in ``ts`` contains either ``0`` or ``1``.
        If the value is ``1``, the tensor is conjugated.

    order: Sequence[int]
        Order in which legs, marked by positive indices in inds, are contracted.
        If None, the legs are contracted following an ascending indices order.
        The default is None.

    Note
    ----
    :meth:`yastn.ncon` and :meth:`yastn.einsum` differ only by syntax.

    Example
    -------

    ::

        # matrix-matrix multiplication where the first matrix is conjugated

        yastn.ncon([a, b], ((-0, 1), (1, -1)), conjs=(1, 0))

        # outer product

        yastn.ncon([a, b], ((-0, -2), (-1, -3)))
    """
    if len(ts) != len(inds):
        raise YastnError('Number of tensors and indices do not match.')
    for tensor, ind in zip(ts, inds):
        if tensor.ndim != len(ind):
            raise YastnError('Number of legs of one of the tensors do not match provided indices.')

    inds = tuple(_clear_axes(*inds))
    if conjs is not None:
        conjs = tuple(conjs)
    if order is not None:
        order = tuple(order)

    meta_tr, meta_dot, meta_transpose = _meta_ncon(inds, conjs, order)
    ts = dict(enumerate(ts))
    for command in meta_tr:
        t, axes = command
        ts[t] = trace(ts[t], axes=axes)
    for command in meta_dot:
        (t1, t2), axes, conj = command
        ts[t1] = tensordot(ts[t1], ts[t2], axes=axes, conj=conj)
        del ts[t2]
    t, axes, to_conj = meta_transpose
    if axes is not None:
        ts[t] = ts[t].transpose(axes=axes)
    return ts[t].conj() if to_conj else ts[t]


@lru_cache(maxsize=1024)
def _meta_ncon(inds, conjs, order):
    r""" Turning information in ``inds`` and ``conjs`` into list of contraction commands. """
    if not all(-256 < x < 256 for x in _flatten(inds)):
        raise YastnError('Ncon requires indices to be between -256 and 256.')

    if order is not None:
        if len(order) != len(set(order)) or not all(o > 0 for o in order):
            raise YastnError("Order should be a list of positive ints with no repetitions.")
        if not set(o for o in _flatten(inds) if o > 0) == set(order):
            raise YastnError("Positive ints in ins and order should match.")
        reorder = {o: k for k, o in enumerate(order, start=1)}
        inds = [[reorder[o] if o > 0 else o for o in xx] for xx in inds]

    edges = [[order, leg, ten] if order > 0 else [-order + 1024, leg, ten]
             for ten, el in enumerate(inds) for leg, order in enumerate(el)]
    edges.append([512, 512, 512])  # this will mark the end of contractions.
    conjs = [0] * len(inds) if conjs is None else list(conjs)

    # order of contraction with info on tensor and axis
    edges = sorted(edges, reverse=True, key=itemgetter(0))
    return _consume_edges(edges, conjs)


def _consume_edges(edges, conjs):
    r""" Consumes edges to generate order of contractions. """
    eliminated, ntensors = [], len(conjs)
    meta_tr, meta_dot = [], []
    order1, leg1, ten1 = edges.pop()
    ax1, ax2 = [], []
    while order1 != 512:  # tensordot two tensors, or trace one tensor; 512 is cutoff marking end of truncation
        order2, leg2, ten2 = edges.pop()
        if order1 != order2:
            raise YastnError('Indices of legs to contract do not match.')
        t1, t2, leg1, leg2 = (ten1, ten2, leg1, leg2) if ten1 < ten2 else (ten2, ten1, leg2, leg1)
        ax1.append(leg1)
        ax2.append(leg2)
        if edges[-1][0] == 512 or min(edges[-1][2], edges[-2][2]) != t1 or max(edges[-1][2], edges[-2][2]) != t2:
            # execute contraction
            if t1 == t2:  # trace
                if len(meta_dot) > 0:
                    raise YastnError("Likely inefficient order of contractions. Do all traces before tensordot. " +
                                     "Call all axes connecting two tensors one after another.")
                meta_tr.append((t1, (tuple(ax1), tuple(ax2))))
                ax12 = ax1 + ax2
                for edge in edges:  # edge = (order, leg, tensor)
                    edge[1] -= sum(i < edge[1] for i in ax12) if edge[2] == t1 else 0
            else:  # tensordot (tensor numbers, axes, conj)
                meta_dot.append(((t1, t2), (tuple(ax1), tuple(ax2)), (conjs[t1], conjs[t2])))
                eliminated.append(t2)
                conjs[t1], conjs[t2] = 0, 0
                lt1 = sum(ii[2] == t1 for ii in edges)  # legs of t1
                for edge in edges:  # edge = (order, leg, tensor)
                    if edge[2] == t1:
                        edge[1] = edge[1] - sum(i < edge[1] for i in ax1)
                    elif edge[2] == t2:
                        edge[1:] = edge[1] + lt1 - sum(i < edge[1] for i in ax2), t1
            ax1, ax2 = [], []
        order1, leg1, ten1 = edges.pop()

    remaining = [i for i in range(ntensors) if i not in eliminated]
    t1 = remaining[0]
    for t2 in remaining[1:]:
        meta_dot.append(((t1, t2), ((), ()), (conjs[t1], conjs[t2])))
        eliminated.append(t2)
        conjs[t1], conjs[t2] = 0, 0
        lt1 = sum(tt == t1 for _, _, tt in edges)
        for edge in edges:  # edge = (order, leg, tensor)
            if edge[2] == t2:
                edge[1:] = edge[1] + lt1, t1
    unique_out = tuple(ed[0] for ed in edges)
    if len(unique_out) != len(set(unique_out)):
        raise YastnError("Repeated non-positive (outgoing) index is ambiguous.")
    axes = tuple(ed[1] for ed in sorted(edges))  # final order for transpose
    if axes == tuple(range(len(axes))):
        axes = None
    meta_transpose = (t1, axes, conjs[t1])
    return tuple(meta_tr), tuple(meta_dot), meta_transpose
