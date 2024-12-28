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
from functools import lru_cache
from itertools import groupby, accumulate, product
from numbers import Number
from operator import itemgetter
import numpy as np
from ._auxliary import _struct, _slc, _clear_axes, _unpack_axes, _flatten, SpecialTensor
from ._tests import YastnError, _test_can_be_combined, _test_axes_match
from ._merging import _merge_to_matrix, _unmerge, _meta_unmerge_matrix
from ._merging import _masks_for_vdot, _masks_for_trace
from. _merging import _meta_fuse_hard, _transpose_and_merge, _mask_tensor_intersect_legs

__all__ = ['tensordot', 'vdot', 'trace', 'swap_gate', 'ncon', 'einsum', 'broadcast', 'apply_mask']


def __matmul__(a, b) -> yastn.Tensor:
    """
    The operation ``A @ B`` uses ``@`` operator to compute tensor dot product.
    The operation contracts the last axis of ``self``, i.e., ``a``,
    with the first axis of ``b``.

    It is equivalent to ``yastn.tensordot(a, b, axes=(a.ndim - 1, 0))``.
    """
    return tensordot(a, b, axes=(a.ndim - 1, 0))


def tensordot(a, b, axes, conj=(0, 0)) -> yastn.Tensor:
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
    needs_mask, (nin_a, nin_b) = _test_axes_match(a, b, sgn=-1, axes=(in_a, in_b))
    if a.isdiag:
        return _tensordot_diag(a, b, in_b, destination=(0,))
    if b.isdiag:
        return _tensordot_diag(b, a, in_a, destination=(-1,))

    _test_can_be_combined(a, b)

    nout_a = tuple(ii for ii in range(a.ndim_n) if ii not in nin_a)  # outgoing native legs
    nout_b = tuple(ii for ii in range(b.ndim_n) if ii not in nin_b)  # outgoing native legs

    s_c = tuple(a.struct.s[i1] for i1 in nout_a) + tuple(b.struct.s[i2] for i2 in nout_b)
    mfs_c = tuple(a.mfs[ii] for ii in range(a.ndim) if ii not in in_a) + tuple(b.mfs[ii] for ii in range(b.ndim) if ii not in in_b)
    hfs_c = tuple(a.hfs[ii] for ii in nout_a) + tuple(b.hfs[ii] for ii in nout_b)

    n_c = a.config.sym.add_charges(a.struct.n, b.struct.n)

    if needs_mask:
        msk_a, msk_b = _mask_tensor_intersect_legs(a, b, nin_a, nin_b)
        a = _apply_mask_to_axes(a, nin_a, msk_a)
        b = _apply_mask_to_axes(b, nin_b, msk_b)

    if a.config.tensordot_policy == 'fuse_to_matrix':
        data, struct_c, slices_c = _tensordot_f2m(a, b, nout_a, nin_a, nin_b, nout_b, s_c)
    elif a.config.tensordot_policy == 'fuse_contracted':
        data, struct_c, slices_c = _tensordot_fc(a, b, nout_a, nin_a, nin_b, nout_b)
    elif a.config.tensordot_policy == 'no_fusion':
        data, struct_c, slices_c = _tensordot_nf(a, b, nout_a, nin_a, nin_b, nout_b)
    else:
        raise YastnError(f"Tensordot policy not recognized. It should be 'fuse_to_matrix', 'fuse_contracted', or 'no_fusion'.")

    struct_c = struct_c._replace(n=n_c)
    return a._replace(data=data, struct=struct_c, slices=slices_c, mfs=mfs_c, hfs=hfs_c)


def _tensordot_f2m(a, b, nout_a, nin_a, nin_b, nout_b, s_c):
    """ Perform tensordot by fuse_to_matrix: merging tensors to matrices, executing dot, and unmerging outgoing legs. """

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


def _apply_mask_to_axes(a, naxes, masks):
    """ Auxlliary function applying mask tensors to native legs. """
    for ax, msk in zip(naxes, masks):
        if msk is not None:
            Dbnew = tuple(a.config.backend.count_nonzero(msk._data[slice(*sl.slcs[0])]) for sl in msk.slices)
            meta, struct, slices = _meta_mask(a.struct, a.slices, a.isdiag, msk.struct, msk.slices, Dbnew, ax)
            data = a.config.backend.mask_diag(a._data, msk._data, meta, struct.size, ax, a.ndim_n)
            a = a._replace(struct=struct, slices=slices, data=data)
    return a


def _tensordot_fc(a, b, nout_a, nin_a, nin_b, nout_b):
    """
    Perform tensordot by fuse_contracted: merging contracted legs, and executing dot.
    Outgoing legs are not merged so unmerge is not needed.
    """

    ind_a, ind_b = _common_inds(a.struct.t, b.struct.t, nin_a, nin_b, a.ndim_n, b.ndim_n, a.config.sym.NSYM)

    axes_a = tuple((x,) for x in nout_a) + (nin_a,)
    order_a = nout_a + nin_a
    struct_a, slices_a, meta_mrg_a, t_a, D_a = _meta_fuse_hard(a.config, a.struct, a.slices, axes_a, ind_a)
    data_a = _transpose_and_merge(a.config, a._data, order_a, struct_a, slices_a, meta_mrg_a)

    axes_b =  (nin_b,) + tuple((x,) for x in nout_b)
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
    """
    Perform tensordot directly: permute blocks and execute dot accumulaing results into result blocks.
    """
    ind_a, ind_b = _common_inds(a.struct.t, b.struct.t, nin_a, nin_b, a.ndim_n, b.ndim_n, a.config.sym.NSYM)
    meta_dot, reshape_a, reshape_b, struct_c, slices_c = _meta_tensordot_nf(a.struct, a.slices, b.struct, b.slices, ind_a, ind_b, nout_a, nin_a, nin_b, nout_b)
    order_a = nout_a + nin_a
    order_b = nin_b + nout_b
    data = a.config.backend.transpose_dot_sum(a.data, b.data, meta_dot, reshape_a, reshape_b, order_a, order_b, struct_c.size)
    return data, struct_c, slices_c


@lru_cache(maxsize=1024)
def _common_inds(t_a, t_b, nin_a, nin_b, ndimn_a, ndimn_b, nsym):
    """ Return row indices of nparray ``a`` that are in ``b``, and vice versa. Outputs tuples."""
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
        meta.append((ta[:nsym] + tb[nsym:], (Da[0], Db[1]), sla, Da, slb, Db, tar, tbl))
    meta = sorted(meta)
    t_c = tuple(x[0] for x in meta)
    D_c = tuple(x[1] for x in meta)
    Dp_c = tuple(D[0] * D[1] for D in D_c)
    slices_c = tuple( _slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(Dp_c), Dp_c, D_c))
    meta = tuple((sl.slcs[0], *mt[1:]) for sl, mt in zip(slices_c, meta))
    s_c = (struct_a.s[0], struct_b.s[1])
    struct_c = _struct(s=s_c, t=t_c, D=D_c, size=sum(Dp_c))
    return meta, struct_c, slices_c


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
    struct_a_resorted = sorted(((tuple(tc), tuple(to), Dc, Dop, tuple(Do), sl.slcs[0]) for tc, to, Dc, Dop, Do, sl in zip(tac, tao, Dac, Daop, Dao, slices_a)))

    ltb, ndimb = len(struct_b.t), len(struct_b.s)
    tb = np.array(struct_b.t, dtype=np.int64).reshape((ltb, ndimb, nsym))
    Db = np.array(struct_b.D, dtype=np.int64).reshape((ltb, ndimb))

    tbo = tb[:, 1:, :].reshape(ltb, (ndimb - 1) * nsym).tolist()
    tbc = tb[:, 0, :].tolist()
    Dbo = Db[:, 1:]
    Dbop = np.prod(Dbo, axis=1, dtype=np.int64).tolist()
    Dbo = Dbo.tolist()
    Dbc = Db[:, 0].tolist()
    struct_b_resorted = [(tuple(tc), tuple(to), Dc, Dop, tuple(Do), sl.slcs[0]) for tc, to, Dc, Dop, Do, sl in zip(tbc, tbo, Dbc, Dbop, Dbo, slices_b)]

    struct_a_resorted = groupby(struct_a_resorted, key=itemgetter(0))
    struct_b_resorted = groupby(struct_b_resorted, key=itemgetter(0))

    meta = []
    for (tar, group_ta), (tbl, group_tb) in zip(struct_a_resorted, struct_b_resorted):
        assert tar == tbl, "Sanity check."
        for (tca, toa, Dca, Dopa, Doa, sla), (tcb, tob, Dcb, Dopb, Dob, slb) in product(group_ta, group_tb):
            meta.append((toa + tob, Doa + Dob, Dopa * Dopb, (Dopa, Dopb), sla, (Dopa, Dca), slb, (Dcb, Dopb), tca, tcb))

    meta = sorted(meta)
    t_c = tuple(x[0] for x in meta)
    D_c = tuple(x[1] for x in meta)
    Dp_c = tuple(x[2] for x in meta)
    slices_c = tuple( _slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(Dp_c), Dp_c, D_c))
    meta = tuple((sl.slcs[0], *mt[3:]) for sl, mt in zip(slices_c, meta))
    s_c = struct_a.s[:-1] + struct_b.s[1:]
    struct_c = _struct(s=s_c, t=t_c, D=D_c, size=sum(Dp_c))
    return meta, struct_c, slices_c


@lru_cache(maxsize=1024)
def _meta_tensordot_nf(struct_a, slices_a, struct_b, slices_b, ind_a, ind_b, nout_a, nin_a, nin_b, nout_b):

    nsym = len(struct_a.n)

    if ind_a is None:
        ta = struct_a.t
        Da = struct_a.D
    else:
        ta = [struct_a.t[ii] for ii in ind_a]
        Da = [struct_a.D[ii] for ii in ind_a]
        slices_a = [slices_a[ii] for ii in ind_a]

    lta, ndima = len(ta), len(struct_a.s)
    ata = np.array(ta, dtype=np.int64).reshape((lta, ndima, nsym))
    aDa = np.array(Da, dtype=np.int64).reshape((lta, ndima))
    tao = ata[:, nout_a, :].reshape(lta, len(nout_a) * nsym).tolist()
    tac = ata[:, nin_a, :].reshape(lta, len(nin_a) * nsym).tolist()
    Dao = aDa[:, nout_a]
    Dac = aDa[:, nin_a]
    Daop = np.prod(Dao, axis=1, dtype=np.int64).tolist()
    Dao = Dao.tolist()
    Dacp = np.prod(Dac, axis=1, dtype=np.int64).tolist()
    Dac = Dac.tolist()
    struct_a_resorted = sorted(((tuple(tc), tuple(Dc), tuple(to), taa, Dop, tuple(Do)) for tc, Dc, to, taa, Dop, Do in zip(tac, Dac, tao, ta, Daop, Dao)))
    reshape_a = tuple((taa, sl.slcs[0], D, (Dop, Dcp)) for taa, sl, D, Dop, Dcp in zip(ta, slices_a, Da, Daop, Dacp))

    if ind_b is None:
        tb = struct_b.t
        Db = struct_b.D
    else:
        tb = [struct_b.t[ii] for ii in ind_b]
        Db = [struct_b.D[ii] for ii in ind_b]
        slices_b = [slices_b[ii] for ii in ind_b]
    ltb, ndimb = len(tb), len(struct_b.s)
    atb = np.array(tb, dtype=np.int64).reshape((ltb, ndimb, nsym))
    aDb = np.array(Db, dtype=np.int64).reshape((ltb, ndimb))
    tbo = atb[:, nout_b, :].reshape(ltb, len(nout_b) * nsym).tolist()
    tbc = atb[:, nin_b, :].reshape(ltb, len(nin_b) * nsym).tolist()
    Dbo = aDb[:, nout_b]
    Dbc = aDb[:, nin_b]
    Dbop = np.prod(Dbo, axis=1, dtype=np.int64).tolist()
    Dbo = Dbo.tolist()
    Dbcp = np.prod(Dbc, axis=1, dtype=np.int64).tolist()
    Dbc = Dbc.tolist()
    struct_b_resorted = sorted((tuple(tc), tuple(Dc), tuple(to), tbb, Dop, tuple(Do)) for tc, Dc, to, tbb, Dop, Do in zip(tbc, Dbc, tbo, tb, Dbop, Dbo))
    reshape_b = tuple((t, sl.slcs[0], D, (Dcp, Dop)) for t, sl, D, Dcp, Dop in zip(tb, slices_b, Db, Dbcp, Dbop))

    struct_a_resorted = groupby(struct_a_resorted, key=itemgetter(0, 1))
    struct_b_resorted = groupby(struct_b_resorted, key=itemgetter(0, 1))

    meta = []
    last_tc = None
    for ((tca, Dca), group_ta), ((tcb, Dcb), group_tb) in zip(struct_a_resorted, struct_b_resorted):
        assert tca == tcb, "Sanity check."
        assert last_tc != tcb, "Sanity check."
        last_tc = tcb
        if Dca != Dcb:
            raise YastnError('Bond dimensions do not match.')
        for (tca, Dca, toa, taa, Dopa, Doa), (tcb, Dcb, tob, tbb, Dopb, Dob) in product(group_ta, group_tb):
            meta.append((toa + tob, Doa + Dob, Dopa * Dopb, (Dopa, Dopb), taa, tbb))

    meta = sorted(meta)
    t_c, D_c, slices_c, meta_dot = [], [], [], []
    start = 0
    for (t, D, Dp, Dslc), group in groupby(meta, key=itemgetter(0, 1, 2, 3)):
        t_c.append(t)
        D_c.append(D)
        stop = start + Dp
        slices_c.append(_slc(((start, stop),), D, Dp))
        meta_dot.append(((start, stop), Dslc, [mt[4:] for mt in group]))
        start = stop
    t_c = tuple(t_c)
    D_c = tuple(D_c)
    slices_c = tuple(slices_c)
    s_c = tuple(struct_a.s[i] for i in nout_a) + tuple(struct_b.s[i] for i in nout_b)
    struct_c = _struct(s=s_c, t=t_c, D=D_c, size=start)
    return meta_dot, reshape_a, reshape_b, struct_c, slices_c


def _tensordot_diag(a, b, in_b, destination):
    """ Executes broadcast and then transpose into order expected by tensordot. """
    if len(in_b) == 1:
        c = a.broadcast(b, axes=in_b[0])
        return c.moveaxis(source=in_b, destination=destination)
    if len(in_b) == 2:
        c = a.broadcast(b, axes=in_b[0])
        return c.trace(axes=in_b)
    raise YastnError('Outer product with diagonal tensor not supported. Use yastn.diag() first.')  # len(in_a) == 0


def broadcast(a, *args, axes=0) -> yastn.Tensor | iterable[yastn.Tensor]:
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
    """ meta information for backend, and new tensor structure for brodcast. """
    nsym = len(a_struct.n)
    ind_tb = tuple(x[axis * nsym: (axis + 1) * nsym] for x in b_struct.t)
    ind_ta = tuple(x[:nsym] for x in a_struct.t)
    sl_a = dict(zip(ind_ta, a_slices))

    meta = tuple((tb, slb.slcs[0], Db, slb.Dp, sl_a[ib].slcs[0]) for tb, slb, Db, ib in \
                 zip(b_struct.t, b_slices, b_struct.D, ind_tb) if ib in ind_ta)

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


def apply_mask(a, *args, axes=0) -> yastn.Tensor | iterable[yastn.Tensor]:
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
    for b, ax in zip(args, axes):
        _test_can_be_combined(a, b)
        ax = _broadcast_input(ax, b.mfs, a.isdiag)
        if b.hfs[ax].tree != (1,):
            raise YastnError('Second tensor`s leg specified by axes cannot be fused.')

        Dbnew = tuple(a.config.backend.count_nonzero(a._data[slice(*sl.slcs[0])]) for sl in a.slices)
        meta, struct, slices = _meta_mask(b.struct, b.slices, b.isdiag, a.struct, a.slices, Dbnew, ax)

        if b.isdiag:
            b_ndim, ax = (1, 0)
            meta = tuple((sln, sla, Da[0], slb) for sln, sla, Da, slb in meta)
        else:
            b_ndim = b.ndim_n
        data = b.config.backend.mask_diag(b._data, a._data, meta, struct.size, ax, b_ndim)
        results.append(b._replace(struct=struct, slices=slices, data=data))
    return results.pop() if len(results) == 1 else results


@lru_cache(maxsize=1024)
def _meta_mask(a_struct, a_slices, a_isdiag, b_struct, b_slices, Dbnew, axis):
    """ meta information for backend, and new tensor structure for mask."""
    nsym = len(a_struct.n)
    ind_tb = {x[:nsym]: (sl.slcs[0], d) for x, d, sl in zip(b_struct.t, Dbnew, b_slices) if d > 0}
    ind_ta = tuple(x[axis * nsym: (axis + 1) * nsym] for x in a_struct.t)

    meta = tuple((ta, sla.slcs[0], Da, *ind_tb[ia]) for ta, sla, Da, ia in \
                zip(a_struct.t, a_slices, a_struct.D, ind_ta) if ia in ind_tb)

    if any(Da[axis] != slb[1] - slb[0] for _, _, Da, slb, _ in meta):
        raise YastnError("Bond dimensions do not match.")

    # mt = (ta, sla, Da, slb, Db)
    c_t = tuple(mt[0] for mt in meta)
    if a_isdiag:
        c_D = tuple((mt[4], mt[4]) for mt in meta)
        c_Dp = tuple(x[0] for x in c_D)
    else:
        c_D = tuple(mt[2][:axis] + (mt[4],) + mt[2][axis + 1:] for mt in meta)
        c_Dp = np.prod(c_D, axis=1, dtype=np.int64).tolist() if len(c_D) > 0 else ()

    c_slices = tuple(_slc(((stop - dp, stop),), ds, dp)  for stop, dp, ds in zip(accumulate(c_Dp), c_Dp, c_D))
    c_struct = a_struct._replace(t=c_t, D=c_D, size=sum(c_Dp))
    meta = tuple((sln.slcs[0], sla, Da, slb) for (_, sla, Da, slb, _), sln in zip(meta, c_slices))
    return meta, c_struct, c_slices


def vdot(a, b, conj=(1, 0)) -> Number:
    r"""
    Compute scalar product :math:`\langle a|b \rangle` between two tensors.

    Parameters
    ----------
    a, b: yastn.Tensor
        Two tensors for operation.

    conj: tuple[int, int]
        shows which tensor to conjugate: ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, or ``(1, 1)``.
        The default is ``(1, 0)``, i.e., tensor ``a`` is conjugated.
    """
    _test_can_be_combined(a, b)
    if conj[0] == 1:
        a = a.conj()
    if conj[1] == 1:
        b = b.conj()
    needs_mask, _ = _test_axes_match(a, b, sgn=-1)
    if a.struct.t == b.struct.t and a.slices == b.slices:
        Adata, Bdata = a._data, b._data
        struct_a, slices_a = a.struct, a.slices
        struct_b, slices_b = b.struct, b.slices
    else:
        ia, ib, ta, Da, Dpa, inter_sla,  tb, Db, Dpb, inter_slb = 0, 0, [], [], [], [], [], [], [], []
        while ia < len(a.struct.t) and ib < len(b.struct.t):
            if a.struct.t[ia] == b.struct.t[ib]:
                ta.append(a.struct.t[ia])
                tb.append(b.struct.t[ib])
                Da.append(a.struct.D[ia])
                Db.append(b.struct.D[ib])
                Dpa.append(a.slices[ia].Dp)
                Dpb.append(b.slices[ib].Dp)
                inter_sla.append(a.slices[ia].slcs[0])
                inter_slb.append(b.slices[ib].slcs[0])
                ia += 1
                ib += 1
            elif a.struct.t[ia] < b.struct.t[ib]:
                ia += 1
            else:
                ib += 1
        sla = tuple((stop - dp, stop) for stop, dp in zip(accumulate(Dpa), Dpa))
        slb = tuple((stop - dp, stop) for stop, dp in zip(accumulate(Dpb), Dpb))
        struct_a = a.struct._replace(t=tuple(ta), D=tuple(Da), size=sum(Dpa))
        struct_b = b.struct._replace(t=tuple(tb), D=tuple(Db), size=sum(Dpb))
        slices_a = tuple(_slc((x,), y, z) for x, y, z in zip(sla, Da, Dpa))
        slices_b = tuple(_slc((x,), y, z) for x, y, z in zip(slb, Db, Dpb))
        Adata = a.config.backend.apply_slice(a._data, sla, inter_sla)
        Bdata = b.config.backend.apply_slice(b._data, slb, inter_slb)
    if needs_mask:
        msk_a, msk_b, struct_a, slices_a, struct_b, slices_b = _masks_for_vdot(a.config, struct_a, slices_a, a.hfs, struct_b, slices_b, b.hfs)
        Adata = Adata[msk_a]
        Bdata = Bdata[msk_b]
    if struct_a.D != struct_b.D:
        raise YastnError('Bond dimensions do not match.')

    n_c = a.config.sym.add_charges(a.struct.n, b.struct.n)
    if len(struct_a.D) > 0 and n_c == a.config.sym.zero():
        return a.config.backend.vdot(Adata, Bdata)
    return a.zero_of_dtype()


def trace(a, axes=(0, 1)) -> yastn.Tensor:
    """
    Compute trace of legs specified by axes.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Legs to be traced out, e.g., ``axes=(0, 1)``; or ``axes=((2, 3, 4), (0, 1, 5))``.
    """
    lin1, lin2 = _clear_axes(*axes)  # contracted legs
    if len(set(lin1) & set(lin2)) > 0:
        raise YastnError('The same axis in axes[0] and axes[1].')
    needs_mask, (in1, in2) = _test_axes_match(a, a, sgn=-1, axes=(lin1, lin2))

    if len(in1) == 0:
        return a

    order = in1 + in2
    out = tuple(i for i in range(a.ndim_n) if i not in order)
    order = order + out
    mfs = tuple(a.mfs[i] for i in range(a.ndim) if i not in lin1 + lin2)
    hfs = tuple(a.hfs[ii] for ii in out)

    if a.isdiag:
        # if needs_mask: raise YastnError('Should not have happend')
        struct = a.struct._replace(s=(), diag=False, t=((),), D=((),), size=1)
        data = a.config.backend.sum_elements(a._data)
        return a._replace(struct=struct, slices=(_slc(((0, 1),), (), 1),), mfs=mfs, hfs=hfs, isdiag=False, data=data)

    meta, struct, slices, tcon, D1, D2 = _meta_trace(a.struct, a.slices, in1, in2, out)
    if needs_mask:
        msk12 = _masks_for_trace(a.config, tcon, D1, D2, a.hfs, in1, in2)
        data = a.config.backend.trace_with_mask(a._data, order, meta, struct.size, tcon, msk12)
    else:
        if D1 != D2:
            raise YastnError('Bond dimensions do not match.')
        data = a.config.backend.trace(a._data, order, meta, struct.size)
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, slices=slices, data=data)


@lru_cache(maxsize=1024)
def _meta_trace(struct, slices, in1, in2, out):
    """ meta-information for backend and struct of traced tensor. """
    lt, nsym = len(struct.t), len(struct.n)
    tset = np.array(struct.t, dtype=np.int64).reshape((lt, len(struct.s), nsym))
    Dset = np.array(struct.D, dtype=np.int64).reshape((lt, len(struct.s)))
    t1 = tset[:, in1, :].reshape(lt, len(in1) * nsym)
    t2 = tset[:, in2, :].reshape(lt, len(in2) * nsym)
    tn = tset[:, out, :].reshape(lt, len(out) * nsym)
    D1 = Dset[:, in1]
    D2 = Dset[:, in2]
    Dn = Dset[:, out]
    Dnp = np.prod(Dn, axis=1, dtype=np.int64)
    pD1 = np.prod(D1, axis=1, dtype=np.int64).reshape(lt, 1)
    pD2 = np.prod(D2, axis=1, dtype=np.int64).reshape(lt, 1)
    ind = (np.all(t1 == t2, axis=1)).nonzero()[0]
    Drsh = np.hstack([pD1, pD2, Dn])
    t12 = tuple(map(tuple, t1[ind].tolist()))
    tn = tuple(map(tuple, tn[ind].tolist()))
    D1 = tuple(map(tuple, D1[ind].tolist()))
    D2 = tuple(map(tuple, D2[ind].tolist()))
    Dn = tuple(map(tuple, Dn[ind].tolist()))
    Dnp = Dnp[ind].tolist()
    slo = tuple(slices[n].slcs[0] for n in ind)
    Do = tuple(struct.D[n] for n in ind)
    Drsh = tuple(map(tuple, Drsh[ind].tolist()))

    meta = tuple(sorted(zip(tn, Dn, Dnp, t12, slo, Do, Drsh), key=itemgetter(0)))

    low, high = 0, 0
    c_t, c_D, c_slices, meta2, tcon = [], [], [], [], []
    for t, group in groupby(meta, key=itemgetter(0)):
        c_t.append(t)
        mt = next(group)
        c_D.append(mt[1])
        Dp = mt[2]
        high = low + Dp
        sl = (low, high)
        low = high
        c_slices.append(_slc((sl,), c_D[-1], Dp))
        tcon.append(mt[3])
        meta2.append((sl, *mt[4:]))
        for mt in group:
            tcon.append(mt[3])
            meta2.append((sl, *mt[4:]))
    c_s = tuple(struct.s[i] for i in out)
    c_struct = _struct(s=c_s, n=struct.n, t=tuple(c_t), D=tuple(c_D), size=high)
    return tuple(meta2), c_struct, tuple(c_slices), tuple(tcon), D1, D2


def swap_gate(a, axes, charge=None) -> yastn.Tensor:
    """
    Return tensor after application of a swap gate.

    The function's action is controlled by the ``fermionic`` flag in the tensor :ref:`config <tensor/configuration:YASTN configuration>`.
    Multiply blocks with odd charges on swapped legs by :math:`-1`.
    The ``fermionic`` flag selects which individual charges (in case of a direct product of a few symmetries) are tested for oddity,
    where the contributions from each selected charge get multiplied.
    See :class:`yastn.operators.SpinfulFermions` for an example.
    For ``fermionic=True``, all charges are considered.
    For ``fermionic=False``,  swap_gate returns ``a``.

    Parameters
    ----------
    axes: Sequence[int | Sequence[int]]
        Tuple with groups of legs. Consecutive pairs of grouped legs that are to be swapped.
        For instance, ``axes = (0, 1)`` apply swap gate between 0th and 1st leg.
        ``axes = ((0, 1), (2, 3), 4, 5)`` swaps ``(0, 1)`` with ``(2, 3)``, and ``4`` with ``5``.

    charge: Optional[Sequence[int]]
        If provided, the swap gate is applied between a virtual one-dimensional leg
        of specified charge, e.g., a fermionic string, and tensor legs specified in axes.
        In this case, there is no application of a swap gates between legs specified in axes.
    """
    if not a.config.fermionic:
        return a
    nsym = a.config.sym.NSYM
    fss = (True,) * nsym if a.config.fermionic is True else a.config.fermionic
    if charge is None:
        axes = tuple(_clear_axes(*axes))  # swapped groups of legs
        tp = _meta_swap_gate(a.struct.t, a.mfs, a.ndim_n, nsym, axes, fss)
    else:
        axes, = _clear_axes(axes)  # swapped groups of legs
        tp = _meta_swap_gate_charge(a.struct.t, charge, a.mfs, a.ndim_n, nsym, axes, fss)
    c = a.clone()
    for sl, odd in zip(c.slices, tp):
        if odd:
            c._data[slice(*sl.slcs[0])] *= -1
    return c


@lru_cache(maxsize=1024)
def _meta_swap_gate(t, mf, ndim, nsym, axes, fss):
    """ Calculate which blocks to negate. """
    axes = _unpack_axes(mf, *axes)
    tset = np.array(t, dtype=np.int64).reshape((len(t), ndim, nsym))
    iaxes = iter(axes)
    tp = np.zeros(len(t), dtype=np.int64)

    if len(axes) % 2 == 1:
        raise YastnError('Odd number of elements in axes. Elements of axes should come in pairs.')
    for l1, l2 in zip(*(iaxes, iaxes)):
        # if len(set(l1) & set(l2)) > 0:
        #     raise YastnError('Cannot swap the same index.')
        t1 = np.sum(tset[:, l1, :], axis=1, dtype=np.int64) % 2
        t2 = np.sum(tset[:, l2, :], axis=1, dtype=np.int64) % 2
        tp += np.sum(t1[:, fss] * t2[:, fss], axis=1, dtype=np.int64)
    return tuple((tp % 2).tolist())


@lru_cache(maxsize=1024)
def _meta_swap_gate_charge(t, charge, mf, ndim, nsym, axes, fss):
    """ Calculate which blocks to negate. """
    axes, = _unpack_axes(mf, axes)
    tset = np.array(t, dtype=np.int64).reshape((len(t), ndim, nsym))
    if len(charge) != nsym:
        raise YastnError(f'Length of charge {charge} does not match sym.NSYM = {nsym}.')

    charge = np.array(charge, dtype=np.int64).reshape(1, nsym) % 2
    tp = np.sum(tset[:, axes, :], axis=1, dtype=np.int64) % 2
    fp = np.sum(tp[:, fss] * charge[:, fss], axis=1, dtype=np.int64) % 2
    return tuple(fp.tolist())


def einsum(subscripts, *operands, order=None) -> yastn.Tensor:
    """
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


def ncon(ts, inds, conjs=None, order=None) -> yastn.Tensor:
    """
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
    """ Turning information in ``inds`` and ``conjs`` into list of contraction commands. """
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
    """ Consumes edges to generate order of contractions. """
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
