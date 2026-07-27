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
r""" Support for merging blocks in yastn.Tensor """
from __future__ import annotations

from functools import lru_cache
from itertools import groupby, product
from operator import itemgetter
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from ._auxiliary import _struct, _flatten, _clear_axes, _unpack_legs, get_blocks, find_matching_indices, get_trimmed_struct
from ._legbasic import LegBasic
from ._tests import YastnError, _test_axes_all

if TYPE_CHECKING:
    from . import Tensor

__all__ = ['fuse_legs', 'unfuse_legs', 'fuse_meta_to_hard', '_Fusion']


class _LegSlices(NamedTuple):
    r""" Immutable structure with information how to decompose a leg. """
    t: tuple = ()  # list of effective charges
    D: tuple = ()  # list of their bond dimensions
    dec: tuple = ()  # and their decompositions


class _DecRecord(NamedTuple):
    r""" Single record in _LegSlices.dec[i]"""
    t: tuple = ()  # charge
    Dslc: tuple = (None, None)  # slice
    Dprod: int = 0  # size of slice, equal to product of Drsh
    Drsh: tuple = ()  # original shape of fused dims in a block


class _Fusion(NamedTuple):
    r"""
    Information identifying the structure of hard fusion

    tree gives linearized structure of the fusion tree.
    E.g., tree = (6, 4, 1, 2, 1, 1, 1, 2, 1, 1) means that 6 original legs are fused
    followed by information about the number of legs constituting each sub-branch,
    where 1 corresponds to original legs.
    The above example represents the fusion of ((o, (o, o), o), (o, o))
    op gives the type of fusion:
    'p' product of spaces, 's' direct sum, 'o' original space with no internal structure.

    Charges t and dimensions D are given for fused spaces only.
    Their indexing is shifted by one compared to tree, op, s.
    For the top level, charges and dimensions follow from existing blocks.
    """
    tree: tuple = (1,)  # order of fusions
    op: str = 'o'  # type of node; 'o' original; 'p' product; 's' sum  len(node) = len(tree)
    s: tuple = (1,)  # signatures len(s) = len(tree)
    t: tuple = ()  # fused leg charges at each step len(t) = len(tree) - 1
    D: tuple = ()  # fused dimensions  at each step len(t) = len(tree) - 1

    def conj(self):
        return self._replace(s=tuple(-x for x in self.s))

    def is_fused(self):
        return self.tree[0] > 1


#  =========== fuse legs ======================


def fuse_legs(a, axes, mode=None) -> 'Tensor':
    r"""
    Fuse groups of legs into effective legs, reducing the rank of the tensor.

    .. note::
        Fusion can be reverted back by :meth:`yastn.Tensor.unfuse_legs`

    First, the legs are permuted into desired order. Then, selected groups of consecutive legs
    are fused. The desired order of the legs is given by a tuple ``axes``
    of leg indices where the groups of legs to be fused are denoted by inner tuples ::

        axes=(0,1,(2,3,4),5)  keep leg order, fuse legs (2,3,4) into new leg
        ->   (0,1, 2,     3)
            __              __
        0--|  |--3      0--|  |--3<-5
        1--|  |--4  =>  1--|__|
        2--|__|--5          |
                            2<-(2,3,4)


        axes=(2,(3,1),4,(7,6),5,0)  permute indices, then fuse legs (3,1) into new leg
        ->   (0, 1,   2, 3,   4,5)  and legs (7,6) into another new leg
            __                   __
        0--|  |--4        0->5--|  |--2<-4
        1--|  |--5 =>           |  |--4<-5
        2--|  |--6        2->0--|  |--3<-(7,6)
        3--|__|--7    (3,1)->1--|__|


    Two types of fusion are supported: `meta` and `hard`:

        * ``'hard'`` changes both the structure and data by aggregating smaller blocks into larger ones. Such fusion allows to balance number of non-zero blocks and typical block size.

        * ``'meta'`` performs the fusion only at the level of syntax, where it operates as a tensor with lower rank. Tensor structure and data (blocks) are not affected - apart from a transpose that may be needed for consistency.

    It is possible to use both `meta` and `hard` fusion of legs on the same tensor.
    Applying hard fusion on tensor turns all previous meta fused legs into hard fused
    ones.

    Parameters
    ----------
    axes: Sequence[int | Sequence[int]]
        tuple of leg indices. Groups of legs to be fused together are accumulated within inner tuples.

    mode: str
        can select ``'hard'`` or ``'meta'`` fusion. If ``None``, uses ``default_fusion``
        from tensor's :doc:`configuration </tensor/configuration>`.
        Configuration option ``force_fusion`` can be used to override ``mode`` (introduced for debugging purposes).
    """
    if a.isdiag:
        raise YastnError('Cannot fuse legs of a diagonal tensor.')
    if mode is None:
        mode = a.config.default_fusion
    if a.config.force_fusion is not None:
        mode = a.config.force_fusion

    order = tuple(_flatten(axes))
    _test_axes_all(a, order)
    axes = tuple(_clear_axes(*axes))
    if any(len(x) == 0 for x in axes):
        raise YastnError(f'Empty axis in {axes=}. To add a new dim-1 leg, use add_leg().')

    if mode == 'meta':
        mfs = []
        for group in axes:
            if len(group) == 1:
                mfs.append(a.mfs[group[0]])
            else:
                new_mf = [sum(a.mfs[ii][0] for ii in group)]
                for ii in group:
                    new_mf.extend(a.mfs[ii])
                mfs.append(tuple(new_mf))
        c = a.transpose(axes=order)
        c.mfs = tuple(mfs)
        return c
    if mode == 'hard':
        c = fuse_meta_to_hard(a)
        return _fuse_legs_hard(c, axes, order)
    raise YastnError('mode not in (`meta`, `hard`). Mode can be specified in config file.')


def _fuse_legs_hard(a, axes, order):
    r"""
    Function performing hard fusion. axes are for native legs and are cleaned outside.
    a.trans is accounted for here
    """
    order = tuple(a.trans[ax] for ax in order)
    axes = tuple(tuple(a.trans[ax] for ax in group) for group in axes)
    meta_mrg, size, struct_new, legs_old = _meta_fuse_hard(a.config.sym, a.struct, axes)
    data = a.config.backend.transpose_and_merge(a._data, order, meta_mrg, size)

    mfs = ((1,),) * len(struct_new.legs)
    hfs = []
    for leg_new, axs, legs in zip(struct_new.legs, axes, legs_old):
        t_in = tuple(leg.t for leg in legs)
        D_in = tuple(leg.D for leg in legs)
        hfs_axs = tuple(a.hfs[ax] for ax in axs)
        hfs.append(_combine_hfs_prod(hfs_axs, t_in, D_in, leg_new.s))
    out = a._replace(mfs=mfs, hfs=hfs, struct=struct_new, data=data, trans=None)
    return out


def _fuse_blocks(config, data, struct, axes, struct_sub=None, connector_first=True):
    order = sum(axes, start=())
    sub_legs = struct_sub.legs if struct_sub is not None else None
    meta_mrg, size, struct_mrg, legs_group = _meta_fuse_hard(config.sym, struct, axes, sub_legs, connector_first)
    data = config.backend.transpose_and_merge(data, order, meta_mrg, size)
    hfs = []
    for leg_new, legs in zip(struct_mrg.legs, legs_group):
        t_in = tuple(leg.t for leg in legs)
        D_in = tuple(leg.D for leg in legs)
        hfs_axs = tuple(_Fusion(s=(leg.s,)) for leg in legs)
        hfs.append(_combine_hfs_prod(hfs_axs, t_in, D_in, leg_new.s))
    return data, struct_mrg, tuple(hfs)


@lru_cache(maxsize=1024)
def _meta_fuse_hard(sym, struct, axes, legs_sub=None, connector_first=True):
    r""" Meta information for backend needed to hard-fuse some legs. """
    assert not struct.isdiag, "Sanity check. Contact developers."
    #
    st_full = get_blocks(sym, struct)
    if legs_sub is None or legs_sub == struct.legs:
        st = st_full
        slc = st.slc
    else:
        struct = get_trimmed_struct(sym, struct, legs_sub)
        st = get_blocks(sym, struct)
        slc = st_full.slc[find_matching_indices(st_full.t, st.t, both=False)]
    #
    slegs = tuple(tuple(struct.legs[n].s for n in axis) for axis in axes)
    s_eff = tuple(ss[0] if ss else -1 for ss in slegs)
    if axes and not axes[0] and connector_first:
        s_eff = (1,) + s_eff[1:]
    #
    teff = np.zeros((st.nblocks, len(s_eff), sym.NSYM), dtype=np.int64)
    for n, axs in enumerate(axes):
        teff[:, n, :] = sym.fuse(st.t[:, axs, :], slegs[n], s_eff[n]) if axs else sym.zero()
    #
    lls, legs_old = [], []
    for n, axs in enumerate(axes):
        legs_old.append(tuple(struct.legs[ax] for ax in axes[n]))
        if len(axs) > 1:
            teff_set = tuple(set(map(tuple, teff[:, n, :].tolist())))
            t_a = tuple(struct.legs[ia].t for ia in axs)
            D_a = tuple(struct.legs[ia].D for ia in axs)
            lls.append(_leg_structure_combine_charges_prod(sym, t_a, D_a, slegs[n], teff_set, s_eff[n]))
        elif len(axs) == 1:
            t, D = struct.legs[axs[0]].t, struct.legs[axs[0]].D
            dec = tuple((_DecRecord(tt, (0, DD), DD, (DD,)),) for tt, DD in zip(t, D))
            lls.append(_LegSlices(t, D, dec))
        else:  # len(a) == 0
            t, D = (sym.zero(),), (1,)
            dec = ((_DecRecord((), (0, 1), 1, (1,)),),)
            lls.append(_LegSlices(t, D, dec))

    legs_new = tuple(LegBasic(s=s, t=ll.t, D=ll.D) for s, ll in zip(s_eff, lls))

    struct_new = _struct(legs=legs_new, n=struct.n, isdiag=struct.isdiag)
    struct_new = get_trimmed_struct(sym, struct_new)
    bl_new = get_blocks(sym, struct_new)

    teff_split = list(tuple(map(tuple, x)) for x in teff.tolist())
    if len(axes) > 0:
        told_split = list(zip(*[st.t[:, axs, :].reshape(st.nblocks, len(axs) * sym.NSYM).tolist() for axs in axes]))
        told_split = list((tuple(map(tuple, x)) for x in told_split))
    else:
        told_split = tuple(map(tuple, st.t.reshape(st.nblocks, len(struct.legs) * sym.NSYM).tolist()))
    teff = list(map(tuple, teff.reshape(st.nblocks, len(axes) * sym.NSYM).tolist()))

    smeta = sorted((tes, tn, tos, slo, Do) for tes, tn, tos, slo, Do
                   in zip(teff_split, teff, told_split, slc, st.D))

    meta, tnew = [], []
    # ic = 0
    for tes, tn, tos, slo, Do in smeta:
        # while tuple(bl_new.t[ic].ravel()) != tn:
        #     ic += 1
        # sln, Dn = bl_new.slc[ic], bl_new.D[ic]

        ind = tuple(ls.t.index(te) for ls, te in zip(lls, tes))
        decs = tuple(ls.dec[ii] for ls, ii in zip(lls, ind))

        jjj = tuple([d.t for d in dd].index(tt) for tt, dd in zip(tos, decs))
        de = [dd[jj] for jj, dd in zip(jjj, decs)]
        sub_slc = tuple(x for d in de for x in d.Dslc)
        Dsln = tuple(d.Dprod for d in de)
        tnew.append(tn)
        meta.append((*slo, *Do, *sub_slc, *Dsln))

    ndimo = len(struct.legs)
    ndimn = len(struct_new.legs)
    meta = np.array(meta, dtype=np.int64).reshape(len(meta), 2 + 3 * ndimn + ndimo)
    tnew = np.array(tnew, dtype=np.int64).reshape(len(tnew), ndimn, sym.NSYM)
    ind = find_matching_indices(bl_new.t, tnew, both=False)  # tnew is not sorted

    ind_u, inv_c = np.unique(ind, return_inverse=True)
    struct_new = struct_new.mask_from_ind(bl_new.nblocks, ind_u)
    bl_new = get_blocks(sym, struct_new)
    meta = np.column_stack([bl_new.slc[inv_c], bl_new.D[inv_c], meta])

    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('Dn',  np.int64, (ndimn,)),
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (ndimo,)),
        ('Dslc', np.int64, (ndimn, 2)),
        ('Drsh', np.int64, (ndimn,))])
    meta = meta.view(meta_dt).reshape(-1)
    return meta, bl_new.size, struct_new, legs_old


def fuse_meta_to_hard(a):
    r""" Changes all meta fusions into hard fusions. If there are no meta fusions, return self. """
    while any(mf != (1,) for mf in a.mfs):
        axes, new_mfs = _consume_mfs_lowest(a.mfs)
        order = tuple(range(a.ndim_n))
        a = _fuse_legs_hard(a, axes, order)
        a.mfs = new_mfs
    return a

#  =========== unfuse legs ======================


def unfuse_legs(a, axes) -> 'Tensor':
    r"""
    Unfuse legs, reverting one layer of fusion.

    If the tensor has been obtained by fusing some legs together, `unfuse_legs`
    can revert such fusion. The legs to be unfused are passed in ``axes`` as `int`
    or `tuple[int]` in case of more legs to be unfused. The unfused legs are inserted at
    the positions of the fused legs. The remaining legs are shifted accordingly ::

        axes=2              unfuse leg 2 into legs 2,3,4
        ->   (0,1,2,3,4,5)
            __                 __
        0--|  |--3         0--|  |--3
        1--|__|        =>  1--|  |--4
            |              2--|__|--5<-3
            2=(2,3,4)


        axes=(    2,      5  )  unfuse leg 2 into legs 2,3 and leg 5 into legs 6,7
        ->   (0,1,2,3,4,5,6,7)
                  __                   __
              0--|  |--3           0--|  |--4
                 |  |--4       =>  1--|  |--5
              1--|  |--5=(6,7)     2--|  |--6
        (2,3)=2--|__|              3--|__|--7


    Unfusing a leg obtained by fusing together other previously fused legs, unfuses
    only the last fusion. ::

        axes=2              unfuse leg 2 into legs 2,3
        ->   (0,1,2,3,4)
            __                    __
        0--|  |--3            0--|  |--3=(3,4)
        1--|__|           =>  1--|  |
            |                 2--|__|--4<-3
            2=(2,3=(3,4))

    `fuse_legs` may involve leg transposition, which is not undone by `unfuse_legs`.

    Parameters
    ----------
    axes: int | Sequence[int]
        leg(s) to unfuse.
    """
    if a.isdiag:
        raise YastnError('Cannot unfuse legs of a diagonal tensor.')
    if isinstance(axes, int):
        axes = (axes,)
    ui, mfs, axes_mf, axes_uf, axes_hf = 0, [], [], [], []
    for mi in range(a.ndim):
        hi = a.trans[ui]
        if mi not in axes or (a.mfs[mi][0] == 1 and a.hfs[hi].tree[0] == 1):
            mfs.append(a.mfs[mi])
        elif a.mfs[mi][0] > 1:
            stack = a.mfs[mi]
            lstack = len(stack)
            pos_init, cum = 1, 0
            for pos in range(1, lstack):
                if cum == 0:
                    cum = stack[pos]
                if stack[pos] == 1:
                    cum = cum - 1
                    if cum == 0:
                        mfs.append(stack[pos_init: pos + 1])
                        pos_init = pos + 1
        elif a.hfs[hi].op[0] == 'p':  # and a.hfs[ni].tree[0] > 1 and a.mfs[mi][0] == 1 and mi in axes
            axes_hf.append(hi)
            axes_uf.append(ui)
            axes_mf.append(len(mfs))
            mfs.append((1,))  # a.mfs[mi] == (1,)
        else:  # c.hfs[ni].op == 's':
            raise YastnError('Cannot unfuse a leg obtained as a result of yastn.block()')
        ui += a.mfs[mi][0]
    if axes_hf:
        data, struct, nlegs, hfs = _unfuse_blocks(a.config, a._data, a.struct, tuple(axes_hf), tuple(a.hfs), return_hfs=True)

        for unfused, n in zip(nlegs[::-1], axes_mf[::-1]):
            mfs = mfs[:n] + [(1,)] * unfused + mfs[n+1:]

        trans = list(a.trans)
        for unfused, n in zip(nlegs[::-1], axes_uf[::-1]):
            trans = trans[:n] + [trans[n]] * unfused + trans[n+1:]
        for ax in range(max(trans), -1, -1):
            newax = sum(i < ax for i in trans)
            ii = trans.index(ax)
            while ii < len(trans) and trans[ii] == ax:
                trans[ii] = newax
                newax += 1
                ii += 1

        out = a._replace(struct=struct, mfs=tuple(mfs), hfs=hfs, data=data, trans=trans)
        return out
    out = a._replace(mfs=tuple(mfs))
    return out


def _unfuse_blocks(config, data, struct, axes, hfsm, return_hfs=False):
    axes_trim = tuple(ax for ax in axes if hfsm[ax].tree[0] > 1)
    if axes_trim:
        meta, size, struct, nlegs, hfs = _meta_unfuse_hard(config.sym, struct, axes_trim, hfsm)
        data = config.backend.unmerge(data, meta, size)
    if return_hfs:
        return data, struct, nlegs, hfs
    return data, struct


@lru_cache(maxsize=1024)
def _meta_unfuse_hard(sym, struct, axes, hfs):
    r""" Meta information for backend needed to hard-unfuse some legs. """
    assert not struct.isdiag, "Sanity check. Contact developers."

    lls, hfs_new, nlegs_unfused = [], [], []
    legs_new = []
    for n, hf in enumerate(hfs):
        if n in axes and hf.tree[0] > 1:
            t_part, D_part, s_part, hfs_part = _unfuse_Fusion(hf)
            legs_new.extend(LegBasic(s=s, t=t, D=D) for s, t, D in zip(s_part, t_part, D_part))
            lls.append(_leg_structure_combine_charges_prod(sym, t_part, D_part, s_part, struct.legs[n].t, struct.legs[n].s))
            hfs_new.extend(hfs_part)
            nlegs_unfused.append(len(hfs_part))
        else:
            legs_new.append(struct.legs[n])
            dec = tuple((_DecRecord(tt, (0, DD), DD, (DD,)),) for tt, DD in struct.legs[n].tD.items())
            lls.append(_LegSlices(struct.legs[n].t, struct.legs[n].D, dec))
            hfs_new.append(hf)

    bl_old = get_blocks(sym, struct)

    struct_new = _struct(legs=legs_new, n=struct.n, isdiag=struct.isdiag)
    struct_new = get_trimmed_struct(sym, struct_new)
    bl_new = get_blocks(sym, struct_new)

    tnew, meta = [], []

    old_t = bl_old.t.tolist()
    old_slc = map(tuple, bl_old.slc)

    for to, slo, Do in zip(old_t, old_slc, bl_old.D):
        ind = tuple(ls.t.index(tuple(to[n])) for n, ls in enumerate(lls))
        decs = tuple(tuple(ls.dec[ii]) for ls, ii in zip(lls, ind))
        for tt in product(*decs):
            tn = sum((x.t for x in tt), ())
            sub_slc = tuple(y for x in tt for y in x.Dslc)
            Dsln = tuple(x.Dprod for x in tt)
            tnew.append(tn)
            meta.append((*Dsln, *slo, *Do, *sub_slc))

    ndimo = len(struct.legs)
    meta = np.array(meta, dtype=np.int64).reshape(len(meta), 2 + 4 * ndimo)
    tnew = np.array(tnew, dtype=np.int64).reshape(len(tnew), len(legs_new), sym.NSYM)
    ind_c = find_matching_indices(bl_new.t, tnew, both=False)  # tnew is not sorted
    arg_c = np.argsort(ind_c)

    struct_new = struct_new.mask_from_ind(bl_new.nblocks, ind_c)
    bl_new = get_blocks(sym, struct_new)
    meta = np.column_stack([bl_new.slc, meta[arg_c]])

    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('Dn',  np.int64, (ndimo,)),
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (ndimo,)),
        ('sub_slc', np.int64, (ndimo, 2))])
    meta = meta.view(meta_dt).reshape(-1)
    return meta, bl_new.size, struct_new, tuple(nlegs_unfused), tuple(hfs_new)


#  =========== masks ======================


@lru_cache(maxsize=1024)
def _meta_mask(sym, struct, mask_t, mask_D, axis):
    r""" meta information for backend, and new tensor structure for mask."""
    leg_a = struct.legs[axis]
    mask_tD = {t: D for t, D in sorted(zip(mask_t, mask_D)) if D > 0 and t in leg_a}
    leg_c = LegBasic(s=leg_a.s, t=tuple(mask_tD.keys()), D=tuple(mask_tD.values()))

    if struct.isdiag:
        if axis == 0:
            legs_c = (leg_c, leg_c.conj())
        else:
            legs_c = (leg_c.conj(), leg_c)
            axis = 0
        ndim = 1
    else:
        legs_c = struct.legs[:axis] + (leg_c,) + struct.legs[axis + 1:]
        ndim = len(legs_c)

    struct_c = get_trimmed_struct(sym, struct, legs_c)
    bl_c = get_blocks(sym, struct_c)
    D_c = bl_c.D[:, :1] if struct.isdiag else bl_c.D

    bl_a = get_blocks(sym, struct)
    D_a = bl_a.D[:, :1] if struct.isdiag else bl_a.D
    taxis = bl_a.t[:, axis, :]

    ind_a = find_matching_indices(bl_a.t, bl_c.t, both=False)
    meta = np.column_stack([bl_c.slc, D_c, bl_a.slc[ind_a], D_a[ind_a], taxis[ind_a]])
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('Dn', np.int64, (ndim,)),
        ('sla', np.int64, (2,)),
        ('Da', np.int64, (ndim,)),
        ('tm', np.int64, (sym.NSYM,))])
    meta = meta.view(meta_dt).reshape(-1)
    return meta, bl_c.size, struct_c, axis, ndim


def _mask_nonzero(mask):
    r"""
    Change boolean masks into masks of indices.
    Fow trivial mask with all true, return None.
    """
    if all(np.all(v) for v in mask.values()):
        return None
    mask = {k: v.nonzero()[0] for k, v in mask.items()}
    mask = {k: v for k, v in mask.items() if len(v) > 0}
    return mask


def _mask_tensors_leg_intersection(a, b, axa, axb):
    r""" masks to get the intersecting parts of legs from two tensors a and b, for legs axa, axb. """
    msk_a, msk_b = [], []
    a_hfs, b_hfs = list(a.hfs), list(b.hfs)
    for i1, i2 in zip(axa, axb):
        ma, mb, axes_hfs = _masks_hfs_intersection(a.config.sym, (a.struct.legs[i1].t, b.struct.legs[i2].t), (a.struct.legs[i1].D, b.struct.legs[i2].D), (a.hfs[i1], b.hfs[i2]))
        msk_a.append(_mask_nonzero(ma))
        msk_b.append(_mask_nonzero(mb))
        a_hfs[i1], b_hfs[i2] = axes_hfs[0], axes_hfs[1]

    return msk_a, msk_b, tuple(a_hfs), tuple(b_hfs)


def _embed_tensor(a, legs, legs_new):
    r"""
    Embed tensor to fill in zero block in fusion mismatch.
    here legs are contained in legs_new that result from legs_union
    legs_new is a dict = {n: leg}
    """

    legs_new = [legs_new[n] if n in legs_new else legs[n] for n in range(len(legs))]
    legs, _ = _unpack_legs(legs)
    legs_new, _ = _unpack_legs(legs_new)

    hfs = tuple(lb.hf for lb in legs_new)
    assert a.ndim_n == len(hfs), "Sanity check. Contact developers."

    for axis, (la, lb) in enumerate(zip(legs, legs_new)):
        if la.hf != lb.hf:  # mask needed
            mb = _mask_embed_in_union(a.config.sym, la.t, la.hf, lb.hf)
            mask_tD = {t: len(v) for t, v in mb.items()}
            mask = _mask_nonzero(mb)
            if mask is not None:
                mask_t = tuple(mask_tD.keys())
                mask_D = tuple(mask_tD.values())
                meta, size, struct, axis, ndim = _meta_mask(a.config.sym, a.struct, mask_t, mask_D, axis)
                data = a.config.backend.embed_mask(a._data, mask, meta, size, axis, ndim)
                a = a._replace(struct=struct, data=data, hfs=hfs)
    return a

#  =========== auxiliary functions handling fusion logic ======================


@lru_cache(maxsize=1024)
def _leg_structure_combine_charges_prod(sym, t_in, D_in, s_in, t_out, s_out):
    r"""
    Combine effective charges and dimensions from a list of charges and dimensions for a few legs,
    forming product of spaces.
    """
    comb_t = list(product(*t_in))
    comb_t = np.array(comb_t, dtype=np.int64).reshape((len(comb_t), len(s_in), sym.NSYM))
    comb_D = list(product(*D_in))
    comb_D = np.array(comb_D, dtype=np.int64).reshape((len(comb_D), len(s_in)))
    teff = sym.fuse(comb_t, s_in, s_out)
    ind = np.array([ii for ii, te in enumerate(teff.tolist()) if tuple(te) in t_out], dtype=np.int64)
    comb_D, comb_t, teff = comb_D[ind], comb_t[ind], teff[ind]
    Deff = tuple(np.prod(comb_D, axis=1, dtype=np.int64).tolist())
    Dlegs = tuple(map(tuple, comb_D.tolist()))
    teff = tuple(map(tuple, teff.tolist()))
    tlegs = tuple(map(tuple, comb_t.reshape(len(comb_t), len(s_in) * sym.NSYM).tolist()))
    return _leg_structure_merge(teff, tlegs, Deff, Dlegs)


def _leg_structure_combine_charges_sum(t_in, D_in, pos=None):
    r"""
    Combine effective charges and dimensions from a list of charges and dimensions for a few legs,
    forming direct sum of spaces.
    """
    if pos is None:
        pos = range(len(t_in))
    teff, plegs, Deff, Dlegs = [], [], [], []
    for n, tns, Dns in zip(pos, t_in, D_in):
        for tn, Dn in zip(tns, Dns):
            plegs.append(n)
            teff.append(tn)
            Deff.append(Dn)
            Dlegs.append((Dn,))
    return _leg_structure_merge(teff, plegs, Deff, Dlegs)


def _leg_structure_merge(teff, tlegs, Deff, Dlegs):
    r""" LegDecomposition for merging into a single leg. """
    tt = sorted(set(zip(teff, tlegs, Deff, Dlegs)))
    t, D, dec = [], [], []
    for te, grp in groupby(tt, key=itemgetter(0)):
        Dlow, dect = 0, []
        for _, tl, De, Dl in grp:
            Dhigh = Dlow + De
            dect.append(_DecRecord(tl, (Dlow, Dhigh), De, Dl))
            Dlow = Dhigh
        t.append(te)
        D.append(Dhigh)
        dec.append(tuple(dect))
    return _LegSlices(tuple(t), tuple(D), tuple(dec))


def _combine_hfs_prod(hfs, t_in, D_in, s_out):
    r""" Combine _Fusion(s) forming product of space, adding charges and dimensions present on the fused legs. """
    axes = list(range(len(hfs)))
    if len(axes) == 0:
        return _Fusion(s=(s_out,))
    if len(axes) == 1:
        return hfs[0]
    tfl, Dfl, sfl = [], [], [s_out]
    opfl = 'p'  # product
    treefl = [sum(hfs[n].tree[0] for n in axes)]
    for n in axes:
        tfl.append(t_in[n])
        tfl.extend(hfs[n].t)
        Dfl.append(D_in[n])
        Dfl.extend(hfs[n].D)
        sfl.extend(hfs[n].s)
        treefl.extend(hfs[n].tree)
        opfl += hfs[n].op
    return _Fusion(tree=tuple(treefl), op=opfl, s=tuple(sfl), t=tuple(tfl), D=tuple(Dfl))


def _combine_hfs_sum(hfs, t_in, D_in, s_out):
    r""" Combine _Fusion(s) forming direct sum of space. """
    if len(hfs) == 1:
        return hfs[0]
    tfl, Dfl, sfl = [], [], [s_out]
    opfl = 's'  # sum
    treefl = [sum(hf.tree[0] for hf in hfs)]
    for t, D, hf in zip(t_in, D_in, hfs):
        if hf.op[0] != 's':
            ds = 0
            tfl.append(t)
            Dfl.append(D)
        else:  # hf.op[0] == 's':
            ds = 1
        tfl.extend(hf.t)
        Dfl.extend(hf.D)
        sfl.extend(hf.s[ds:])
        treefl.extend(hf.tree[ds:])
        opfl += hf.op[ds:]
    return _Fusion(tree=tuple(treefl), op=opfl, s=tuple(sfl), t=tuple(tfl), D=tuple(Dfl))


def _merge_masks_prod(sym, ls, ms):
    r""" Perform product of spaces / leg fusion, Combining masks using information from LegSlices. """
    msk = {tt: np.ones(Dt, dtype=bool) for tt, Dt in zip(ls.t, ls.D)}
    nsym = sym.NSYM
    for tt, dect in zip(ls.t, ls.dec):
        for rec in dect:
            msi = ms[0]
            tmp = msi[rec.t[:nsym]]
            for i, msi in enumerate(ms[1:], start=1):
                tmp = np.outer(tmp, msi[rec.t[i * nsym: (i + 1) * nsym]]).ravel()
            msk[tt][slice(*rec.Dslc)] = tmp
    return msk


def _merge_masks_sum(ls, ms):
    r""" Perform sum of spaces / blocking of legs, combining masks using information from LegSlices. """
    msk = {tt: np.ones(Dt, dtype=bool) for tt, Dt in zip(ls.t, ls.D)}
    for tt, dect in zip(ls.t, ls.dec):
        for rec in dect:
            msk[tt][slice(*rec.Dslc)] = ms[rec.t][tt]
    return msk


def _mask_falsify_mismatches_(ms1, ms2):
    r""" Multiply masks by False for indices that are not in both dictionaries ms1 and ms2. """
    set1, set2 = set(ms1), set(ms2)
    for t in set1 - set2:
        ms1[t] *= False
    for t in set2 - set1:
        ms2[t] *= False
    return ms1, ms2


@lru_cache(maxsize=1024)
def _masks_hfs_intersection(sym, ts, Ds, hfs):
    r"""
    Calculate two masks that project onto intersection of two spaces.
    ts = tuple[ts0, ts1], where ts0, ts1 are top-layer charges in two intersected legs.
    Ds = tuple[Ds0, Ds1] with corresponding top-lyer bond dimensions.
    hfs = tuple[hfs0, hfs1], where hfs0, hfs1 are hard fusion data for two spaces
    """
    teff = tuple(sorted(set(ts[0]) & set(ts[1])))
    tree = list(hfs[0].tree)

    if len(tree) == 1:
        ma0 = {t: np.ones(D, dtype=bool) for t, D in zip(ts[0], Ds[0]) if t in teff}
        ma1 = {t: np.ones(D, dtype=bool) for t, D in zip(ts[1], Ds[1]) if t in teff}
        if any(ma0[t].size != ma1[t].size for t in teff):
            raise YastnError('Bond dimensions of some charges do not match.')
        return ma0, ma1, hfs

    msks = [[{t: np.ones(D, dtype=bool) for t, D in zip(hf.t[i], hf.D[i])} for i, l in enumerate(tree[1:]) if l == 1]
            for hf in hfs]

    keeped_ts, keeped_Ds = [], []
    for ma0, ma1 in zip(*msks):
        keeped_t, keeped_D = [], []
        for t in set(ma0) & set(ma1):
            if ma0[t].size != ma1[t].size:
                raise YastnError('Bond dimensions of fused legs do not match.')
            keeped_t.append(t)
            keeped_D.append(ma0[t].size)
        keeped_ts.append(tuple(keeped_t))
        keeped_Ds.append(tuple(keeped_D))
        _mask_falsify_mismatches_(ma0, ma1)

    # lists to be consumed during parsing of the tree
    op = list(hfs[0].op)
    s = [list(hf.s) for hf in hfs]
    t = [[teff] + list(hf.t) for hf in hfs]
    D = [[()] + list(hf.D) for hf in hfs]

    # parse the tree, building masks
    while len(tree) > 1:
        it, io, no = _tree_cut_contiguous_leafs_(tree)
        # Remove original leafs to be fused; collect info for fusion
        del op[it: it + no]
        ss = [tuple(s1.pop(it) for _ in range(no)) for s1 in s]
        tt = [tuple(t1.pop(it) for _ in range(no)) for t1 in t]
        DD = [tuple(D1.pop(it) for _ in range(no)) for D1 in D]
        mss = [[msk.pop(io) for _ in range(no)] for msk in msks]
        assert op[it - 1] in 'sp', 'Sanity check. Contact developers.'
        if op[it - 1] == 'p':
            lss = [_leg_structure_combine_charges_prod(sym, tt1, DD1, ss1, t1[it - 1], s1[it - 1])
                   for tt1, DD1, ss1, t1, s1 in zip(tt, DD, ss, t, s)]
            ma = [_merge_masks_prod(sym, ls1, ms1) for ls1, ms1 in zip(lss, mss)]
            reduced_ls = _leg_structure_combine_charges_prod(sym, tuple(keeped_ts[:no]), tuple(keeped_Ds[:no]), ss[0], t[0][it - 1], s[0][it - 1])
        else:  # op[it - 1] == 's':
            lss = [_leg_structure_combine_charges_sum(tt1, DD1) for tt1, DD1, in zip(tt, DD)]
            ma = [_merge_masks_sum(ls1, ms1) for ls1, ms1 in zip(lss, mss)]
            reduced_ls = _leg_structure_combine_charges_sum(tuple(keeped_ts[:no]), tuple(keeped_Ds[:no]))
        _mask_falsify_mismatches_(ma[0], ma[1])
        msks[0].insert(io, ma[0])
        msks[1].insert(io, ma[1])

        keeped_ts.insert(0, reduced_ls.t)
        keeped_Ds.insert(0, reduced_ls.D)
    # Only the final leaf is left in msks[0] and msks[1]
    new_hfs = [_Fusion(hf.tree, hf.op, hf.s, tuple(keeped_ts[1:]), tuple(keeped_Ds[1:])) for hf in hfs]
    return msks[0].pop(), msks[1].pop(), new_hfs


def _mask_embed_in_union(sym, t0, hf0, hfu):
    r"""
    Return a mask to embed hard-fusion hf0 into hfu.
    hf0 should be a subspace of hfu, and consistent with it.
    The above is not checked. hfu should follow from _hfs_union including hf0.
    t0 are the top-layer charges for which masks are calculated.
    Return a dictionty {charge: mask}
    """
    # to be consumed during parsing of the tree
    tree = list(hfu.tree)
    ss = list(hfu.s)
    op = list(hfu.op)
    tus = [t0] + list(hfu.t)
    Dus = [()] + list(hfu.D)
    t0s = [t0] + list(hf0.t)

    msk = [{t: np.ones(D, dtype=bool) * (t in t0s[i]) for t, D in zip(tus[i], Dus[i])}
           for i, leafs in enumerate(tree) if leafs == 1]

    # parse the tree, building mask
    while len(tree) > 1:
        it, io, no = _tree_cut_contiguous_leafs_(tree)
        # Remove original leafs to be fused; collect info for fusion
        del op[it: it + no]
        del t0s[it: it + no]
        s_in = tuple(ss.pop(it) for _ in range(no))
        t_in = tuple(tus.pop(it) for _ in range(no))
        D_in = tuple(Dus.pop(it) for _ in range(no))
        ms_in = [msk.pop(io) for _ in range(no)]
        assert op[it - 1] in 'sp', 'Sanity check. Contact developers.'
        if op[it - 1] == 'p':
            ls = _leg_structure_combine_charges_prod(sym, t_in, D_in, s_in, tus[it - 1], ss[it - 1])
            ma = _merge_masks_prod(sym, ls, ms_in)
        else:  # op[it - 1] == 's':
            ls = _leg_structure_combine_charges_sum(t_in, D_in)
            ma = _merge_masks_sum(ls, ms_in)

        for t in ma.keys():
            if t not in t0s[it - 1]:
                ma[t] *= False
        msk.insert(io, ma)
    # Only the final leaf is left in msk
    return msk.pop()


def _hfs_union(sym, ts, hfs):
    r"""
    Consumes fusion trees from the bottom, while building the union of fused spaces.

    Parameters
    ----------
    sym: module
        symmetry class
    ts: Sequence[Sequence[tuple[int]]]
        List of top-level changes in each fused space.
    hfs: Sequence[_Fusion]
        List of _Fusion characterizing each fused space.
        They should have matching fusion tree, op and signatures.

    Returns
    -------
    tu, Du, hfs
        top-level charges in the union, their corresponding dimensions, _Fusion describing the union.
    """
    if any(hfs[0].tree != hf.tree or hfs[0].op != hf.op for hf in hfs):
        raise YastnError("Inconsistent numbers of hard-fused legs or sub-fusions order.")
    if any(hfs[0].s != hf.s for hf in hfs):
        raise YastnError("Inconsistent signatures of fused legs.")

    # to be consumed during parsing of the tree
    tree = list(hfs[0].tree)
    s = list(hfs[0].s)
    op = list(hfs[0].op)

    tu, Du, hfu = [], [], []
    for i, leafs in enumerate(tree[1:]):
        if leafs == 1:
            tDs = [list(zip(hf.t[i], hf.D[i])) for hf in hfs]
            alltD = {t: D for tD in tDs for t, D in tD}
            if any(alltD[t] != D for tD in tDs for t, D in tD):
                raise YastnError('Bond dimensions of fused legs do not match.')
            alltD = dict(sorted(alltD.items()))
            tu.append(tuple(alltD.keys()))
            Du.append(tuple(alltD.values()))
            hfu.append(_Fusion(s=(s[i + 1],)))  # len(s) == 1 + len(t)

    tss = [tuple(sorted({t for tl in ts for t in tl}))]  # len(tss) == len(tree)
    tss += [tuple(sorted({t for hf in hfs for t in hf.t[i]})) for i in range(len(tree) - 1)]

    while len(tree) > 1:
        it, io, no = _tree_cut_contiguous_leafs_(tree)
        # Remove original leafs to be fused; collect info for fusion
        del op[it: it + no]
        del tss[it: it + no]
        s_in = tuple(s.pop(it) for _ in range(no))
        t_in = tuple(tu.pop(io) for _ in range(no))
        D_in = tuple(Du.pop(io) for _ in range(no))
        hf_in = [hfu.pop(io) for _ in range(no)]
        # it - 1 is the index of new fused space in the tree
        t_out = tss[it - 1]
        s_out = s[it - 1]
        assert op[it - 1] in 'sp', 'Sanity check. Contact developers.'
        # Perform fusion and collect results for new lowest leaf
        if op[it - 1] == 'p':
            ls = _leg_structure_combine_charges_prod(sym, t_in, D_in, s_in, t_out, s_out)
            hf = _combine_hfs_prod(hf_in, t_in, D_in, s_out)
        else:  # op[it - 1] == 's':
            ls = _leg_structure_combine_charges_sum(t_in, D_in)
            hf = _combine_hfs_sum(hf_in, t_in, D_in, s_out)
        tu.insert(io, ls.t)
        Du.insert(io, ls.D)
        hfu.insert(io, hf)
    # Only the final leaf is left in tu, Du, and hfu
    return tu.pop(), Du.pop(), hfu.pop()


def _tree_cut_contiguous_leafs_(tree):
    r""" Parse tree searching for the first group of contiguous legs/leafs to merge. """
    fo, count, parents = 0, 1, []
    for ind, nleafs in enumerate(tree):
        if nleafs > 1:  # fused space
            no = 0  # the number of elements in the grouop
            count = nleafs
            parents.append(ind)
        else:  # original leg/leaf
            fo += 1
            no += 1
            count -= 1
        if count == 0:
            break

    it = ind - no + 1  # index of the first group element in the tree
    io = fo - no  # first element of the gruop among original legs

    # update tree in place; tree should be a list
    del tree[it: it + no]
    for nn in parents:  # parent nodes are before deleted ones
        tree[nn] -= (no - 1)

    return it, io, no


def _unfuse_Fusion(hf):
    r""" One layer of unfuse. """
    tt, DD, ss, hfs = [], [], [], []
    n_init, cum = 1, 0
    for n in range(1, len(hf.tree)):
        if cum == 0:
            cum = hf.tree[n]
        if hf.tree[n] == 1:
            cum -= 1
            if cum == 0:
                tt.append(hf.t[n_init - 1])
                DD.append(hf.D[n_init - 1])
                ss.append(hf.s[n_init])
                hfs.append(_Fusion(tree=hf.tree[n_init: n + 1], op=hf.op[n_init: n + 1],
                                   s=hf.s[n_init: n + 1], t=hf.t[n_init: n], D=hf.D[n_init: n]))
                n_init = n + 1
    return tuple(tt), tuple(DD), tuple(ss), hfs


def _consume_mfs_lowest(mfs):
    r"""
    Collects all fusions to be done in the lowest layer, based on fusion trees.
    Returns axes and new mfs.
    """
    new_mfs, axes, leg = [], [], 0
    for mf in mfs:
        if mf == (1,):
            new_mfs.append((1,))
            axes.append((leg,))
            leg += 1
        else:
            group, count = [], 0
            for nlegs in mf:  # parsing m tree to identify the lowest layer
                if nlegs > 1:
                    count = nlegs
                    for x in group:
                        axes.append((x,))
                    group = []
                else:
                    group.append(leg)
                    leg += 1
                    count -= 1
                if count == 0:
                    axes.append(tuple(group))
                    group = []
            for x in group:
                axes.append((x,))
            nt = _mf_to_ntree(mf)
            _ntree_eliminate_lowest(nt)
            new_mfs.append(_ntree_to_mf(nt))
    return tuple(axes), tuple(new_mfs)


def _ntree_eliminate_lowest(ntree):
    r""" Eliminates lowest possible layer of merges """
    if all(len(x) == 0 for x in ntree):
        ntree.clear()
    else:
        for x in ntree:
            _ntree_eliminate_lowest(x)


def _mf_to_ntree(mf):
    r""" Change linear fusion tree into nested lists. """
    ntree = []
    if mf[0] > 1:
        pos_init, cum = 1, 0
        for pos, nlegs in enumerate(mf[1:]):
            if cum == 0:
                cum = nlegs
            if nlegs == 1:
                cum -= 1
                if cum == 0:
                    ntree.append(_mf_to_ntree(mf[pos_init: pos + 2]))
                    pos_init = pos + 2
    return ntree


def _ntree_to_mf(ntree):
    """ Change nested lists into linear fusion tree. """
    mf = ()
    for subtree in ntree:
        mf = mf + _ntree_to_mf(subtree)
    nlegs = max(1, sum(x == 1 for x in mf))
    return (nlegs,) + mf
