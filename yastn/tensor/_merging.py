""" Support for merging blocks in yastn tensors. """

from functools import lru_cache
from itertools import groupby, product
from operator import itemgetter
from typing import NamedTuple
import numpy as np
from ._auxliary import _slc, _flatten, _clear_axes, _ntree_to_mf, _mf_to_ntree, _unpack_legs
from ._tests import YastnError, _test_axes_all, _get_tD_legs


__all__ = ['fuse_legs', 'unfuse_legs', 'fuse_meta_to_hard']


class _LegSlices(NamedTuple):
    """ Immutable version of LegDec """
    t: tuple = ()  # list of effective charges
    D: tuple = ()  # list of their bond dimensions
    dec: tuple = ()  # and their decompositions


class _DecRecord(NamedTuple):
    """ Single record in _LegSlices.dec[i]"""
    t : tuple = ()  # charge
    Dslc: tuple = (None, None)  # slice
    Dprod: int = 0  # size of slice, equal to product of Drsh
    Drsh: tuple = ()  # original shape of fused dims in a block


class _Fusion(NamedTuple):
    """ Information identifying the structure of hard fusion"""
    tree: tuple = (1,)  # order of fusions
    op: str = 'o'  # type of node; 'o' original; 'p' product; 's' sum  len(node) = len(tree)
    s: tuple = (1,)  # signatures len(s) = len(tree)
    t: tuple = ()  # fused leg charges at each step len(t) = len(tree) - 1
    D: tuple = ()  # fused dimensions  at each step len(t) = len(tree) - 1

    def conj(self):
        return self._replace(s=tuple(-x for x in self.s))

    def is_fused(self):
        return self.tree[0] > 1


#  =========== merging blocks ======================

def _merge_to_matrix(a, axes, inds=None):
    """ Main function merging tensor into effective block matrix. """
    order = axes[0] + axes[1]
    struct, slices, meta_mrg, ls_l, ls_r = _meta_merge_to_matrix(a.config, a.struct, a.slices, axes, inds)
    data = _transpose_and_merge(a.config, a._data, order, struct, slices, meta_mrg, inds)
    return data, struct, slices, ls_l, ls_r


def _transpose_and_merge(config, data, order, struct, slices, meta_mrg, inds=None):
    meta_new = tuple((x, y, z.slcs[0]) for x, y, z in zip(struct.t, struct.D, slices))
    if inds is None and tuple(range(len(order))) == order and struct.size == len(data) \
       and _no_change_in_transpose_and_merge(meta_mrg, meta_new, struct.size):
        return data
    return config.backend.transpose_and_merge(data, order, meta_new, meta_mrg, struct.size)


def _no_change_in_transpose_and_merge(meta_mrg, meta_new, Dsize):
    """ Assumes C ordering on backend reshape """
    low = 0
    for _, slo, _, _, _ in meta_mrg:
        if slo[0] != low:
            return False
        low = slo[1]
    if low != Dsize:
        return False
    for (_, Dn, _), (_, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        low = 0
        for _, _, _, Dslc, _ in gr:
            if Dslc[0][0] != low:
                return False
            low = Dslc[0][1]
        if low != Dn[0]:
            return False
    return True


def _unmerge(config, data, meta):
    Dsize = meta[-1][0][1] if len(meta) > 0 else 0
    assert len(data) == Dsize, "This should not have happen"
    if _no_change_in_unmerge(meta):
        return data
    return config.backend.unmerge(data, meta)


def _no_change_in_unmerge(meta):
    local_low, Dn_last, sl_last, sln = 0, 0, (0, 0), (0, 0)
    for sln, Dn, slo, _, sub_slc in meta:
        if slo != sl_last:  # new group
            if (slo[0] != sl_last[1]) or (local_low != Dn_last):
                return False
            sl_last, Dn_last, local_low = slo, Dn[0], 0
        if local_low != sub_slc[0][0]:
            return False
        local_low = sub_slc[0][1]
    if sl_last[1] != sln[1]:
        return False
    return True


@lru_cache(maxsize=1024)
def _meta_merge_to_matrix(config, struct, slices, axes, inds):
    """ Meta information for backend needed to merge tensor into effective block matrix. """
    s_eff = []
    s_eff.append(struct.s[axes[0][0]] if len(axes[0]) > 0 else 1)
    s_eff.append(struct.s[axes[1][0]] if len(axes[1]) > 0 else -1)

    t_old = struct.t if inds is None else [struct.t[ii] for ii in inds]
    D_old = struct.D if inds is None else [struct.D[ii] for ii in inds]
    sl_old = slices if inds is None else [slices[ii] for ii in inds]
    tset = np.array(t_old, dtype=int).reshape((len(t_old), len(struct.s), config.sym.NSYM))
    Dset = np.array(D_old, dtype=int).reshape(len(D_old), len(struct.s))
    legs, t, D, Deff, teff, s, ls = [], [], [], [], [], [], []
    for n in (0, 1):
        legs.append(np.array(axes[n], int))
        t.append(tset[:, legs[n], :])
        D.append(Dset[:, legs[n]])
        Deff.append(np.prod(D[n], axis=1, dtype=int))
        s.append(np.array([struct.s[ii] for ii in axes[n]], dtype=int))
        teff.append(config.sym.fuse(t[n], s[n], s_eff[n]))
        teff[n] = tuple(tuple(t.flat) for t in teff[n])
        t[n] = tuple(tuple(t.flat) for t in t[n])
        D[n] = tuple(tuple(x) for x in D[n])
        ls.append(_leg_structure_merge(teff[n], t[n], Deff[n], D[n]))

    smeta = sorted((tel, ter, tl, tr, slo.slcs[0], Do)
                for tel, ter, tl, tr, slo, Do in zip(teff[0], teff[1], t[0], t[1], sl_old, D_old))

    meta_mrg, t_new, D_new, slices_new, Dlow = [], [], [], [],  0
    for (tel, ter), gr in groupby(smeta, key=lambda x: x[:2]):
        ind0 = ls[0].t.index(tel)
        ind1 = ls[1].t.index(ter)
        tn = tel + ter
        t_new.append(tn)
        D0, D1 = ls[0].D[ind0], ls[1].D[ind1]
        D_new.append((D0, D1))
        Dp = D0 * D1
        Dhigh = Dlow + Dp
        slices_new.append(_slc(((Dlow, Dhigh),), (D0, D1), Dp))
        Dlow = Dhigh
        try:
            _, _, tl, tr, slo, Do = next(gr)
            for d0, d1 in product(ls[0].dec[ind0], ls[1].dec[ind1]):
                if d0.t == tl and d1.t == tr:
                    meta_mrg.append((tn, slo, Do, (d0.Dslc, d1.Dslc), (d0.Dprod, d1.Dprod)))
                    _, _, tl, tr, slo, Do = next(gr)
        except StopIteration:
            pass
    struct_new = struct._replace(t=tuple(t_new), D=tuple(D_new), s=tuple(s_eff), size=Dlow)
    slices_new = tuple(slices_new)
    return struct_new, slices_new, tuple(meta_mrg), ls[0], ls[1]


def _leg_struct_trivial(struct, axis=0):
    """ trivial LegDecomposition for unfused leg. """
    nsym = len(struct.n)
    tD = sorted((tt[nsym * axis: nsym * (axis + 1)], DD[axis]) for tt, DD in zip(struct.t, struct.D))
    t = tuple(x[0] for x in tD)
    D = tuple(x[1] for x in tD)
    dec = tuple((_DecRecord(tt, (0, DD), DD, (DD,)),) for tt, DD in zip(t, D))
    return _LegSlices(t, D, dec)


#  =========== fuse legs ======================

def fuse_legs(a, axes, mode=None):
    r"""
    Fuse groups of legs into effective legs, reducing the rank of the tensor.

    .. note::
        Fusion can be reverted back by :meth:`yastn.Tensor.unfuse_legs`

    First, the legs are permuted into desired order. Then, selected groups of consecutive legs
    are fused. The desired order of the legs is given by a tuple `axes`
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

        * ``'meta'`` performs the fusion only at the level of syntax, where it operates as a tensor with lower rank. Tensor structure and data (blocks) are not affected - apart from a transpose that may be needed for consistency.

        * ``'hard'`` changes both the structure and data by aggregating smaller blocks into larger ones. Such fusion allows to balance number of non-zero blocks and typical block size.

    It is possible to use both `meta` and `hard` fusion of legs on the same tensor.
    Applying hard fusion on tensor turns all previous meta fused legs into hard fused
    ones.

    Parameters
    ----------
    axes: tuple[tuple[int]]
        tuple of leg indices. Groups of legs to be fused together are accumulated within inner tuples.

    mode: str
        can select ``'hard'`` or ``'meta'`` fusion. If ``None``, uses ``default_fusion``
        from tensor's :doc:`configuration </tensor/configuration>`.
        Configuration option ``force_fusion`` can be used to override `mode`, typically for debugging purposes.

    Returns
    -------
    yastn.Tensor
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
    """ Function performing hard fusion. axes are for native legs and are cleaned outside."""
    struct, slices, meta_mrg, t_in, D_in = _meta_fuse_hard(a.config, a.struct, a.slices, axes)
    data = _transpose_and_merge(a.config, a._data, order, struct, slices, meta_mrg)
    mfs = ((1,),) * len(struct.s)
    hfs = tuple(_fuse_hfs(a.hfs, t_in, D_in, struct.s[n], axis) if len(axis) > 1 else a.hfs[axis[0]]
                for n, axis in enumerate(axes))
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, slices=slices, data=data)


@lru_cache(maxsize=1024)
def _meta_fuse_hard(config, struct, slices, axes):
    """ Meta information for backend needed to hard-fuse some legs. """
    nblocks, nsym = len(struct.t), len(struct.n)
    t_in, D_in, tD_dict, tset, Dset = _get_tD_legs(struct)
    slegs = tuple(tuple(struct.s[n] for n in a) for a in axes)
    s_eff = tuple(struct.s[axis[0]] for axis in axes)
    teff = np.zeros((nblocks, len(s_eff), nsym), dtype=int)
    Deff = np.zeros((nblocks, len(s_eff)), dtype=int)
    for n, a in enumerate(axes):
        teff[:, n, :] = config.sym.fuse(tset[:, a, :], slegs[n], s_eff[n])
        Deff[:, n] = np.prod(Dset[:, a], axis=1, dtype=int)

    lls = []
    for n, a in enumerate(axes):
        if len(a) > 1:
            teff_set = tuple(set(tuple(x.flat) for x in teff[:, n, :]))
            t_a = tuple(t_in[n] for n in a)
            D_a = tuple(D_in[n] for n in a)
            lls.append(_leg_structure_combine_charges_prod(config.sym, t_a, D_a, slegs[n], teff_set, s_eff[n]))
        else:
            t, D = tuple(tD_dict[a[0]].keys()), tuple(tD_dict[a[0]].values())
            dec = tuple((_DecRecord(tt, (0, DD), DD, (DD,)),) for tt, DD in zip(t, D))
            lls.append(_LegSlices(t, D, dec))

    teff_split = [tuple(tuple(y.flat) for y in x) for x in teff]
    told_split = [tuple(tuple(x[a, :].flat) for a in axes) for x in tset]
    teff = tuple(tuple(x.flat) for x in teff)

    smeta = sorted((tes, tn, tos, slo.slcs[0], Do) for tes, tn, tos, slo, Do
                in zip(teff_split, teff, told_split, slices, struct.D))

    meta_mrg, t_new, D_new = [], [], []
    for (tes, tn), gr in groupby(smeta, key=lambda x: x[:2]):
        ind = tuple(ls.t.index(te) for ls, te in zip(lls, tes))
        decs = tuple(ls.dec[ii] for ls, ii in zip(lls, ind))
        t_new.append(tn)
        D_new.append(tuple(ls.D[ii] for ls, ii in zip(lls, ind)))
        try:
            _, _, tos, slo, Do = next(gr)
            for de in product(*decs):
                if tuple(d.t for d in de) == tos:
                    sub_slc = tuple(d.Dslc for d in de)
                    Dsln = tuple(d.Dprod for d in de)
                    meta_mrg.append((tn, slo, Do, sub_slc, Dsln))
                    _, _, tos, slo, Do = next(gr)
        except StopIteration:
            pass
    Dp_new = np.prod(D_new, axis=1, dtype=int) if D_new else []
    struct_new = struct._replace(t=tuple(t_new), D=tuple(D_new), s=tuple(s_eff), size=sum(Dp_new))
    slices_new = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(np.cumsum(Dp_new), Dp_new, D_new))
    return struct_new, slices_new, meta_mrg, t_in, D_in


def fuse_meta_to_hard(a):
    """ Changes all meta fusions into a hard fusions. If there are no meta fusions, do nothing. """
    while any(mf != (1,) for mf in a.mfs):
        axes, new_mfs = _consume_mfs_lowest(a.mfs)
        order = tuple(range(a.ndim_n))
        a = _fuse_legs_hard(a, axes, order)
        a.mfs = new_mfs
    return a

#  =========== unfuse legs ======================

def unfuse_legs(a, axes):
    r"""
    Unfuse legs, reverting one layer of fusion.

    If the tensor has been obtained by fusing some legs together, `unfuse_legs`
    can revert such fusion. The legs to be unfused are passed in `axes` as `int`
    or `tuple[int]` in case of more legs to be unfused. The unfused legs follow
    the original position of the fused legs. The remaining legs are shifted accordingly ::

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
    only the last fusion ::

        axes=2              unfuse leg 2 into legs 2,3
        ->   (0,1,2,3,4)
            __                    __
        0--|  |--3            0--|  |--3=(3,4)
        1--|__|           =>  1--|  |
            |                 2--|__|--4<-3
            2=(2,3=(3,4))

    Parameters
    ----------
    axes: int or tuple[int]
        leg(s) to unfuse.

    Returns
    -------
    yastn.Tensor
    """
    if a.isdiag:
        raise YastnError('Cannot unfuse legs of a diagonal tensor.')
    if isinstance(axes, int):
        axes = (axes,)
    ni, mfs, axes_hf = 0, [], []
    for mi in range(a.ndim):
        dni = a.mfs[mi][0]
        if mi not in axes or (a.mfs[mi][0] == 1 and a.hfs[ni].tree[0] == 1):
            mfs.append(a.mfs[mi])
        elif a.mfs[mi][0] > 1:  #and mi in axes
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
        elif a.hfs[ni].op[0] == 'p':  # and a.hfs[ni].tree[0] > 1 and a.mfs[mi][0] == 1 and mi in axes
            axes_hf.append(ni)
            mfs.append(a.mfs[mi])
        else: # c.hfs[ni].op == 's':
            raise YastnError('Cannot unfuse a leg obtained as a result of yastn.block()')
        ni += dni
    if axes_hf:
        meta, struct, slices, nlegs, hfs = _meta_unfuse_hard(a.config, a.struct, a.slices, tuple(axes_hf), tuple(a.hfs))
        data = _unmerge(a.config, a._data, meta)
        for unfused, n in zip(nlegs[::-1], axes_hf[::-1]):
            mfs = mfs[:n] + [mfs[n]] * unfused + mfs[n+1:]
        return a._replace(struct=struct, slices=slices, mfs=tuple(mfs), hfs=hfs, data=data)
    return a._replace(mfs=tuple(mfs))


@lru_cache(maxsize=1024)
def _meta_unfuse_hard(config, struct, slices, axes, hfs):
    """ Meta information for backend needed to hard-unfuse some legs. """
    t_in, _, tD_dict, _, _ = _get_tD_legs(struct)
    lls, hfs_new, snew, nlegs_unfused = [], [], [], []
    for n, hf in enumerate(hfs):
        if n in axes:
            t_part, D_part, s_part, hfs_part = _unfuse_Fusion(hf)
            lls.append(_leg_structure_combine_charges_prod(config.sym, t_part, D_part, s_part, t_in[n], struct.s[n]))
            hfs_new.extend(hfs_part)
            nlegs_unfused.append(len(hfs_part))
            snew.extend(s_part)
        else:
            t, D = tuple(tD_dict[n].keys()), tuple(tD_dict[n].values())
            dec = tuple((_DecRecord(tt, (0, DD), DD, (DD,)),) for tt, DD in zip(t, D))
            lls.append(_LegSlices(t, D, dec))
            hfs_new.append(hf)
            snew.append(struct.s[n])

    meta, nsym = [], config.sym.NSYM
    for to, slo, Do in zip(struct.t, slices, struct.D):
        ind = tuple(ls.t.index(to[n * nsym: (n + 1) * nsym]) for n, ls in enumerate(lls))
        decs = tuple(tuple(ls.dec[ii]) for ls, ii in zip(lls, ind))
        for tt in product(*decs):
            tn = sum((x.t for x in tt), ())
            sub_slc = tuple(x.Dslc for x in tt)
            Dn = sum((x.Drsh for x in tt), ())
            Dsln = tuple(x.Dprod for x in tt)
            meta.append((tn, Dn, Dsln, slo.slcs[0], Do, sub_slc))

    meta = sorted(meta, key=lambda x: x[0])
    tnew = tuple(x[0] for x in meta)
    Dnew = tuple(x[1] for x in meta)
    Dpnew = tuple(np.prod(np.array(Dnew, dtype=int).reshape(len(Dnew), len(snew)), axis=1, dtype=int))
    struct_new = struct._replace(s=tuple(snew), t=tnew, D=Dnew, size=sum(Dpnew))
    slices_new = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(np.cumsum(Dpnew), Dpnew, Dnew))
    meta = tuple((x.slcs[0], *y[2:]) for x, y in zip(slices_new, meta))
    return meta, struct_new, slices_new, tuple(nlegs_unfused), tuple(hfs_new)


@lru_cache(maxsize=1024)
def _meta_unmerge_matrix(config, struct, slices, ls0, ls1, snew):
    meta, nsym = [], config.sym.NSYM
    for to, slo, Do in zip(struct.t, slices, struct.D):
        ind0 = ls0.t.index(to[:nsym])
        ind1 = ls1.t.index(to[nsym:])
        for d0, d1 in product(ls0.dec[ind0], ls1.dec[ind1]):
            tn = d0.t + d1.t
            sub_slc = (d0.Dslc, d1.Dslc)
            Dn = d0.Drsh + d1.Drsh
            Dsln = (d0.Dprod, d1.Dprod)
            Dp = d0.Dprod * d1.Dprod
            meta.append((tn, Dn, Dp, Dsln, slo.slcs[0], Do, sub_slc))

    meta = sorted(meta, key=lambda x: x[0])
    tnew = tuple(x[0] for x in meta)
    Dnew = tuple(x[1] for x in meta)
    Dpnew = tuple(x[2] for x in meta)
    struct_new = struct._replace(s=tuple(snew), t=tnew, D=Dnew, size=sum(Dpnew))
    slices_new = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(np.cumsum(Dpnew), Dpnew, Dnew))
    meta = tuple((x.slcs[0], *y[3:]) for x, y in zip(slices_new, meta))
    return meta, struct_new, slices_new


#  =========== masks ======================

# def _masks_for_axes(config, structa, hfa, axa, structb, hfb, axb, tcon):
#     """ masks to get the intersecting parts of single legs from two tensors. """
#     msk_a, msk_b = [], []
#     tla, Dla, _, _, _ = _get_tD_legs(structa)
#     tlb, Dlb, _, _, _ = _get_tD_legs(structb)
#     for i1, i2 in zip(axa, axb):
#         ma, mb = _intersect_hfs(config, (tla[i1], tlb[i2]), (Dla[i1], Dlb[i2]), (hfa[i1], hfb[i2]))
#         msk_a.append(ma)
#         msk_b.append(mb)
#     nsym = config.sym.NSYM
#     mam = {t: config.backend.to_mask(_outer_masks(t, msk_a, nsym)) for t in tcon}
#     mbm = {t: config.backend.to_mask(_outer_masks(t, msk_b, nsym)) for t in tcon}
#     return mam, mbm


def _masks_for_tensordot(config, structa, hfa, axa, lsa, structb, hfb, axb, lsb):
    """ masks to get the intersecting parts of legs from two tensors as single masks """
    msk_a, msk_b = [], []
    tla, Dla, _, _, _ = _get_tD_legs(structa)
    tlb, Dlb, _, _, _ = _get_tD_legs(structb)
    for i1, i2 in zip(axa, axb):
        ma, mb = _intersect_hfs(config, (tla[i1], tlb[i2]), (Dla[i1], Dlb[i2]), (hfa[i1], hfb[i2]))
        msk_a.append(ma)
        msk_b.append(mb)
    msk_a = _merge_masks_outer(config, lsa, msk_a)
    msk_b = _merge_masks_outer(config, lsb, msk_b)
    for t, x in msk_a.items():
        msk_a[t] = config.backend.to_mask(x)
    for t, x in msk_b.items():
        msk_b[t] = config.backend.to_mask(x)
    return msk_a, msk_b


def _merge_masks_intersect(config, struct, slices, ms):
    """ combine masks using information from struct"""
    msk = np.ones((struct.size,), dtype=bool)
    nsym = config.sym.NSYM
    Dnew, slices_new, low, high = [], [], 0, 0
    for t, slo in zip(struct.t, slices):
        x = ms[0][t[:nsym]]
        Dt = [np.sum(x)]
        for i in range(1, len(ms)):
            xi = ms[i][t[i * nsym: (i + 1) * nsym]]
            Dt.append(np.sum(xi))
            x = np.outer(x, xi).ravel()
        msk[slice(*slo.slcs[0])] = x
        Dnew.append(tuple(Dt))
        Dp = np.prod(Dnew[-1])
        high = low + Dp
        slices_new.append(_slc(((low, high),), Dnew[-1], Dp))
        low = high
    struct_new = struct._replace(D=tuple(Dnew), size=high)
    return msk, struct_new, tuple(slices_new)


def _masks_for_vdot(config, structa, slicesa, hfa, structb, slicesb, hfb):
    """ masks to get the intersecting parts on all legs from two tensors """
    msk_a, msk_b = [], []
    tla, Dla, _, _, _ = _get_tD_legs(structa)
    tlb, Dlb, _, _, _ = _get_tD_legs(structb)
    ndim = len(tla)
    for n in range(ndim):
        ma, mb = _intersect_hfs(config, (tla[n], tlb[n]), (Dla[n], Dlb[n]), (hfa[n], hfb[n]))
        msk_a.append(ma)
        msk_b.append(mb)
    msk_a, struct_a, slices_a = _merge_masks_intersect(config, structa, slicesa, msk_a)
    msk_b, struct_b, slices_b = _merge_masks_intersect(config, structb, slicesb, msk_b)
    msk_a = config.backend.to_mask(msk_a)
    msk_b = config.backend.to_mask(msk_b)
    return msk_a, msk_b, struct_a, slices_a, struct_b, slices_b


def _masks_for_trace(config, t12, D1, D2, hfs, ax1, ax2):
    """ masks to get the intersecting part of a combination of legs. """
    nsym = config.sym.NSYM
    msk1, msk2 = [], []

    tDDset = set(zip(t12, D1, D2))
    tset = set(t12)

    for n, (i1, i2) in enumerate(zip(ax1, ax2)):
        tdn = tuple(set((t[n * nsym: (n + 1) * nsym], d1[n], d2[n]) for t, d1, d2 in tDDset))
        tn, D1n, D2n = zip(*tdn) if len(tdn) > 0 else ((), (), ())
        m1, m2 = _intersect_hfs(config, (tn, tn), (D1n, D2n), (hfs[i1], hfs[i2]))
        msk1.append(m1)
        msk2.append(m2)

    msk12 = {}
    for t in tset:
        m1 = msk1[0][t[:nsym]]
        m2 = msk2[0][t[:nsym]]
        for n in range(1, len(msk1)):
            ind = t[n * nsym: (n + 1) * nsym]
            m1 = np.outer(m1, msk1[n][ind]).ravel()
            m2 = np.outer(m2, msk2[n][ind]).ravel()
        msk12[t] = (config.backend.to_mask(m1), config.backend.to_mask(m2))
    return msk12


def _merge_masks_embed(config, struct, slices, ms):  # TODO
    """ combine masks using information from struct"""
    nsym = config.sym.NSYM
    msk = []
    Dnew, slices_new, low, high = [], [], 0, 0
    for t in struct.t:
        x = ms[0][t[:nsym]]
        Dt = [len(x)]
        for i in range(1, len(ms)):
            xi = ms[i][t[i * nsym: (i + 1) * nsym]]
            Dt.append(len(xi))
            x = np.outer(x, xi).ravel()
        msk.append(x)
        Dnew.append(tuple(Dt))
        Dp = np.prod(Dnew[-1])
        high = low + Dp
        slices_new.append(_slc(((low, high),), Dnew[-1], Dp))
        low = high
    struct_new = struct._replace(D=tuple(Dnew), size=high)
    msk = np.hstack(msk) if len(msk) > 0 else np.zeros((0,), dtype=bool)
    return msk, struct_new, tuple(slices_new)


def _masks_for_add(config, structa, slicesa, hfa, structb, slicesb, hfb):
    msk_a, msk_b, hfs = [], [], []
    tla, Dla, _, _, _ = _get_tD_legs(structa)
    tlb, Dlb, _, _, _ = _get_tD_legs(structb)
    for n in range(len(structa.s)):
        ma, mb, hf = _union_hfs(config, (tla[n], tlb[n]), (Dla[n], Dlb[n]), (hfa[n], hfb[n]))
        msk_a.append(ma)
        msk_b.append(mb)
        hfs.append(hf)
    msk_a, struct_a, slices_a = _merge_masks_embed(config, structa, slicesa, msk_a)
    msk_b, struct_b, slices_b = _merge_masks_embed(config, structb, slicesb, msk_b)
    msk_a = config.backend.to_mask(msk_a)
    msk_b = config.backend.to_mask(msk_b)
    return msk_a, msk_b, struct_a, slices_a, struct_b, slices_b, tuple(hfs)


def _embed_tensor(a, legs, legs_new):
    """
    Embed tensor to fill in zero block in fusion mismatch.
    here legs are contained in legs_new that result from leg_union
    legs_new is a dict = {n: leg}
    """

    legs_new = [legs_new[n] if n in legs_new else legs[n] for n in range(len(legs))]
    legs, _ = _unpack_legs(legs)
    legs_new, _ = _unpack_legs(legs_new)

    msk_a, hfs = [], []
    for la, lb in zip(legs, legs_new):
        ma, mb, hf = _union_hfs(a.config, (la.t, lb.t), (la.D, lb.D), (la.legs[0], lb.legs[0]))
        msk_a.append(ma)
        hfs.append(hf)
        assert all(sum(m) == m.size for m in mb.values()), "here legs_new should contain legs"
        assert hf == lb.legs[0], "here legs_new should contain legs"
    msk_a, struct_a, slices_a = _merge_masks_embed(a.config, a.struct, a.slices, msk_a)
    data = a.config.backend.embed_msk(a._data, msk_a, struct_a.size)
    return a._replace(struct=struct_a, slices=slices_a, data=data, hfs=tuple(hfs))


#  =========== auxliary functions handling fusion logic ======================

@lru_cache(maxsize=1024)
def _leg_structure_combine_charges_prod(sym, t_in, D_in, s_in, t_out, s_out):
    """ Combine effective charges and dimensions from a list of charges and dimensions for a few legs. """
    comb_t = tuple(product(*t_in))
    comb_t = np.array(comb_t, dtype=int).reshape((len(comb_t), len(s_in), sym.NSYM))
    comb_D = tuple(product(*D_in))
    comb_D = np.array(comb_D, dtype=int).reshape((len(comb_D), len(s_in)))
    teff = sym.fuse(comb_t, s_in, s_out)
    ind = np.array([ii for ii, te in enumerate(teff) if tuple(te.flat) in t_out], dtype=int)
    comb_D, comb_t, teff = comb_D[ind], comb_t[ind], teff[ind]
    Deff = tuple(np.prod(comb_D, axis=1, dtype=int))
    Dlegs = tuple(tuple(x.flat) for x in comb_D)
    teff = tuple(tuple(x.flat) for x in teff)
    tlegs = tuple(tuple(x.flat) for x in comb_t)
    return _leg_structure_merge(teff, tlegs, Deff, Dlegs)


def _leg_structure_combine_charges_sum(t_in, D_in, pos=None):
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
    """ LegDecomposition for merging into a single leg. """
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


def _fuse_hfs(hfs, t_in, D_in, s_out, axes=None):
    """ Fuse _Fusion(s), including charges and dimensions present on the fused legs. """
    if axes is None:
        axes = list(range(len(hfs)))
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


def _sum_hfs(hfs, t_in, D_in, s_out):
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
        else:  #  hf.op[0] == 's':
            ds = 1
        tfl.extend(hf.t)
        Dfl.extend(hf.D)
        sfl.extend(hf.s[ds:])
        treefl.extend(hf.tree[ds:])
        opfl += hf.op[ds:]
    return _Fusion(tree=tuple(treefl), op=opfl, s=tuple(sfl), t=tuple(tfl), D=tuple(Dfl))


def _merge_masks_outer(config, ls, ms):
    """ combine masks using information from LegDec; perform product of spaces / leg fusion. """
    msk = {tt: np.ones(Dt, dtype=bool) for tt, Dt in zip(ls.t, ls.D)}
    nsym = config.sym.NSYM
    for tt, dect in zip(ls.t, ls.dec):
        for rec in dect:
            msk[tt][slice(*rec.Dslc)] = _outer_masks(rec.t, ms, nsym)
    return msk


def _merge_masks_sum(ls, ms):
    """ combine masks using information from LegDec; perform sum of spaces / blocking of legs."""
    msk = {tt: np.ones(Dt, dtype=bool) for tt, Dt in zip(ls.t, ls.D)}
    for tt, dect in zip(ls.t, ls.dec):
        for rec in dect:
            msk[tt][slice(*rec.Dslc)] = ms[rec.t][tt]
    return msk


def _outer_masks(t, ms, nsym):
    """ Outer product of masks for different axes, mask of reshaped block. """
    x = ms[0][t[:nsym]]
    for i in range(1, len(ms)):
        x = np.outer(x, ms[i][t[i * nsym: (i + 1) * nsym]]).ravel()
    return x


def _mask_falsify_mismatches(ms1, ms2):
    """ Multiply masks by False for indices that are not in both dictionaries ms1 and ms2. """
    set1, set2 = set(ms1), set(ms2)
    for t in set1 - set2:
        ms1[t] *= False
    for t in set2 - set1:
        ms2[t] *= False
    return ms1, ms2


@lru_cache(maxsize=1024)
def _intersect_hfs(config, ts, Ds, hfs):
    """
    Returns mask1 and mask2, finding common leg indices for each teff.
    Consumes fusion trees from the bottom, identifying common elements and building the masks.
    """
    teff = tuple(sorted((set(ts[0]) & set(ts[1]))))
    tree = list(hfs[0].tree)

    if len(tree) == 1:
        msks = [[{t: np.ones(D, dtype=bool) for t, D in zip(ts[n], Ds[n]) if t in teff}] for n in (0, 1)]
    else:
        msks = [[{t: np.ones(D, dtype=bool) for t, D in zip(hf.t[i], hf.D[i])}
                    for i, l in enumerate(tree[1:]) if l == 1] for hf in hfs]

    for ms1, ms2 in zip(*msks):
        for t in set(ms1) & set(ms2):
            if ms1[t].size != ms2[t].size:
                if len(tree) == 1:
                    raise YastnError('Bond dimensions do not match.')
                raise YastnError('Bond dimensions of fused legs do not match.')
        _mask_falsify_mismatches(ms1, ms2)

    if len(tree) == 1:
        return msks[0].pop(), msks[1].pop()

    op = list(hfs[0].op) # to be consumed during parsing of the tree
    s = [list(hf.s) for hf in hfs]  # to be consumed during parsing of the tree
    t = [[teff] + list(hf.t) for hf in hfs]  # to be consumed during parsing of the tree
    D = [[()] + list(hf.D) for hf in hfs]  # to be consumed during parsing of the tree
    while len(tree) > 1:
        leg, nlegs, count, parents, ltree = 0, 0, 1, [], -1
        for cnt in tree:  # partisng tree to search for a group of legs to merge
            ltree += 1
            if cnt > 1:
                nlegs, count = 0, cnt
                parents.append(ltree)
            else:
                leg, nlegs, count = leg + 1, nlegs + 1, count - 1
            if count == 0:
                break
        del op[ltree - nlegs + 1: ltree + 1]  # remove leafs
        del tree[ltree - nlegs + 1: ltree + 1]  # remove leafs
        for i in parents:
            tree[i] -= nlegs - 1
        ss = [tuple(s1.pop(ltree - nlegs + 1) for _ in range(nlegs)) for s1 in s]
        tt = [tuple(t1.pop(ltree - nlegs + 1) for _ in range(nlegs)) for t1 in t]
        DD = [tuple(D1.pop(ltree - nlegs + 1) for _ in range(nlegs)) for D1 in D]
        mss = [[msk.pop(leg - nlegs) for _ in range(nlegs)] for msk in msks]
        if op[ltree - nlegs] == 'p':
            lss = [_leg_structure_combine_charges_prod(config.sym, tt1, DD1, ss1, t1[ltree - nlegs], s1[ltree - nlegs])
                   for tt1, DD1, ss1, t1, s1 in zip(tt, DD, ss, t, s)]
            ma = [_merge_masks_outer(config, ls1, ms1) for ls1, ms1 in zip(lss, mss)]
        else: # op[ltree - nlegs] == 's':
            lss = [_leg_structure_combine_charges_sum(tt1, DD1) for tt1, DD1, in zip(tt, DD)]
            ma = [_merge_masks_sum(ls1, ms1) for ls1, ms1 in zip(lss, mss)]
        _mask_falsify_mismatches(ma[0], ma[1])
        msks[0].insert(leg - nlegs, ma[0])
        msks[1].insert(leg - nlegs, ma[1])
    return msks[0].pop(), msks[1].pop()


def _union_hfs(config, ts, Ds, hfs):
    """
    Returns mask1 and mask2 that can be applied on the union of the two fusions
    to map the contribution of each leg to the union.
    Consumes fusion trees from the bottom, building the union
    and identifying contribution of each to the union.
    """
    tree = list(hfs[0].tree)
    s = list(hfs[0].s)  # to be consumed during parsing of the tree
    op = list(hfs[0].op) # to be consumed during parsing of the tree

    if len(tree) == 1:
        hfu = _Fusion(s=hfs[0].s)
        msk1 = {t: np.ones(D, dtype=bool) for t, D in zip(ts[0], Ds[0])}
        msk2 = {t: np.ones(D, dtype=bool) for t, D in zip(ts[1], Ds[1])}
        if any(msk1[t].size != msk2[t].size for t in set(msk1) & set(msk2)):
            raise YastnError('Bond dimensions do not match.')
        return msk1, msk2, hfu

    ind_native = [i for i, l in enumerate(tree[1:]) if l == 1]
    msk1, msk2, hfu, tu, Du = [], [], [], [], []
    for i in ind_native:
        tD1 = dict(zip(hfs[0].t[i], hfs[0].D[i]))
        tD2 = dict(zip(hfs[1].t[i], hfs[1].D[i]))
        if any(tD1[t] != tD2[t] for t in set(tD1) & set(tD2)):
            raise YastnError('Bond dimensions of fused legs do not match.')
        tD12 = {**tD1, **tD2}
        tu.append(tuple(sorted(tD12)))
        Du.append(tuple(tD12[t] for t in tu[-1]))
        msk1.append({t: np.ones(D, dtype=bool) if t in tD1 else np.zeros(D, dtype=bool) for t, D in zip(tu[-1], Du[-1])})
        msk2.append({t: np.ones(D, dtype=bool) if t in tD2 else np.zeros(D, dtype=bool) for t, D in zip(tu[-1], Du[-1])})
        hfu.append(_Fusion(s=(s[i + 1],)))  # len(s) == 1 + len(t)

    teff = tuple(sorted(set(ts[0]) | set(ts[1])))
    t1 = [teff] + list(hfs[0].t)  # to be consumed during parsing of the tree
    t2 = [teff] + list(hfs[1].t)
    while len(tree) > 1:
        leg, nlegs, count, parents, ltree = 0, 0, 1, [], -1
        for cnt in tree:  # partisng tree to search for a group of legs to merge
            ltree += 1
            if cnt > 1:
                nlegs, count = 0, cnt
                parents.append(ltree)
            else:
                leg, nlegs, count = leg + 1, nlegs + 1, count - 1
            if count == 0:
                break
        del op[ltree - nlegs + 1: ltree + 1]  # remove leafs
        del tree[ltree - nlegs + 1: ltree + 1]
        del t1[ltree - nlegs + 1: ltree + 1]
        del t2[ltree - nlegs + 1: ltree + 1]
        for i in parents:
            tree[i] -= nlegs - 1
        s_in = tuple(s.pop(ltree - nlegs + 1) for _ in range(nlegs))
        t_in = tuple(tu.pop(leg - nlegs) for _ in range(nlegs))
        D_in = tuple(Du.pop(leg - nlegs) for _ in range(nlegs))
        t_out = tuple(sorted(set(t1[ltree - nlegs]) | set(t2[ltree - nlegs])))
        hh = [hfu.pop(leg - nlegs) for _ in range(nlegs)]
        ms1_in = [msk1.pop(leg - nlegs) for _ in range(nlegs)]
        ms2_in = [msk2.pop(leg - nlegs) for _ in range(nlegs)]
        s_out = s[ltree - nlegs]

        if op[ltree - nlegs] == 'p':
            ls = _leg_structure_combine_charges_prod(config.sym, t_in, D_in, s_in, t_out, s_out)
            ma1 = _merge_masks_outer(config, ls, ms1_in)
            ma2 = _merge_masks_outer(config, ls, ms2_in)
            hfu.insert(leg - nlegs, _fuse_hfs(hh, t_in, D_in, s_out))
        else:  # op[ltree - nlegs] == 's':
            ls = _leg_structure_combine_charges_sum(t_in, D_in)
            ma1 = _merge_masks_sum(ls, ms1_in)
            ma2 = _merge_masks_sum(ls, ms2_in)
            hfu.insert(leg - nlegs, _sum_hfs(hh, t_in, D_in, s_out))
        for ind in ma1.keys() - set(t1[ltree - nlegs]):
            ma1[ind] *= False
        for ind in ma2.keys() - set(t2[ltree - nlegs]):
            ma2[ind] *= False
        msk1.insert(leg - nlegs, ma1)
        msk2.insert(leg - nlegs, ma2)
        tu.insert(leg - nlegs, ls.t)
        Du.insert(leg - nlegs, ls.D)
    return msk1.pop(), msk2.pop(), hfu.pop()


def _pure_hfs_union(sym, ts, hfs):
    """
    Consumes fusion trees from the bottom, building the union.
    """
    tree = list(hfs[0].tree)
    s = list(hfs[0].s)  # to be consumed during parsing of the tree
    op = list(hfs[0].op) # to be consumed during parsing of the tree

    if any(hfs[0].tree != hf.tree or hfs[0].op != hf.op for hf in hfs):
        raise YastnError("Inconsistent numbers of hard-fused legs or sub-fusions order.")
    if any(hfs[0].s != hf.s for hf in hfs):
        raise YastnError("Inconsistent signatures of fused legs.")
    ind_native = [i for i, l in enumerate(tree[1:]) if l == 1]

    hfu, tu, Du = [], [], []
    for i in ind_native:
        tDs = [dict(zip(hf.t[i], hf.D[i])) for hf in hfs]
        alltD = {}
        for tD in tDs:
            alltD.update(tD)
        if any(alltD[t] != D for tD in tDs for t, D in tD.items()):
            raise YastnError('Bond dimensions of fused legs do not match.')
        tu.append(tuple(sorted(alltD)))
        Du.append(tuple(alltD[t] for t in tu[-1]))
        hfu.append(_Fusion(s=(s[i + 1],)))  # len(s) == 1 + len(t)

    teff = tuple(sorted(set.union(*(set(x) for x in ts))))
    tss = [[teff] + list(hf.t) for hf in hfs]
    while len(tree) > 1:
        leg, nlegs, count, parents, ltree = 0, 0, 1, [], -1
        for cnt in tree:  # partisng tree to search for a group of legs to merge
            ltree += 1
            if cnt > 1:
                nlegs, count = 0, cnt
                parents.append(ltree)
            else:
                leg, nlegs, count = leg + 1, nlegs + 1, count - 1
            if count == 0:
                break
        del op[ltree - nlegs + 1: ltree + 1]  # remove leafs
        del tree[ltree - nlegs + 1: ltree + 1]
        for ti in tss:
            del ti[ltree - nlegs + 1: ltree + 1]
        for i in parents:
            tree[i] -= nlegs - 1
        s_in = tuple(s.pop(ltree - nlegs + 1) for _ in range(nlegs))
        t_in = tuple(tu.pop(leg - nlegs) for _ in range(nlegs))
        D_in = tuple(Du.pop(leg - nlegs) for _ in range(nlegs))
        t_out = tuple(sorted(set.union(*(set(x[ltree - nlegs]) for x in tss))))
        s_out = s[ltree - nlegs]
        hh = [hfu.pop(leg - nlegs) for _ in range(nlegs)]
        if op[ltree - nlegs] == 'p':
            ls = _leg_structure_combine_charges_prod(sym, t_in, D_in, s_in, t_out, s_out)
            hfu.insert(leg - nlegs, _fuse_hfs(hh, t_in, D_in, s_out))
        else:  # op[ltree - nlegs] == 's':
            ls = _leg_structure_combine_charges_sum(t_in, D_in)
            hfu.insert(leg - nlegs, _sum_hfs(hh, t_in, D_in, s_out))
        tu.insert(leg - nlegs, ls.t)
        Du.insert(leg - nlegs, ls.D)
    return tu.pop(), Du.pop(), hfu.pop()


def _unfuse_Fusion(hf):
    """ one layer of unfuse """
    tt, DD, ss, hfs = [], [], [], []
    n_init, cum = 1, 0
    for n in range(1, len(hf.tree)):
        if cum == 0:
            cum = hf.tree[n]
        if hf.tree[n] == 1:
            cum = cum - 1
            if cum == 0:
                tt.append(hf.t[n_init - 1])
                DD.append(hf.D[n_init - 1])
                ss.append(hf.s[n_init])
                hfs.append(_Fusion(tree=hf.tree[n_init: n + 1], op=hf.op[n_init: n + 1],
                                   s=hf.s[n_init: n + 1], t=hf.t[n_init: n], D=hf.D[n_init: n]))
                n_init = n + 1
    return tuple(tt), tuple(DD), tuple(ss), hfs


def _consume_mfs_lowest(mfs):
    """
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
    """ Eliminates lowest possible layer of merges """
    if all(len(x) == 0 for x in ntree):
        ntree.clear()
    else:
        for x in ntree:
            _ntree_eliminate_lowest(x)
