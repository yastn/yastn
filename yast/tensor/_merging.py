""" Support for merging blocks in yast tensors. """

from functools import lru_cache
from itertools import groupby, product
from operator import itemgetter
from typing import NamedTuple
import numpy as np
from ._auxliary import _flatten, _clear_axes, _ntree_to_mf, _mf_to_ntree
from ._tests import YastError, _test_axes_all, _get_tD_legs


__all__ = ['fuse_legs', 'unfuse_legs', 'fuse_meta_to_hard']


class _LegDec(NamedTuple):
    """ Internal structure of leg resulting from fusions, including slices"""
    dec: dict = None  # decomposition of each effective charge
    Dtot: dict = None  # bond dimensions of effective charges


class _DecRec(NamedTuple):
    """ Single record in _LegDec.dec[effective charge]"""
    Dslc: tuple = (None, None)  # slice
    Dprod: int = 0  # size of slice, equal to product of Drsh
    Drsh: tuple = ()  # original shape of fused dims in a block


class _Fusion(NamedTuple):
    """ Information identifying the structure of hard fusion"""
    tree: tuple = (1,)  # order of fusions
    s: tuple = (1,)  # signatures len(s) = len(tree)
    t: tuple = ()  # fused leg charges at each step len(t) = len(tree) - 1
    D: tuple = ()  # fused dimensions  at each step len(t) = len(tree) - 1


#  =========== merging blocks ======================

def _merge_to_matrix(a, axes, s_eff, inds=None, sort_r=False):
    """ Main function merging tensor into effective block matrix. """
    order = axes[0] + axes[1]
    struct, meta_mrg, ls_l, ls_r = _meta_merge_to_matrix(a.config, a.struct, axes, s_eff, inds, sort_r)
    meta_1d = tuple(sorted(zip(struct.t, struct.D, struct.sl)))
    Dsize = struct.sl[-1][1] if len(struct.sl) > 0 else 0
    newdata = a.config.backend.merge_to_1d(a._data, order, meta_1d, meta_mrg, Dsize)
    return newdata, struct, ls_l, ls_r


@lru_cache(maxsize=1024)
def _meta_merge_to_matrix(config, struct, axes, s_eff, inds, sort_r):
    """ Meta information for backend needed to merge tensor into effective block matrix. """
    told = struct.t if inds is None else [struct.t[ii] for ii in inds]
    Dold = struct.D if inds is None else [struct.D[ii] for ii in inds]
    slold = struct.sl if inds is None else [struct.sl[ii] for ii in inds]
    tset = np.array(told, dtype=int).reshape((len(told), len(struct.s), config.sym.NSYM))
    Dset = np.array(Dold, dtype=int).reshape(len(Dold), len(struct.s))
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

    tnew = tuple(t1 + t2 for t1, t2 in zip(teff[0], teff[1]))
    # meta_mrg = ((tnew, told, Dslc, Drsh), ...)
    meta_mrg = tuple(sorted((tn, slo, Do,
                             (ls[0].dec[tel][tl].Dslc, ls[1].dec[ter][tr].Dslc),
                             (ls[0].dec[tel][tl].Dprod, ls[1].dec[ter][tr].Dprod))
                    for tn, slo, Do, tel, tl, ter, tr in zip(tnew, slold, Dold, teff[0], t[0], teff[1], t[1])))
    if sort_r:
        unew_r, unew_l, unew = zip(*sorted(set(zip(teff[1], teff[0], tnew)))) if len(tnew) > 0 else ((), (), ())
    else:
        unew, unew_l, unew_r = zip(*sorted(set(zip(tnew, teff[0], teff[1])))) if len(tnew) > 0 else ((), (), ())
    # meta_new = ((unew, Dnew), ...)
    D_new = tuple((ls[0].Dtot[il], ls[1].Dtot[ir]) for il, ir in zip(unew_l, unew_r))
    Dp_new = tuple(x[0] * x[1] for x in D_new)
    sl_new = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(Dp_new), Dp_new))
    struct_new = struct._replace(t=unew, D=D_new, Dp=Dp_new, sl=sl_new, s=tuple(s_eff))
    return struct_new, meta_mrg, ls[0], ls[1]


def _leg_struct_trivial(struct, axis=0):
    """ trivial LegDecomposition for unfused leg. """
    nsym = len(struct.n)
    dec, Dtot = {}, {}
    for tt, DD in zip(struct.t, struct.D):
        t = tt[nsym * axis: nsym * (axis + 1)]
        D = DD[axis]
        dec[t] = {t: _DecRec((0, D), D, (D,))}
        Dtot[t] = D
    return _LegDec(dec, Dtot)


def _leg_struct_trivial2(config, Am, axis=0):
    """ trivial LegDecomposition for unfused leg. """
    nsym = config.sym.NSYM
    dec, Dtot = {}, {}
    for ind, val in Am.items():
        t = ind[nsym * axis: nsym * (axis + 1)]
        D = config.backend.get_shape(val)[axis]
        dec[t] = {t: _DecRec((0, D), D, (D,))}
        Dtot[t] = D
    return _LegDec(dec, Dtot)


def _leg_struct_truncation(config, Sdata, St, Ssl,
                            tol=0., tol_block=0, D_block=np.inf, D_total=np.inf,
                            keep_multiplets=False, eps_multiplet=1e-12, ordering='eigh'):
    r"""
    Gives slices for truncation of 1d matrices according to tolerance, D_block, D_total.

    A should be dict of ordered 1d arrays.
    Sorting gives information about ordering outputed by a particular splitting funcion:
    Usual convention is that for svd A[ind][0] is largest; and for eigh A[ind][-1] is largest.
    """
    maxS = 0 if len(Sdata) == 0 else config.backend.max_abs(Sdata)
    Dmax, D_keep = {}, {}

    nsym = config.sym.NSYM
    St = tuple(x[:nsym] for x in St)

    for t, sl, in zip(St, Ssl):
        Dmax[t] = sl[1] - sl[0]
        D_keep[t] = min(D_block, Dmax[t])

    if (tol > 0) and (maxS > 0):  # truncate to relative tolerance
        for t, sl, in zip(St, Ssl):
            local_maxS = config.backend.max_abs(Sdata[slice(*sl)])
            local_tol = max(local_maxS * tol_block, maxS * tol) if tol_block > 0 else maxS * tol
            D_keep[t] = min(D_keep[t], config.backend.count_greater(Sdata[slice(*sl)], local_tol))

    if sum(D_keep.values()) > D_total:  # truncate to total bond dimension
        order = config.backend.select_global_largest(Sdata, St, Ssl, D_keep, D_total, keep_multiplets, eps_multiplet, ordering)
        low = 0
        for ind, D_ind in D_keep.items():
            high = low + D_ind
            D_keep[ind] = sum((low <= order) & (order < high)).item()
            low = high
    if keep_multiplets:  # check symmetry related blocks and truncate to equal length
        for t in D_keep:
            tn = np.array(t, dtype=int).reshape((1, 1, -1))
            tn = tuple(config.sym.fuse(tn, np.array([1], dtype=int), -1)[0])
            minD_sector = min(D_keep[t], D_keep[tn])
            D_keep[t] = D_keep[tn] = minD_sector
    dec, Dtot = {}, {}
    for ind, D_ind in D_keep.items():
        if D_ind > 0:
            Dslc = config.backend.range_largest(D_ind, Dmax[ind], ordering)
            dec[ind] = {ind: _DecRec(Dslc, D_ind, (D_ind,))}
            Dtot[ind] = D_ind
    return _LegDec(dec, Dtot)

#  =========== fuse legs ======================

def fuse_legs(a, axes, mode=None):
    r"""
    Fuse groups of legs into effective legs, reducing the rank of the tensor.
    
        .. note::
            Fusion can be reverted back by :meth:`yast.Tensor.unfuse_legs`

    First, the legs are permuted into desired order. Then, selected groups of consecutive legs 
    are fused. The desired order of the legs is given by a tuple `axes`
    of leg indices where the groups of legs to be fused are denoted by inner tuples ::

        axes=(0,1,(2,3,4),5)  keep order, fuse legs (2,3,4) into new leg
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

    * `meta` performs the fusion only at the level of tensor structure: changing its rank,
      signature, charge sectors. The tensor data (blocks) is not affected. 

    * `hard` changes both the structure and data, by aggregating smaller blocks into larger
      ones. Such fusion allows to balance number of non-zero blocks and typical block size.

    Parameters
    ----------
    axes: tuple(tuple(int))
        tuple of leg's indices. Groups of legs to be fused together are accumulated within inner tuples.

    mode: str
        can select 'hard' or 'meta' fusion. If None, use default from config.
        It can also be overriden by config.force_fuse.
        Applying hard fusion of tensor with meta fusion
        first replaces meta fusion with hard fusion.
        Meta fusion can be applied on top of hard fusion.

    Returns
    -------
    tensor : Tensor
    """
    if a.isdiag:
        raise YastError('Cannot fuse legs of a diagonal tensor.')
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
    raise YastError('mode not in (`meta`, `hard`). Mode can be specified in config file.')


def _fuse_legs_hard(a, axes, order):
    """ Funtion performing hard fusion. axes are for native legs and are cleaned outside."""
    assert all(isinstance(x, tuple) for x in axes)
    for x in axes:
        assert all(isinstance(y, int) for y in x)

    struct, meta_mrg, t_in, D_in = _meta_fuse_hard(a.config, a.struct, axes)
    meta_new = tuple(zip(struct.t, struct.D, struct.sl))
    Dsize = struct.sl[-1][1] if len(struct.sl) > 0 else 0

    mfs = ((1,),) * len(struct.s)
    hfs = tuple(_fuse_hfs(a.hfs, t_in, D_in, struct.s[n], axis) if len(axis) > 1 else a.hfs[axis[0]]
                for n, axis in enumerate(axes))
    data = a.config.backend.merge_to_1d(a._data, order, meta_new, meta_mrg, Dsize)
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, data=data)


@lru_cache(maxsize=1024)
def _meta_fuse_hard(config, struct, axes):
    """ Meta information for backend needed to hard-fuse some legs. """
    nblocks, nsym = len(struct.t), len(struct.n)
    t_in, D_in, tD_dict, tset, Dset = _get_tD_legs(struct)
    slegs = tuple(tuple(struct.s[n] for n in a) for a in axes)
    snew = tuple(struct.s[axis[0]] for axis in axes)
    teff = np.zeros((nblocks, len(snew), nsym), dtype=int)
    Deff = np.zeros((nblocks, len(snew)), dtype=int)
    for n, a in enumerate(axes):
        teff[:, n, :] = config.sym.fuse(tset[:, a, :], slegs[n], snew[n])
        Deff[:, n] = np.prod(Dset[:, a], axis=1, dtype=int)

    ls = []
    for n, a in enumerate(axes):
        if len(a) > 1:
            teff_set = tuple(set(tuple(x.flat) for x in teff[:, n, :]))
            t_a = tuple(t_in[n] for n in a)
            D_a = tuple(D_in[n] for n in a)
            ls.append(_leg_structure_combine_charges(config, t_a, D_a, slegs[n], teff_set, snew[n]))
        else:
            ls.append(_LegDec({t: {t: _DecRec((0, D), D, (D,))} for t, D in tD_dict[a[0]].items()}, tD_dict[a[0]]))

    teff_split = [tuple(tuple(y.flat) for y in x) for x in teff]
    told_split = [tuple(tuple(x[a, :].flat) for a in axes) for x in tset]
    teff = tuple(tuple(x.flat) for x in teff)

    tnew = tuple(sorted(set(teff)))
    ndimnew = len(snew)
    tnew_split = [tuple(x[i * nsym: (i + 1) * nsym] for i in range(ndimnew)) for x in tnew]
    Dnew = tuple(tuple(l.Dtot[y] for l, y in zip(ls, x)) for x in tnew_split)
    Dpnew = np.prod(Dnew, axis=1, dtype=int)
    slnew = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(Dpnew), Dpnew))
    Dpnew = tuple(Dpnew)

    meta_mrg = tuple(sorted(((tn, slo, Do, tuple(l.dec[e][o].Dslc for l, e, o in zip(ls, tes, tos)),
                             tuple(l.dec[e][o].Dprod for l, e, o in zip(ls, tes, tos)))
                             for tn, slo, Do, tes, tos in zip(teff, struct.sl, struct.D, teff_split, told_split)), key=lambda x : x[0]))
    struct_new = struct._replace(t=tnew, D=Dnew, Dp=Dpnew, sl=slnew, s=tuple(snew))
    return struct_new, meta_mrg, t_in, D_in


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
    or `tuple(int)` in case of more legs to be unfused. The unfused legs follow 
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


    This operation can be done in-place.

    Parameters
    ----------
    axis: int or tuple of ints
        leg(s) to unfuse.

    Returns
    -------
    tensor : Tensor
    """
    if a.isdiag:
        raise YastError('Cannot unfuse legs of a diagonal tensor.')
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
        else:  # c.hfs[ni].tree[0] > 1 and c.mfs[mi][0] == 1 and mi in axes
            axes_hf.append(ni)
            mfs.append(a.mfs[mi])
        ni += dni
    if axes_hf:
        meta, struct, nlegs, hfs = _meta_unfuse_hard(a.config, a.struct, tuple(axes_hf), tuple(a.hfs))
        data = a.config.backend.unmerge_from_1d(a._data, meta)
        for unfused, n in zip(nlegs[::-1], axes_hf[::-1]):
            mfs = mfs[:n] + [mfs[n]] * unfused + mfs[n+1:]
        return a._replace(struct=struct, mfs=tuple(mfs), hfs=hfs, data=data)
    return a._replace(mfs=tuple(mfs))


@lru_cache(maxsize=1024)
def _meta_unfuse_hard(config, struct, axes, hfs):
    """ Meta information for backend needed to hard-unfuse some legs. """
    t_in, _, tD_dict, _, _ = _get_tD_legs(struct)
    ls, hfs_new, snew, nlegs_unfused = [], [], [], []
    for n, hf in enumerate(hfs):
        if n in axes:
            t_part, D_part, s_part, hfs_part = _unfuse_Fusion(hf)
            ls.append(_leg_structure_combine_charges(config, t_part, D_part, s_part, t_in[n], struct.s[n]))
            hfs_new.extend(hfs_part)
            nlegs_unfused.append(len(hfs_part))
            snew.extend(s_part)
        else:
            dec = {t: {t: _DecRec((0, D), D, (D,))} for t, D in tD_dict[n].items()}
            ls.append(_LegDec(dec, tD_dict[n]))
            hfs_new.append(hf)
            snew.append(struct.s[n])
    meta, new_struct = _meta_unfuse_legdec(config, struct, ls, snew)
    return meta, new_struct, tuple(nlegs_unfused), tuple(hfs_new)


def _meta_unfuse_legdec(config, struct, ls, snew):
    meta, nsym = [], config.sym.NSYM
    for to, slo, Do in zip(struct.t, struct.sl, struct.D):
        tfused = tuple(to[n * nsym: (n + 1) * nsym] for n in range(len(struct.s)))
        if all(ts in l.dec for l, ts in zip(ls, tfused)):
            tunfused = tuple(tuple(l.dec[ts].items()) for l, ts in zip(ls, tfused))
            for tt in product(*tunfused):
                tn = sum((x[0] for x in tt), ())
                sub_slc = tuple(x[1].Dslc for x in tt)
                Dn = sum((x[1].Drsh for x in tt), ())
                Dsln = tuple(x[1].Dprod for x in tt)
                Dp = np.prod(Dsln, dtype=int)
                meta.append((tn, Dn, Dp, Dsln, slo, Do, sub_slc))

    meta = sorted(meta, key=lambda x: x[0])
    tnew = tuple(x[0] for x in meta)
    Dnew = tuple(x[1] for x in meta)
    Dpnew = tuple(x[2] for x in meta)
    slnew = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(Dpnew), Dpnew))
    meta = tuple((x, *y[3:]) for x, y in zip(slnew, meta))
    if struct.diag:
        meta = tuple((sln, Dslc, slo, Do[0], sub_slc) for sln, Dslc, slo, Do, sub_slc in meta)
        tnew = tuple(t + t for t in tnew)
        Dnew = tuple(D + D for D in Dnew)
    new_struct = struct._replace(s=tuple(snew), t=tnew, D=Dnew, Dp=Dpnew, sl=slnew)
    return meta, new_struct


#  =========== masks ======================

def _masks_for_axes(config, structa, hfa, axa, structb, hfb, axb, tcon):
    """ masks to get the intersecting parts of single legs from two tensors. """
    msk_a, msk_b = [], []
    tla, Dla, _, _, _ = _get_tD_legs(structa)
    tlb, Dlb, _, _, _ = _get_tD_legs(structb)
    for i1, i2 in zip(axa, axb):
        ma, mb = _intersect_hfs(config, (tla[i1], tlb[i2]), (Dla[i1], Dlb[i2]), (hfa[i1], hfb[i2]))
        msk_a.append(ma)
        msk_b.append(mb)
    nsym = config.sym.NSYM
    mam = {t: config.backend.to_mask(_outer_masks(t, msk_a, nsym)) for t in tcon}
    mbm = {t: config.backend.to_mask(_outer_masks(t, msk_b, nsym)) for t in tcon}
    return mam, mbm


def _masks_for_tensordot(config, structa, hfa, axa, lsa, structb, hfb, axb, lsb):
    """ masks to get the intersecting parts of legs from two tensors as single masks """
    msk_a, msk_b = [], []
    tla, Dla, _, _, _ = _get_tD_legs(structa)
    tlb, Dlb, _, _, _ = _get_tD_legs(structb)
    for i1, i2 in zip(axa, axb):
        ma, mb = _intersect_hfs(config, (tla[i1], tlb[i2]), (Dla[i1], Dlb[i2]), (hfa[i1], hfb[i2]))
        msk_a.append(ma)
        msk_b.append(mb)
    msk_a = _merge_masks(config, lsa, msk_a)
    msk_b = _merge_masks(config, lsb, msk_b)
    for t, x in msk_a.items():
        msk_a[t] = config.backend.to_mask(x)
    for t, x in msk_b.items():
        msk_b[t] = config.backend.to_mask(x)
    return msk_a, msk_b


def _merge_masks_intersect(config, struct, ms):
    """ combine masks using information from struct"""
    Dsize = struct.sl[-1][1] if len(struct.sl) > 0 else 0
    msk = np.ones((Dsize,), dtype=bool)
    nsym = config.sym.NSYM
    Dnew, Dpnew, slnew, low, high = [], [], [], 0, 0
    for t, sl in zip(struct.t, struct.sl):
        x = ms[0][t[:nsym]]
        Dt = [np.sum(x)]
        for i in range(1, len(ms)):
            xi = ms[i][t[i * nsym: (i + 1) * nsym]]
            Dt.append(np.sum(xi))
            x = np.outer(x, xi).ravel()
        msk[slice(*sl)] = x
        Dnew.append(tuple(Dt))
        Dpnew.append(np.prod(Dnew[-1]))
        high = low + Dpnew[-1]
        slnew.append((low, high))
        low = high
    structnew = struct._replace(D=tuple(Dnew), Dp=tuple(Dpnew), sl=tuple(slnew))
    return msk, structnew


def _masks_for_vdot(config, structa, hfa, structb, hfb):
    """ masks to get the intersecting parts on all legs from two tensors """
    msk_a, msk_b = [], []
    tla, Dla, _, _, _ = _get_tD_legs(structa)
    tlb, Dlb, _, _, _ = _get_tD_legs(structb)
    ndim = len(tla)
    for n in range(ndim):
        ma, mb = _intersect_hfs(config, (tla[n], tlb[n]), (Dla[n], Dlb[n]), (hfa[n], hfb[n]))
        msk_a.append(ma)
        msk_b.append(mb)
    msk_a, struct_a = _merge_masks_intersect(config, structa, msk_a)
    msk_b, struct_b = _merge_masks_intersect(config, structb, msk_b)
    msk_a = config.backend.to_mask(msk_a)
    msk_b = config.backend.to_mask(msk_b)
    return msk_a, msk_b, struct_a, struct_b


def _masks_for_trace(config, t12, D1, D2, hfs, ax1, ax2):
    """ masks to get the intersecting part of a combination of legs. """
    nsym = config.sym.NSYM
    msk1, msk2 = [], []

    tDDset = set(zip(t12, D1, D2))
    tset = set(t12)
    # if len(tset) != len(tDDset):
    #     raise YastError('CRITICAL ERROR. Bond dimensions of a tensor are inconsistent. This should not have happend.')

    for n, (i1, i2) in enumerate(zip(ax1, ax2)):
        tdn = tuple(set((t[n * nsym: (n + 1) * nsym], d1[n], d2[n]) for t, d1, d2 in tDDset))
        tn, D1n, D2n = zip(*tdn)  # this can triggerexception when tdn is empty -- fix and test in test_trace
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


def _merge_masks_embed(config, struct, ms):
    """ combine masks using information from struct"""
    nsym = config.sym.NSYM
    msk = []
    Dnew, Dpnew, slnew, low, high = [], [], [], 0, 0
    for t in struct.t:
        x = ms[0][t[:nsym]]
        Dt = [len(x)]
        for i in range(1, len(ms)):
            xi = ms[i][t[i * nsym: (i + 1) * nsym]]
            Dt.append(len(xi))
            x = np.outer(x, xi).ravel()
        msk.append(x)
        Dnew.append(tuple(Dt))
        Dpnew.append(np.prod(Dnew[-1]))
        high = low + Dpnew[-1]
        slnew.append((low, high))
        low = high
    structnew = struct._replace(D=tuple(Dnew), Dp=tuple(Dpnew), sl=tuple(slnew))
    msk = np.hstack(msk)
    return msk, structnew


def _masks_for_add(config, structa, hfa, structb, hfb):
    msk_a, msk_b, hfs = [], [], []
    tla, Dla, _, _, _ = _get_tD_legs(structa)
    tlb, Dlb, _, _, _ = _get_tD_legs(structb)
    nsym, ndim = config.sym.NSYM, len(tla)
    for n in range(ndim):
        ma, mb, hf = _union_hfs(config, (tla[n], tlb[n]), (Dla[n], Dlb[n]), (hfa[n], hfb[n]))
        msk_a.append(ma)
        msk_b.append(mb)
        hfs.append(hf)
    msk_a, struct_a = _merge_masks_embed(config, structa, msk_a)
    msk_b, struct_b = _merge_masks_embed(config, structb, msk_b)
    msk_a = config.backend.to_mask(msk_a)
    msk_b = config.backend.to_mask(msk_b)
    return msk_a, msk_b, struct_a, struct_b, tuple(hfs)

#  =========== auxliary functions handling fusion logic ======================

def _flip_hf(x):
    """ _Fusion with fliped signature. """
    return x._replace(s=tuple(-s for s in x.s))


@lru_cache(maxsize=1024)
def _leg_structure_combine_charges(config, t_in, D_in, s_in, t_out, s_out):
    """ Combine effectibe charges and dimensions from a list of charges and dimensions for a few legs. """
    comb_t = tuple(product(*t_in))
    comb_t = np.array(comb_t, dtype=int).reshape((len(comb_t), len(s_in), config.sym.NSYM))
    comb_D = tuple(product(*D_in))
    comb_D = np.array(comb_D, dtype=int).reshape((len(comb_D), len(s_in)))
    teff = config.sym.fuse(comb_t, s_in, s_out)
    ind = np.array([ii for ii, te in enumerate(teff) if tuple(te.flat) in t_out], dtype=int)
    comb_D, comb_t, teff = comb_D[ind], comb_t[ind], teff[ind]
    Deff = tuple(np.prod(comb_D, axis=1, dtype=int))
    Dlegs = tuple(tuple(x.flat) for x in comb_D)
    teff = tuple(tuple(x.flat) for x in teff)
    tlegs = tuple(tuple(x.flat) for x in comb_t)
    return _leg_structure_merge(teff, tlegs, Deff, Dlegs)


def _leg_structure_merge(teff, tlegs, Deff, Dlegs):
    """ LegDecomposition for merging into a single leg. """
    tt = sorted(set(zip(teff, tlegs, Deff, Dlegs)))
    dec, Dtot = {}, {}
    for te, grp in groupby(tt, key=itemgetter(0)):
        Dlow = 0
        dec[te] = {}
        for _, tl, De, Dl in grp:
            Dtop = Dlow + De
            dec[te][tl] = _DecRec((Dlow, Dtop), De, Dl)
            Dlow = Dtop
        Dtot[te] = Dtop
    return _LegDec(dec, Dtot)


def _fuse_hfs(hfs, t_in, D_in, s_out, axis=None):
    """ Fuse _Fusion(s), including charges and dimensions present on the fused legs. """
    if axis is None:
        axis = list(range(len(hfs)))
    tfl, Dfl, sfl = [], [], [s_out]
    treefl = [sum(hfs[n].tree[0] for n in axis)]
    for n in axis:
        tfl.append(t_in[n])
        tfl.extend(hfs[n].t)
        Dfl.append(D_in[n])
        Dfl.extend(hfs[n].D)
        sfl.extend(hfs[n].s)
        treefl.extend(hfs[n].tree)
    return _Fusion(tuple(treefl), tuple(sfl), tuple(tfl), tuple(Dfl))


def _merge_masks(config, ls, ms):
    """ combine masks using information from LegDec"""
    msk = {te: np.ones(D, dtype=bool) for te, D in ls.Dtot.items()}
    nsym = config.sym.NSYM
    for te, dec in ls.dec.items():
        for t, Dr in dec.items():
            msk[te][slice(*Dr.Dslc)] = _outer_masks(t, ms, nsym)
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
                    raise YastError('Bond dimensions do not match.')
                raise YastError('Bond dimensions of fused legs do not match.')
        _mask_falsify_mismatches(ms1, ms2)

    if len(tree) == 1:
        return msks[0].pop(), msks[1].pop()

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
        for _ in range(nlegs):  # remove leafs
            tree.pop(ltree - nlegs + 1)
        for i in parents:
            tree[i] -= nlegs - 1
        ss = [tuple(s1.pop(ltree - nlegs + 1) for _ in range(nlegs)) for s1 in s]
        tt = [tuple(t1.pop(ltree - nlegs + 1) for _ in range(nlegs)) for t1 in t]
        DD = [tuple(D1.pop(ltree - nlegs + 1) for _ in range(nlegs)) for D1 in D]
        lss = [_leg_structure_combine_charges(config, tt1, DD1, ss1, t1[ltree - nlegs], s1[ltree - nlegs])
                for tt1, DD1, ss1, t1, s1 in zip(tt, DD, ss, t, s)]
        mss = [[msk.pop(leg - nlegs) for _ in range(nlegs)] for msk in msks]
        ma = [_merge_masks(config, ls1, ms1) for ls1, ms1 in zip(lss, mss)]
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

    if len(tree) == 1:
        hfu = _Fusion(s=hfs[0].s)
        msk1 = {t: np.ones(D, dtype=bool) for t, D in zip(ts[0], Ds[0])}
        msk2 = {t: np.ones(D, dtype=bool) for t, D in zip(ts[1], Ds[1])}
        if any(msk1[t].size != msk2[t].size for t in set(msk1) & set(msk2)):
            raise YastError('Bond dimensions do not match.')
        return msk1, msk2, hfu

    ind_native = [i for i, l in enumerate(tree[1:]) if l == 1]
    msk1, msk2, hfu, tu, Du = [], [], [], [], []
    for i in ind_native:
        tD1 = dict(zip(hfs[0].t[i], hfs[0].D[i]))
        tD2 = dict(zip(hfs[1].t[i], hfs[1].D[i]))
        if any(tD1[t] != tD2[t] for t in set(tD1) & set(tD2)):
            raise YastError('Bond dimensions of fused legs do not match.')
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
        for _ in range(nlegs):  # remove leafs
            tree.pop(ltree - nlegs + 1)
            t1.pop(ltree - nlegs + 1)
            t2.pop(ltree - nlegs + 1)
        for i in parents:
            tree[i] -= nlegs - 1
        s_in = tuple(s.pop(ltree - nlegs + 1) for _ in range(nlegs))
        t_in = tuple(tu.pop(leg - nlegs) for _ in range(nlegs))
        D_in = tuple(Du.pop(leg - nlegs) for _ in range(nlegs))
        t_out = tuple(sorted(set(t1[ltree - nlegs]) | set(t2[ltree - nlegs])))
        s_out = s[ltree - nlegs]
        ls = _leg_structure_combine_charges(config, t_in, D_in, s_in, t_out, s_out)
        ma1 = _merge_masks(config, ls, [msk1.pop(leg - nlegs) for _ in range(nlegs)])
        ma2 = _merge_masks(config, ls, [msk2.pop(leg - nlegs) for _ in range(nlegs)])
        for ind in ma1.keys() - set(t1[ltree - nlegs]):
            ma1[ind] *= False
        for ind in ma2.keys() - set(t2[ltree - nlegs]):
            ma2[ind] *= False
        msk1.insert(leg - nlegs, ma1)
        msk2.insert(leg - nlegs, ma2)
        hh = [hfu.pop(leg - nlegs) for _ in range(nlegs)]
        hfu.insert(leg - nlegs, _fuse_hfs(hh, t_in, D_in, s_out))
        t_out, D_out = zip(*ls.Dtot.items())
        tu.insert(leg - nlegs, t_out)
        Du.insert(leg - nlegs, D_out)
    return msk1.pop(), msk2.pop(), hfu.pop()


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
                hfs.append(_Fusion(hf.tree[n_init: n + 1], hf.s[n_init: n + 1],
                                    hf.t[n_init: n], hf.D[n_init: n]))
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
