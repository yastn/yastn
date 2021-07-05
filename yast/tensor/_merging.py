""" Support for merging blocks in yast tensors. """

from functools import lru_cache
from itertools import groupby, product
from operator import itemgetter
from typing import NamedTuple
import numpy as np
from ._auxliary import _flatten, _struct, _clear_axes, _ntree_to_mf, _mf_to_ntree
from ._tests import YastError,  _test_all_axes


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
    ms: tuple = (-1,)  # minus s
    t: tuple = ()  # fused leg charges at each step len(t) = len(tree) - 1
    D: tuple = ()  # fused dimensions  at each step len(t) = len(tree) - 1


def _flip_sign_hf(x):
    """ _Fusion with fliped signature. """
    return x._replace(s=x.ms, ms=x.s)

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
    Deff = tuple(np.prod(comb_D, axis=1))
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
    tfl, Dfl, sfl, msfl = [], [], [s_out], [-s_out]
    treefl = [sum(hfs[n].tree[0] for n in axis)]
    for n in axis:
        tfl.append(t_in[n])
        tfl.extend(hfs[n].t)
        Dfl.append(D_in[n])
        Dfl.extend(hfs[n].D)
        sfl.extend(hfs[n].s)
        msfl.extend(hfs[n].ms)
        treefl.extend(hfs[n].tree)
    return _Fusion(tuple(treefl), tuple(sfl), tuple(msfl), tuple(tfl), tuple(Dfl))


def _merge_masks(config, ls, ms):
    """ combine masks using information from LegDec"""
    msk = {te: np.ones(D, dtype=bool) for te, D in ls.Dtot.items()}
    nsym = config.sym.NSYM
    for te, dec in ls.dec.items():
        for t, Dr in dec.items():
            x = ms[0][t[:config.sym.NSYM]]
            for i in range(1, len(ms)):
                x = np.outer(x, ms[i][t[i * nsym: (i + 1) * nsym]]).ravel()
            msk[te][slice(*Dr.Dslc)] = x
    return msk


def _get_tD_legs(struct):
    """ different views on struct.t and struct.D """
    tset = np.array(struct.t, dtype=int).reshape((len(struct.t), len(struct.s), len(struct.n)))
    Dset = np.array(struct.D, dtype=int).reshape((len(struct.t), len(struct.s)))
    tD_legs = [sorted(set((tuple(t.flat), D) for t, D in zip(tset[:, n, :], Dset[:, n]))) for n in range(len(struct.s))]
    tD_dict = [dict(tD) for tD in tD_legs]
    if any(len(x) != len(y) for x, y in zip(tD_legs, tD_dict)):
        raise YastError('CRITICAL ERROR. Bond dimensions of a tensor are inconsistent. This should not have happend.')
    tlegs = [tuple(tD.keys()) for tD in tD_dict]
    Dlegs= [tuple(tD.values()) for tD in tD_dict]
    return tlegs, Dlegs, tD_dict, tset, Dset


def _masks_for_tensordot(config, structa, hfa, axa, lsa, structb, hfb, axb, lsb):
    """ masks to get the intersecting parts of legs from two tensors """
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
    if hfs[0].tree != hfs[1].tree:
        raise YastError('Orders of merges do not match. ')
    teff = tuple(sorted((set(ts[0]) & set(ts[1]))))

    tree = list(hfs[0].tree)
    if len(tree) == 1:
        msks = [[{t: np.ones(D, dtype=bool) for t, D in zip(ts[n], Ds[n]) if t in teff}]for n in (0, 1)]
    else:
        msks = [[{t: np.ones(D, dtype=bool) for t, D in zip(hf.t[i], hf.D[i])}
                    for i, l in enumerate(tree[1:]) if l == 1] for hf in hfs]

    for ms1, ms2 in zip(*msks):
        for t in set(ms1) & set(ms2):
            if ms1[t].size != ms2[t].size:
                raise YastError('Mismatch of bond dimension of native legs for charge %s' %str(t))
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
        for _ in range(nlegs):  # remove leaves
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
                hfs.append(_Fusion(hf.tree[n_init : n + 1], hf.s[n_init : n + 1], hf.ms[n_init : n + 1],\
                                    hf.t[n_init : n], hf.D[n_init : n]))
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
            for nlegs in mf:  # parsing the tree to identify the lowest layer
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


def _merge_to_matrix(a, axes, s_eff, inds=None, sort_r=False):
    """ Main function merging tensor into effective block matrix. """
    order = axes[0] + axes[1]
    meta_new, meta_mrg, ls_l, ls_r, ul, ur = _meta_merge_to_matrix(a.config, a.struct, axes, s_eff, inds, sort_r)
    Anew = a.config.backend.merge_blocks(a.A, order, meta_new, meta_mrg, a.config.device)
    return Anew, ls_l, ls_r, ul, ur


@lru_cache(maxsize=1024)
def _meta_merge_to_matrix(config, struct, axes, s_eff, inds, sort_r):
    """ Meta information for backend needed to merge tensor into effective block matrix. """
    told = struct.t if inds is None else [struct.t[ii] for ii in inds]
    Dold = struct.D if inds is None else [struct.D[ii] for ii in inds]
    tset = np.array(told, dtype=int).reshape((len(told), len(struct.s), config.sym.NSYM))
    Dset = np.array(Dold, dtype=int).reshape(len(Dold), len(struct.s))
    legs, t, D, Deff, teff, s, ls = [], [], [], [], [], [], []
    for n in (0, 1):
        legs.append(np.array(axes[n], int))
        t.append(tset[:, legs[n], :])
        D.append(Dset[:, legs[n]])
        Deff.append(np.prod(D[n], axis=1))
        s.append(np.array([struct.s[ii] for ii in axes[n]], dtype=int))
        teff.append(config.sym.fuse(t[n], s[n], s_eff[n]))
        teff[n] = tuple(tuple(t.flat) for t in teff[n])
        t[n] = tuple(tuple(t.flat) for t in t[n])
        D[n] = tuple(tuple(x) for x in D[n])
        ls.append(_leg_structure_merge(teff[n], t[n], Deff[n], D[n]))

    tnew = tuple(tuple(t.flat) for t in np.hstack([teff[0], teff[1]]))
    # meta_mrg = ((tnew, told, Dslc, Drsh), ...)
    meta_mrg = tuple((tn, to, (ls[0].dec[tel][tl].Dslc, ls[1].dec[ter][tr].Dslc),
                        (ls[0].dec[tel][tl].Dprod, ls[1].dec[ter][tr].Dprod))
                        for tn, to, tel, tl, ter, tr in zip(tnew, told, teff[0], t[0], teff[1], t[1]))
    if sort_r:
        unew_r, unew_l, unew = zip(*sorted(set(zip(teff[1], teff[0], tnew)))) if len(tnew) > 0 else ((), (), ())
    else:
        unew, unew_l, unew_r = zip(*sorted(set(zip(tnew, teff[0], teff[1])))) if len(tnew) > 0 else ((), (), ())
    # meta_new = ((unew, Dnew), ...)
    meta_new = (unew, tuple((ls[0].Dtot[il], ls[1].Dtot[ir]) for il, ir in zip(unew_l, unew_r)))
    return meta_new, meta_mrg, ls[0], ls[1], unew_l, unew_r


@lru_cache(maxsize=1024)
def _meta_fuse_hard(config, struct, axes):
    """ Meta information for backend needed to hard-fuse some legs. """
    nblocks, ndim, nsym = len(struct.t), len(struct.s), len(struct.n)
    t_in, D_in, tD_dict, tset, Dset = _get_tD_legs(struct)
    slegs = tuple(tuple(struct.s[n] for n in a) for a in axes)
    snew = tuple(struct.s[axis[0]] for axis in axes)
    teff = np.zeros((nblocks, len(snew), nsym), dtype=int)
    Deff = np.zeros((nblocks, len(snew)), dtype=int)
    for n, a in enumerate(axes):
        teff[:, n, :] = config.sym.fuse(tset[:, a, :], slegs[n], snew[n])
        Deff[:, n] = np.prod(Dset[:, a], axis=1)

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
    tnew_split = [tuple(x[i: i + nsym] for i in range(0, nsym * ndim, nsym)) for x in tnew]
    Dnew = tuple(tuple(l.Dtot[y] for l, y in zip(ls, x)) for x in tnew_split)
    meta_mrg = tuple((tn, to, tuple(l.dec[e][o].Dslc for l, e, o in zip(ls, tes, tos)),
                        tuple(l.dec[e][o].Dprod for l, e, o in zip(ls, tes, tos)))
                        for tn, to, tes, tos in zip(teff, struct.t, teff_split, told_split))
    struct_new = _struct(t=tnew, D=Dnew, s=tuple(snew), n=struct.n)
    return struct_new, meta_mrg, t_in, D_in


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

    meta = []
    tall = tuple(tuple(l.dec) for l in ls)
    for ind in product(*tall):
        to = sum(ind, ())
        if to in struct.t:
            tpart = tuple(tuple(ls[n].dec[ii].items()) for n, ii in enumerate(ind))
            for tt in product(*tpart):
                tn = sum((x[0] for x in tt), ())
                slc = tuple(x[1].Dslc for x in tt)
                Dn = tuple(_flatten((x[1].Drsh for x in tt)))
                meta.append((tn, to, slc, Dn))
    return meta, tuple(snew), tuple(nlegs_unfused), tuple(hfs_new)


def _fuse_legs_hard(a, axes, order, inplace=False):
    """ Funtion performing hard fusion. axes are for native legs and are cleaned outside."""
    assert all(isinstance(x, tuple) for x in axes)
    for x in axes:
        assert all(isinstance(y, int) for y in x)

    struct_new, meta_mrg, t_in, D_in = _meta_fuse_hard(a.config, a.struct, axes)
    meta_new = (struct_new.t, struct_new.D)
    fm = ((1,),) * len(struct_new.s)
    fh = tuple(_fuse_hfs(a.hard_fusion, t_in, D_in, struct_new.s[n], axis) if len(axis) > 1 else a.hard_fusion[axis[0]]
                for n, axis in enumerate(axes))
    if inplace:
        c = a
        c.struct = struct_new
        c.hard_fusion = tuple(fh)
        c.meta_fusion = tuple(fm)
    else:
        c = a.__class__(config=a.config, meta_fusion=tuple(fm), hard_fusion=tuple(fh), struct=struct_new)
    c.A = a.config.backend.merge_blocks(a.A, order, meta_new, meta_mrg)
    return c


def fuse_meta_to_hard(a, inplace=False):
    """ Changes all meta fusions into a hard fusions. If there are no meta fusions, do nothing. """
    while any(mf != (1,) for mf in a.meta_fusion):
        axes, new_mfs = _consume_mfs_lowest(a.meta_fusion)
        order = tuple(range(a.nlegs))
        a = _fuse_legs_hard(a, axes, order, inplace)
        a.meta_fusion = new_mfs
    return a


def fuse_legs(a, axes, inplace=False, mode=None):
    r"""
    Fuse groups of legs into new effective ones.

    Legs are first permuted -- groups of consequative legs are then fused.
    Two types of fusion are supported: `meta` and `hard`.
    `meta` adds a layer of logical information that allows
    to interpret a group of native legs as a single logical leg.
    `hard` is creating merged blocks characterised by fused charges.

    Parameters
    ----------
    axes: tuple
        tuple of leg's indices for transpose. Groups of legs to be fused together form inner tuples.

    inplace: bool
        If true, perform operation in place.

    mode: str
        can select 'hard' or 'meta' fusion. If None, use default from config.
        It can also be overriden by config.force_fuse.
        Applying hard fusion of tensor with meta fusion first
        replaces meta fusion with hard fusion.
        Meta fusion can be applied on top of hard fusion.

    Returns
    -------
    tensor : Tensor

    Example
    -------
    tensor.fuse_legs(axes=(2, 0, (1, 4), 3)) gives 4 efective legs from original 5; with one metaly non-trivial one
    tensor.fuse_legs(axes=[(2, 0), (1, 4), (3, 5)]) gives 3 effective legs from original 6
    """
    if a.isdiag:
        raise YastError('Cannot fuse legs of a diagonal tensor')
    if mode is None:
        mode = a.config.default_fusion
    if a.config.force_fusion is not None:
        mode = a.config.force_fusion

    order = tuple(_flatten(axes))
    _test_all_axes(a, order)
    axes = tuple(_clear_axes(*axes))

    if mode == 'meta':
        mfs = []
        for group in axes:
            if len(group) == 1:
                mfs.append(a.meta_fusion[group[0]])
            else:
                new_mf = [sum(a.meta_fusion[ii][0] for ii in group)]
                for ii in group:
                    new_mf.extend(a.meta_fusion[ii])
                mfs.append(tuple(new_mf))
        if inplace and order == tuple(ii for ii in range(a.mlegs)):
            c = a
        else:
            c = a.transpose(axes=order, inplace=inplace)
        c.meta_fusion = tuple(mfs)
    elif mode == 'hard':
        c = fuse_meta_to_hard(a, inplace)
        c = _fuse_legs_hard(c, axes, order, inplace)
    else:
        raise YastError('fuse_legs mode should be `meta` or `hard`; can be also set in config file.')
    return c


def unfuse_legs(a, axes, inplace=False):
    """
    Unfuse legs reverting one layer of fusion. Operation can be done in-place.

    New legs are inserted in place of the unfused one.

    Parameters
    ----------
    axis: int or tuple of ints
        leg(s) to ungroup.

    inplace: bool
        If true, perform operation in place.

    Returns
    -------
    tensor : Tensor
    """
    if a.isdiag:
        raise YastError('Cannot unfuse legs of a diagonal tensor')
    if isinstance(axes, int):
        axes = (axes,)
    ni, new_mfs, axes_hf = 0, [], []
    for mi in range(a.mlegs):
        dni = a.meta_fusion[mi][0]
        if mi not in axes or (a.meta_fusion[mi][0] == 1 and a.hard_fusion[ni].tree[0] == 1):
            new_mfs.append(a.meta_fusion[mi])
        elif a.meta_fusion[mi][0] > 1:  #and mi in axes
            stack = a.meta_fusion[mi]
            lstack = len(stack)
            pos_init, cum = 1, 0
            for pos in range(1, lstack):
                if cum == 0:
                    cum = stack[pos]
                if stack[pos] == 1:
                    cum = cum - 1
                    if cum == 0:
                        new_mfs.append(stack[pos_init: pos + 1])
                        pos_init = pos + 1
        else:  # c.hard_fusion[ni].tree[0] > 1 and c.meta_fusion[mi][0] == 1 and mi in axes
            axes_hf.append(ni)
            new_mfs.append(a.meta_fusion[mi])
        ni += dni
    if axes_hf:
        meta, snew, nlegs, new_hfs = _meta_unfuse_hard(a.config, a.struct, tuple(axes_hf), tuple(a.hard_fusion))
        for unfused, n in zip(nlegs[::-1], axes_hf[::-1]):
            new_mfs = new_mfs[:n] + [new_mfs[n]] * unfused + new_mfs[n+1:]
        if inplace:
            c = a
            c.struct = c.struct._replace(s=snew)
            c.hard_fusion = tuple(new_hfs)
            c.meta_fusion = tuple(new_mfs)
        else:
            c = a.__class__(config=a.config, n=a.n, s=snew, meta_fusion=tuple(new_mfs), hard_fusion=tuple(new_hfs))
        c.A = a.config.backend.unmerge_from_array(a.A, meta)
        c.update_struct()
    else:
        c = a if inplace else a.clone()
        c.meta_fusion = tuple(new_mfs)
    return c


def _leg_struct_trivial(a, axis=0):
    """ trivial LegDecomposition for unfused leg. """
    nsym = a.config.sym.NSYM
    dec, Dtot = {}, {}
    for ind, val in a.A.items():
        t = ind[nsym * axis: nsym * (axis + 1)]
        D = a.config.backend.get_shape(val)[axis]
        dec[t] = {t: _DecRec((0, D), D, (D,))}
        Dtot[t] = D
    return _LegDec(dec, Dtot)


def _leg_struct_truncation(a, tol=0., D_block=np.inf, D_total=np.inf,
                            keep_multiplets=False, eps_multiplet=1e-12, ordering='eigh'):
    r"""
    Gives slices for truncation of 1d matrices according to tolerance, D_block, D_total.

    A should be dict of ordered 1d arrays.
    Sorting gives information about ordering outputed by a particular splitting funcion:
    Usual convention is that for svd A[ind][0] is largest; and for eigh A[ind][-1] is largest.
    """
    maxS = 0 if len(a.A) == 0 else a.config.backend.maximum(a.A)
    Dmax, D_keep = {}, {}
    for ind in a.A:
        Dmax[ind] = a.config.backend.get_size(a.A[ind])
        D_keep[ind] = min(D_block, Dmax[ind])
    if (tol > 0) and (maxS > 0):  # truncate to relative tolerance
        for ind in D_keep:
            D_keep[ind] = min(D_keep[ind], a.config.backend.count_greater(a.A[ind], maxS * tol))
    if sum(D_keep.values()) > D_total:  # truncate to total bond dimension
        order = a.config.backend.select_global_largest(a.A, D_keep, D_total, keep_multiplets, eps_multiplet, ordering)
        low = 0
        for ind, D_ind in D_keep.items():
            high = low + D_ind
            D_keep[ind] = sum((low <= order) & (order < high)).item()
            low = high
    if keep_multiplets:  # check symmetry related blocks and truncate to equal length
        ind_list = [np.asarray(k) for k in D_keep]
        for ind in ind_list:
            t = tuple(ind)
            tn = tuple(-ind)
            minD_sector = min(D_keep[t], D_keep[tn])
            D_keep[t] = D_keep[tn] = minD_sector
            # if -ind in ind_list:
            #     ind_list.remove(-ind)  ## this might mess-up iterator
    dec, Dtot = {}, {}
    for ind, D_ind in D_keep.items():
        if D_ind > 0:
            Dslc = a.config.backend.range_largest(D_ind, Dmax[ind], ordering)
            dec[ind] = {ind: _DecRec(Dslc, D_ind, (D_ind,))}
            Dtot[ind] = D_ind
    return _LegDec(dec, Dtot)


def _unmerge_matrix(a, ls_l, ls_r):
    meta = []
    for il, ir in product(ls_l.dec, ls_r.dec):
        ic = il + ir
        if ic in a.A:
            for (tl, rl), (tr, rr) in product(ls_l.dec[il].items(), ls_r.dec[ir].items()):
                meta.append((tl + tr, ic, rl.Dslc, rr.Dslc, rl.Drsh + rr.Drsh))
    a.A = a.config.backend.unmerge_from_matrix(a.A, meta)
    a.update_struct()


def _unmerge_diagonal(a, ls):
    meta = tuple((ta + ta, ia, ra.Dslc) for ia in ls.dec for ta, ra in ls.dec[ia].items())
    a.A = a.config.backend.unmerge_from_diagonal(a.A, meta)
    a.A = {ind: a.config.backend.diag_create(x) for ind, x in a.A.items()}
    a.update_struct()
