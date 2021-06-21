""" Support for merging blocks in yast tensors. """

from functools import lru_cache
from itertools import groupby, product
from operator import itemgetter
from typing import NamedTuple
import numpy as np
from ._auxliary import _flatten, _tarray, _Darray, _hard_fusion, _unpack_axes, _clear_axes


__all__ = ['fuse_legs_hard', 'unfuse_legs_hard']


class _LegDec(NamedTuple):
    """ Internal structure of leg resulting from fusions, including slices"""
    dec: dict = None  # decomposition of each effective charge
    Dtot: dict = None  # bond dimensions of effective charges


class _DecRec(NamedTuple):
    """ Single record in _LegDec.dec[effective charge]"""
    Dslc: tuple = (None, None)  # slice
    Dprod: int = 0  # size of slice, equal to product of Drsh
    Drsh: tuple = ()  # original shape of fused dims in a block


def _leg_structure_combine_charges(config, t_in, D_in, t_out, seff, slegs):
    """ Auxilliary function that takes a combination of charges and dimensions on a number of legs,
        and combines them into effective charges and dimensions.
    """
    comb_t = tuple(product(*t_in))
    comb_t = np.array(comb_t, dtype=int).reshape((len(comb_t), len(slegs), config.sym.NSYM))
    comb_D = tuple(product(*D_in))
    comb_D = np.array(comb_D, dtype=int).reshape((len(comb_D), len(slegs)))
    teff = config.sym.fuse(comb_t, slegs, seff)
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


def _fuse_hfs(hfs, t_legs, D_legs, ss, axis=None):
    """ Fuse _hard_fusions, including charges and dimensions present on the fused legs. """
    if axis is None:
        axis = list(range(len(hfs)))
    tfl, Dfl, sfl, msfl = [], [], [], []
    treefl = [sum(hfs[n].tree[0] for n in axis)]
    for n in axis:
        tfl.append(t_legs[n])
        tfl.extend(hfs[n].t)
        Dfl.append(D_legs[n])
        Dfl.extend(hfs[n].D)
        sfl.append(ss[n])
        sfl.extend(hfs[n].s)
        msfl.append(-ss[n])
        msfl.extend(hfs[n].ms)
        treefl.extend(hfs[n].tree)
    return _hard_fusion(tuple(treefl), tuple(sfl), tuple(msfl), tuple(tfl), tuple(Dfl))


def _unfuse_leg_fusion(lf):
    """ one layer of unfuse """
    tt, DD, ss, lfs = [], [], [], []
    n_init, cum = 1, 0
    for n in range(1, len(lf.tree)):
        if cum == 0:
            cum = lf.tree[n]
        if lf.tree[n] == 1:
            cum = cum - 1
            if cum == 0:
                tt.append(lf.t[n_init - 1])
                DD.append(lf.D[n_init - 1])
                ss.append(lf.s[n_init - 1])
                slc = slice(n_init, n)
                lfs.append(_hard_fusion(lf.tree[n_init : n + 1], lf.s[slc], lf.ms[slc], lf.t[slc], lf.D[slc]))
                n_init = n + 1
    return tt, DD, ss, lfs


def fuse_legs_hard(a, axes, inplace=False):
    """ fuse """
    tset, Dset, ss = _tarray(a), _Darray(a), a.s
    axes = list(_clear_axes(*axes))
    axes = list(_unpack_axes(a, *axes))
    nsym = tset.shape[2]
    t_legs, D_legs = a.get_leg_charges_and_dims(native=True)
    order = tuple(_flatten(axes))
    aaxes, news, slegs, fh, fm, ls = [], [], [], [], [], []
    for axis in axes:
        aaxis = np.array(axis, dtype=int).reshape(-1)
        aaxes.append(aaxis)
        news.append(ss[aaxis[0]])
        slegs.append(ss[aaxis])
        if len(axis) == 1:
            axis = axis[0]
            xx = _LegDec({t: {t: _DecRec((0, D), D, (D,))} for t, D in zip(t_legs[axis], D_legs[axis])},
                                {t: D for t, D in zip(t_legs[axis], D_legs[axis])})
            ls.append(xx)
            fh.append(a.hard_fusion[axis])
            fm.append(a.meta_fusion[axis])
        else:
            teff_set = a.config.sym.fuse(tset[:, aaxis, :], slegs[-1], news[-1])
            teff_set = set(tuple(x.flat) for x in teff_set)
            t_in = tuple(t_legs[n] for n in axis)
            D_in = tuple(D_legs[n] for n in axis)
            ls.append(_leg_structure_combine_charges(a.config, t_in, D_in, teff_set, news[-1], slegs[-1]))
            fh.append(_fuse_hfs(a.hard_fusion, t_legs, D_legs, ss, axis))
            fm.append((1,))
    tset, Dset = _tarray(a), _Darray(a)
    teff = np.zeros((tset.shape[0], len(aaxes), tset.shape[2]), dtype=int)
    Deff = np.zeros((Dset.shape[0], len(aaxes)), dtype=int)

    for n, aa in enumerate(aaxes):
        teff[:, n, :] = a.config.sym.fuse(tset[:, aa, :], slegs[n], news[n])
        Deff[:, n] = np.prod(Dset[:, aa], axis=1)

    told = tuple(tuple(tuple(x[aa, :].flat) for aa in aaxes) for x in tset)
    tnew = tuple(tuple(x.flat) for x in teff)
    teff = tuple(tuple(tuple(x.flat) for x in y) for y in teff)

    meta_mrg = tuple((tn, to, tuple(l.dec[e][o].Dslc for l, e, o in zip(ls, tes, tos)),
                    tuple(l.dec[e][o].Dprod for l, e, o in zip(ls, tes, tos)))
                    for tn, to, tes, tos in zip(tnew, a.struct.t, teff, told))
    meta_new = tuple((x, tuple(l.Dtot[x[i * nsym : (i + 1) * nsym]] for i, l in enumerate(ls))) for x in set(tnew))

    if inplace:
        c = a
        c.struct = c.struct._replace(s=tuple(news))
        c.hard_fusion = tuple(fh)
        c.meta_fusion = tuple(fm)
    else:
        c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=tuple(fm), hard_fusion=tuple(fh), n=a.n, s=news)
    c.A = a.config.backend.merge_blocks(a.A, order, meta_new, meta_mrg)
    c.update_struct()
    return c


def unfuse_legs_hard(a, axes, inplace=False):
    if isinstance(axes, int):
        axes = (axes,)
    axes, = _unpack_axes(a, axes)

    t_legs, D_legs = a.get_leg_charges_and_dims(native=True)
    ss = a.s

    ls, fh, fm, news = [], [], [], []
    for n in range(a.nlegs):
        if n in axes:
            t_in, D_in, slegs, lfs = _unfuse_leg_fusion(a.hard_fusion[n])
            ls.append(_leg_structure_combine_charges(a.config, t_in, D_in, t_legs[n], ss[n], slegs))
            fh.extend(lfs)
            fm.extend([(1,)] * len(lfs))
            news.extend(slegs)
        else:
            dec = {t: {t: _DecRec((0, D), D, (D,))} for t, D in zip(t_legs[n], D_legs[n])}
            Dtot = {t: D for t, D in zip(t_legs[n], D_legs[n])}
            ls.append(_LegDec(dec, Dtot))
            fh.append(a.hard_fusion[n])
            fm.append(a.meta_fusion[n])
            news.append(ss[n])

    meta = []
    tt = tuple(tuple(ls[n].dec) for n in range(a.nlegs))
    for ind in product(*tt):
        to = sum(ind, ())
        if to in a.A:
            kkk = tuple(tuple(ls[n].dec[ii].items()) for n, ii in enumerate(ind))
            for tt in product(*kkk):
                tn = sum((x[0] for x in tt), ())
                slc = tuple(x[1].Dslc for x in tt)
                Dn = tuple(_flatten((x[1].Drsh for x in tt)))
                meta.append((tn, to, slc, Dn))

    if inplace:
        c = a
        c.struct = c.struct._replace(s=tuple(news))
        c.hard_fusion = tuple(fh)
        c.meta_fusion = tuple(fm)
    else:
        c = a.__class__(config=a.config, isdiag=a.isdiag, n=a.n, s=news, meta_fusion=tuple(fm), hard_fusion=tuple(fh))

    c.A = a.config.backend.unmerge_from_array(a.A, meta)
    c.update_struct()
    return c


def _merge_to_matrix(a, axes, s_eff, inds=None, sort_r=False):
    order = axes[0] + axes[1]
    meta_new, meta_mrg, ls_l, ls_r, ul, ur = _meta_merge_to_matrix(a.config, a.struct, axes, s_eff, inds, sort_r)
    Anew = a.config.backend.merge_blocks(a.A, order, meta_new, meta_mrg, a.config.device)
    return Anew, ls_l, ls_r, ul, ur


@lru_cache(maxsize=1024)
def _meta_merge_to_matrix(config, struct, axes, s_eff, inds, sort_r):
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
    meta_mrg = tuple((tn, to, (ls[0].dec[tel][tl].Dslc, ls[1].dec[ter][tr].Dslc), (ls[0].dec[tel][tl].Dprod, ls[1].dec[ter][tr].Dprod))
                     for tn, to, tel, tl, ter, tr in zip(tnew, told, teff[0], t[0], teff[1], t[1]))
    if sort_r:
        unew_r, unew_l, unew = zip(*sorted(set(zip(teff[1], teff[0], tnew)))) if len(tnew) > 0 else ((), (), ())
    else:
        unew, unew_l, unew_r = zip(*sorted(set(zip(tnew, teff[0], teff[1])))) if len(tnew) > 0 else ((), (), ())
    # meta_new = ((unew, Dnew), ...)
    meta_new = tuple((iu, (ls[0].Dtot[il], ls[1].Dtot[ir])) for iu, il, ir in zip(unew, unew_l, unew_r))
    return meta_new, meta_mrg, ls[0], ls[1], unew_l, unew_r


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
    if sum(D_keep[ind] for ind in D_keep) > D_total:  # truncate to total bond dimension
        order = a.config.backend.select_global_largest(a.A, D_keep, D_total, keep_multiplets, eps_multiplet, ordering)
        low = 0
        for ind in D_keep:
            high = low + D_keep[ind]
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
    for ind in D_keep:
        if D_keep[ind] > 0:
            Dslc = a.config.backend.range_largest(D_keep[ind], Dmax[ind], ordering)
            dec[ind] = {ind: _DecRec(Dslc, D_keep[ind], (D_keep[ind],))}
            Dtot[ind] = D_keep[ind]
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
