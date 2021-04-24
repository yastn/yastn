""" Support for merging blocks in yast tensors. """

from functools import lru_cache
from itertools import groupby, product
from operator import itemgetter
import numpy as np


def _merge_to_matrix(a, axes, s_eff, inds=None, sort_r=False):
    order = axes[0] + axes[1]
    meta_new, meta_mrg, ls_l, ls_r, ul, ur = _meta_merge_to_matrix(a.config, a.struct, axes, s_eff, inds, sort_r)
    Anew = a.config.backend.merge_to_matrix(a.A, order, meta_new, meta_mrg, a.config.dtype, a.config.device)
    return Anew, ls_l, ls_r, ul, ur


@lru_cache(maxsize=1024)
def _meta_merge_to_matrix(config, struct, axes, s_eff, inds, sort_r):
    legs_l = np.array(axes[0], int)
    legs_r = np.array(axes[1], int)
    told = struct.t if inds is None else [struct.t[ii] for ii in inds]
    Dold = struct.D if inds is None else [struct.D[ii] for ii in inds]
    tset = np.array(told, dtype=int).reshape((len(told), len(struct.s), config.sym.NSYM))
    Dset = np.array(Dold, dtype=int).reshape(len(Dold), len(struct.s))
    t_l = tset[:, legs_l, :]
    t_r = tset[:, legs_r, :]
    D_l = Dset[:, legs_l]
    D_r = Dset[:, legs_r]
    s_l = np.array([struct.s[ii] for ii in axes[0]], dtype=int)
    s_r = np.array([struct.s[ii] for ii in axes[1]], dtype=int)
    Deff_l = np.prod(D_l, axis=1)
    Deff_r = np.prod(D_r, axis=1)

    teff_l = config.sym.fuse(t_l, s_l, s_eff[0])
    teff_r = config.sym.fuse(t_r, s_r, s_eff[1])
    tnew = np.hstack([teff_l, teff_r])

    tnew = tuple(tuple(t.flat) for t in tnew)
    teff_l = tuple(tuple(t.flat) for t in teff_l)
    teff_r = tuple(tuple(t.flat) for t in teff_r)
    t_l = tuple(tuple(t.flat) for t in t_l)
    t_r = tuple(tuple(t.flat) for t in t_r)
    D_l = tuple(tuple(x) for x in D_l)
    D_r = tuple(tuple(x) for x in D_r)
    ls_l = _leg_structure_merge(config, teff_l, t_l, Deff_l, D_l, s_eff[0], s_l)
    ls_r = _leg_structure_merge(config, teff_r, t_r, Deff_r, D_r, s_eff[1], s_r)
    # meta_mrg = ((tnew, told, Dslc_l, D_l, Dslc_r, D_r), ...)
    meta_mrg = tuple((tn, to, *ls_l.dec[tel][tl][:2], *ls_r.dec[ter][tr][:2])
                     for tn, to, tel, tl, ter, tr in zip(tnew, told, teff_l, t_l, teff_r, t_r))
    if sort_r:
        unew_r, unew_l, unew = zip(*sorted(set(zip(teff_r, teff_l, tnew)))) if len(tnew) > 0 else ((), (), ())
    else:
        unew, unew_l, unew_r = zip(*sorted(set(zip(tnew, teff_l, teff_r)))) if len(tnew) > 0 else ((), (), ())
    # meta_new = ((unew, Dnew), ...)
    meta_new = tuple((iu, (ls_l.D[il], ls_r.D[ir])) for iu, il, ir in zip(unew, unew_l, unew_r))
    return meta_new, meta_mrg, ls_l, ls_r, unew_l, unew_r


def _leg_structure_merge(config, teff, tlegs, Deff, Dlegs, seff, slegs):
    """ LegDecomposition for merging into a single leg. """
    tt = sorted(set(zip(teff, tlegs, Deff, Dlegs)))
    dec, Dtot = {}, {}
    for te, grp in groupby(tt, key=itemgetter(0)):
        Dlow = 0
        dec[te] = {}
        for _, tl, De, Dl in grp:
            Dtop = Dlow + De
            dec[te][tl] = ((Dlow, Dtop), De, Dl)
            Dlow = Dtop
        Dtot[te] = Dtop
    return _LegDecomposition(config, seff, slegs, dec, Dtot)


def _leg_struct_trivial(a, axis=0):
    """ trivial LegDecomposition for unfused leg. """
    nsym = a.config.sym.NSYM
    dec, Dtot = {}, {}
    for ind, val in a.A.items():
        t = ind[nsym * axis: nsym * (axis + 1)]
        D = a.config.backend.get_shape(val)[axis]
        dec[t] = {t: ((0, D), D, (D,))}
        Dtot[t] = D
    return _LegDecomposition(a.config, a.s[axis], a.s[axis], dec, Dtot)


def _leg_struct_truncation(a, tol=0., D_block=np.inf, D_total=np.inf, keep_multiplets=False, eps_multiplet=1e-12, ordering='eigh'):
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
            dec[ind] = {ind: (Dslc, D_keep[ind], (D_keep[ind],))}
            Dtot[ind] = D_keep[ind]
    return _LegDecomposition(a.config, a.s[0], a.s[0], dec, Dtot)


def _unmerge_matrix(a, ls_l, ls_r):
    meta = []
    for il, ir in product(ls_l.dec, ls_r.dec):
        ic = il + ir
        if ic in a.A:
            for (tl, (sl, _, Dl)), (tr, (sr, _, Dr)) in product(ls_l.dec[il].items(), ls_r.dec[ir].items()):
                meta.append((tl + tr, ic, sl, sr, Dl + Dr))
    a.A = ls_l.config.backend.unmerge_from_matrix(a.A, meta)
    a.update_struct()


def _unmerge_diagonal(a, ls):
    meta = tuple((ta + ta, ia, sa) for ia in ls.dec for ta, (sa, _, _) in ls.dec[ia].items())
    a.A = ls.config.backend.unmerge_from_diagonal(a.A, meta)
    a.A = {ind: ls.config.backend.diag_create(x) for ind, x in a.A.items()}
    a.update_struct()


class _LegDecomposition:
    """Information about internal structure of leg resulting from fusions."""
    def __init__(self, config=None, s_eff=1, s=(), dec=None, D=None):
        try:
            self.nlegs = len(s)  # number of fused legs
            self.s = tuple(s)  # signature of fused legs
        except TypeError:
            self.nlegs = 1
            self.s = (s,)
        self.config = config
        self.s_eff = s_eff  # signature of effective leg
        self.D = {} if D is None else D  # bond dimensions of effective charges
        self.dec = {} if dec is None else dec  # leg's structure/ decomposition

    def match(self, other):
        """ Compare if decomposition match. This does not include signatures."""
        return self.nlegs == other.nlegs and self.D == other.D and self.dec == other.dec

    def copy(self):
        """ Copy leg structure. """
        ls = _LegDecomposition(s=self.s, news=self.news)
        for te, de in self.dec.items():
            ls.dec[te] = de.copy()
        ls.D = self.D.copy()
        return ls

    def show(self):
        """ Print information about leg structure. """
        print("Leg structure: fused = ", self.nlegs)
        for te, de in self.dec.items():
            print(te, ":")
            for to, Do in de.items():
                print("   ", to, ":", Do)


# def group_legs(self, axes, new_s=None):
#     """
#     Permutes tensor legs. Next, fuse a specified group of legs into a new single leg.

#     Parameters
#     ----------
#     axes: tuple
#         tuple of leg indices for transpose. Group of legs to be fused forms inner tuple.
#         If there is not internal tuple, fuse given indices and a new leg is placed at the position of the first fused oned.

#     new_s: int
#         signature of a new leg. If not given, the signature of the first fused leg is given.

#     Returns
#     -------
#     tensor : Tensor

#     Example
#     -------
#     For tensor with 5 legs: tensor.fuse_legs1(axes=(2, 0, (1, 4), 3))
#     tensor.fuse_legs1(axes=(2, 0)) is equivalent to tensor.fuse_legs1(axes=(1, (2, 0), 3, 4))
#     """
#     if self.isdiag:
#         raise YastError('Cannot group legs of a diagonal tensor')

#     ituple = [ii for ii, ax in enumerate(axes) if isinstance(ax, tuple)]
#     if len(ituple) == 1:
#         ig = ituple[0]
#         al, ag, ar = axes[:ig], axes[ig], axes[ig+1:]
#     elif len(ituple) == 0:
#         al = tuple(ii for ii in range(axes[0]) if ii not in axes)
#         ar = tuple(ii for ii in range(axes[0]+1, self.nlegs) if ii not in axes)
#         ag = axes
#         ig = len(al)
#     else:
#         raise YastError('Too many groups to fuse')
#     if len(ag) < 2:
#         raise YastError('Need at least two legs to fuse')

#     order = al+ag+ar  # order for permute
#     legs_l = np.array(al, dtype=np.intp)
#     legs_r = np.array(ar, dtype=np.intp)
#     legs_c = np.array(ag, dtype=np.intp)

#     if new_s is None:
#         new_s = self.s[ag[0]]

#     new_ndim = len(al) + 1 + len(ar)

#     t_grp = self.tset[:, legs_c, :]
#     D_grp = self.Dset[:, legs_c]
#     s_grp = self.s[legs_c]
#     t_eff = self.config.sym.fuse(t_grp, s_grp, new_s)
#     D_eff = np.prod(D_grp, axis=1)

#     D_rsh = np.empty((len(self.A), new_ndim), dtype=int)
#     D_rsh[:, :ig] = self.Dset[:, legs_l]
#     D_rsh[:, ig] = D_eff
#     D_rsh[:, ig+1:] = self.Dset[:, legs_r]

#     ls_c = _LegDecomposition(self.config, s_grp, new_s)
#     ls_c.leg_struct_for_merged(t_eff, t_grp, D_eff, D_grp)

#     t_new = np.empty((len(self.A), new_ndim, self.config.sym.NSYM), dtype=int)
#     t_new[:, :ig, :] = self.tset[:, legs_l, :]
#     t_new[:, ig, :] = t_eff
#     t_new[:, ig+1:, :] = self.tset[:, legs_r, :]

#     t_new = t_new.reshape(len(t_new), -1)
#     u_new, iu_new = np.unique(t_new, return_index=True, axis=0)
#     Du_new = D_rsh[iu_new]
#     Du_new[:, ig] = np.array([ls_c.D[tuple(t_eff[ii].flat)] for ii in iu_new], dtype=int)

#     # meta_new = ((u, Du), ...)
#     meta_new = tuple((tuple(u.flat), tuple(Du)) for u, Du in zip(u_new, Du_new))
#     # meta_mrg = ((tn, Ds, to, Do), ...)
#     meta_mrg = tuple((tuple(tn.flat), ls_c.dec[tuple(te.flat)][tuple(tg.flat)][0], tuple(to.flat), tuple(Do))
#         for tn, te, tg, to, Do in zip(t_new, t_eff, t_grp, self.tset, D_rsh))

#     c = self.empty(s=tuple(self.s[legs_l]) + (new_s,) + tuple(self.s[legs_r]), n=self.n, isdiag=self.isdiag)
#     c.A = self.config.backend.merge_one_leg(self.A, ig, order, meta_new , meta_mrg, self.config.dtype)
#     c.update_struct()
#     c.lss[ig] = ls_c
#     for nnew, nold in enumerate(al+ (-1,) + ar):
#         if nold in self.lss:
#             c.lss[nnew] = self.lss[nold].copy()
#     return c

# def ungroup_leg(self, axis):
#     """
#     Unfuse a single tensor leg.

#     New legs are inserted in place of the unfused one.

#     Parameters
#     ----------
#     axis: int
#         index of leg to ungroup.

#     Returns
#     -------
#     tensor : Tensor
#     """
#     try:
#         ls = self.lss[axis]
#     except KeyError:
#         return self

#     meta = []
#     for tt, DD in zip(self.tset, self.Dset):
#         tl = tuple(tt[:axis, :].flat)
#         tc = tuple(tt[axis, :].flat)
#         tr = tuple(tt[axis+1:, :].flat)
#         told = tuple(tt.flat)
#         Dl = tuple(DD[:axis])
#         Dr = tuple(DD[axis+1:])
#         for tcom, (Dsl, _, Dc) in ls.dec[tc].items():
#             tnew = tl + tcom + tr
#             Dnew = Dl + Dc + Dr
#             meta.append((told, tnew, Dsl, Dnew))
#     meta = tuple(meta)
#     s = tuple(self.s[:axis]) + ls.s + tuple(self.s[axis+1:])

#     c = self.empty(s=s, n=self.n, isdiag=self.isdiag)
#     c.A = self.config.backend.unmerge_one_leg(self.A, axis, meta)
#     c.update_struct()
#     for ii in range(axis):
#         if ii in self.lss:
#             c.lss[ii]=self.lss[ii].copy()
#     for ii in range(axis+1, self.nlegs):
#         if ii in self.lss:
#             c.lss[ii+ls.nlegs]=self.lss[ii].copy()
#     return c
