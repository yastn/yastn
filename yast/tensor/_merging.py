""" Support for merging blocks in yast tensors. """

from functools import lru_cache
from itertools import groupby, product
from operator import itemgetter
import numpy as np


def _merge_to_matrix(a, axes, s_eff, inds=None, rsort=False):
    order = axes[0] + axes[1]
    meta_new, meta_mrg, ls_l, ls_r, ul, ur = _meta_merge_to_matrix(a.config, a.struct, axes, s_eff, inds, rsort)
    Anew = a.config.backend.merge_to_matrix(a.A, order, meta_new, meta_mrg, a.config.dtype, a.config.device)
    return Anew, ls_l, ls_r, ul, ur


@lru_cache(maxsize=256)
def _meta_merge_to_matrix(config, struct, axes, s_eff, inds, sort_r):
    legs_l = np.array(axes[0], np.int)
    legs_r = np.array(axes[1], np.int)
    nsym = len(struct.n)
    nleg = len(struct.s)
    told = struct.t if inds is None else [struct.t[ii] for ii in inds]
    Dold = struct.D if inds is None else [struct.D[ii] for ii in inds]
    tset = np.array(told, dtype=int).reshape((len(told), nleg, nsym))
    Dset = np.array(Dold, dtype=int).reshape(len(Dold), nleg)
    t_l = tset[:, legs_l, :]
    t_r = tset[:, legs_r, :]
    D_l = Dset[:, legs_l]
    D_r = Dset[:, legs_r]
    s_l = np.array([struct.s[ii] for ii in axes[0]], dtype=int)
    s_r = np.array([struct.s[ii] for ii in axes[1]], dtype=int)
    Deff_l = np.prod(D_l, axis=1)
    Deff_r = np.prod(D_r, axis=1)

    te_l = config.sym.fuse(t_l, s_l, s_eff[0])
    te_r = config.sym.fuse(t_r, s_r, s_eff[1])
    tnew = np.hstack([te_l, te_r])

    tnew = tuple(tuple(t.flat) for t in tnew)
    te_l = tuple(tuple(t.flat) for t in te_l)
    te_r = tuple(tuple(t.flat) for t in te_r)
    t_l = tuple(tuple(t.flat) for t in t_l)
    t_r = tuple(tuple(t.flat) for t in t_r)
    D_l = tuple(tuple(x) for x in D_l)
    D_r = tuple(tuple(x) for x in D_r)
    dec_l, Dtot_l = _leg_structure_merge(te_l, t_l, Deff_l, D_l)
    dec_r, Dtot_r = _leg_structure_merge(te_r, t_r, Deff_r, D_r)

    ls_l = _LegDecomposition(config, s_l, s_eff[0])
    ls_r = _LegDecomposition(config, s_r, s_eff[1])
    ls_l.dec = dec_l
    ls_r.dec = dec_r
    ls_l.D = Dtot_l
    ls_r.D = Dtot_r

    # meta_mrg = ((tnew, told, Dslc_l, D_l, Dslc_r, D_r), ...)
    meta_mrg = tuple((tn, to, *dec_l[tel][tl][:2], *dec_r[ter][tr][:2])
                     for tn, to, tel, tl, ter, tr in zip(tnew, told, te_l, t_l, te_r, t_r))

    if sort_r:
        tt = sorted(set(zip(te_r, te_l, tnew)))
        unew_r, unew_l, unew = zip(*tt) if len(tt) > 0 else ((), (), ())
    else:
        tt = sorted(set(zip(tnew, te_l, te_r)))
        unew, unew_l, unew_r = zip(*tt) if len(tt) > 0 else ((), (), ())
    # meta_new = ((unew, Dnew), ...)

    meta_new = tuple((u, (ls_l.D[l], ls_r.D[r])) for u, l, r in zip(unew, unew_l, unew_r))
    return meta_new, meta_mrg, ls_l, ls_r, unew_l, unew_r


def _leg_structure_merge(teff, tlegs, Deff, Dlegs):
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
    return dec, Dtot


def _unmerge_from_matrix(A, ls_l, ls_r):
    meta = []
    for il, ir in product(ls_l.dec, ls_r.dec):
        ic = il + ir
        if ic in A:
            for (tl, (sl, _, Dl)), (tr, (sr, _, Dr)) in product(ls_l.dec[il].items(), ls_r.dec[ir].items()):
                meta.append((tl + tr, ic, sl, sr, Dl + Dr))
    return ls_l.config.backend.unmerge_from_matrix(A, meta)


def _unmerge_from_diagonal(A, ls):
    meta = tuple((ta + ta, ia, sa) for ia in ls.dec for ta, (sa, _, _) in ls.dec[ia].items())
    Anew = ls.config.backend.unmerge_from_diagonal(A, meta)
    return {ind: ls.config.backend.diag_create(Anew[ind]) for ind in Anew}


class _LegDecomposition:
    """Information about internal structure of leg resulting from fusions."""
    def __init__(self, config=None, s=(), news=1):
        self.s =  (s,) if isinstance(s, int) else tuple(s) # signature of fused legs
        self.nlegs = len(self.s)  # number of fused legs
        self.config = config
        self.news = news  # signature of effective leg
        self.D = {}
        self.dec = {}  # leg's structure/ decomposition

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

    def leg_struct_for_merged(self, teff, tlegs, Deff, Dlegs):
        """ Calculate meta-information about bond dimensions for merging into one leg. """
        shape_t = list(tlegs.shape)
        shape_t[1] = shape_t[1] + 1
        tcom = np.empty(shape_t, dtype=int)
        tcom[:, 0, :] = teff
        tcom[:, 1:, :] = tlegs
        tcom = tcom.reshape((shape_t[0], shape_t[1]*shape_t[2]))
        ucom, icom = np.unique(tcom, return_index=True, axis=0)
        Dlow = 0
        for ii, tt in zip(icom, ucom):
            t0 = tuple(tt[:self.config.sym.nsym])
            t1 = tuple(tt[self.config.sym.nsym:])
            if t0 not in self.dec:
                self.dec[t0] = {}
                Dlow = 0
            Dtop = Dlow + Deff[ii]
            self.dec[t0][t1] = ((Dlow, Dtop), Deff[ii], tuple(Dlegs[ii]))
            Dlow = Dtop
            self.D[t0] = Dtop

    def leg_struct_trivial(self, A, axis):
        """ Meta-information for single leg. """
        nsym = self.config.sym.nsym
        for ind, val in A.items():
            t = ind[nsym * axis: nsym * (axis + 1)]
            D = self.config.backend.get_shape(val)[axis]
            self.dec[t] = {t: ((0, D), D, (D,))}

    def leg_struct_for_truncation(self, A, opts, sorting='svd'):
        r"""Gives slices for truncation of 1d matrices according to tolerance, D_block, D_total.

        A should be dict of ordered 1d arrays.
        Sorting gives information about ordering outputed by a particular splitting funcion:
        Usual convention is that for svd A[ind][0] is largest; and for eigh A[ind][-1] is largest.
        """
        maxS = 0 if len(A) == 0 else self.config.backend.maximum(A)
        Dmax, D_keep = {}, {}
        for ind in A:
            Dmax[ind] = self.config.backend.get_size(A[ind])
            D_keep[ind] = min(opts['D_block'], Dmax[ind])
        if (opts['tol'] > 0) and (maxS > 0):  # truncate to relative tolerance
            for ind in D_keep:
                D_keep[ind] = min(D_keep[ind], self.config.backend.count_greater(A[ind], maxS * opts['tol']))
        if sum(D_keep[ind] for ind in D_keep) > opts['D_total']:  # truncate to total bond dimension
            if 'keep_multiplets' in opts.keys():
                order = self.config.backend.select_global_largest(A, D_keep, opts['D_total'], sorting,
                                                                  keep_multiplets=opts['keep_multiplets'], eps_multiplet=opts['eps_multiplet'])
            else:
                order = self.config.backend.select_global_largest(A, D_keep, opts['D_total'], sorting)
            low = 0
            for ind in D_keep:
                high = low + D_keep[ind]
                D_keep[ind] = sum((low <= order) & (order < high))
                low = high

        # check symmetry related blocks and truncate to equal length
        if 'keep_multiplets' in opts.keys() and opts['keep_multiplets']:
            ind_list = [np.asarray(k) for k in D_keep]
            for ind in ind_list:
                t = tuple(ind)
                tn = tuple(-ind)
                minD_sector = min(D_keep[t], D_keep[tn])
                D_keep[t] = D_keep[tn] = minD_sector
                if -ind in ind_list:
                    ind_list.remove(-ind)

        for ind in D_keep:
            if D_keep[ind] > 0:
                Dslc = self.config.backend.range_largest(D_keep[ind], Dmax[ind], sorting)
                self.dec[ind] = {ind: (Dslc, D_keep[ind], (D_keep[ind],))}


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

#     t_new = np.empty((len(self.A), new_ndim, self.config.sym.nsym), dtype=int)
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
