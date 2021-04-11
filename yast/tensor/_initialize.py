""" Methods creating a new yast tensor """

from itertools import chain, repeat, accumulate, product
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _flatten, _tarray, YastError

__all__ = ['match_legs', 'block', 'matching_tensor']


def copy_empty(a):
    """ Return a copy of the tensor, but without copying blocks. """
    return a.__class__(config=a.config, s=a.s, n=a.n, isdiag=a.isdiag, meta_fusion=a.meta_fusion)


def fill_tensor(a, t=(), D=(), val='rand'):
    r"""
    Create all possible blocks based on s, n and list of charges for all legs.

    Brute-force check all possibilities and select the ones satisfying f(t@s) == n for each symmetry generator f.
    Initialize each possible block with sizes given by D.

    Parameters
    ----------
    t : list
        All possible combination of charges for each leg:
        t = [[(leg1sym1, leg1sym2), ... ], [(leg2sym1, leg2sym2), ... )]
        If nsym is 0, it is not taken into account.
        When somewhere there is only one value and it is unambiguous, tuple can typically be replaced by int, see examples.

    D : tuple
        list of bond dimensions on all legs
        If nsym == 0, D = [leg1, leg2, leg3]
        If nsym >= 1, it should match the form of t
        When somewhere there is only one value tuple can typically be replaced by int.

    val : str
        'randR', 'rand' (use current dtype float or complex), 'ones', 'zeros'

    Examples
    --------
    D = 5  # ndim = 1
    D = (1, 2, 3)  # nsym = 0, ndim = 3
    t = [0, (-2, 0), (2, 0)], D = [1, (1, 2), (1, 3)]  # nsym = 1 ndim = 3
    t = [[(0, 0)], [(-2, -2), (0, 0), (-2, 0), (0, -2)], [(2, 2), (0, 0), (2, 0), (0, 2)]], \
    D = [1, (1, 4, 2, 2), (1, 9, 3, 3)]  # nsym = 2 ndim = 3
    """
    D = (D,) if isinstance(D, int) else D
    t = (t,) if isinstance(t, int) else t

    if a.config.sym.nsym == 0:
        if a.isdiag and len(D) == 1:
            D = D + D
        if len(D) != a.nlegs:
            raise YastError("Number of elements in D does not match tensor rank.")
        tset = np.zeros((1, a.nlegs, a.config.sym.nsym))
        Dset = np.array(D, dtype=int).reshape(1, a.nlegs)
    else:  # a.config.sym.nsym >= 1
        D = (D,) if (a.nlegs == 1 or a.isdiag) and isinstance(D[0], int) else D
        t = (t,) if (a.nlegs == 1 or a.isdiag) and isinstance(t[0], int) else t
        D = D + D if a.isdiag and len(D) == 1 else D
        t = t + t if a.isdiag and len(t) == 1 else t

        D = list((x,) if isinstance(x, int) else x for x in D)
        t = list((x,) if isinstance(x, int) else x for x in t)

        if len(D) != a.nlegs:
            raise YastError("Number of elements in D does not match tensor rank.")
        if len(t) != a.nlegs:
            raise YastError("Number of elements in t does not match tensor rank.")
        for x, y in zip(D, t):
            if len(x) != len(y):
                raise YastError("Elements of t and D do not match")

        # comb_t = list(tuple(tuple(t) if isinstance(t, int) else t for t in ts) for ts in product(*t))
        comb_D = list(product(*D))
        comb_t = list(product(*t))
        lcomb_t = len(comb_t)
        comb_t = list(_flatten(comb_t))
        comb_t = np.array(comb_t, dtype=int).reshape(lcomb_t, a.nlegs, a.config.sym.nsym)
        comb_D = np.array(comb_D, dtype=int).reshape(lcomb_t, a.nlegs)
        ind = np.all(a.config.sym.fuse(comb_t, a.s, 1) == a.n, axis=1)
        tset = comb_t[ind]
        Dset = comb_D[ind]

    for ts, Ds in zip(tset, Dset):
        set_block(a, tuple(ts.flat), tuple(Ds), val)


def set_block(a, ts=(), Ds=None, val='zeros'):
    """
    Add new block to tensor or change the existing one.

    This is the intended way to add new blocks by hand.
    Checks if bond dimensions of the new block are consistent with the existing ones.
    Updates meta-data.

    Parameters
    ----------
    ts : tuple
        charges identifing the block, t = (sym1leg1, sym2leg1, sym1leg2, sym2leg2, ...)
        If nsym == 0, it is not taken into account.

    Ds : tuple
        bond dimensions of the block. Ds = (leg1, leg2, leg3)
        If Ds not given, tries to read it from existing blocks.

    val : str, nparray, list
        'randR', 'rand' (use current dtype float or complex), 'ones', 'zeros'
        for nparray setting Ds is needed.
    """
    if isinstance(Ds, int):
        Ds = (Ds,)
    if isinstance(ts, int):
        ts = (ts,)
    if a.isdiag and Ds is not None and len(Ds) == 1:
        Ds = Ds + Ds
    if a.isdiag and len(ts) == a.config.sym.nsym:
        ts = ts + ts

    if len(ts) != a.nlegs * a.config.sym.nsym:
        raise YastError('Wrong size of ts.')
    if Ds is not None and len(Ds) != a.nlegs:
        raise YastError('Wrong size of Ds.')

    ats = np.array(ts, dtype=int).reshape(1, a.nlegs, a.config.sym.nsym)
    if not np.all(a.config.sym.fuse(ats, a.s, 1) == a.n):
        raise YastError('Charges ts are not consistent with the symmetry rules: t @ s - n != 0')

    if isinstance(val, str):
        if Ds is None:  # attempt to read Ds from existing block
            Ds = []
            tD = [a.get_leg_structure(n, native=True) for n in range(a.nlegs)]
            for n in range(a.nlegs):
                try:
                    Ds.append(tD[n][tuple(ats[0, n, :].flat)])
                except KeyError as err:
                    raise YastError('Provided Ds. Cannot infer all bond dimensions from existing blocks.') from err
            Ds = tuple(Ds)

        if val == 'zeros':
            a.A[ts] = a.config.backend.zeros(Ds, dtype=a.config.dtype, device=a.config.device)
        elif val == 'randR':
            a.A[ts] = a.config.backend.randR(Ds, dtype=a.config.dtype, device=a.config.device)
        elif val == 'rand':
            a.A[ts] = a.config.backend.rand(Ds, dtype=a.config.dtype, device=a.config.device)
        elif val == 'ones':
            a.A[ts] = a.config.backend.ones(Ds, dtype=a.config.dtype, device=a.config.device)
        if a.isdiag:
            a.A[ts] = a.config.backend.diag_get(a.A[ts])
            a.A[ts] = a.config.backend.diag_create(a.A[ts])
    else:  # enforce that Ds is provided to increase clarity of the code
        if a.isdiag and val.ndim == 1 and np.prod(Ds) == (val.size**2):
            a.A[ts] = a.config.backend.to_tensor(np.diag(val), Ds, dtype=a.config.dtype, device=a.config.device)
        else:
            a.A[ts] = a.config.backend.to_tensor(val, Ds=Ds, dtype=a.config.dtype, device=a.config.device)
    a.update_struct()
    tD = [a.get_leg_structure(n, native=True) for n in range(a.nlegs)]  # here checks the consistency of bond dimensions


def match_legs(tensors=None, legs=None, conjs=None, val='ones', n=None, isdiag=False):
    r"""
    Initialize tensor matching legs of existing tensors, so that it can be contracted with those tensors.

    Finds all matching symmetry sectors and their bond dimensions and passes it to :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    tensors: list
        list of tensors -- they should not be diagonal to properly identify signature.
    legs: list
        and their corresponding legs to match
    conjs: list
        if tensors are entering dot as conjugated
    val: str
        'randR', 'rand', 'ones', 'zeros'
    """
    t, D, s, lf = [], [], [], []
    if conjs is None:
        conjs = (0,) * len(tensors)
    for nf, te, cc in zip(legs, tensors, conjs):
        lf.append(te.meta_fusion[nf])
        un, = _unpack_axes(te, (nf,))
        for nn in un:
            tdn = te.get_leg_structure(nn, native=True)
            t.append(tuple(tdn.keys()))
            D.append(tuple(tdn.values()))
            s.append(te.s[nn] * (2 * cc - 1))
    a = tensors[0].__class__(config=tensors[0].config, s=s, n=n, isdiag=isdiag, meta_fusion=lf)
    a.fill_tensor(t=t, D=D, val=val)
    return a


matching_tensor = match_legs


def block(tensors, common_legs=None):
    """ Assemble new tensor by blocking a set of tensors.

        Parameters
        ----------
        tensors : dict
            dictionary of tensors {(x,y,...): tensor at position x,y,.. in the new, blocked super-tensor}.
            Length of tuple should be equall to tensor.ndim - len(common_legs)

        common_legs : list
            Legs that are not blocked.
            This is equivalently to all tensors having the same position (not specified) in the supertensor on that leg.
    """
    out_s, = ((),) if common_legs is None else _clear_axes(common_legs)
    tn0 = next(iter(tensors.values()))  # first tensor; used to initialize new objects and retrive common values
    out_b = tuple((ii,) for ii in range(tn0.mlegs) if ii not in out_s)
    pos = list(_clear_axes(*tensors))
    lind = tn0.mlegs - len(out_s)
    for ind in pos:
        if len(ind) != lind:
            raise YastError('Wrong number of coordinates encoded in tensors.keys()')

    out_s, =  _unpack_axes(tn0, out_s)
    u_b = tuple(_unpack_axes(tn0, *out_b))
    out_b = tuple(chain(*u_b))
    pos = tuple(tuple(chain.from_iterable(repeat(x, len(u)) for x, u in zip(ind, u_b))) for ind in pos)

    for ind, tn in tensors.items():
        ind, = _clear_axes(ind)
        if tn.nlegs != tn0.nlegs or tn.meta_fusion != tn0.meta_fusion or\
           not np.all(tn.s == tn0.s) or not np.all(tn.n == tn0.n) or\
           tn.isdiag != tn0.isdiag:
            raise YastError('Ndims, signatures, total charges or fusion trees of blocked tensors are inconsistent.')

    posa = np.ones((len(pos), tn0.nlegs), dtype=int)
    posa[:, np.array(out_b, dtype=np.intp)] = np.array(pos, dtype=int).reshape(len(pos), -1)

    tDs = []  # {leg: {charge: {position: D, 'D' : Dtotal}}}
    for n in range(tn0.nlegs):
        tDl = {}
        for tn, pp in zip(tensors.values(), posa):
            tDn = tn.get_leg_structure(n, native=True)
            for t, D in tDn.items():
                if t in tDl:
                    if (pp[n] in tDl[t]) and (tDl[t][pp[n]] != D):
                        raise YastError('Dimensions of blocked tensors are not consistent')
                    tDl[t][pp[n]] = D
                else:
                    tDl[t] = {pp[n]: D}
        for t, pD in tDl.items():
            ps = sorted(pD.keys())
            Ds = [pD[p] for p in ps]
            tDl[t] = {p: (aD-D, aD) for p, D, aD in zip(ps, Ds, accumulate(Ds))}
            tDl[t]['Dtot'] = sum(Ds)
        tDs.append(tDl)

    # all unique blocks
    # meta_new = {tind: Dtot};  #meta_block = [(tind, pos, Dslc)]
    meta_new, meta_block = {}, []
    for pind, pa in zip(tensors, posa):
        a = tensors[pind]
        tset = _tarray(a)
        for tind, t in zip(a.struct.t, tset):
            if tind not in meta_new:
                meta_new[tind] = tuple(tDs[n][tuple(t[n].flat)]['Dtot'] for n in range(a.nlegs))
            meta_block.append((tind, pind, tuple(tDs[n][tuple(t[n].flat)][pa[n]] for n in range(a.nlegs))))
    meta_new = tuple((ts, Ds) for ts, Ds in meta_new.items())

    c = tn0.__class__(config=a.config, s=a.s, isdiag=a.isdiag, n=a.n, meta_fusion=tn0.meta_fusion)
    c.A = c.config.backend.merge_super_blocks(tensors, meta_new, meta_block, a.config.dtype, c.config.device)
    c.update_struct()
    return c
