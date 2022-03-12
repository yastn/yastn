""" Methods creating a new yast tensor """

from itertools import chain, repeat, accumulate, product
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _flatten, _struct
from ._merging import _flip_hf
from ._tests import YastError

__all__ = ['match_legs', 'block']


def fill_tensor(a, t=(), D=(), val='rand', dtype=None):
    r"""
    Create all possible blocks based on s, n and list of charges for all legs.

    Brute-force check all possibilities and select the ones satisfying f(t@s) == n for each symmetry generator f.
    Initialize each possible block with sizes given by D.
    Old data in the tensor are reset.

    Parameters
    ----------
    a : Tensor

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

    dtype : str
        desired dtype, overrides default_dtype specified in config of tensor `a`

    Examples
    --------
    D = 5  # ndim = 1
    D = (1, 2, 3)  # nsym = 0, ndim = 3
    t = [0, (-2, 0), (2, 0)], D = [1, (1, 2), (1, 3)]  # nsym = 1 ndim = 3
    t = [[(0, 0)], [(-2, -2), (0, 0), (-2, 0), (0, -2)], [(2, 2), (0, 0), (2, 0), (0, 2)]], \
    D = [1, (1, 4, 2, 2), (1, 9, 3, 3)]  # nsym = 2 ndim = 3
    """

    if not dtype:
        assert hasattr(a.config, 'default_dtype'), "Either dtype or valid config has to be provided"
        dtype = a.config.default_dtype

    D = (D,) if isinstance(D, int) else D
    t = (t,) if isinstance(t, int) else t

    if a.config.sym.NSYM == 0:
        if a.isdiag and len(D) == 1:
            D = D + D
        if len(D) != a.ndim_n:
            raise YastError("Number of elements in D does not match tensor rank.")
        tset = np.zeros((1, a.ndim_n, a.config.sym.NSYM))
        Dset = np.array(D, dtype=int).reshape(1, a.ndim_n)
    else:  # a.config.sym.NSYM >= 1
        D = (D,) if (a.ndim_n == 1 or a.isdiag) and isinstance(D[0], int) else D
        t = (t,) if (a.ndim_n == 1 or a.isdiag) and isinstance(t[0], int) else t
        D = D + D if a.isdiag and len(D) == 1 else D
        t = t + t if a.isdiag and len(t) == 1 else t

        D = list((x,) if isinstance(x, int) else x for x in D)
        t = list((x,) if isinstance(x, int) else x for x in t)

        if len(D) != a.ndim_n:
            raise YastError("Number of elements in D does not match tensor rank.")
        if len(t) != a.ndim_n:
            raise YastError("Number of elements in t does not match tensor rank.")
        for x, y in zip(D, t):
            if len(x) != len(y):
                raise YastError("Elements of t and D do not match")

        comb_D = list(product(*D))
        comb_t = list(product(*t))
        lcomb_t = len(comb_t)
        comb_t = list(_flatten(comb_t))
        comb_t = np.array(comb_t, dtype=int).reshape((lcomb_t, len(a.struct.s), len(a.struct.n)))
        comb_D = np.array(comb_D, dtype=int).reshape((lcomb_t, len(a.struct.s)))
        sa = np.array(a.struct.s, dtype=int)
        na = np.array(a.struct.n, dtype=int)
        ind = np.all(a.config.sym.fuse(comb_t, sa, 1) == na, axis=1)
        tset = comb_t[ind]
        Dset = comb_D[ind]

    if a.isdiag and np.any(Dset[:, 0] != Dset[:, 1]):
        raise YastError("Diagonal tensor requires the same bond dimensions on both legs.")
    Dp = Dset[:, 0] if a.isdiag else np.prod(Dset, axis=1, dtype=int)
    Dsize = np.sum(Dp)

    meta = [(tuple(ts.flat), tuple(Ds), dp) for ts, Ds, dp in zip(tset, Dset, Dp)]
    meta = sorted(meta, key=lambda x: x[0])

    a_t, a_D, a_Dp = zip(*meta) if len(meta) > 0 else ((), (), ())
    a_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(a_Dp), a_Dp))
    a.struct = a.struct._replace(t=a_t, D=a_D, Dp=a_Dp, sl=a_sl)

    a._data = _init_block(a.config, Dsize, val, dtype)
    for n in range(a.ndim_n):
        a.get_leg_structure(n, native=True)  # here checks the consistency of bond dimensions


def set_block(a, ts=(), Ds=None, val='zeros', dtype=None):
    """
    Add new block to tensor or change the existing one.

    This is the intended way to add new blocks by hand.
    Checks if bond dimensions of the new block are consistent with the existing ones.
    Updates meta-data.

    Parameters
    ----------
    a : Tensor

    ts : tuple
        charges identifing the block, t = (sym1leg1, sym2leg1, sym1leg2, sym2leg2, ...)
        If nsym == 0, it is not taken into account.

    Ds : tuple
        bond dimensions of the block. Ds = (leg1, leg2, leg3)
        If Ds not given, tries to read it from existing blocks.

    val : str, nparray, list
        'randR', 'rand' (use current dtype float or complex), 'ones', 'zeros'
        for nparray setting Ds is needed.

    dtype : str
        desired dtype, overrides default_dtype specified in config of tensor `a`
    """
    if dtype is None:
        assert hasattr(a.config, 'default_dtype'), "Either dtype or valid config has to be provided"
        dtype = a.config.default_dtype

    if isinstance(Ds, int):
        Ds = (Ds,)
    if isinstance(ts, int):
        ts = (ts,)
    if a.isdiag and Ds is not None and len(Ds) == 1:
        Ds = Ds + Ds
    ts = tuple(_flatten(ts))
    if a.isdiag and len(ts) == a.config.sym.NSYM:
        ts = ts + ts

    if len(ts) != a.ndim_n * a.config.sym.NSYM:
        raise YastError('Wrong size of ts.')
    if Ds is not None and len(Ds) != a.ndim_n:
        raise YastError('Wrong size of Ds.')

    ats = np.array(ts, dtype=int).reshape((1, a.ndim_n, a.config.sym.NSYM))
    sa = np.array(a.struct.s, dtype=int)
    na = np.array(a.struct.n, dtype=int)
    if not np.all(a.config.sym.fuse(ats, sa, 1) == na):
        raise YastError('Charges ts are not consistent with the symmetry rules: f(t @ s) == n')

    if Ds is None:  # attempt to read Ds from existing blocks.
        Ds = []
        tD = [a.get_leg_structure(n, native=True) for n in range(a.ndim_n)]
        for n in range(a.ndim_n):
            try:
                Ds.append(tD[n][tuple(ats[0, n, :].flat)])
            except KeyError as err:
                raise YastError('Provided Ds. Cannot infer all bond dimensions from existing blocks.') from err
        Ds = tuple(Ds)

    if a.isdiag and Ds[0] != Ds[1]:
        raise YastError("Diagonal tensor requires the same bond dimensions on both legs.")
    Dsize = Ds[0] if a.isdiag else np.prod(Ds, dtype=int)

    ind = sum(t < ts for t in a.struct.t)
    ind2 = ind
    if ind < len(a.struct.t) and a.struct.t[ind] == ts:
        ind2 += 1
        a._data = a.config.backend.delete(a._data, slice(*a.struct.sl[ind]))

    pos = 0 if ind == 0 else a.struct.sl[ind - 1][1]
    a._data = a.config.backend.insert(a._data, pos, _init_block(a.config, Dsize, val, dtype))

    a_t = a.struct.t[:ind] + (ts,) + a.struct.t[ind2:]
    a_D = a.struct.D[:ind] + (Ds,) + a.struct.D[ind2:]
    a_Dp = a.struct.Dp[:ind] + (Dsize,) + a.struct.Dp[ind2:]
    a_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(a_Dp), a_Dp))
    a.struct = a.struct._replace(t=a_t, D=a_D, Dp=a_Dp, sl=a_sl)

    for n in range(a.ndim_n):
        a.get_leg_structure(n, native=True)  # here checks the consistency of bond dimensions


def _init_block(config, Dsize, val, dtype):
    if isinstance(val, str):
        if val == 'zeros':
            return config.backend.zeros((Dsize,), dtype=dtype, device=config.device)
        if val in ('randR', 'rand'):
            return config.backend.randR((Dsize,), device=config.device)
        elif val == 'randC':
            return config.backend.randC((Dsize,), device=config.device)
        elif val == 'ones':
            return config.backend.ones((Dsize,), dtype=dtype, device=config.device)
    else:
        x = config.backend.to_tensor(val, Ds=Dsize, dtype=dtype, device=config.device)
        if config.backend.get_size(x) == Dsize ** 2:
            x = config.backend.diag_get(x.reshape(Dsize, Dsize))
        return x


def match_legs(tensors=None, legs=None, conjs=None, val='ones', n=None, isdiag=False):
    r"""
    Initialize tensor matching legs of existing tensors, so that it can be contracted with those tensors.

    Finds all matching symmetry sectors and their bond dimensions and passes it to :meth:`Tensor.fill_tensor`.

    TODO: the set of tensors should reside on the same device. Optional destination device might be added

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
    t, D, s, lf, hf = [], [], [], [], []
    if conjs is None:
        conjs = (0,) * len(tensors)
    for nf, te, cc in zip(legs, tensors, conjs):
        lf.append(te.meta_fusion[nf])
        un, = _unpack_axes(te.meta_fusion, (nf,))
        for nn in un:
            tdn = te.get_leg_structure(nn, native=True)
            t.append(tuple(tdn.keys()))
            D.append(tuple(tdn.values()))
            s.append(te.struct.s[nn] * (2 * cc - 1))
            hf.append(te.hard_fusion[nn] if cc == 1 else _flip_hf(te.hard_fusion[nn]))
    a = tensors[0].__class__(config=tensors[0].config, s=s, n=n, isdiag=isdiag, meta_fusion=lf, hard_fusion=hf)
    a.fill_tensor(t=t, D=D, val=val)
    return a


def block(tensors, common_legs=None):
    """
    Assemble new tensor by blocking a set of tensors.

    TODO: the set of tensors should reside on the same device. Optional destination device might be added

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
    out_b = tuple((ii,) for ii in range(tn0.ndim) if ii not in out_s)
    pos = list(_clear_axes(*tensors))
    lind = tn0.ndim - len(out_s)
    for ind in pos:
        if len(ind) != lind:
            raise YastError('Wrong number of coordinates encoded in tensors.keys()')

    out_s, =  _unpack_axes(tn0.meta_fusion, out_s)
    u_b = tuple(_unpack_axes(tn0.meta_fusion, *out_b))
    out_b = tuple(chain(*u_b))
    pos = tuple(tuple(chain.from_iterable(repeat(x, len(u)) for x, u in zip(ind, u_b))) for ind in pos)

    for ind, tn in tensors.items():
        ind, = _clear_axes(ind)
        if tn.struct.s != tn0.struct.s:
            raise YastError('Signatues of blocked tensors are inconsistent.')
        if tn.struct.n != tn0.struct.n:
            raise YastError('Tensor charges of blocked tensors are inconsistent.')
        if tn.meta_fusion != tn0.meta_fusion or tn.hard_fusion != tn0.hard_fusion:
            raise YastError('Fusion structures of blocked tensors are inconsistent.')
        if tn.isdiag != tn0.isdiag:
            raise YastError('Block can talk either only diagonal of only nondiagonal tensors.')

    posa = np.ones((len(pos), tn0.ndim_n), dtype=int)
    posa[:, np.array(out_b, dtype=np.intp)] = np.array(pos, dtype=int).reshape(len(pos), -1)

    tDs = []  # {leg: {charge: {position: D, 'D' : Dtotal}}}
    for n in range(tn0.ndim_n):
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
            tDl[t] = {p: (aD - D, aD) for p, D, aD in zip(ps, Ds, accumulate(Ds))}
            tDl[t]['Dtot'] = sum(Ds)
        tDs.append(tDl)

    # all unique blocks
    # meta_new = {tind: Dtot};  #meta_block = [(tind, pos, Dslc)]
    meta_new, meta_block = {}, []
    for pind, pa in zip(tensors, posa):
        a = tensors[pind]
        tset = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
        for tind, slind, Dind, t in zip(a.struct.t, a.struct.sl, a.struct.D, tset):
            if tind not in meta_new:
                meta_new[tind] = tuple(tDs[n][tuple(t[n].flat)]['Dtot'] for n in range(a.ndim_n))
            meta_block.append((tind, slind, Dind, pind, tuple(tDs[n][tuple(t[n].flat)][pa[n]] for n in range(a.ndim_n))))
    meta_block = tuple(sorted(meta_block, key=lambda x: x[0]))
    meta_new = tuple(sorted(meta_new.items()))
    c_t = tuple(t for t, _ in meta_new)
    c_D = tuple(D for _, D in meta_new)
    c_Dp = tuple(np.prod(c_D, axis=1))
    c_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(c_Dp), c_Dp))
    c_struct = _struct(n=a.struct.n, s=a.struct.s, t=c_t, D=c_D, Dp=c_Dp, sl=c_sl)
    meta_new = tuple(zip(c_t, c_D, c_sl))
    Dsize = c_sl[-1][1] if len(c_sl) > 0 else 0

    c = tn0.__class__(config=a.config, isdiag=a.isdiag, struct=c_struct,
                        meta_fusion=tn0.meta_fusion, hard_fusion=tn0.hard_fusion)
    c._data = c.config.backend.merge_super_blocks(tensors, meta_new, meta_block, Dsize)
    return c
