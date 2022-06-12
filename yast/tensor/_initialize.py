""" Methods creating a new yast tensor """

from itertools import product
import numpy as np
from ._auxliary import _flatten
from ._tests import YastError, _test_tD_consistency


def __setitem__(a, key, newvalue):
    """
    Parameters
    ----------
    key : tuple(int)
        charges of the block

    Update data corresponding the block. The data should be consistent with shape
    """
    key = tuple(_flatten(key))
    try:
        ind = a.struct.t.index(key)
    except ValueError as exc:
        raise YastError('Tensor does not have a block specify by the key.') from exc
    a._data[slice(*a.struct.sl[ind])] = newvalue.reshape(-1)


def fill_tensor(a, t=(), D=(), val='rand'):  # dtype = None
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
        'rand' (use current dtype float or complex), 'ones', 'zeros'

    dtype : str
        desired dtype, overrides current dtype of 'a'

    Examples
    --------
    D = 5  # ndim = 1
    D = (1, 2, 3)  # nsym = 0, ndim = 3
    t = [0, (-2, 0), (2, 0)], D = [1, (1, 2), (1, 3)]  # nsym = 1 ndim = 3
    t = [[(0, 0)], [(-2, -2), (0, 0), (-2, 0), (0, -2)], [(2, 2), (0, 0), (2, 0), (0, 2)]], \
    D = [1, (1, 4, 2, 2), (1, 9, 3, 3)]  # nsym = 2 ndim = 3
    """

    # if dtype is None:

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
    a._data = _init_block(a.config, Dsize, val, dtype=a.yast_dtype, device=a.device)
    _test_tD_consistency(a.struct)


def set_block(a, ts=(), Ds=None, val='zeros'):
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
        'rand' (use current dtype float or complex), 'ones', 'zeros'
        for nparray setting Ds is needed.

    dtype : str
        desired dtype, overrides default_dtype specified in config of tensor `a`
    """
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
        raise YastError('Size of ts is not consistent with tensor rank and the number of symmetry sectors.')
    if Ds is not None and len(Ds) != a.ndim_n:
        raise YastError('Size of Ds is not consistent with tensor rank.')

    ats = np.array(ts, dtype=int).reshape((1, a.ndim_n, a.config.sym.NSYM))
    sa = np.array(a.struct.s, dtype=int)
    na = np.array(a.struct.n, dtype=int)
    if not np.all(a.config.sym.fuse(ats, sa, 1) == na):
        raise YastError('Charges ts are not consistent with the symmetry rules: f(t @ s) == n')

    if Ds is None:  # attempt to read Ds from existing blocks.
        Ds = []
        legs = a.get_legs(range(a.ndim_n), native=True)
        for n, leg in enumerate(legs):
            try:
                Ds.append(leg.D[leg.t.index(tuple(ats[0, n, :].flat))])
            except ValueError as err:
                raise YastError('Provided Ds. Cannot infer all bond dimensions from existing blocks.') from err
        Ds = tuple(Ds)

    if a.isdiag and Ds[0] != Ds[1]:
        raise YastError("Diagonal tensor requires the same bond dimensions on both legs.")
    Dsize = Ds[0] if a.isdiag else np.prod(Ds, dtype=int)

    ind = sum(t < ts for t in a.struct.t)
    ind2 = ind
    if ind < len(a.struct.t) and a.struct.t[ind] == ts:
        ind2 += 1
        a._data = a.config.backend.delete(a._data, a.struct.sl[ind])

    pos = 0 if ind == 0 else a.struct.sl[ind - 1][1]
    new_block = _init_block(a.config, Dsize, val, dtype=a.yast_dtype, device=a.device)
    a._data = a.config.backend.insert(a._data, pos, new_block)

    a_t = a.struct.t[:ind] + (ts,) + a.struct.t[ind2:]
    a_D = a.struct.D[:ind] + (Ds,) + a.struct.D[ind2:]
    a_Dp = a.struct.Dp[:ind] + (Dsize,) + a.struct.Dp[ind2:]
    a_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(a_Dp), a_Dp))
    a.struct = a.struct._replace(t=a_t, D=a_D, Dp=a_Dp, sl=a_sl)
    _test_tD_consistency(a.struct)


def _init_block(config, Dsize, val, dtype, device):
    if isinstance(val, str):
        if val == 'zeros':
            return config.backend.zeros((Dsize,), dtype=dtype, device=device)
        if val == 'rand':
            return config.backend.rand((Dsize,), dtype=dtype, device=device)
        if val == 'ones':
            return config.backend.ones((Dsize,), dtype=dtype, device=device)
        raise YastError('val should be in ("zeros", "ones", "rand") or an array of the correct size')
    x = config.backend.to_tensor(val, Ds=Dsize, dtype=dtype, device=device)
    if config.backend.get_size(x) == Dsize ** 2:
        x = config.backend.diag_get(x.reshape(Dsize, Dsize))
    return x
