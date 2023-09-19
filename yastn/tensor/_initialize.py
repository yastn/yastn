""" Methods creating a new yastn tensor """

from itertools import product
import numpy as np
from ._auxliary import _flatten
from ._tests import YastnError, _test_tD_consistency


def __setitem__(a, key, newvalue):
    """
    Update data of the selected block.

    The data (its shape) should be consistent with
    the dimensions of the charge sectors where the block belongs.

    Parameters
    ----------
    key : tuple(int)
        charges of the block
    """
    key = tuple(_flatten(key))
    try:
        ind = a.struct.t.index(key)
    except ValueError as exc:
        raise YastnError('Tensor does not have a block specified by the key.') from exc
    a._data[slice(*a.struct.sl[ind])] = newvalue.reshape(-1)


def _fill_tensor(a, t=(), D=(), val='rand'):  # dtype = None
    r"""
    Create all allowed blocks based on signature ``s``, total charge ``n``,
    and a set of charge sectors ``t`` for each leg of the tensor.

    First, all allowed blocks are identified by checking the
    :ref:`selection rule<symmetry selection rule>`.
    Then each allowed block is created as a tensor with
    sizes specified in ``D`` and filled with value ``val``.

    .. note::
        This operation overwrites the data of the tensor.

    Parameters
    ----------
    a : yastn.Tensor

    t : list[list[int]] or list[list[list[int]]]
        list of charge sectors for each leg of the tensor, see examples.
        In case of tensor without symmetry this argument is ignored.

    D : list[int] or list[list[int]]
        list of sector sizes for each leg of the tensor, see examples.

    val : str
        ``'rand'``, ``'ones'``, or  ``'zeros'``
    """
    D = (D,) if isinstance(D, int) else D
    t = (t,) if isinstance(t, int) else t

    if a.config.sym.NSYM == 0:
        if a.isdiag and len(D) == 1:
            D = D + D
        if len(D) != a.ndim_n:
            raise YastnError("Number of elements in D does not match tensor rank.")
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
            raise YastnError("Number of elements in D does not match tensor rank.")
        if len(t) != a.ndim_n:
            raise YastnError("Number of elements in t does not match tensor rank.")
        for x, y in zip(D, t):
            if len(x) != len(y):
                raise YastnError("Elements of t and D do not match")

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
        raise YastnError("Diagonal tensor requires the same bond dimensions on both legs.")
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
    Checks if bond dimensions of the new block are consistent with the existing ones
    and updates the legs of the tensors accordingly.

    Parameters
    ----------
    ts : tuple(int) or tuple(tuple(int))
        Charges identifing the block. Ignored if tensor has no symmetry.

    Ds : tuple(int)
        Dimensions of the block. If ``None``, tries to infer
        dimensions from legs of the tensor.

    val : str, tensor-like
        recognized string values are ``'rand'``, ``'ones'``,`or  ``'zeros'``.
        Otherwise any tensor-like format such as nested list, numpy.ndarray, etc.,
        can be used provided it is supported by :doc:`tensor's backend </tensor/configuration>`.
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
        raise YastnError('Size of ts is not consistent with tensor rank and the number of symmetry sectors.')
    if Ds is not None and len(Ds) != a.ndim_n:
        raise YastnError('Size of Ds is not consistent with tensor rank.')

    ats = np.array(ts, dtype=int).reshape((1, a.ndim_n, a.config.sym.NSYM))
    sa = np.array(a.struct.s, dtype=int)
    na = np.array(a.struct.n, dtype=int)
    if not np.all(a.config.sym.fuse(ats, sa, 1) == na):
        raise YastnError('Charges ts are not consistent with the symmetry rules: f(t @ s) == n')

    if Ds is None:  # attempt to read Ds from existing blocks.
        Ds = []
        legs = a.get_legs(range(a.ndim_n), native=True)
        for n, leg in enumerate(legs):
            try:
                Ds.append(leg.D[leg.t.index(tuple(ats[0, n, :].flat))])
            except ValueError as err:
                raise YastnError('Provided Ds. Cannot infer all bond dimensions from existing blocks.') from err
        Ds = tuple(Ds)

    if a.isdiag and Ds[0] != Ds[1]:
        raise YastnError("Diagonal tensor requires the same bond dimensions on both legs.")
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
        raise YastnError('val should be in ("zeros", "ones", "rand") or an array of the correct size')
    x = config.backend.to_tensor(val, Ds=Dsize, dtype=dtype, device=device)
    if config.backend.get_size(x) == Dsize ** 2:
        x = config.backend.diag_get(x.reshape(Dsize, Dsize))
    return x
