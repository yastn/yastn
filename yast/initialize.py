# Methods creating new YAST tensors from scratch
# and importing tensors from different formats
# such as 1D+metadata or dictionary representation
from ast import literal_eval
import numpy as np
from .tensor import Tensor, YastError
from .tensor._auxliary import _unpack_axes, _struct
from .tensor._merging import _Fusion


__all__ = ['rand', 'rand2', 'randR', 'randC', 'zeros', 'ones', 'eye',
           'load_from_dict', 'load_from_hdf5',  'decompress_from_1d']


def rand2(config=None, n=None, isdiag=False, legs=(), **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with the random numbers.

    Draws from a uniform distribution in [-1, 1] or [-1, 1] + 1j * [-1, 1], depending on dtype.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    s : tuple
        Signature of tensor. Also determines the number of legs
    n : int
        Total charge of the tensor
    t : list
        List of charges for each leg,
        see :meth:`Tensor.fill_tensor` for description.
    D : list
        List of corresponding bond dimensions
    isdiag : bool
        Makes tensor diagonal
    dtype : str
        Desired dtype, overrides default_dtype specified in config
    device : str
        Device on which the tensor should be initialized, overrides default_device specified in config
    legs : list
        Specify t and D based on a list of lists of tensors and their legs.
        e.q., legs = [[a, 0, b, 0], [a, 1], [b, 1]] gives tensor with 3 legs, whose
        charges and dimension are consistent with specific legs of tensors a and b (simultaniously for the first leg).
        Overrides t and D.

    Returns
    -------
    tensor : yast.Tensor
        a random instance of a :meth:`Tensor`
    """
    mfs = None
    s = tuple(leg.s for leg in legs)
    t = tuple(leg.t for leg in legs)
    D = tuple(leg.D for leg in legs)
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, mfs=mfs, **kwargs)
    a.fill_tensor(t=t, D=D, val='randR')
    return a



def rand(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with the random numbers.

    Draws from a uniform distribution in [-1, 1] or [-1, 1] + 1j * [-1, 1], depending on dtype.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    s : tuple
        Signature of tensor. Also determines the number of legs
    n : int
        Total charge of the tensor
    t : list
        List of charges for each leg,
        see :meth:`Tensor.fill_tensor` for description.
    D : list
        List of corresponding bond dimensions
    isdiag : bool
        Makes tensor diagonal
    dtype : str
        Desired dtype, overrides default_dtype specified in config
    device : str
        Device on which the tensor should be initialized, overrides default_device specified in config
    legs : list
        Specify t and D based on a list of lists of tensors and their legs.
        e.q., legs = [[a, 0, b, 0], [a, 1], [b, 1]] gives tensor with 3 legs, whose
        charges and dimension are consistent with specific legs of tensors a and b (simultaniously for the first leg).
        Overrides t and D.

    Returns
    -------
    tensor : yast.Tensor
        a random instance of a :meth:`Tensor`
    """
    dtype = kwargs['dtype'] if 'dtype' in kwargs else config.default_dtype
    if dtype == 'float64':
        return randR(config, s, n, t, D, isdiag, **kwargs)
    if dtype == 'complex128':
        return randC(config, s, n, t, D, isdiag, **kwargs)
    raise YastError('dtype should be "float64" or "complex128"')


def randR(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    """ Shortcut for rand(..., dtype='float64')"""
    mfs = None
    if 'legs' in kwargs:
        t, D, s, mfs = _tD_from_legs(kwargs['legs'])
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, mfs=mfs, **kwargs)
    a.fill_tensor(t=t, D=D, val='randR')
    return a


def randC(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    """ Shortcut for rand(..., dtype='complex128')"""
    mfs = None
    if 'legs' in kwargs:
        t, D, s, mfs = _tD_from_legs(kwargs['legs'])
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, mfs=mfs, **kwargs)
    a.fill_tensor(t=t, D=D, val='randC')
    return a


def zeros(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with zeros.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    s : tuple
        a signature of tensor. Also determines the number of legs
    n : int
        total charge of the tensor
    t : list
        a list of charges for each leg,
        see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal
    dtype : str
        desired dtype, overrides default_dtype specified in config
    device : str
        device on which the tensor should be initialized, overrides default_device
        specified in config

    Returns
    -------
    tensor : yast.Tensor
        an instance of a tensor filled with zeros
    """
    mfs = None
    if 'legs' in kwargs:
        t, D, s, mfs = _tD_from_legs(kwargs['legs'])
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, mfs=mfs, **kwargs)
    a.fill_tensor(t=t, D=D, val='zeros')
    return a


def ones(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    s : tuple
        a signature of tensor. Also determines the number of legs
    n : int
        total charge of the tensor
    t : list
        a list of charges for each leg,
        see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    dtype : str
        desired dtype, overrides default_dtype specified in config
    device : str
        device on which the tensor should be initialized, overrides default_device
        specified in config

    Returns
    -------
    tensor : yast.Tensor
        an instance of a tensor filled with ones
    """
    mfs = None
    if 'legs' in kwargs:
        t, D, s, mfs = _tD_from_legs(kwargs['legs'])
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, mfs=mfs, **kwargs)
    a.fill_tensor(t=t, D=D, val='ones')
    return a


def eye(config=None, t=(), D=(), legs=None, **kwargs):
    r"""
    Initialize diagonal tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    t : list
        a list of charges for each leg,
        see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    dtype : str
        desired dtype, overrides default_dtype specified in config
    device : str
        device on which the tensor should be initialized, overrides default_device
        specified in config

    Returns
    -------
    tensor : yast.Tensor
        an instance of diagonal tensor filled with ones
    """
    s = ()
    if legs is not None:
        t, D, s, _ = _tD_from_legs(legs)
    a = Tensor(config=config, s=s, isdiag=True, **kwargs)
    a.fill_tensor(t=t, D=D, val='ones')
    return a


def load_from_dict(config=None, d=None):
    """
    Generate tensor based on information in dictionary `d`.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`

    d : dict
        Tensor stored in form of a dictionary. Typically provided by an output
        of :meth:`~yast.Tensor.save_to_dict`
    """
    if d is not None:
        c_isdiag = bool(d['isdiag'])
        c_Dp = tuple(x[0] for x in d['D']) if c_isdiag else tuple(np.prod(d['D'], axis=1))
        c_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(c_Dp), c_Dp))
        struct = _struct(s=d['s'], n=d['n'], diag=c_isdiag, t=d['t'], D=d['D'], Dp=c_Dp, sl=c_sl)
        hfs = tuple(_Fusion(**hf) for hf in d['hfs'])
        c = Tensor(config=config, struct=struct,
                    hfs=hfs, mfs=d['mfs'])
        if 'SYM_ID' in d and c.config.sym.SYM_ID != d['SYM_ID']:
            raise YastError("Symmetry rule in config do not match loaded one.")
        if 'fermionic' in d and c.config.fermionic != d['fermionic']:
            raise YastError("Fermionic statistics in config do not match loaded one.")
        c._data = c.config.backend.to_tensor(d['_d'], dtype=d['_d'].dtype.name, device=c.device)
        c.is_consistent()
        return c
    raise YastError("Dictionary d is required.")


def load_from_hdf5(config, file, path):
    """
    Create tensor from hdf5 file.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    file: TODO
    path: TODO
    """
    g = file.get(path)
    c_isdiag = bool(g.get('isdiag')[:][0])
    c_n = tuple(g.get('n')[:])
    c_s = tuple(g.get('s')[:])
    c_t = tuple(tuple(x.flat) for x in g.get('ts')[:])
    c_D = tuple(tuple(x.flat) for x in g.get('Ds')[:])
    c_Dp = tuple(x[0] for x in c_D) if c_isdiag else tuple(np.prod(c_D, axis=1))
    c_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(c_Dp), c_Dp))
    struct = _struct(s=c_s, n=c_n, diag=c_isdiag, t=c_t, D=c_D, Dp=c_Dp, sl=c_sl)

    mfs = eval(tuple(file.get(path+'/mfs').keys())[0])
    hfs = tuple(_Fusion(**hf) for hf in literal_eval(tuple(g.get('hfs').keys())[0]))
    c = Tensor(config=config, struct=struct, mfs=mfs, hfs=hfs)

    vmat = g.get('matrix')[:]
    c._data = c.config.backend.to_tensor(vmat, dtype=vmat.dtype.name, device=c.device)
    c.is_consistent()
    return c


def decompress_from_1d(r1d, config, meta):
    """
    Generate tensor from dictionary `meta` describing the structure of the tensor,
    charges and dimensions of its non-zero blocks, and 1-D array `r1d` containing
    serialized data of non-zero blocks.

    Typically, the pair `r1d` and `meta` is obtained from :func:`~yast.Tensor.compress_to_1d`.

    Parameters
    ----------
    r1d : rank-1 tensor
        1-D array (of backend type) holding serialized blocks.

    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`

    meta : dict
        structure of symmetric tensor. Non-zero blocks are indexed by associated charges.
        Each such entry contains block's dimensions and the location of its data
        in rank-1 tensor `r1d`
    """
    a = Tensor(config=config, **meta)
    a._data = r1d
    return a


def _tD_from_legs(legs):
    r""" Translates input specified in legs into charges t and block dimensions D. """
    tlegs, Dlegs, slegs, lflegs = [], [], [], []
    for leg in legs:
        leg = [leg] if isinstance(leg, dict) else leg
        tns, lgs, fps, lss = [], [], [], []
        ileg = iter(leg)
        a = next(ileg, None)
        while a is not None:
            if isinstance(a, Tensor):
                tns.append(a)
                a = next(ileg, None)
                if not isinstance(a, int):
                    raise YastError('Specifying leg number is required')
                lgs.append(a)
                a = next(ileg, None)
                fps.append(-1 if isinstance(a, str) and a in ('flip', 'flip_s', 'f') else 1)
                if isinstance(a, int):
                    raise YastError('Two leg numbers one after another not understood.')
                if isinstance(a, str):
                    a = next(ileg, None)
            if isinstance(a, dict):
                lss.append(a)
                a = next(ileg, None)
        lf = set(a.mfs[n] for a, n in zip(tns, lgs))
        if len(lf) > 1:
            raise YastError('Provided tensors fusions do not match.')
        if len(lf) == 0:
            d, s = _dict_union(lss)
            if s is None:
                raise YastError('Dictionary should include singnature s.')
            tlegs.append(tuple(d.keys()))
            Dlegs.append(tuple(d.values()))
            slegs.append(s)
            lflegs.append((1,))
        else:
            lf = lf.pop()
            lflegs.append(lf)
            if (lf[0] > 1) and len(lss) > 0:
                raise YastError('For fused legs, do not support mix input. ')
            for nn in range(lf[0]):
                ss = []
                for t, l, f in zip(tns, lgs, fps):
                    un, = _unpack_axes(t.mfs, (l,))
                    lss.append(t.get_leg_structure(un[nn], native=True))
                    ss.append(f * np.array(t.s_n, dtype=int)[un[nn]])
                d, s = _dict_union(lss)
                if s is not None:
                    ss.append(s)
                ss = set(ss)
                if len(ss) > 1:
                    raise YastError('Signature of tensors do not match.')
                tlegs.append(tuple(d.keys()))
                Dlegs.append(tuple(d.values()))
                slegs.append(ss.pop())
    return tlegs, Dlegs, slegs, lflegs


def _dict_union(ldict):
    d = {}
    for pd in ldict:
        for k, v in pd.items():
            k = (k,) if isinstance(k, int) else k
            if k in d and d[k] != v:
                raise YastError('provided dimensions of charge %s do not match' % str(k))
            d[k] = v
    s = d.pop('s') if 's' in d else None
    return d, s
