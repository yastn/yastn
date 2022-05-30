# Methods creating new YAST tensors from scratch
# and importing tensors from different formats
# such as 1D+metadata or dictionary representation
from ast import literal_eval
import numpy as np
from .tensor import Tensor, YastError
from .tensor._auxliary import _struct, _config
from .tensor._merging import _Fusion

__all__ = ['rand', 'randR', 'randC', 'zeros', 'ones', 'eye',
           'load_from_dict', 'load_from_hdf5',  'decompress_from_1d']


# def make_config(backend= backend_np, sym=sym_none, default_device='cpu', 
#     default_dtype='float64', fermionic= False,    
#     default_fusion= 'meta', force_fusion= None,
#     default_tensordot= 'hybrid', force_tensordot= None, **kwargs):
def make_config(**kwargs):
    r"""
    Parameters
    ----------
        backend : backend module or compatible object
            Specify ``backend`` providing Linear algebra and base dense tensors.
            Currently support backends

                * NumPy as ``yast.backend.backend_np``
                * PyTorch as ``yast.backend.backend_torch``

            Defaults to NumPy backend.

        sym : symmetry module or compatible object
            Specify abelian symmetry. To see how YAST defines symmetries,
            see :class:`yast.sym.sym_abelian`.
            Defaults to ``yast.sym.sym_none``, effectively a dense tensor.
        default_device : str
            Base tensors can be stored on various devices as supported by ``backend``

                * NumPy supports only ``'cpu'`` device
                * PyTorch supports multiple devices, see
                  https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

            If not specified, the default device is ``'cpu'``.

        default_dtype: str
            Default data type (dtype) of YAST tensors. Supported options are: 'float64', 'complex128'.
            If not specified, the default dtype is ``'float64'``.
        fermionic : bool or tuple[bool,...]
            Specify behavior of swap_gate function, allowing to introduce fermionic symmetries.
            Allowed values: ``False``, ``True``, or a tuple ``(True, False, ...)`` with one bool for each component
            charge vector i.e. of length sym.NSYM. Default is ``False``.
        default_fusion: str
            Specify default strategy to handle leg fusion: 'hard' or 'meta'. See yast.tensor.fuse_legs
            for details. Default is ``'meta'``.
        force_fusion : str
            Overrides fusion strategy provided in yast.tensor.fuse_legs. Default is ``None``.
    """
    if "backend" not in kwargs:
        from .backend import backend_np
        kwargs["backend"] = backend_np
    if "sym" not in kwargs:
        from .sym import sym_none
        kwargs["sym"] = sym_none
    return _config(**{a: kwargs[a] for a in _config._fields if a in kwargs})



def _fill(config=None, legs=(), n=None, isdiag=False, val='rand', **kwargs):
    if 's' in kwargs or 't' in kwargs or 'D' in kwargs:
        s = kwargs.pop('s') if 's' in kwargs else ()
        t = kwargs.pop('t') if 't' in kwargs else ()
        D = kwargs.pop('D') if 'D' in kwargs else ()
        mfs, hfs = None, None
    else:
        ulegs, mfs = [], []
        for leg in legs:
            if hasattr(leg, 'legs'):  # _metaLeg
                if isdiag:
                    raise YastError('Diagonal tensor cannot be initialized with fused legs')
                ulegs.extend(leg.legs)
                mfs.append(leg.mf)
            else:  #_Leg
                ulegs.append(leg)
                mfs.append((1,))
        if any(config.sym.SYM_ID != leg.sym.SYM_ID for leg in ulegs):
            raise YastError('Different symmetry of initialized tensor and some of the legs.')
        s = tuple(leg.s for leg in ulegs)
        t = tuple(leg.t for leg in ulegs)
        D = tuple(leg.D for leg in ulegs)
        hfs = tuple(leg.hf for leg in ulegs)
        mfs = tuple(mfs)

    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, mfs=mfs, hfs=hfs, **kwargs)
    a.fill_tensor(t=t, D=D, val=val)
    return a


def rand(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with the random numbers.

    Draws from a uniform distribution in [-1, 1] or [-1, 1] + 1j * [-1, 1], depending on dtype.

    Parameters
    ----------
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
    tensor : tensor
        a random instance of a :meth:`Tensor`
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', **kwargs)


def randR(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with real random numbers, see `rand`.
    """
    kwargs['dtype'] = 'float64'
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', **kwargs)


def randC(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with complex random numbers, see `rand`.
    """
    kwargs['dtype'] = 'complex128'
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', **kwargs)


def zeros(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with zeros.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
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
    tensor : tensor
        an instance of a tensor filled with zeros
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='zeros', **kwargs)


def ones(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
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
    tensor : tensor
        an instance of a tensor filled with ones
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='ones', **kwargs)


def eye(config=None, legs=(), n=None, **kwargs):
    r"""
    Initialize diagonal tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
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
    tensor : tensor
        an instance of diagonal tensor filled with ones
    """
    return _fill(config=config, legs=legs, n=n, isdiag=True, val='ones', **kwargs)


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

    mfs = literal_eval(tuple(file.get(path+'/mfs').keys())[0])
    hfs = tuple(_Fusion(*hf) if isinstance(hf, tuple) else _Fusion(**hf) \
                for hf in literal_eval(tuple(g.get('hfs').keys())[0]))
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
