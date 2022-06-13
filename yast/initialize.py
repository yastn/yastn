# Methods creating new YAST tensors from scratch
# and importing tensors from different formats
# such as 1D+metadata or dictionary representation
from ast import literal_eval
from itertools import chain, repeat, accumulate
import numpy as np
from .tensor import Tensor, YastError
from .tensor._auxliary import _struct, _config, _clear_axes, _unpack_axes,  _unpack_legs
from .tensor._merging import _Fusion
from .tensor._legs import Leg


__all__ = ['rand', 'randR', 'randC', 'zeros', 'ones', 'eye', 'block',
           'make_config', 'load_from_dict', 'load_from_hdf5', 'decompress_from_1d']


# def make_config(backend= backend_np, sym=sym_none, default_device='cpu',
#     default_dtype='float64', fermionic= False,
#     default_fusion= 'meta', force_fusion= None, **kwargs):
def make_config(**kwargs):
    r"""
    Parameters
    ----------
    backend : backend module or compatible object
        Specify ``backend`` providing Linear algebra and base dense tensors.
        Currently supported backends are

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
    
    Returns
    -------
    typing.NamedTuple
        YAST configuration
    
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
    else:  # use legs for initialization
        if isinstance(legs, Leg):
            legs = (legs,)
        if isdiag and len(legs) == 1:
            legs = (legs[0], legs[0].conj())
        ulegs, mfs = _unpack_legs(legs)
        s = tuple(leg.s for leg in ulegs)
        t = tuple(leg.t for leg in ulegs)
        D = tuple(leg.D for leg in ulegs)
        hfs = tuple(leg.legs[0] for leg in ulegs)

        if any(config.sym.SYM_ID != leg.sym.SYM_ID for leg in ulegs):
            raise YastError('Different symmetry of initialized tensor and some of the legs.')
        if isdiag and any(mf != (1,) for mf in mfs):
            raise YastError('Diagonal tensor cannot be initialized with fused legs.')
        if isdiag and any(hf.tree != (1,) for hf in hfs):
            raise YastError('Diagonal tensor cannot be initialized with fused legs.')

    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, mfs=mfs, hfs=hfs, **kwargs)
    a.fill_tensor(t=t, D=D, val=val)
    return a


def rand(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all allowed blocks filled with random numbers.

    Draws from a uniform distribution in [-1, 1] or [-1, 1] + 1j * [-1, 1], 
    depending on desired ``dtype``.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    legs : list[yast.Leg]
        If specified, overrides `t`, `D`, and `s` arguments. Specify legs of the tensor
        directly by passing a list of :class:`~yast.Leg`.
    s : tuple
        Signature of tensor. Also determines the number of legs
    n : int
        Total charge of the tensor
    t : list
        List of charges for each leg,
        see :meth:`Tensor.fill_tensor` for description.
    D : list
        List of corresponding bond dimensions.
    isdiag : bool
        Makes tensor diagonal
    dtype : str
        Desired dtype, overrides default_dtype specified in config
    device : str
        Device on which the tensor should be initialized, overrides default_device specified in config

    Returns
    -------
    yast.Tensor
        new random tensor
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', **kwargs)


def randR(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all allowed blocks filled with real random numbers, 
    see :meth:`yast.rand`.
    """
    kwargs['dtype'] = 'float64'
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', **kwargs)


def randC(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all allowed blocks filled with complex random numbers, \
    see :meth:`yast.rand`.
    """
    kwargs['dtype'] = 'complex128'
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', **kwargs)


def zeros(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all allowed blocks filled with zeros.
    First, initializes empty tensor then calls :meth:`yast.Tensor.fill_tensor`.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    legs : list[yast.Leg]
        If specified, overrides `t`, `D`, and `s` arguments. Specify legs of the tensor
        directly by passing a list of :class:`~yast.Leg`.
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
    yast.Tensor
        new tensor filled with zeros
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='zeros', **kwargs)


def ones(config=None, legs=(), n=None, isdiag=False, **kwargs):
    r"""
    Initialize tensor with all allowed blocks filled with ones.
    First, initializes empty tensor then calls :meth:`yast.Tensor.fill_tensor`.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    legs : list[yast.Leg]
        If specified, overrides `t`, `D`, and `s` arguments. Specify legs of the tensor
        directly by passing a list of :class:`~yast.Leg`.
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
    yast.Tensor
        new tensor filled with ones
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='ones', **kwargs)


def eye(config=None, legs=(), n=None, **kwargs):
    r"""
    Initialize `diagonal` tensor with all possible blocks filled with ones.

    Initializes tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    legs : list[yast.Leg]
        If specified, overrides `t`, `D`, and `s` arguments. Specify legs of the tensor
        directly by passing a list of :class:`~yast.Leg`.
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
    return _fill(config=config, legs=legs, n=n, isdiag=True, val='ones', **kwargs)


def load_from_dict(config=None, d=None):
    """
    Create tensor the dictionary `d`.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`

    d : dict
        Tensor stored in form of a dictionary. Typically provided by an output
        of :meth:`~yast.save_to_dict`

    Returns
    -------
    yast.Tensor
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

    Returns
    -------
    yast.Tensor
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

    Typically, the pair `r1d` and `meta` is obtained from :meth:`~yast.compress_to_1d`.

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

    Returns
    -------
    yast.Tensor
    """
    a = Tensor(config=config, **meta)
    a._data = r1d
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
        This is equivalently to all tensors having the same position (not specified explicitly) in the super-tensor on that leg.
    """
    out_s, = ((),) if common_legs is None else _clear_axes(common_legs)
    tn0 = next(iter(tensors.values()))  # first tensor; used to initialize new objects and retrive common values
    out_b = tuple((ii,) for ii in range(tn0.ndim) if ii not in out_s)
    pos = list(_clear_axes(*tensors))
    lind = tn0.ndim - len(out_s)
    for ind in pos:
        if len(ind) != lind:
            raise YastError('Wrong number of coordinates encoded in tensors.keys()')

    out_s, =  _unpack_axes(tn0.mfs, out_s)
    u_b = tuple(_unpack_axes(tn0.mfs, *out_b))
    out_b = tuple(chain(*u_b))
    pos = tuple(tuple(chain.from_iterable(repeat(x, len(u)) for x, u in zip(ind, u_b))) for ind in pos)

    for tn in tensors.values():
        if tn.struct.s != tn0.struct.s:
            raise YastError('Signatues of blocked tensors are inconsistent.')
        if tn.struct.n != tn0.struct.n:
            raise YastError('Tensor charges of blocked tensors are inconsistent.')
        if tn.mfs != tn0.mfs:
            raise YastError('Meta fusions of blocked tensors are inconsistent.')
        #if any(tn.hfs[n].tree != (1,) for n in range(tn.ndim_n) if n not in out_s):
        #    raise YastError('Blocking of hard-fused legs is currently not supported. Go through meta-fusion. Only common_legs can be hard-fused.')
        #if any(tn.hfs[n] != tn0.hfs[n] for n in out_s):
        #    raise YastError('Hard-fusions of common_legs do not match.')
        if tn.isdiag:
            raise YastError('Block does not support diagonal tensors. Use .diag() first.')

    posa = np.ones((len(pos), tn0.ndim_n), dtype=int)
    posa[:, np.array(out_b, dtype=np.intp)] = np.array(pos, dtype=int).reshape(len(pos), -1)

    tDs = []  # {leg: {charge: {position: D, 'D' : Dtotal}}}
    for n in range(tn0.ndim_n):
        tDl = {}
        for tn, pp in zip(tensors.values(), posa):
            leg = tn.get_legs(n, native=True)
            for t, D in zip(leg.t, leg.D):
                if t in tDl:
                    if (pp[n] in tDl[t]) and (tDl[t][pp[n]] != D):
                        raise YastError('Dimensions of blocked tensors are not consistent.')
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

    data = tn0.config.backend.merge_super_blocks(tensors, meta_new, meta_block, Dsize)
    hfs = tuple(_Fusion(s=(tn0.struct.s[n],)) if n not in out_s else tn0.hfs[n] for n in range(tn0.ndim_n))
    return tn0._replace(struct=c_struct, data=data, hfs=hfs)
