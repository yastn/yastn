""" Methods creating a new yast tensor """
from .tensor import Tensor, YastError
from .tensor._auxliary import _unpack_axes


__all__ = ['rand', 'randR', 'randC', 'zeros', 'ones', 'eye',
           'import_from_dict', 'decompress_from_1d']


def rand(config=None, s=(), n=None, t=(), D=(), isdiag=False, legs=None, dtype=None, device=None):
    r"""
    Initialize tensor with all possible blocks filled with the random numbers.

    dtype is specified in config drawing from [-1, 1] or [-1, 1] + 1j * [-1, 1]
    Initialize tensor and call :meth:`yast.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
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
        a random instance of a tensor
    """
    if not dtype:
        assert hasattr(config,'default_dtype'), "Either dtype or valid config has to be provided"
        dtype= config.default_dtype
    if not device:
        assert hasattr(config,'default_device'), "Either device or valid config has to be provided"
        device= config.default_device
    if dtype == 'float64':
        return randR(config, s, n, t, D, isdiag, legs, dtype, device)
    if dtype == 'complex128':
        return randC(config, s, n, t, D, isdiag, legs, dtype, device)
    raise YastError('dtype should be float64 or complex128')


def randR(config=None, s=(), n=None, t=(), D=(), isdiag=False, legs=None, dtype=None, device=None):
    """ Shortcut for rand(..., dtype='float64')"""
    meta_fusion = None
    if legs is not None:
         t, D, s, meta_fusion = _tD_from_legs(legs)
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, device=device, meta_fusion=meta_fusion)
    a.fill_tensor(t=t, D=D, val='randR', dtype=dtype)
    return a


def randC(config=None, s=(), n=None, t=(), D=(), isdiag=False, legs=None, dtype=None, device=None):
    """ Shortcut for rand(..., dtype='complex128')"""
    meta_fusion = None
    if legs is not None:
         t, D, s, meta_fusion = _tD_from_legs(legs)
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, device=device, meta_fusion=meta_fusion)
    a.fill_tensor(t=t, D=D, val='randC', dtype=dtype)
    return a


def zeros(config=None, s=(), n=None, t=(), D=(), isdiag=False, legs=None, dtype=None, device=None):
    r"""
    Initialize tensor with all possible blocks filled with zeros.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
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
    if not dtype:
        assert hasattr(config,'default_dtype'), "Either dtype or valid config has to be provided"
        dtype= config.default_dtype
    if not device:
        assert hasattr(config,'default_device'), "Either device or valid config has to be provided"
        device= config.default_device
    meta_fusion = None
    if legs is not None:
         t, D, s, meta_fusion = _tD_from_legs(legs)
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, device=device, meta_fusion=meta_fusion)
    a.fill_tensor(t=t, D=D, val='zeros', dtype=dtype)
    return a


def ones(config=None, s=(), n=None, t=(), D=(), isdiag=False, legs=None, dtype=None, device=None):
    r"""
    Initialize tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
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
    if not dtype:
        assert hasattr(config,'default_dtype'), "Either dtype or valid config has to be provided"
        dtype= config.default_dtype
    if not device:
        assert hasattr(config,'default_device'), "Either device or valid config has to be provided"
        device= config.default_device
    meta_fusion = None
    if legs is not None:
         t, D, s, meta_fusion = _tD_from_legs(legs)
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, device=device, meta_fusion=meta_fusion)
    a.fill_tensor(t=t, D=D, val='ones', dtype=dtype)
    return a


def eye(config=None, t=(), D=(), legs=None, dtype=None, device=None):
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
    if not dtype:
        assert hasattr(config,'default_dtype'), "Either dtype or valid config has to be provided"
        dtype= config.default_dtype
    if not device:
        assert hasattr(config,'default_device'), "Either device or valid config has to be provided"
        device= config.default_device
    s = ()
    if legs is not None:
         t, D, s, _ = _tD_from_legs(legs)
    a = Tensor(config=config, s=s, isdiag=True, device=device)
    a.fill_tensor(t=t, D=D, val='ones', dtype=dtype)
    return a


def import_from_dict(config=None, d=None):
    """
    Generate tensor based on information in dictionary d.

    Parameters
    ----------
    config: module
            configuration with backend, symmetry, etc.

    d : dict
        information about tensor stored with :meth:`Tensor.to_dict`
    """
    if d is not None:
        a = Tensor(config=config, **d)
        for ind in d['A']:
            a.set_block(ts=ind, Ds=d['A'][ind].shape, val=d['A'][ind])
        return a
    raise YastError("Dictionary d is required.")


def decompress_from_1d(r1d, config, meta):
    """
    Generate tensor based on information in dictionary d and 1D array
    r1d containing the serialized blocks

    Parameters
    ----------
    config: module
            configuration with backend, symmetry, etc.

    d : dict
        information about tensor stored with :meth:`Tensor.to_dict`
    """
    a = Tensor(config=config, **meta)
    A = {(): r1d}
    a.A = a.config.backend.unmerge_one_leg(A, 0, meta['meta_unmerge'])
    a.update_struct()
    return a


def _tD_from_legs(legs):
    r""" Translates input of legs into charges t and block dimensions D """
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
                fps.append(-1 if isinstance(a, str) and (a == 'flip' or a == 'f') else 1)
                if isinstance(a, int):
                    raise YastError('Two leg numbers after each not understood.')
                if isinstance(a, str):
                    a = next(ileg, None)
            if isinstance(a, dict):
                lss.append(a)
                a = next(ileg, None)
        lf = set(a.meta_fusion[n] for a, n in zip(tns, lgs))
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
                    un, = _unpack_axes(t, (l,))
                    lss.append(t.get_leg_structure(un[nn], native=True))
                    ss.append(f * t.s[un[nn]])
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
