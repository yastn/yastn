""" Methods creating a new yast tensor """

from .core import Tensor, YastError

__all__ = ['rand', 'randR', 'zeros', 'ones', 'eye', 'import_from_dict', 'decompress_from_1d', 'match_legs']


def rand(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with the random numbers in [-1, 1] and type specified in config.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        a random instance of a tensor
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, **kwargs)
    a.fill_tensor(t=t, D=D, val='rand')
    return a


def randR(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with real random numbers in [-1, 1].

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        a random instance of a tensor
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, **kwargs)
    a.fill_tensor(t=t, D=D, val='randR')
    return a


def zeros(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
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
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        an instance of a tensor filled with zeros
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, **kwargs)
    a.fill_tensor(t=t, D=D, val='zeros')
    return a


def ones(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
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
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions

    Returns
    -------
    tensor : tensor
        an instance of a tensor filled with ones
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, **kwargs)
    a.fill_tensor(t=t, D=D, val='ones')
    return a


def eye(config=None, t=(), D=(), **kwargs):
    r"""
    Initialize diagonal tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions

    Returns
    -------
    tensor : tensor
        an instance of diagonal tensor filled with ones
    """
    a = Tensor(config=config, isdiag=True, **kwargs)
    a.fill_tensor(t=t, D=D, val='ones')
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
    a._update_tD_arrays()
    return a


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
        un, = te._unpack_axes((nf,))
        for nn in un:
            tdn = te.get_leg_structure(nn, native=True)
            t.append(tuple(tdn.keys()))
            D.append(tuple(tdn.values()))
            s.append(te.s[nn] * (2 * cc - 1))
    a = Tensor(config=tensors[0].config, s=s, n=n, isdiag=isdiag, meta_fusion=lf)
    a.fill_tensor(t=t, D=D, val=val)
    return a

