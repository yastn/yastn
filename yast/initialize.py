""" Methods creating a new yast tensor """
from .tensor import Tensor, YastError


__all__ = ['rand', 'randR', 'zeros', 'ones', 'eye',
           'import_from_dict', 'decompress_from_1d']


def rand(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
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
    Initialize tensor with all possible blocks filled with real random numbers.

    drawing from uniform distribution in [-1, 1].
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
        a list of charges for each leg,
        see :meth:`Tensor.fill_tensor` for description.
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
        a list of charges for each leg,
        see :meth:`Tensor.fill_tensor` for description.
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
        a list of charges for each leg,
        see :meth:`Tensor.fill_tensor` for description.
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
    a.update_struct()
    return a
