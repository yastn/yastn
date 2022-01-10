""" Linear operations and operations on a single yast tensor. """
from ._auxliary import _struct
from ._merging import _masks_for_add
from ._tests import YastError, _test_configs_match, _get_tD_legs, _test_axes_match

__all__ = ['apxb', 'absolute', 'real', 'imag', 'sqrt', 'rsqrt', 'reciprocal', 'exp']


def __add__(a, b):
    """
    Add two tensors, use: tensor + tensor.

    Signatures and total charges should match.

    Parameters
    ----------
    b: Tensor

    Returns
    -------
    tensor : Tensor
        result of addition as a new tensor
    """
    _test_configs_match(a, b)
    aA, bA, hfs, meta, c_struct = _addition_meta(a, b)
    c = a.__class__(config=a.config, isdiag=a.isdiag, struct=c_struct,
                    meta_fusion=a.meta_fusion, hard_fusion=hfs)
    c.A = c.config.backend.add(aA, bA, meta)
    return c


def __sub__(a, b):
    """
    Subtract two tensors, use: tensor - tensor.

    Both signatures and total charges should match.

    Parameters
    ----------
    b: Tensor

    Returns
    -------
    tensor : Tensor
        result of subtraction as a new tensor
    """
    _test_configs_match(a, b)
    aA, bA, hfs, meta, c_struct = _addition_meta(a, b)
    c = a.__class__(config=a.config, isdiag=a.isdiag, struct=c_struct,
                    meta_fusion=a.meta_fusion, hard_fusion=hfs)
    c.A = c.config.backend.sub(aA, bA, meta)
    return c


def apxb(a, b, x=1):
    r"""
    Directly compute the result of :math:`a + x \times b`.
    This `composite` operation is faster than first performing multiplication
    and then addition.

    Parameters
    ----------
    a, b: Tensor
    x : scalar

    Returns
    -------
    tensor : Tensor
    """
    _test_configs_match(a, b)
    aA, bA, hfs, meta, c_struct = _addition_meta(a, b)
    c = a.__class__(config=a.config, isdiag=a.isdiag, struct=c_struct,
                    meta_fusion=a.meta_fusion, hard_fusion=hfs)
    c.A = c.config.backend.apxb(aA, bA, x, meta)
    return c


def _addition_meta(a, b):
    """ meta-information for backend and new tensor charges and dimensions. """
    if a.struct.n != b.struct.n:
        raise YastError('Error in add: tensor charges do not match')
    needs_mask, _ = _test_axes_match(a, b, sgn=1)
    if needs_mask:
        sla, tDa, slb, tDb, hfs = _masks_for_add(a.config, a.struct, a.hard_fusion, b.struct, b.hard_fusion)
        aA = a.config.backend.embed(a.A, sla, tDa)
        bA = a.config.backend.embed(b.A, slb, tDb)
        aDset = tuple(tDa[t] for t in a.struct.t)
        bDset = tuple(tDb[t] for t in b.struct.t)
    else:
        aA, bA, hfs = a.A, b.A, a.hard_fusion
        aDset, bDset = a.struct.D, b.struct.D

    if a.struct.t == b.struct.t:
        if aDset != bDset:
            raise YastError('Bond dimensions do not match.')
        c_struct = a.struct._replace(D=aDset)
        meta = tuple((ta, 'AB') for ta in a.struct.t)
        return aA, bA, hfs, meta, c_struct

    ia, ib, meta = 0, 0, []
    while ia < len(aDset) and ib < len(bDset):
        ta, Da = a.struct.t[ia], aDset[ia]
        tb, Db = b.struct.t[ib], bDset[ib]
        if ta == tb:
            if Da != Db:
                raise YastError('Bond dimensions do not match.')
            meta.append((ta, Da, 'AB'))
            ia += 1
            ib += 1
        elif ta < tb:
            meta.append((ta, Da, 'A'))
            ia += 1
        else:
            meta.append((tb, Db, 'B'))
            ib += 1
    for ta, Da in zip(a.struct.t[ia:], aDset[ia:]):
        meta.append((ta, Da, 'A'))
    for tb, Db in zip(b.struct.t[ib:], bDset[ib:]):
        meta.append((tb, Db, 'B'))

    c_t = tuple(t for t, _, _ in meta)
    c_D = tuple(D for _, D, _ in meta)
    meta = tuple((t, ab) for t, _, ab in meta)
    c_struct = _struct(t=c_t, D=c_D, s=a.struct.s, n=a.struct.n)
    if any(ab != 'AB' for _, ab in meta):
        _get_tD_legs(c_struct)
    return aA, bA, hfs, meta, c_struct


def __lt__(a, number):
    """
    Logical tensor with elements less-than a number (if it makes sense for backend data tensors)

    Parameters
    ----------
    number: number

    Intended for diagonal tensor to be applied as a truncation mask.

    Returns
    -------
    tensor : Tensor
        result of logical element-wise operation as a new tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: x < number for ind, x in a.A.items()}
    return c


def __gt__(a, number):
    """
    Logical tensor with elements greater-than a number (if it makes sense for backend data tensors)

    Intended for diagonal tensor to be applied as a truncation mask.

    Parameters
    ----------
    number: number

    Returns
    -------
    tensor : Tensor
        result of logical element-wise operation as a new tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: x > number for ind, x in a.A.items()}
    return c


def __le__(a, number):
    """
    Logical tensor with elements less-than-or-equal-to a number (if it makes sense for backend data tensors)

    Intended for diagonal tensor to be applied as a truncation mask.

    Parameters
    ----------
    number: number

    Returns
    -------
    tensor : Tensor
        result of logical element-wise operation as a new tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: x <= number for ind, x in a.A.items()}
    return c


def __ge__(a, number):
    """
    Logical tensor with elements greater-than-or-equal-to a number (if it makes sense for backend data tensors)

    Intended for diagonal tensor to be applied as a truncation mask.

    Parameters
    ----------
    number: number

    Returns
    -------
    tensor : Tensor
        result of logical element-wise operation as a new tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: x >= number for ind, x in a.A.items()}
    return c


def __mul__(a, number):
    """
    Multiply tensor by a number, use: number * tensor.

    Parameters
    ----------
    number: number

    Returns
    -------
    tensor : Tensor
        result of multipcilation as a new tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: number * x for ind, x in a.A.items()}
    return c


def __rmul__(a, number):
    """
    Multiply tensor by a number, use: tensor * number.

    Parameters
    ----------
    number: number

    Returns
    -------
    tensor : Tensor
        result of multipcilation as a new tensor
    """
    return __mul__(a, number)


def __pow__(a, exponent):
    """
    Element-wise exponent of tensor, use: tensor ** exponent.

    Parameters
    ----------
    exponent: number

    Returns
    -------
    tensor : Tensor
        result of element-wise exponentiation as a new tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: x**exponent for ind, x in a.A.items()}
    return c


def __truediv__(a, number):
    """
    Divide tensor by a scalar, use: tensor / scalar.

    Parameters
    ----------
    number: number

    Returns
    -------
    tensor : Tensor
        result of element-wise division  as a new tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ind: x / number for ind, x in a.A.items()}
    return c


def absolute(a):
    r"""
    Return tensor with element-wise absolute values

    Returns
    -------
    tensor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,\
        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.absolute(a.A)
    return c


def real(a):
    r"""
    Return tensor with imaginary part set to zero.

    .. note::
        Returned :class:`yast.Tensor` has the same dtype

    Returns
    -------
    tensor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,\
        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {t: a.config.backend.real(x) for t, x in a.A.items()}
    return c


def imag(a):
    r"""
    Return tensor with real part set to zero.

    .. note::
        Returned :class:`yast.Tensor` has the same dtype

    Returns
    -------
    tensor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,\
        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {t: a.config.backend.imag(x) for t, x in a.A.items()}
    return c


def sqrt(a):
    """
    Return element-wise sqrt(A).

    Returns
    -------
    tensor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, \
        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.sqrt(a.A)
    return c


def rsqrt(a, cutoff=0):
    """
    Return element-wise 1/sqrt(A).

    The tensor elements with square root of absolute value below the cutoff are set to zero.

    Parameters
    ----------
    cutoff: real scalar
        (element-wise) cutoff for inversion

    Returns
    -------
    tensor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, \
        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.rsqrt(a.A, cutoff=cutoff)
    return c


def reciprocal(a, cutoff=0):
    """
    Return element-wise 1/A.

    The tensor elements with absolute value below the cutoff are set to zero.

    Parameters
    ----------
    cutoff: real scalar
        (element-wise) cutoff for inversion

    Returns
    -------
    tensor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, \
        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.reciprocal(a.A, cutoff=cutoff)
    return c


def exp(a, step=1.):
    r"""
    Return element-wise `exp(step * A)`.

    .. note::
        This applies only to non-empty blocks of A

    Parameters
    ----------
    step: scalar

    Returns
    -------
    tensor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, \
        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.exp(a.A, step)
    return c
