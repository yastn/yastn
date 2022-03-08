""" Linear operations and operations on a single yast tensor. """
from ._auxliary import _struct
from ._merging import _masks_for_add
from ._tests import YastError, _test_configs_match, _get_tD_legs, _test_axes_match

__all__ = ['apxb', 'real', 'imag', 'sqrt', 'rsqrt', 'reciprocal', 'exp']


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
    aA, bA, hfs, meta, c_struct, Dsize = _addition_meta(a, b)
    c = a.__class__(config=a.config, isdiag=a.isdiag, struct=c_struct,
                    meta_fusion=a.meta_fusion, hard_fusion=hfs)
    c._data = c.config.backend.add(aA, bA, meta, Dsize)
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
    aA, bA, hfs, meta, c_struct, Dsize = _addition_meta(a, b)
    c = a.__class__(config=a.config, isdiag=a.isdiag, struct=c_struct,
                    meta_fusion=a.meta_fusion, hard_fusion=hfs)
    c._data = c.config.backend.sub(aA, bA, meta, Dsize)
    return c


def apxb(a, b, x=1):
    r"""
    Directly compute the result of :math:`a + x b`.
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
    aA, bA, hfs, meta, c_struct, Dsize = _addition_meta(a, b)
    c = a.__class__(config=a.config, isdiag=a.isdiag, struct=c_struct,
                    meta_fusion=a.meta_fusion, hard_fusion=hfs)
    c._data = c.config.backend.apxb(aA, bA, x, meta, Dsize)
    return c


def _addition_meta(a, b):
    """ meta-information for backend and new tensor charges and dimensions. """
    if a.struct.n != b.struct.n:
        raise YastError('Error in add: tensor charges do not match')
    needs_mask, _ = _test_axes_match(a, b, sgn=1)
    if needs_mask:
        sla, tDa, slb, tDb, hfs = _masks_for_add(a.config, a.struct, a.hard_fusion, b.struct, b.hard_fusion)
        aA = a.config.backend.embed(a._data, sla, tDa)
        bA = a.config.backend.embed(b._data, slb, tDb)
        aDset = tuple(tDa[t] for t in a.struct.t)
        bDset = tuple(tDb[t] for t in b.struct.t)
    else:
        aA, bA, hfs = a._data, b._data, a.hard_fusion
        aDset, bDset = a.struct.D, b.struct.D
        aDpset, bDpset = a.struct.Dp, b.struct.Dp
        aslset, bslset = a.struct.sl, b.struct.sl

    if a.struct.t == b.struct.t:
        if aDset != bDset:
            raise YastError('Bond dimensions do not match.')
        c_struct = a.struct._replace(D=aDset, Dp=aDpset, sl=aslset)
        high = aslset[-1][1]
        one_sl = slice(0, high)
        meta = ((one_sl, one_sl, one_sl, 'AB'),)
        return aA, bA, hfs, meta, c_struct, high

    ia, ib, meta, c_t, c_D, c_Dp, c_sl = 0, 0, [], [], [], [], []
    low = 0
    while ia < len(aDset) and ib < len(bDset):
        ta, Da, Dpa, asl = a.struct.t[ia], aDset[ia], aDpset[ia], aslset[ia]
        tb, Db, Dpb, bsl = b.struct.t[ib], bDset[ib], bDpset[ib], bslset[ib]
        if ta == tb:
            if Da != Db:
                raise YastError('Bond dimensions do not match.')
            high = low + Dpa
            meta.append((slice(low, high), asl, bsl, 'AB'))
            c_t.append(ta)
            c_D.append(Da)
            c_Dp.append(Dpa)
            c_sl.append((low, high))
            low = high
            ia += 1
            ib += 1
        elif ta < tb:
            high = low + Dpa
            meta.append((slice(low, high), asl, None, 'A'))
            c_t.append(ta)
            c_D.append(Da)
            c_Dp.append(Dpa)
            c_sl.append((low, high))
            low = high
            ia += 1
        else:
            high = low + Dpb
            meta.append((slice(low, high), None, bsl, 'B'))
            c_t.append(tb)
            c_D.append(Db)
            c_Dp.append(Dpb)
            c_sl.append((low, high))
            low = high
            ib += 1
    for ta, Da, Dpa, asl in zip(a.struct.t[ia:], aDset[ia:], aDpset[ia:], aslset[ia:]):
        high = low + Dpa
        meta.append((slice(low, high), asl, None, 'A'))
        c_t.append(ta)
        c_D.append(Da)
        c_Dp.append(Dpa)
        c_sl.append((low, high))
        low = high
    for tb, Db, Dpb, bsl in zip(b.struct.t[ib:], bDset[ib:], bDpset[ib:], bslset[ib:]):
        high = low + Dpa
        meta.append((slice(low, high), None, bsl, 'B'))
        c_t.append(tb)
        c_D.append(Db)
        c_Dp.append(Dpb)
        c_sl.append((low, high))
        low = high

    c_struct = _struct(s=a.struct.s, n=a.struct.n, t=tuple(c_t), D=tuple(c_D), Dp=tuple(c_Dp), sl=tuple(c_sl))
    if any(mt[3] != 'AB' for mt in meta):
        _get_tD_legs(c_struct)
    return aA, bA, hfs, meta, c_struct, high


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


def __abs__(a):
    r"""
    Return tensor with element-wise absolute values. 
    Can be on called on tensor as ``abs(tensor)``. 

    Returns
    -------
    tensor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,
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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,
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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,
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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,
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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,
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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,
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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,
        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.exp(a.A, step)
    return c
