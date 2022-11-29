""" Linear operations and operations on a single yast tensor. """
from ._merging import _masks_for_add
from ._tests import YastError, _test_can_be_combined, _get_tD_legs, _test_axes_match

__all__ = ['apxb', 'real', 'imag', 'sqrt', 'rsqrt', 'reciprocal', 'exp', 'bitwise_not']


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
    _test_can_be_combined(a, b)
    aA, bA, hfs, meta, struct, Dsize = _addition_meta(a, b)
    data = a.config.backend.add(aA, bA, meta, Dsize)
    return a._replace(hfs=hfs, struct=struct, data=data)


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
    _test_can_be_combined(a, b)
    aA, bA, hfs, meta, struct, Dsize = _addition_meta(a, b)
    data = a.config.backend.sub(aA, bA, meta, Dsize)
    return a._replace(hfs=hfs, struct=struct, data=data)


def apxb(a, b, x=1):
    r"""
    Directly compute the result of :math:`a + x b`.

    Parameters
    ----------
    a, b: Tensor
    x : scalar

    Returns
    -------
    tensor : Tensor
    """
    _test_can_be_combined(a, b)
    aA, bA, hfs, meta, struct, Dsize = _addition_meta(a, b)
    data = a.config.backend.apxb(aA, bA, x, meta, Dsize)
    return a._replace(hfs=hfs, struct=struct, data=data)


def _addition_meta(a, b):
    """ meta-information for backend and new tensor charges and dimensions. """
    if a.struct.n != b.struct.n:
        raise YastError('Tensor charges do not match.')
    if a.isdiag != b.isdiag:
        raise YastError('Cannot add diagonal tensor to non-diagonal one.')

    needs_mask, _ = _test_axes_match(a, b, sgn=1)
    if needs_mask:
        msk_a, msk_b, struct_a, struct_b, hfs = _masks_for_add(a.config, a.struct, a.hfs, b.struct, b.hfs)
        Dsize = struct_a.sl[-1][1] if len(struct_a.sl) > 0 else 0
        Adata = a.config.backend.embed_msk(a._data, msk_a, Dsize)
        Dsize = struct_b.sl[-1][1] if len(struct_b.sl) > 0 else 0
        Bdata = a.config.backend.embed_msk(b._data, msk_b, Dsize)
        del Dsize
    else:
        Adata, Bdata = a._data, b._data
        struct_a, struct_b = a.struct, b.struct
        hfs = a.hfs

    if struct_a == struct_b:
        c_struct = struct_a
        Dsize = c_struct.sl[-1][1] if len(c_struct.sl) > 0 else 0
        metaA = (((0, Dsize), (0, Dsize)),)
        metaB = (((0, Dsize), (0, Dsize)),)
        return Adata, Bdata, hfs, (metaA, metaB), c_struct, Dsize

    ia, ib, metaA, metaB, str_c, check_structure = 0, 0, [], [], [], False
    low = 0
    while ia < len(struct_a.t) and ib < len(struct_b.t):
        ta, Da, Dpa, asl = struct_a.t[ia], struct_a.D[ia], struct_a.Dp[ia], struct_a.sl[ia]
        tb, Db, Dpb, bsl = struct_b.t[ib], struct_b.D[ib], struct_b.Dp[ib], struct_b.sl[ib]
        if ta == tb:
            if Da != Db and not a.isdiag:
                raise YastError('Bond dimensions do not match.')
            metaA.append(((low, low + Dpa), asl))
            metaB.append(((low, low + Dpb), bsl))
            Dp = max(Dpa, Dpb)
            high = low + Dp
            str_c.append((ta, max(Da, Db), Dp, (low, high)))
            low = high
            ia += 1
            ib += 1
        elif ta < tb:
            high = low + Dpa
            metaA.append(((low, high), asl))
            str_c.append((ta, Da, Dpa, (low, high)))
            low = high
            ia += 1
            check_structure = True
        else:
            high = low + Dpb
            metaB.append(((low, high), bsl))
            str_c.append((tb, Db, Dpb, (low, high)))
            low = high
            ib += 1
            check_structure = True
    for ta, Da, Dpa, asl in zip(struct_a.t[ia:], struct_a.D[ia:], struct_a.Dp[ia:], struct_a.sl[ia:]):
        high = low + Dpa
        metaA.append(((low, high), asl))
        str_c.append((ta, Da, Dpa, (low, high)))
        low = high
        check_structure = True
    for tb, Db, Dpb, bsl in zip(struct_b.t[ib:], struct_b.D[ib:], struct_b.Dp[ib:], struct_b.sl[ib:]):
        high = low + Dpb
        metaB.append(((low, high), bsl))
        str_c.append((tb, Db, Dpb, (low, high)))
        low = high
        check_structure = True

    c_t, c_D, c_Dp, c_sl = zip(*str_c) if len(str_c) > 0 else ((), (), (), ())
    c_struct = struct_a._replace(t=c_t, D=c_D, Dp=c_Dp, sl=c_sl)
    if check_structure:
        _get_tD_legs(c_struct)
    return Adata, Bdata, hfs, (metaA, metaB), c_struct, high



# def _addition_meta(a, b):
#     """ meta-information for backend and new tensor charges and dimensions. """
#     if a.struct.n != b.struct.n:
#         raise YastError('Tensor charges do not match.')
#     needs_mask, _ = _test_axes_match(a, b, sgn=1)
#     if needs_mask:
#         msk_a, msk_b, struct_a, struct_b, hfs = _masks_for_add(a.config, a.struct, a.hfs, b.struct, b.hfs)
#         Dsize = struct_a.sl[-1][1] if len(struct_a.sl) > 0 else 0
#         Adata = a.config.backend.embed_msk(a._data, msk_a, Dsize)
#         Dsize = struct_b.sl[-1][1] if len(struct_b.sl) > 0 else 0
#         Bdata = a.config.backend.embed_msk(b._data, msk_b, Dsize)
#         del Dsize
#     else:
#         Adata, Bdata = a._data, b._data
#         struct_a, struct_b = a.struct, b.struct
#         hfs = a.hfs

#     if struct_a.t == struct_b.t:
#         if struct_a != struct_b:
#             raise YastError('Bond dimensions do not match.')
#         c_struct = struct_a
#         Dsize = c_struct.sl[-1][1] if len(c_struct.sl) > 0 else 0
#         meta = (((0, Dsize), (0, Dsize), (0, Dsize), 'AB'),)
#         return Adata, Bdata, hfs, meta, c_struct, Dsize

#     ia, ib, meta, c_t, c_D, c_Dp, c_sl = 0, 0, [], [], [], [], []
#     low = 0
#     while ia < len(struct_a.t) and ib < len(struct_b.t):
#         ta, Da, Dpa, asl = struct_a.t[ia], struct_a.D[ia], struct_a.Dp[ia], struct_a.sl[ia]
#         tb, Db, Dpb, bsl = struct_b.t[ib], struct_b.D[ib], struct_b.Dp[ib], struct_b.sl[ib]
#         if ta == tb:
#             if Da != Db:
#                 raise YastError('Bond dimensions do not match.')
#             high = low + Dpa
#             meta.append(((low, high), asl, bsl, 'AB'))
#             c_t.append(ta)
#             c_D.append(Da)
#             c_Dp.append(Dpa)
#             c_sl.append((low, high))
#             low = high
#             ia += 1
#             ib += 1
#         elif ta < tb:
#             high = low + Dpa
#             meta.append(((low, high), asl, None, 'A'))
#             c_t.append(ta)
#             c_D.append(Da)
#             c_Dp.append(Dpa)
#             c_sl.append((low, high))
#             low = high
#             ia += 1
#         else:
#             high = low + Dpb
#             meta.append(((low, high), None, bsl, 'B'))
#             c_t.append(tb)
#             c_D.append(Db)
#             c_Dp.append(Dpb)
#             c_sl.append((low, high))
#             low = high
#             ib += 1
#     for ta, Da, Dpa, asl in zip(struct_a.t[ia:], struct_a.D[ia:], struct_a.Dp[ia:], struct_a.sl[ia:]):
#         high = low + Dpa
#         meta.append(((low, high), asl, None, 'A'))
#         c_t.append(ta)
#         c_D.append(Da)
#         c_Dp.append(Dpa)
#         c_sl.append((low, high))
#         low = high
#     for tb, Db, Dpb, bsl in zip(struct_b.t[ib:], struct_b.D[ib:], struct_b.Dp[ib:], struct_b.sl[ib:]):
#         high = low + Dpb
#         meta.append(((low, high), None, bsl, 'B'))
#         c_t.append(tb)
#         c_D.append(Db)
#         c_Dp.append(Dpb)
#         c_sl.append((low, high))
#         low = high

#     c_struct = struct_a._replace(t=tuple(c_t), D=tuple(c_D), Dp=tuple(c_Dp), sl=tuple(c_sl))
#     if any(mt[3] != 'AB' for mt in meta):
#         _get_tD_legs(c_struct)
#     return Adata, Bdata, hfs, meta, c_struct, high

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
    data = a._data < number
    return a._replace(data=data)


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
    data = a._data > number
    return a._replace(data=data)


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
    data = a._data <= number
    return a._replace(data=data)


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
    data = a._data >= number
    return a._replace(data=data)


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
    data = number * a._data
    return a._replace(data=data)


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
    data = a._data ** exponent
    return a._replace(data=data)


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
    data = a._data / number
    return a._replace(data=data)


def __abs__(a):
    r"""
    Return tensor with element-wise absolute values.
    Can be on called on tensor as ``abs(tensor)``.

    Returns
    -------
    tensor: Tensor
    """
    data = a.config.backend.absolute(a._data)
    return a._replace(data=data)


def real(a):
    r"""

    Return tensor with imaginary part set to zero.

        .. note::
            Follows the behavior of the backend.real()
            when it comes to creating a new copy of the data or handling dtype.

    Returns
    -------
    tensor: Tensor
    """
    data = a.config.backend.real(a._data)
    return a._replace(data=data)


def imag(a):
    r"""
    Return tensor with real part set to zero.

        .. note::
            Follows the behavior of the backend.imag()
            when it comes to creating a new copy of the data or handling dtype.


    Returns
    -------
    tensor: Tensor
    """
    data = a.config.backend.imag(a._data)
    return a._replace(data=data)


def sqrt(a):
    """
    Return element-wise sqrt(A).

    Returns
    -------
    tensor: Tensor
    """
    data = a.config.backend.sqrt(a._data)
    return a._replace(data=data)


def rsqrt(a, cutoff=0):
    """
    Return element-wise 1/sqrt(A).

    The tensor elements with absolute value below the cutoff are set to zero.

    Parameters
    ----------
    cutoff: real scalar
        (element-wise) cutoff for inversion

    Returns
    -------
    tensor: Tensor
    """
    data = a.config.backend.rsqrt(a._data, cutoff=cutoff)
    return a._replace(data=data)


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
    data = a.config.backend.reciprocal(a._data, cutoff=cutoff)
    return a._replace(data=data)


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
    data = a.config.backend.exp(a._data, step)
    return a._replace(data=data)


def bitwise_not(a):
    r"""
    Return element-wise bit-wise not.

        .. note::
            This applies only to non-empty blocks of A.

    Returns
    -------
    tensor: Tensor
    """
    data = a.config.backend.bitwise_not(a._data)
    return a._replace(data=data)
