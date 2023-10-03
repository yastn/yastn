""" Linear operations and operations on a single yastn tensor. """
from ._merging import _masks_for_add
from ._tests import YastnError, _test_can_be_combined, _get_tD_legs, _test_axes_match
from ._auxliary import _slc


__all__ = ['apxb', 'real', 'imag', 'sqrt', 'rsqrt', 'reciprocal', 'exp', 'bitwise_not']


def __add__(a, b):
    """
    Add two tensors, use: :math:`a + b`.

    Signatures and total charges should match.

    Returns
    -------
    yastn.Tensor
    """
    _test_can_be_combined(a, b)
    aA, bA, hfs, meta, struct, slices = _addition_meta(a, b)
    data = a.config.backend.add(aA, bA, meta, struct.size)
    return a._replace(hfs=hfs, struct=struct, slices=slices, data=data)


def __sub__(a, b):
    """
    Subtract two tensors, use: :math:`a - b`.

    Both signatures and total charges should match.

    Returns
    -------
    yastn.Tensor
    """
    _test_can_be_combined(a, b)
    aA, bA, hfs, meta, struct, slices = _addition_meta(a, b)
    data = a.config.backend.sub(aA, bA, meta, struct.size)
    return a._replace(hfs=hfs, struct=struct, slices=slices, data=data)


def apxb(a, b, x=1):
    r"""
    Directly compute the result of :math:`a + x b`.

    Parameters
    ----------
    a, b : yastn.Tensor
    x : number

    Returns
    -------
    yastn.Tensor
    """
    _test_can_be_combined(a, b)
    aA, bA, hfs, meta, struct, slices = _addition_meta(a, b)
    data = a.config.backend.apxb(aA, bA, x, meta, struct.size)
    return a._replace(hfs=hfs, struct=struct, slices=slices, data=data)


def _addition_meta(a, b):
    """ meta-information for backend and new tensor charges and dimensions. """
    if a.struct.n != b.struct.n:
        raise YastnError('Tensor charges do not match.')
    if a.isdiag != b.isdiag:
        raise YastnError('Cannot add diagonal tensor to non-diagonal one.')

    needs_mask, _ = _test_axes_match(a, b, sgn=1)
    if needs_mask:  # TODO
        msk_a, msk_b, struct_a, slices_a, struct_b, slices_b, hfs = _masks_for_add(a.config, a.struct, a.slices, a.hfs, b.struct, b.slices, b.hfs)
        Adata = a.config.backend.embed_msk(a._data, msk_a, struct_a.size)
        Bdata = a.config.backend.embed_msk(b._data, msk_b, struct_b.size)
    else:
        Adata, Bdata = a._data, b._data
        struct_a, slices_a = a.struct, a.slices
        struct_b, slices_b = b.struct, b.slices
        hfs = a.hfs

    if struct_a == struct_b and slices_a == slices_b:
        Dsize = struct_a.size
        metaA = (((0, Dsize), (0, Dsize)),)
        metaB = (((0, Dsize), (0, Dsize)),)
        return Adata, Bdata, hfs, (metaA, metaB), struct_a, slices_a

    ia, ib, metaA, metaB, str_c, check_structure = 0, 0, [], [], [], False
    low = 0
    while ia < len(struct_a.t) and ib < len(struct_b.t):
        ta, Da, asl = struct_a.t[ia], struct_a.D[ia], slices_a[ia]
        tb, Db, bsl = struct_b.t[ib], struct_b.D[ib], slices_b[ib]
        if ta == tb:
            if Da != Db and not a.isdiag:
                raise YastnError('Bond dimensions do not match.')
            metaA.append(((low, low + asl.Dp), asl.slcs[0]))
            metaB.append(((low, low + bsl.Dp), bsl.slcs[0]))
            Dp = max(asl.Dp, bsl.Dp)
            high = low + Dp
            str_c.append((ta, max(Da, Db), Dp, (low, high)))
            low = high
            ia += 1
            ib += 1
        elif ta < tb:
            high = low + asl.Dp
            metaA.append(((low, high), asl.slcs[0]))
            str_c.append((ta, Da, asl.Dp, (low, high)))
            low = high
            ia += 1
            check_structure = True
        else:
            high = low + bsl.Dp
            metaB.append(((low, high), bsl.slcs[0]))
            str_c.append((tb, Db, bsl.Dp, (low, high)))
            low = high
            ib += 1
            check_structure = True
    for ta, Da, asl in zip(struct_a.t[ia:], struct_a.D[ia:], slices_a[ia:]):
        high = low + asl.Dp
        metaA.append(((low, high), asl.slcs[0]))
        str_c.append((ta, Da, asl.Dp, (low, high)))
        low = high
        check_structure = True
    for tb, Db, bsl in zip(struct_b.t[ib:], struct_b.D[ib:], slices_b[ib:]):
        high = low + bsl.Dp
        metaB.append(((low, high), bsl.slcs[0]))
        str_c.append((tb, Db, bsl.Dp, (low, high)))
        low = high
        check_structure = True

    c_t, c_D, c_Dp, c_sl = zip(*str_c) if len(str_c) > 0 else ((), (), (), ())
    slices_c = tuple(_slc((x,), y, z)  for x, y, z in zip(c_sl, c_D, c_Dp))
    struct_c = struct_a._replace(t=c_t, D=c_D, size=sum(c_Dp))
    if check_structure:
        _get_tD_legs(struct_c)
    return Adata, Bdata, hfs, (metaA, metaB), struct_c, slices_c


def __lt__(a, number):
    """
    Logical tensor with elements less-than a number (if it makes sense for backend data tensors),
    use: `tensor < number`

    Intended for diagonal tensor to be applied as a truncation mask.

    Returns
    -------
    yastn.Tensor(bool)
    """
    data = a._data < number
    return a._replace(data=data)


def __gt__(a, number):
    """
    Logical tensor with elements greater-than a number (if it makes sense for backend data tensors),
    use: `tensor > number`

    Intended for diagonal tensor to be applied as a truncation mask.

    Returns
    -------
    yastn.Tensor(bool)
    """
    data = a._data > number
    return a._replace(data=data)


def __le__(a, number):
    """
    Logical tensor with elements less-than-or-equal-to a number (if it makes sense for backend data tensors),
    use: `tensor <= number`

    Intended for diagonal tensor to be applied as a truncation mask.

    Returns
    -------
    yastn.Tensor(bool)
    """
    data = a._data <= number
    return a._replace(data=data)


def __ge__(a, number):
    """
    Logical tensor with elements greater-than-or-equal-to a number (if it makes sense for backend data tensors),
    use: `tensor >= number`

    Intended for diagonal tensor to be applied as a truncation mask.

    Returns
    -------
    yastn.Tensor(bool)
    """
    data = a._data >= number
    return a._replace(data=data)


def __mul__(a, number):
    """
    Multiply tensor by a number, use: `number * tensor`.

    Returns
    -------
    yastn.Tensor
    """
    data = number * a._data
    return a._replace(data=data)


def __rmul__(a, number):
    """
    Multiply tensor by a number, use: `tensor * number`.

    Returns
    -------
    yastn.Tensor
    """
    return __mul__(a, number)


def __pow__(a, exponent):
    """
    Element-wise exponent of tensor, use: `tensor ** exponent`.

    Returns
    -------
    yastn.Tensor
    """
    data = a._data ** exponent
    return a._replace(data=data)


def __truediv__(a, number):
    """
    Divide tensor by a scalar, use: `tensor / number`.

    Returns
    -------
    yastn.Tensor
    """
    data = a._data / number
    return a._replace(data=data)


def __abs__(a):
    r"""
    Return tensor with element-wise absolute values.
    Can be on called on tensor as ``abs(tensor)``.

    Returns
    -------
    yastn.Tensor
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
    yastn.Tensor
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
    yastn.Tensor
    """
    data = a.config.backend.imag(a._data)
    return a._replace(data=data)


def sqrt(a):
    """
    Return element-wise sqrt(tensor).

    Returns
    -------
    yastn.Tensor
    """
    data = a.config.backend.sqrt(a._data)
    return a._replace(data=data)


def rsqrt(a, cutoff=0):
    """
    Return element-wise 1/sqrt(tensor).

    The tensor elements with absolute value below the cutoff are set to zero.

    Parameters
    ----------
    cutoff: real scalar
        (element-wise) cutoff for inversion

    Returns
    -------
    yastn.Tensor
    """
    data = a.config.backend.rsqrt(a._data, cutoff=cutoff)
    return a._replace(data=data)


def reciprocal(a, cutoff=0):
    """
    Return element-wise 1/tensor.

    The tensor elements with absolute value below the cutoff are set to zero.

    Parameters
    ----------
    cutoff: real scalar
        (element-wise) cutoff for inversion

    Returns
    -------
    yastn.Tensor
    """
    data = a.config.backend.reciprocal(a._data, cutoff=cutoff)
    return a._replace(data=data)


def exp(a, step=1.):
    r"""
    Return element-wise `exp(step * tensor)`.

        .. note::
            This applies only to non-empty blocks of tensor

    Returns
    -------
    yastn.Tensor
    """
    data = a.config.backend.exp(a._data, step)
    return a._replace(data=data)


def bitwise_not(a):
    r"""
    Return element-wise bit-wise not.

        .. note::
            This applies only to non-empty blocks of tensor with tensor data dtype allowing for bitwise operation, i.e. intended for masks used to truncate tensor legs.

    Returns
    -------
    yastn.Tensor
    """
    data = a.config.backend.bitwise_not(a._data)
    return a._replace(data=data)
