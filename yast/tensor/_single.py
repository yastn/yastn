""" Linear operations and operations on a single yast tensor. """
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _common_keys, _tarray, _Darray, _struct
from ._merging import _Fusion, _flip_sign_hf, _masks_for_add
from ._tests import YastError, _test_configs_match, _test_tensors_match, _test_all_axes, _get_tD_legs

__all__ = ['conj', 'conj_blocks', 'flip_signature', 'transpose', 'moveaxis', 'diag', 'remove_zero_blocks',
           'absolute', 'real', 'imag', 'sqrt', 'rsqrt', 'reciprocal', 'exp', 'apxb', 'add_leg',
           'copy', 'clone', 'detach', 'to', 'requires_grad_']


def copy(a):
    r""" 
    Return a copy of the tensor.

    .. warning::
        this operation doesn't preserve autograd on returned :class:`yast.Tensor`

    Returns
    -------
    tensor : Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ts: a.config.backend.copy(x) for ts, x in a.A.items()}
    return c


def clone(a):
    r""" 
    Return a copy of the tensor preserving the autograd (resulting clone is a part 
    of the computational graph) 

    Returns
    -------
    tensor : Tensor 
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ts: a.config.backend.clone(x) for ts, x in a.A.items()}
    return c


# TODO inplace ?
def detach(a, inplace=False):
    r""" 
    Detach tensor from computational graph.

    Returns
    -------
    tensor : Tensor
    """
    if inplace:
        for x in a.A.values():
            a.config.backend.detach_(x)
        return a
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ts: a.config.backend.detach(x) for ts, x in a.A.items()}
    return c


def to(a, device, dtype=None):
    r"""
    Move the ``Tensor`` to ``device``. Returns a copy of the tensor on `device``.

    If the tensor already resides on ``device``, return a

    Parameters
    ----------
    device: str
        device identifier
    dtype: str
        desired dtype

    Returns
    -------
    tensor : Tensor
        returns a clone of the tensor residing on ``device`` in desired ``dtype``. If tensor already 
        resides on ``device``, returns ``self``. This operation preserves autograd.
    """
    a_dtype= a.unique_dtype()
    if (dtype is None and device is None) or \
        (dtype is None and a.config.device == device) or \
        (a_dtype == dtype and device is None) or \
        (a_dtype == dtype and a.config.device == device):
        return a
    c = a.__class__(config=a.config, isdiag=a.isdiag, device=device, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.move_to(a.A, dtype=dtype, device=device)
    return c


def requires_grad_(a, requires_grad=True):
    r"""
    Turn on recording of operations for the tensor for automatic differentiation.

    Parameters
    ----------
    requires_grad: bool
    """
    a.config.backend.requires_grad_(a.A, requires_grad=requires_grad)


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
    _test_tensors_match(a, b)
    meta = _common_keys(a.A, b.A)
    masks_needed = any(ha != hb for ha, hb in zip(a.hard_fusion, b.hard_fusion))
    if masks_needed:
        sla, tDa, slb, tDb, hfs = _masks_for_add(a.config, a.struct, a.hard_fusion, b.struct, b.hard_fusion)
        aA = a.config.backend.embed(a.A, sla, tDa)
        bA = a.config.backend.embed(b.A, slb, tDb)
    else:
        aA, bA = a.A, b.A
        hfs = a.hard_fusion
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=hfs, struct=a.struct)
    c.A = c.config.backend.add(aA, bA, meta)
    if len(meta[1]) > 0 or len(meta[2]) > 0 or masks_needed:
        c.update_struct()
    if len(meta[1]) > 0 or len(meta[2]) > 0:
        _get_tD_legs(c.struct)
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
    _test_tensors_match(a, b)
    meta = _common_keys(a.A, b.A)
    masks_needed = any(ha != hb for ha, hb in zip(a.hard_fusion, b.hard_fusion))
    if masks_needed:
        sla, tDa, slb, tDb, hfs = _masks_for_add(a.config, a.struct, a.hard_fusion, b.struct, b.hard_fusion)
        aA = a.config.backend.embed(a.A, sla, tDa)
        bA = a.config.backend.embed(b.A, slb, tDb)
    else:
        aA, bA = a.A, b.A
        hfs = a.hard_fusion
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=hfs, struct=a.struct)
    c.A = c.config.backend.sub(aA, bA, meta)
    if len(meta[1]) > 0 or len(meta[2]) > 0 or masks_needed:
        c.update_struct()
    if len(meta[1]) > 0 or len(meta[2]) > 0:
        _get_tD_legs(c.struct)
    return c


def apxb(a, b, x=1):
    """
    Directly calculate tensor: a + x * b

    Signatures and total charges should match.

    Parameters
    ----------
    a, b: yast tensors
    x : number

    Returns
    -------
    tensor : Tensor
        result of addition as a new tensor
    """
    _test_configs_match(a, b)
    _test_tensors_match(a, b)
    meta = _common_keys(a.A, b.A)
    masks_needed = any(ha != hb for ha, hb in zip(a.hard_fusion, b.hard_fusion))
    if masks_needed:
        sla, tDa, slb, tDb, hfs = _masks_for_add(a.config, a.struct, a.hard_fusion, b.struct, b.hard_fusion)
        aA = a.config.backend.embed(a.A, sla, tDa)
        bA = a.config.backend.embed(b.A, slb, tDb)
    else:
        aA, bA = a.A, b.A
        hfs = a.hard_fusion
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=hfs, struct=a.struct)
    c.A = c.config.backend.apxb(aA, bA, x, meta)
    if len(meta[1]) > 0 or len(meta[2]) > 0 or masks_needed:
        c.update_struct()
    if len(meta[1]) > 0 or len(meta[2]) > 0:
        _get_tD_legs(c.struct)
    return c


def conj(a, inplace=False):
    r"""
    Return conjugated tensor. In particular, change the sign of the signature `s` to `-s`, 
    the total charge `n` to `-n`, and complex conjugate each block of the tensor.    
    
    Parameters
    ----------
    inplace : bool

    Returns
    -------
    tensor : Tensor
    """
    an = np.array(a.struct.n, dtype=int).reshape((1, 1, -1))
    newn = tuple(a.config.sym.fuse(an, np.array([1], dtype=int), -1)[0])
    news = tuple(-x for x in a.struct.s)
    struct = a.struct._replace(s=news, n=newn)
    new_hf = tuple(_flip_sign_hf(x) for x in a.hard_fusion)
    if inplace:
        c = a
        c.struct = struct
        a.hard_fusion = new_hf
    else:
        c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=new_hf, struct=struct)
    c.A = c.config.backend.conj(a.A, inplace)
    return c


def conj_blocks(a, inplace=False):
    """
    Complex-conjugate each block leaving symmetry structure (signature, blocks charge, and 
    total charge) unchanged.

    Parameters
    ----------
    inplace : bool

    Returns
    -------
    tensor : Tensor
    """
    c = a if inplace else a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,
                                        hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = c.config.backend.conj(a.A, inplace)
    return c

# TODO add axis
def flip_signature(a, inplace=False):
    """
    Conjugated each block, leaving symmetry structure unchanged.

    Returns
    -------
    tensor : Tensor
    """
    an = np.array(a.struct.n, dtype=int).reshape((1, 1, -1))
    newn = tuple(a.config.sym.fuse(an, np.array([1], dtype=int), -1)[0])
    news = tuple(-x for x in a.struct.s)
    struct = a.struct._replace(s=news, n=newn)
    new_hf = tuple(_flip_sign_hf(x) for x in a.hard_fusion)
    if inplace:
        a.struct = struct
        a.hard_fusion = new_hf
        return a
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion,\
        hard_fusion=new_hf, struct=struct)
    c.A = {ind: a.config.backend.clone(a.A[ind]) for ind in a.A}
    return c


def transpose(a, axes, inplace=False):
    r"""
    Return transposed tensor.

    Operation can be done in-place, in which case copying of the data is not forced.
    Otherwise, new tensor is created and the data are copied.

    Parameters
    ----------
    axes: tuple of ints
        New order of the legs. Should be a permutation of (0, 1, ..., ndim-1)
    """
    _test_all_axes(a, axes, native=False)
    uaxes, = _unpack_axes(a.meta_fusion, axes)
    order = np.array(uaxes, dtype=np.intp)
    new_meta_fusion = tuple(a.meta_fusion[ii] for ii in axes)
    new_hard_fusion = tuple(a.hard_fusion[ii] for ii in uaxes)
    news = tuple(a.struct.s[ii] for ii in uaxes)
    struct = _struct(s=news, n=a.struct.n)
    tset = _tarray(a)
    newt = tset[:, order, :]
    meta_transpose = tuple((told, tuple(tnew.flat)) for told, tnew in zip(a.struct.t, newt))
    if inplace:
        a.struct = struct
        a.meta_fusion = new_meta_fusion
        a.hard_fusion = new_hard_fusion
    c = a if inplace else a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=new_meta_fusion, hard_fusion=new_hard_fusion, struct=struct)
    if a.isdiag and not inplace:
        c.A = {k: c.config.backend.clone(v) for k, v in a.A.items()}
    elif not a.isdiag:
        c.A = c.config.backend.transpose(a.A, uaxes, meta_transpose, inplace)
    c.update_struct()
    return c


def moveaxis(a, source, destination, inplace=False):
    r"""
    Change the position of an axis (or a group of axes) of the tensor.

    Operation can be done in-place, in which case copying of the data is not forced.
    Othersiwe, new tensor is created and the data are copied. Calls transpose.

    Parameters
    ----------
    source, destination: ints
    """
    lsrc, ldst = _clear_axes(source, destination)
    lsrc = tuple(xx + a.ndim if xx < 0 else xx for xx in lsrc)
    ldst = tuple(xx + a.ndim if xx < 0 else xx for xx in ldst)
    if lsrc == ldst:
        return a if inplace else a.clone()
    axes = [ii for ii in range(a.ndim) if ii not in lsrc]
    ds = sorted(((d, s) for d, s in zip(ldst, lsrc)))
    for d, s in ds:
        axes.insert(d, s)
    return transpose(a, axes, inplace)


def remove_zero_blocks(a, rtol=1e-12, atol=0, inplace=False):
    r"""
    Remove from the tensor blocks where all elements are below a cutoff.
    Cutoff is a combination of absolut tolerance and relative tolerance with respect to maximal element in the tensor.
    """
    cutoff = atol + rtol * a.norm(p='inf')
    c = a if inplace else a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {k: t if inplace else a.config.backend.clone(t) for k, t in a.A.items() if a.config.backend.max_abs(t) > cutoff}
    c.update_struct()
    return c


def add_leg(a, axis=-1, s=1, t=None, inplace=False):
    r"""
    Creates a new auxiliary leg that explicitly carries charge (or part of it) associated with the tensor.

    Parameters
    ----------
        axis: int
            index of the new leg

        s : int
            signature of the new leg

        t : charge on the new leg.
            If None takes the charge of the tensor, making it zero

        inplace : bool
            If true, perform operation in place
    """
    if a.isdiag:
        raise YastError('Cannot add a new leg to a diagonal tensor.')
    tset, Dset = _tarray(a), _Darray(a)

    axis = axis % (a.mlegs + 1)

    new_meta_fusion = a.meta_fusion[:axis] + ((1,),) + a.meta_fusion[axis:]

    axis = sum(a.meta_fusion[ii][0] for ii in range(axis))  # unpack

    if s not in (-1, 1):
        raise YastError('The signature s should be equal to 1 or -1.')
    an = np.array(a.struct.n, dtype=int)
    if t is None:
        t = a.config.sym.fuse(an.reshape((1, 1, -1)), np.array([1], dtype=int), -1)[0] if s == 1 else an  # s == -1
    else:
        t = a.config.sym.fuse(np.array(t, dtype=int).reshape((1, 1, -1)), np.array([1], dtype=int), 1)[0]
    if len(t) != a.config.sym.NSYM:
        raise YastError('t does not have the proper number of symmetry charges')

    news = np.insert(np.array(a.struct.s, dtype=int), axis, s)
    newn = a.config.sym.fuse(np.hstack([an, t]).reshape((1, 2, -1)), np.array([1, s], dtype=int), 1)[0]
    new_tset = np.insert(tset, axis, t, axis=1)
    new_Dset = np.insert(Dset, axis, 1, axis=1)

    news = tuple(news)
    newn = tuple(newn)
    new_tset = tuple(tuple(x.flat) for x in new_tset)
    new_Dset = tuple(tuple(x.flat) for x in new_Dset)

    c = a if inplace else a.clone()
    c.A = {tnew: a.config.backend.expand_dims(c.A[told], axis) for tnew, told in zip(new_tset, a.struct.t)}
    c.struct = _struct(new_tset, new_Dset, news, newn)
    c.meta_fusion = new_meta_fusion
    c.hard_fusion = c.hard_fusion[:axis] + (_Fusion(s=(s,), ms=(-s,)),) + c.hard_fusion[axis:]

    return c


def diag(a):
    """Select diagonal of 2d tensor and output it as a diagonal tensor, or vice versa. """
    if a.isdiag:
        c = a.__class__(config=a.config, isdiag=False, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
        c.A = {ind: a.config.backend.diag_create(a.A[ind]) for ind in a.A}
        return c
    if a.ndim_n == 2 and all(x == 0 for x in a.struct.n):
        c = a.__class__(config=a.config, isdiag=True, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
        c.A = {ind: a.config.backend.diag_get(a.A[ind]) for ind in a.A}
        return c
    raise YastError('Tensor cannot be changed into a diagonal one')


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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.sqrt(a.A)
    return c


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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
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
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.reciprocal(a.A, cutoff=cutoff)
    return c


def exp(a, step=1.):
    r"""
    Return element-wise `exp(step * A)`.

    ..note ::
        This applies only to non-empty blocks of A

    Parameters
    ----------
    step: scalar

    Returns
    -------
    tensor: Tensor
    """
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.exp(a.A, step)
    return c


def remove_zero_blocks(a, rtol=1e-12, atol=0, inplace=False):
    r"""
    Remove from the tensor blocks where all elements are below a cutoff.
    Cutoff is a combination of absolut tolerance and relative tolerance with respect to maximal element in the tensor.
    """
    cutoff = atol + rtol * a.norm(p='inf')
    c = a if inplace else a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {k: t if inplace else a.config.backend.clone(t) for k, t in a.A.items() if a.config.backend.max_abs(t) > cutoff}
    c.update_struct()
    return c


def add_leg(a, axis=-1, s=1, t=None, inplace=False):
    r"""
    Creates a new auxiliary leg that explicitly carries charge (or part of it) associated with the tensor.

    Parameters
    ----------
        axis: int
            index of the new leg

        s : int
            signature of the new leg

        t : charge on the new leg.
            If None takes the charge of the tensor, making it zero

        inplace : bool
            If true, perform operation in place
    """
    if a.isdiag:
        raise YastError('Cannot add a new leg to a diagonal tensor.')
    tset, Dset = _tarray(a), _Darray(a)

    axis = axis % (a.ndim + 1)

    new_meta_fusion = a.meta_fusion[:axis] + ((1,),) + a.meta_fusion[axis:]

    axis = sum(a.meta_fusion[ii][0] for ii in range(axis))  # unpack

    if s not in (-1, 1):
        raise YastError('The signature s should be equal to 1 or -1.')
    an = np.array(a.struct.n, dtype=int)
    if t is None:
        t = a.config.sym.fuse(an.reshape((1, 1, -1)), np.array([1], dtype=int), -1)[0] if s == 1 else an  # s == -1
    else:
        t = a.config.sym.fuse(np.array(t, dtype=int).reshape((1, 1, -1)), np.array([1], dtype=int), 1)[0]
    if len(t) != a.config.sym.NSYM:
        raise YastError('t does not have the proper number of symmetry charges')

    news = np.insert(np.array(a.struct.s, dtype=int), axis, s)
    newn = a.config.sym.fuse(np.hstack([an, t]).reshape((1, 2, -1)), np.array([1, s], dtype=int), 1)[0]
    new_tset = np.insert(tset, axis, t, axis=1)
    new_Dset = np.insert(Dset, axis, 1, axis=1)

    news = tuple(news)
    newn = tuple(newn)
    new_tset = tuple(tuple(x.flat) for x in new_tset)
    new_Dset = tuple(tuple(x.flat) for x in new_Dset)

    c = a if inplace else a.clone()
    c.A = {tnew: a.config.backend.expand_dims(c.A[told], axis) for tnew, told in zip(new_tset, a.struct.t)}
    c.struct = _struct(new_tset, new_Dset, news, newn)
    c.meta_fusion = new_meta_fusion
    c.hard_fusion = c.hard_fusion[:axis] + (_Fusion(s=(s,), ms=(-s,)),) + c.hard_fusion[axis:]

    return c
