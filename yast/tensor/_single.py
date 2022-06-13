""" Linear operations and operations on a single yast tensor. """
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes
from ._merging import _Fusion
from ._tests import YastError, _test_axes_all
from ._legs import leg_union, _unpack_legs

__all__ = ['conj', 'conj_blocks', 'flip_signature', 'transpose', 'moveaxis', 'move_leg', 'diag', 'remove_zero_blocks',
           'add_leg', 'remove_leg', 'copy', 'clone', 'detach', 'to', 'requires_grad_', 'grad']


def copy(a):
    r"""
    Return a copy of the tensor. Data of the resulting tensor is independent 
    from the original.

    .. warning::
        this operation does not preserve autograd on returned :class:`yast.Tensor`

    Returns
    -------
    yast.Tensor
    """
    data = a.config.backend.copy(a._data)
    return a._replace(data=data)


def clone(a):
    r"""
    Return a clone of the tensor preserving the autograd - resulting clone is a part
    of the computational graph. Data of the resulting tensor is indepedent
    from the original.

    Returns
    -------
    yast.Tensor
    """
    data = a.config.backend.clone(a._data)
    return a._replace(data=data)


def to(a, device=None, dtype=None):
    r"""
    Move tensor to device and cast to dtype.

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

        If no change is needed, makes only a shallow copy of the tensor data.
    """
    if dtype in (None, a.yast_dtype) and device in (None, a.device):
        return a._replace()
    data = a.config.backend.move_to(a._data, dtype=dtype, device=device)
    return a._replace(data=data)


def detach(a):
    r"""
    Detach tensor from the computational graph returning a `view`. Data of the resulting
    tensor is a `view` of the original data. 

    .. warning::
        this operation does not preserve autograd on returned :class:`yast.Tensor`

    Returns
    -------
    yast.Tensor
        In case of NumPy backend, returns ``self``.
    """
    data = a.config.backend.detach(a._data)
    return a._replace(data=data)


def grad(a):
    """
    TODO ADD description
    """
    data = a.config.backend.grad(a._data)
    return a._replace(data=data)


def requires_grad_(a, requires_grad=True):
    r"""
    Activate or deactivate recording of operations on the tensor for automatic differentiation.

    .. note::
        This operation sets requires_grad flag for `all` non-empty blocks of the tensor.

    Parameters
    ----------
    requires_grad: bool
    """
    a.config.backend.requires_grad_(a._data, requires_grad=requires_grad)


def conj(a):
    r"""
    Return conjugated tensor. In particular, change the sign of the signature `s` to `-s`,
    the total charge `n` to `-n`, and complex conjugate each block of the tensor.

    Follows the behavior of the backend.conj() when it comes to creating a new copy of the data.

    Returns
    -------
    tensor : Tensor
    """
    an = np.array(a.struct.n, dtype=int).reshape((1, 1, -1))
    newn = tuple(a.config.sym.fuse(an, np.array([1], dtype=int), -1)[0])
    news = tuple(-x for x in a.struct.s)
    struct = a.struct._replace(s=news, n=newn)
    hfs = tuple(hf.conj() for hf in a.hfs)
    data = a.config.backend.conj(a._data)
    return a._replace(hfs=hfs, struct=struct, data=data)


def conj_blocks(a):
    """
    Complex-conjugate all blocks leaving symmetry structure (signature, blocks charge, and
    total charge) unchanged.

    Follows the behavior of the backend.conj() when it comes to creating a new copy of the data.

    Returns
    -------
    tensor : Tensor
    """
    data = a.config.backend.conj(a._data)
    return a._replace(data=data)


def flip_signature(a):
    r"""
    Change the signature of the tensor, `s` to `-s` or equivalently
    reverse the direction of in- and out-going legs, and also the total charge
    of the tensor `n` to `-n`. Does not complex-conjugate the elements of the tensor.

    Creates a shallow copy of the data.

    Returns
    -------
    tensor : Tensor
        clone of the tensor with modified signature `-s` and total
        charge `-n`.
    """
    an = np.array(a.struct.n, dtype=int).reshape((1, 1, -1))
    newn = tuple(a.config.sym.fuse(an, np.array([1], dtype=int), -1)[0])
    news = tuple(-x for x in a.struct.s)
    struct = a.struct._replace(s=news, n=newn)
    hfs = tuple(hf.conj() for hf in a.hfs)
    return a._replace(hfs=hfs, struct=struct)


def transpose(a, axes):
    r"""
    Transpose tensor by permuting the order of its legs (spaces).
    Transpose can be done in-place, in which case copying of the data is not forced.
    Otherwise, new tensor is created and its data (blocks) is cloned.

    Makes a shallow copy of Tensor data if the order is not changed.

    Parameters
    ----------
    axes: tuple(int)
        new order of legs. Has to be a valid permutation of (0, 1, ..., ndim-1)

    Returns
    -------
    tensor : Tensor
        transposed tensor
    """
    _test_axes_all(a, axes, native=False)
    if axes == tuple(range(a.ndim)):
        return a._replace()
    uaxes, = _unpack_axes(a.mfs, axes)
    order = np.array(uaxes, dtype=np.intp)
    mfs = tuple(a.mfs[ii] for ii in axes)
    hfs = tuple(a.hfs[ii] for ii in uaxes)
    c_s = tuple(a.struct.s[ii] for ii in uaxes)
    tset = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
    Dset = np.array(a.struct.D, dtype=int).reshape((len(a.struct.D), len(a.struct.s)))
    newt = tuple(tuple(x.flat) for x in tset[:, order, :])
    newD = tuple(tuple(x.flat) for x in Dset[:, order])

    meta = sorted(zip(newt, a.struct.Dp, newD, a.struct.sl, a.struct.D), key=lambda x: x[0])

    c_t = tuple(mt[0] for mt in meta)
    c_D = tuple(mt[2] for mt in meta)
    c_Dp = tuple(mt[1] for mt in meta)
    c_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(c_Dp), c_Dp))

    meta = tuple((sln, *mt[2:]) for sln, mt, in zip(c_sl, meta))
    struct = a.struct._replace(s=c_s, t=c_t, D=c_D, Dp=c_Dp, sl=c_sl)
    data = a._data if a.isdiag else a.config.backend.transpose(a._data, uaxes, meta)
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, data=data)


def move_leg(a, source, destination):
    r"""
    Change the position of an axis (or a group of axes) of the tensor.
    This is a convenience function for subset of possible permutations. It
    computes the corresponding permutation and then calls :meth:`yast.Tensor.transpose`.

    Makes a shallow copy of Tensor data if the order is not changed.

    Parameters
    ----------
    source, destination: int or tuple(int)

    Returns
    -------
    tensor : Tensor
    """
    lsrc, ldst = _clear_axes(source, destination)
    lsrc = tuple(xx + a.ndim if xx < 0 else xx for xx in lsrc)
    ldst = tuple(xx + a.ndim if xx < 0 else xx for xx in ldst)
    if lsrc == ldst:
        return a._replace()
    axes = [ii for ii in range(a.ndim) if ii not in lsrc]
    ds = sorted(((d, s) for d, s in zip(ldst, lsrc)))
    for d, s in ds:
        axes.insert(d, s)
    return transpose(a, axes)


def moveaxis(a, source, destination):
    r"""
    Change the position of an axis (or a group of axes) of the tensor.
    This is a convenience function for subset of possible permutations. It
    computes the corresponding permutation and then calls :meth:`yast.Tensor.transpose`.

    This function is an alias for :meth:`yast.Tensor.move_leg`.

    Parameters
    ----------
    source, destination: int or tuple(int)

    Returns
    -------
    tensor : Tensor
    """
    return move_leg(a, source, destination)


def add_leg(a, axis=-1, s=1, t=None):
    r"""
    Creates a new auxiliary leg that explicitly carries charge
    (or part of it) associated with the tensor.

    Makes a shallow copy of Tensor data.

    Parameters
    ----------
        axis: int
            index of the new axis

        s : int
            signature :math:`\pm1` of the new axis

        t : tuple
            charge carried by the new leg. If ``None``, takes the total charge `n`
            of the original tensor resulting in uncharged tensor with `n=0`.
    """
    if a.isdiag:
        raise YastError('Cannot add axis to a diagonal tensor.')
    if s not in (-1, 1):
        raise YastError('Signature of the new axis should be 1 or -1.')

    axis = axis % (a.ndim + 1)
    mfs = a.mfs[:axis] + ((1,),) + a.mfs[axis:]

    axis = sum(a.mfs[ii][0] for ii in range(axis))  # unpack mfs
    nsym = a.config.sym.NSYM
    if t is None:
        t = tuple(a.config.sym.fuse(np.array(a.struct.n, dtype=int).reshape((1, 1, nsym)), (-1,), s).flat)
    else:
        if (isinstance(t, int) and nsym != 1) or len(t) != nsym:
            raise YastError('len(t) does not match the number of symmetry charges.')
        t = tuple(a.config.sym.fuse(np.array(t, dtype=int).reshape((1, 1, nsym)), (s,), s).flat)

    news = a.struct.s[:axis] + (s,) + a.struct.s[axis:]
    newn = tuple(a.config.sym.fuse(np.array(a.struct.n + t, dtype=int).reshape((1, 2, nsym)), (1, s), 1).flat)
    newt = tuple(x[:axis * nsym] + t + x[axis * nsym:] for x in a.struct.t)
    newD = tuple(x[:axis] + (1,) + x[axis:] for x in a.struct.D)
    struct = a.struct._replace(t=newt, D=newD, s=news, n=newn)
    hfs = a.hfs[:axis] + (_Fusion(s=(s,)),) + a.hfs[axis:]
    return a._replace(mfs=mfs, hfs=hfs, struct=struct)


def remove_leg(a, axis=-1):
    r"""
    Removes leg of single charge with dimension one.

    The charge carried by that axis is added to the tensors charge.

    Makes a shallow copy of Tensor data.

    Parameters
    ----------
        axis: int
            index of the axis to be removed
    """
    if a.isdiag:
        raise YastError('Cannot remove axis to a diagonal tensor.')
    if a.ndim == 0:
        raise YastError('Cannot remove axis of a scalar tensor.')

    axis = axis % a.ndim
    if a.mfs[axis] != (1,):
        raise YastError('Axis to be removed cannot be fused.')
    mfs = a.mfs[:axis] + a.mfs[axis + 1:]

    axis = sum(a.mfs[ii][0] for ii in range(axis))  # unpack mfs
    if a.hfs[axis].tree != (1,):
        raise YastError('Axis to be removed cannot be fused.')

    nsym = a.config.sym.NSYM
    t = a.struct.t[0][axis * nsym: (axis + 1) * nsym] if len(a.struct.t) > 0 else (0,) * nsym
    if any(x[axis] != 1 for x in a.struct.D) or any(x[axis * nsym: (axis + 1) * nsym] != t for x in a.struct.t):
        raise YastError('Axis to be removed must have single charge of dimension one.')

    news = a.struct.s[:axis] + a.struct.s[axis + 1:]
    newn = tuple(a.config.sym.fuse(np.array(a.struct.n + t, dtype=int).reshape((1, 2, nsym)), (-1, a.struct.s[axis]), -1).flat)
    newt = tuple(x[: axis * nsym] + x[(axis + 1) * nsym:] for x in a.struct.t)
    newD = tuple(x[: axis] + x[axis + 1:] for x in a.struct.D)
    struct = a.struct._replace(t=newt, D=newD, s=news, n=newn)

    hfs = a.hfs[:axis] + a.hfs[axis + 1:]
    return a._replace(mfs=mfs, hfs=hfs, struct=struct)


def diag(a):
    """
    Select diagonal of 2d tensor and output it as a diagonal tensor, or vice versa.
    """
    if not a.isdiag:  # isdiag=False -> isdiag=True
        if a.ndim_n != 2 or sum(a.struct.s) != 0:
            raise YastError('Diagonal tensor requires 2 legs with opposite signatures.')
        if any(x != 0 for x in a.struct.n):
            raise YastError('Diagonal tensor requires zero tensor charge.')
        if any(mf != (1,) for mf in a.mfs) or any(hf.tree != (1,) for hf in a.hfs):
            raise YastError('Diagonal tensor cannot have fused legs.')
        if any(d0 != d1 for d0, d1 in a.struct.D):
            raise YastError('yast.diag() allowed only for square blocks.')
        #     isdiag=True -> isdiag=False                    isdiag=False -> isdiag=True
    Dp = tuple(x ** 2 for x in a.struct.Dp) if a.isdiag else tuple(D[0] for D in a.struct.D)
    sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(Dp), Dp))
    struct = a.struct._replace(Dp=Dp, sl=sl, diag=not a.isdiag)

    Dsize = sl[-1][1] if len(sl) > 0 else 0
    if a.isdiag:  # isdiag=True -> isdiag=False
        meta = tuple(zip(sl, a.struct.sl))
        data = a.config.backend.diag_1dto2d(a._data, meta, Dsize)
    else:  # isdiag=False -> isdiag=True
        meta = tuple(zip(sl, a.struct.sl, a.struct.D))
        data = a.config.backend.diag_2dto1d(a._data, meta, Dsize)
    return a._replace(struct=struct, data=data)


def remove_zero_blocks(a, rtol=1e-12, atol=0):
    r"""
    Remove from the tensor blocks where all elements are below a cutoff.
    Cutoff is a combination of absolut tolerance and relative tolerance with respect to maximal element in the tensor.
    """
    cutoff = atol + rtol * a.norm(p='inf')
    meta = [(t, D, Dp, sl) for t, D, Dp, sl in zip(a.struct.t, a.struct.D, a.struct.Dp, a.struct.sl) \
             if a.config.backend.max_abs(a._data[slice(*sl)]) > cutoff]
    c_t = tuple(mt[0] for mt in meta)
    c_D = tuple(mt[1] for mt in meta)
    c_Dp = tuple(mt[2] for mt in meta)
    old_sl = tuple(mt[3] for mt in meta)
    c_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(c_Dp), c_Dp))
    struct = a.struct._replace(t=c_t, D=c_D, Dp=c_Dp, sl=c_sl)
    data = a.config.backend.apply_slice(a._data, c_sl, old_sl)
    return a._replace(struct=struct, data=data)


# def embed(a, legs=None, return_legs=False):
#     r"""
#     Create equivalent ``yast.Tensor``, where some zero blocks are explicitly added,
#     to include all charges specified in provided legs.

#     .. note::
#         Utility function used in to_nonsymmetric(). Might potentially be usefull also for yast.block.

#     Parameters
#     ----------
#     legs : dict
#         {n: Leg} specify charges and dimensions to include on some legs (indicated by keys n).

#     return_legs : bool
#         if True, additionally return a tuple of legs of a new tensor.

#     Returns
#     -------
#     out : yast.Tensor
#         If embeding is not needed, return self
#     """
#     if legs is None:
#         return a
#     if any((n < 0) or (n >= a.ndim) for n in legs.keys()):
#         raise YastError('Specified leg out of ndim')

#     a_legs, perform_embeding = list(a.get_legs()), False
#     for n, leg in legs.items():
#         new_leg = leg_union(a_legs[n], leg)
#         if new_leg != a_legs[n]:
#             perform_embeding = True
#         a_legs[n] = new_leg
#     if perform_embeding:
#         ulegs, _ = _unpack_legs(a_legs)
#         t = tuple(leg.t for leg in ulegs)
#         D = tuple(leg.D for leg in ulegs)
#         hfs = tuple(leg.legs[0] for leg in ulegs)
#         b = a._replace(hfs=hfs)
#         b.fill_tensor(t=t, D=D, val='zeros')
#         a = a + b
#     return a
