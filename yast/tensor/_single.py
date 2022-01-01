""" Linear operations and operations on a single yast tensor. """
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _tarray, _Darray, _struct
from ._merging import _Fusion, _flip_hf
from ._tests import YastError, _test_all_axes

__all__ = ['conj', 'conj_blocks', 'flip_signature', 'transpose', 'moveaxis', 'diag', 'remove_zero_blocks',
           'add_leg', 'copy', 'clone', 'detach', 'to', 'requires_grad_']


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


def to(a, device=None, dtype=None):
    r"""
    Move tensor to device.

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
    c_config = a.config
    if device is not None:
        c_config = c_config._replace(device=device)
    if dtype is not None:
        c_config = c_config._replace(dtype=dtype)
    c = a.__class__(config=c_config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = a.config.backend.move_to(a.A, dtype=dtype, device=device)
    return c


def requires_grad_(a, requires_grad=True):
    r"""
    Turn on recording of operations on the tensor for automatic differentiation.

    .. note::
        This operation sets autograd for `all` non-empty blocks of the tensor.

    Parameters
    ----------
    requires_grad: bool
    """
    a.config.backend.requires_grad_(a.A, requires_grad=requires_grad)


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
    new_hf = tuple(_flip_hf(x) for x in a.hard_fusion)
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
    r"""
    Change the signature of the tensor, `s` to `-s` or equivalently
    reverse the direction of in- and out-going legs, and also the total charge
    of the tensor `n` to `-n`. Does not complex-conjugate the elements of the tensor.

    Parameters
    ----------
    inplace : bool

    Returns
    -------
    tensor : Tensor
        clone of the tensor with modified signature `-s` and total 
        charge `-n`. If inplace is ``True`` modify only the structural data of tensor, 
        not its blocks, and return ``self``.

    """
    an = np.array(a.struct.n, dtype=int).reshape((1, 1, -1))
    newn = tuple(a.config.sym.fuse(an, np.array([1], dtype=int), -1)[0])
    news = tuple(-x for x in a.struct.s)
    struct = a.struct._replace(s=news, n=newn)
    new_hf = tuple(_flip_hf(x) for x in a.hard_fusion)
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
    Transpose tensor by permuting the order of its legs (spaces). 
    Transpose can be done in-place, in which case copying of the data is not forced.
    Otherwise, new tensor is created and its data (blocks) is cloned.

    Parameters
    ----------
    axes: tuple(int)
        new order of legs. Has to be a valid permutation of (0, 1, ..., ndim-1)

    inplace: bool

    Returns
    -------
    tensor : Tensor
        transposed tensor
    """
    _test_all_axes(a, axes, native=False)
    uaxes, = _unpack_axes(a.meta_fusion, axes)
    order = np.array(uaxes, dtype=np.intp)
    new_mf = tuple(a.meta_fusion[ii] for ii in axes)
    new_hf = tuple(a.hard_fusion[ii] for ii in uaxes)
    c_s = tuple(a.struct.s[ii] for ii in uaxes)
    tset, Dset = _tarray(a), _Darray(a)
    newt = tset[:, order, :]
    newD = Dset[:, order]
    meta = sorted(((tuple(tn.flat), to, tuple(Dn)) for tn, to, Dn in zip(newt, a.struct.t, newD)), key=lambda x: x[0])
    c_t = tuple(tn for tn, _, _ in meta)
    c_D = tuple(Dn for _, _, Dn in meta)
    meta = tuple((tn, to) for tn, to, _ in meta)
    struct = _struct(t=c_t, D=c_D, s=c_s, n=a.struct.n)

    if inplace:
        a.struct = struct
        a.meta_fusion = new_mf
        a.hard_fusion = new_hf
    c = a if inplace else a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=new_mf, hard_fusion=new_hf, struct=struct)
    if a.isdiag and not inplace:
        c.A = {k: c.config.backend.clone(v) for k, v in a.A.items()}
    elif not a.isdiag:
        c.A = c.config.backend.transpose(a.A, uaxes, meta, inplace)
    return c


def moveaxis(a, source, destination, inplace=False):
    r"""
    Change the position of an axis (or a group of axes) of the tensor.
    This is a convenience function for subset of possible permutations. It
    computes the corresponding permutation and then calls :meth:`yast.Tensor.transpose`. 

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
        return a if inplace else a.clone()
    axes = [ii for ii in range(a.ndim) if ii not in lsrc]
    ds = sorted(((d, s) for d, s in zip(ldst, lsrc)))
    for d, s in ds:
        axes.insert(d, s)
    return transpose(a, axes, inplace)


def add_leg(a, axis=-1, s=1, t=None, inplace=False):
    r"""
    Creates a new auxiliary leg that explicitly carries charge 
    (or part of it) associated with the tensor.

    Parameters
    ----------
        axis: int
            index of the new leg

        s : int
            signature :math:`\pm1` of the new leg

        t : ?
            charge carried by the new leg. If ``None``, takes the total charge `n` 
            of the original tensor resulting in uncharged tensor with `n=0`.

        inplace : bool
            If ``True``, perform operation in place
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


def diag(a):
    """
    Select diagonal of 2d tensor and output it as a diagonal tensor, or vice versa. """
    if a.isdiag:
        c = a.__class__(config=a.config, isdiag=False, meta_fusion=a.meta_fusion, \
            hard_fusion=a.hard_fusion, struct=a.struct)
        c.A = {ind: a.config.backend.diag_create(a.A[ind]) for ind in a.A}
        return c
    if a.ndim_n == 2 and all(x == 0 for x in a.struct.n):
        c = a.__class__(config=a.config, isdiag=True, meta_fusion=a.meta_fusion, \
            hard_fusion=a.hard_fusion, struct=a.struct)
        c.A = {ind: a.config.backend.diag_get(a.A[ind]) for ind in a.A}
        return c
    raise YastError('Tensor cannot be changed into a diagonal one')


def remove_zero_blocks(a, rtol=1e-12, atol=0, inplace=False):
    r"""
    Remove from the tensor blocks where all elements are below a cutoff.
    Cutoff is a combination of absolut tolerance and relative tolerance with respect to maximal element in the tensor.
    """
    cutoff = atol + rtol * a.norm(p='inf')
    newtD = [(t, D) for t, D in zip(a.struct.t, a.struct.D) if a.config.backend.max_abs(a.A[t]) > cutoff]
    c_t = tuple(t for t, _ in newtD)
    c_D = tuple(D for _, D in newtD)
    struct = a.struct._replace(t=c_t, D=c_D)
    if inplace:
        c = a
        a.struct = struct
    else:
        c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=struct)
    c.A = {t: a.A[t] if inplace else a.config.backend.clone(a.A[t]) for t in c_t}
    return c
