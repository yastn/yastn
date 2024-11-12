# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Linear operations and operations on a single yastn.Tensor. """
from __future__ import annotations
from itertools import accumulate
import numpy as np
from ._auxliary import _slc, _clear_axes, _unpack_axes
from ._merging import _Fusion
from ._tests import YastnError, _test_axes_all


__all__ = ['conj', 'conj_blocks', 'flip_signature', 'flip_charges',
           'transpose', 'moveaxis', 'move_leg', 'diag', 'remove_zero_blocks',
           'add_leg', 'remove_leg', 'copy', 'clone', 'detach', 'to',
           'requires_grad_', 'grad', 'drop_leg_history']


def copy(a) -> yastn.Tensor:
    r"""
    Return a copy of the tensor. Data of the resulting tensor is independent
    from the original.

    .. warning::
        This operation does not preserve autograd on returned :class:`yastn.Tensor`.

    Returns
    -------
    yastn.Tensor
    """
    data = a.config.backend.copy(a._data)
    return a._replace(data=data)


def clone(a) -> yastn.Tensor:
    r"""
    Return a clone of the tensor preserving the autograd - resulting clone is a part
    of the computational graph. Data of the resulting tensor is indepedent
    from the original.
    """
    data = a.config.backend.clone(a._data)
    return a._replace(data=data)


def to(a, device=None, dtype=None) -> yastn.Tensor:
    r"""
    Move tensor to device and cast to given datatype.

    Returns a clone of the tensor residing on ``device`` in desired datatype ``dtype``.
    If tensor already resides on ``device``, returns ``self``. This operation preserves autograd.
    If no change is needed, makes only a shallow copy of the tensor data.

    Parameters
    ----------
    device: str
        device identifier
    dtype: str
        desired dtype
    """
    if dtype in (None, a.yast_dtype) and device in (None, a.device):
        return a._replace()
    data = a.config.backend.move_to(a._data, dtype=dtype, device=device)
    return a._replace(data=data)


def detach(a) -> yastn.Tensor:
    r"""
    Detach tensor from the computational graph returning a `view`.

    Data of the resulting tensor is a `view` of the original data.
    In case of NumPy backend, returns ``self``.

    .. warning::
        This operation does not preserve autograd on returned :class:`yastn.Tensor`.
    """
    data = a.config.backend.detach(a._data)
    return a._replace(data=data)


def grad(a) -> yastn.Tensor:
    """
    TODO ADD description
    """
    data = a.config.backend.grad(a._data)
    return a._replace(data=data)


def requires_grad_(a, requires_grad=True) -> Never:
    r"""
    Activate or deactivate recording of operations on the tensor for automatic differentiation.

    Parameters
    ----------
    requires_grad: bool
        if ``True`` activates autograd.
    """
    a.config.backend.requires_grad_(a._data, requires_grad=requires_grad)


def conj(a) -> yastn.Tensor:
    r"""
    Return conjugated tensor. In particular, change the sign of the signature `s` to `-s`,
    the total charge `n` to `-n`, and complex conjugate each block of the tensor.

    Follows the behavior of the :code:`backend.conj()` when it comes to creating a new copy of the data.
    """
    newn = a.config.sym.add_charges(a.struct.n, new_signature=-1)
    news = tuple(-x for x in a.struct.s)
    struct = a.struct._replace(s=news, n=newn)
    hfs = tuple(hf.conj() for hf in a.hfs)
    data = a.config.backend.conj(a._data)
    return a._replace(hfs=hfs, struct=struct, data=data)


def conj_blocks(a) -> yastn.Tensor:
    """
    Complex-conjugate all blocks leaving symmetry structure (signature, blocks charge, and
    total charge) unchanged.

    Follows the behavior of the :code:`backend.conj()` when it comes to creating a new copy of the data.
    """
    data = a.config.backend.conj(a._data)
    return a._replace(data=data)


def flip_signature(a) -> yastn.Tensor:
    r"""
    Change the signature of the tensor, `s` to `-s` or equivalently
    reverse the direction of in- and out-going legs, and also the total charge
    of the tensor `n` to `-n`. Does not complex-conjugate the elements of the tensor.

    Creates a shallow copy of the data.
    """
    newn = a.config.sym.add_charges(a.struct.n, new_signature=-1)
    news = tuple(-x for x in a.struct.s)
    struct = a.struct._replace(s=news, n=newn)
    hfs = tuple(hf.conj() for hf in a.hfs)
    return a._replace(hfs=hfs, struct=struct)


def flip_charges(a, axes=None) -> yastn.Tensor:
    r"""
    Flip signs of charges and signatures on specified legs.

    Flipping charges/signature of hard-fused legs is not supported.

    Parameters
    ----------
    axes: int | Sequence[int]
        index of the leg, or a group of legs. If None, flips all legs.
    """
    if a.isdiag:
        raise YastnError('Cannot flip charges of a diagonal tensor. Use diag() first.')
    if axes is None:
        axes = tuple(range(a.ndim))
    else:
        try:
            axes = tuple(axes)
        except TypeError:
            axes = (axes,)
    uaxes, = _unpack_axes(a.mfs, axes)

    snew = list(a.struct.s)
    hfs = list(a.hfs)
    lt, ndim_n, nsym = len(a.struct.t), len(a.struct.s), len(a.struct.n)
    tnew = np.array(a.struct.t, dtype=np.int64).reshape(lt, ndim_n, nsym)
    for ax in uaxes:
        if hfs[ax].is_fused():
            raise YastnError('Flipping charges of hard-fused leg is not supported.')
        s = snew[ax]
        tnew[:, ax, :] = a.config.sym.fuse(tnew[:, (ax,), :], (s,), -s)
        snew[ax] = -s
        hfs[ax] = hfs[ax].conj()
    snew = tuple(snew)
    hfs = tuple(hfs)
    tnew = tuple(map(tuple, tnew.reshape(lt, ndim_n * nsym).tolist()))
    meta = sorted((x, y, z.Dp, z.slcs[0]) for x, y, z in zip(tnew, a.struct.D, a.slices))
    tnew, Dnew, Dpnew, slold = zip(*meta) if len(meta) > 0 else ((), (), (), ())

    slices = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(Dpnew), Dpnew, Dnew))
    struct = a.struct._replace(s=snew, t=tnew, D=Dnew)
    meta = tuple((x.slcs[0], y) for x, y in zip(slices, slold))
    data = a.config.backend.embed_slc(a.data, meta, struct.size)
    return a._replace(struct=struct, slices=slices, data=data, hfs=hfs)


def drop_leg_history(a, axes=None) -> yastn.Tensor:
    r"""
    Drops information about original structure of fused or blocked legs
    that have been combined into a selected tensor leg(s).

    Makes a shallow copy of tensor data.

    Parameters
    ----------
    axes: int | Sequence[int]
        index of the leg, or a group of legs. If :code:`None`, drops information from all legs.
    """
    if axes is None:
        axes = tuple(range(a.ndim))
    else:
        try:
            axes = tuple(axes)
        except TypeError:
            axes = (axes,)
    uaxes, = _unpack_axes(a.mfs, axes)
    hfs = tuple(_Fusion(s=(a.struct.s[n],)) if n in uaxes else a.hfs[n] for n in range(a.ndim_n))
    return a._replace(hfs=hfs)


def transpose(a, axes=None) -> yastn.Tensor:
    r"""
    Transpose tensor by permuting the order of its legs (spaces).
    Makes a shallow copy of tensor data if the order is not changed.

    Parameters
    ----------
    axes: Sequence[int]
        new order of legs. Has to be a valid permutation of :code:`(0, 1, ..., ndim-1)`
        where :code:`ndim` is tensor order (number of legs).
        By default is :code:`range(a.ndim)[::-1]`, which reverses the order of the axes.
    """
    if axes is None:
        axes = tuple(range(a.ndim-1, -1, -1))
    _test_axes_all(a, axes, native=False)
    if axes == tuple(range(a.ndim)):
        return a._replace()
    uaxes, = _unpack_axes(a.mfs, axes)
    order = np.array(uaxes, dtype=np.int64)
    mfs = tuple(a.mfs[ii] for ii in axes)
    hfs = tuple(a.hfs[ii] for ii in uaxes)
    c_s = tuple(a.struct.s[ii] for ii in uaxes)
    lt, ndim_n, nsym = len(a.struct.t), len(a.struct.s), len(a.struct.n)

    tset = np.array(a.struct.t, dtype=np.int64).reshape(lt, ndim_n, nsym)
    Dset = np.array(a.struct.D, dtype=np.int64).reshape(lt, ndim_n)
    newt = tuple(map(tuple, tset[:, order, :].reshape(lt, ndim_n * nsym).tolist()))
    newD = tuple(map(tuple, Dset[:, order].tolist()))

    meta = sorted(zip(newt, newD, a.slices), key=lambda x: x[0])

    c_t = tuple(mt[0] for mt in meta)
    c_D = tuple(mt[1] for mt in meta)
    c_Dp = tuple(mt[2].Dp for mt in meta)
    c_sl = tuple((stop - dp, stop) for stop, dp in zip(accumulate(c_Dp), c_Dp))

    slices = tuple(_slc((x,), y, z) for x, y, z in zip(c_sl, c_D, c_Dp))
    struct = a.struct._replace(s=c_s, t=c_t, D=c_D)
    meta = tuple((sln.slcs[0], sln.D, mt[2].slcs[0], mt[2].D) for sln, mt, in zip(slices, meta))

    data = a._data if a.isdiag else a.config.backend.transpose(a._data, uaxes, meta)
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, slices=slices, data=data)


def move_leg(a, source, destination) -> yastn.Tensor:
    r"""
    Change the position of an axis (or a group of axes) of the tensor.
    This is a convenience function for subset of possible permutations. It
    computes the corresponding permutation and then calls :meth:`yastn.Tensor.transpose`.

    Makes a shallow copy of tensor data if the order is not changed.

    Parameters
    ----------
    source, destination: int | Sequence[int]
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


def moveaxis(a, source, destination) -> yastn.Tensor:
    r"""
    Change the position of an axis (or a group of axes) of the tensor.
    This is a convenience function for subset of possible permutations. It
    computes the corresponding permutation and then calls :meth:`yastn.Tensor.transpose`.

    This function is an alias for :meth:`yastn.Tensor.move_leg`.

    Parameters
    ----------
    source, destination: int | Sequence[int]
    """
    return move_leg(a, source, destination)


def add_leg(a, axis=-1, s=-1, t=None, leg=None) -> yastn.Tensor:
    r"""
    Creates a new tensor with an extra leg that carries the charge (or part of it)
    of the orignal tensor. This is achieved by the extra leg having a single charge sector
    of dimension `D=1`. The total charge of the tensor :attr:`yastn.Tensor.n` can be modified this way.

    Makes a shallow copy of tensor data.

    Parameters
    ----------
    axis: int
        index of the new leg

    s : int
        signature of the new leg, +1 or -1.
        The default is -1, where the leg charge is equal to the tensor charge for :code:`t=None`.

    t : int | Sequence[int]
        charge carried by the new leg. If ``None``, takes the total charge `n`
        of the original tensor resulting in a tensor with `n=0`.

    leg : Optional[Leg]
        It is possible to provide a new leg directly.
        It has to be of dimension one but can contain information about the fusion of other dimension-one legs.
        If given (not None), it overrides information provided in ``s`` and ``t``.
    """
    if a.isdiag:
        raise YastnError('Cannot add axis to a diagonal tensor.')

    if leg is not None:
        if len(leg.t) != 1 or leg.D[0] != 1:
            raise YastnError("Only the leg of dimension one can be added to the tensor.")
        if isinstance(leg.fusion, tuple):  # meta fused leg
            for ll in leg.legs[::-1]:
                a = a.add_leg(axis=axis, leg=ll)
            mfs = a.mfs[:axis] + (leg.fusion,) + a.mfs[axis + len(leg.legs):]
            return a._replace(mfs=mfs)
        s = leg.s
        t = leg.t[0]
        hfsa = leg.legs[0]
    else:
        hfsa = _Fusion(s=(s,))

    if s not in (-1, 1):
        raise YastnError('Signature of the new axis should be 1 or -1.')
    s = int(s)

    axis = axis % (a.ndim + 1)
    mfs = a.mfs[:axis] + ((1,),) + a.mfs[axis:]

    axis = sum(a.mfs[ii][0] for ii in range(axis))  # unpack mfs
    nsym = a.config.sym.NSYM
    if t is None:
        t = a.config.sym.add_charges(a.struct.n, signatures=(-1,), new_signature=s)
    else:
        if (isinstance(t, int) and nsym != 1) or len(t) != nsym:
            raise YastnError('len(t) does not match the number of symmetry charges.')
        t = a.config.sym.add_charges(t, signatures=(s,), new_signature=s)

    news = a.struct.s[:axis] + (s,) + a.struct.s[axis:]
    newn = a.config.sym.add_charges(a.struct.n, t, signatures=(1, s))
    newt = tuple(x[:axis * nsym] + t + x[axis * nsym:] for x in a.struct.t)
    newD = tuple(x[:axis] + (1,) + x[axis:] for x in a.struct.D)
    struct = a.struct._replace(t=newt, D=newD, s=news, n=newn)
    slices = tuple(_slc(x.slcs, y, x.Dp) for x, y in zip(a.slices, newD))
    hfs = a.hfs[:axis] + (hfsa,) + a.hfs[axis:]
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, slices=slices)


def remove_leg(a, axis=-1) -> yastn.Tensor:
    r"""
    Removes leg with a single charge sector of dimension one from tensor.
    The charge carried by that leg (if any) is added to the
    tensor's total charge :attr:`yastn.Tensor.n`.

    Makes a shallow copy of tensor data.

    Parameters
    ----------
    axis: int
        index of the leg to be removed.
    """
    if a.isdiag:
        raise YastnError('Cannot remove axis to a diagonal tensor.')
    if a.ndim == 0:
        raise YastnError('Cannot remove axis of a scalar tensor.')

    axis = axis % a.ndim
    if a.mfs[axis] != (1,):
        raise YastnError('Axis to be removed cannot be fused.')
    mfs = a.mfs[:axis] + a.mfs[axis + 1:]

    axis = sum(a.mfs[ii][0] for ii in range(axis))  # unpack mfs
    if a.hfs[axis].tree != (1,):
        raise YastnError('Axis to be removed cannot be fused.')

    nsym = a.config.sym.NSYM
    if len(a.struct.t) > 0:
        t = a.struct.t[0][axis * nsym: (axis + 1) * nsym]
    else:
        t = a.config.sym.zero()

    if any(x[axis] != 1 for x in a.struct.D) or any(x[axis * nsym: (axis + 1) * nsym] != t for x in a.struct.t):
        raise YastnError('Axis to be removed must have single charge of dimension one.')

    news = a.struct.s[:axis] + a.struct.s[axis + 1:]
    newn = a.config.sym.add_charges(a.struct.n, t, signatures=(-1, a.struct.s[axis]), new_signature=-1)
    newt = tuple(x[: axis * nsym] + x[(axis + 1) * nsym:] for x in a.struct.t)
    newD = tuple(x[: axis] + x[axis + 1:] for x in a.struct.D)
    struct = a.struct._replace(t=newt, D=newD, s=news, n=newn)
    slices = tuple(_slc(x.slcs, y, x.Dp) for x, y in zip(a.slices, newD))
    hfs = a.hfs[:axis] + a.hfs[axis + 1:]
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, slices=slices)


def diag(a) -> yastn.Tensor:
    """
    Select diagonal of 2D tensor and output it as a diagonal tensor, or vice versa.
    """
    if not a.isdiag:  # isdiag=False -> isdiag=True
        if a.ndim_n != 2 or sum(a.struct.s) != 0:
            raise YastnError('Diagonal tensor requires 2 legs with opposite signatures.')
        if any(x != 0 for x in a.struct.n):
            raise YastnError('Diagonal tensor requires zero tensor charge.')
        if any(mf != (1,) for mf in a.mfs) or any(hf.tree != (1,) for hf in a.hfs):
            raise YastnError('Diagonal tensor cannot have fused legs.')
        if any(d0 != d1 for d0, d1 in a.struct.D):
            raise YastnError('yastn.diag() allowed only for square blocks.')
        #     isdiag=True -> isdiag=False                    isdiag=False -> isdiag=True
    Dp = tuple(x.Dp ** 2 for x in a.slices) if a.isdiag else tuple(D[0] for D in a.struct.D)
    slices = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(Dp), Dp, a.struct.D))
    struct = a.struct._replace(diag=not a.isdiag, size=sum(Dp))

    if a.isdiag:  # isdiag=True -> isdiag=False
        meta = tuple((x.slcs[0], y.slcs[0]) for x, y in zip(slices, a.slices))
        data = a.config.backend.diag_1dto2d(a._data, meta, struct.size)
    else:  # isdiag=False -> isdiag=True
        meta = tuple((x.slcs[0], y.slcs[0], y.D) for x, y in zip(slices, a.slices))
        data = a.config.backend.diag_2dto1d(a._data, meta, struct.size)
    return a._replace(struct=struct, slices=slices, data=data)


def remove_zero_blocks(a, rtol=1e-12, atol=0) -> yastn.Tensor:
    r"""
    Remove blocks where all elements are below a cutoff.

    Cutoff is a combination of absolut tolerance and
    relative tolerance with respect to maximal element in the tensor.
    """
    cutoff = atol + rtol * a.norm(p='inf')
    meta = [(t, D, sl) for t, D, sl in zip(a.struct.t, a.struct.D, a.slices) \
             if a.config.backend.max_abs(a._data[slice(*sl.slcs[0])]) > cutoff]
    c_t = tuple(mt[0] for mt in meta)
    c_D = tuple(mt[1] for mt in meta)
    old_sl = tuple(mt[2] for mt in meta)
    c_Dp = tuple(x.Dp for x in old_sl)
    old_sl = tuple(x.slcs[0] for x in old_sl)
    slices = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(c_Dp), c_Dp, c_D))
    c_sl = tuple(x.slcs[0] for x in slices)
    struct = a.struct._replace(t=c_t, D=c_D, size=sum(c_Dp))
    data = a.config.backend.apply_slice(a._data, c_sl, old_sl)
    return a._replace(struct=struct, slices=slices, data=data)
