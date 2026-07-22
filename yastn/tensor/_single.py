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

from typing import Sequence, TYPE_CHECKING, Union

import numpy as np

from ._auxiliary import _clear_axes, _unpack_axes, get_blocks, argsort_t
from ._einsum import ncon
from ._legbasic import LegBasic
from ._legs import LegMeta, Leg, leg_product
from ._merging import _Fusion
from ._tests import YastnError, _test_axes_all

if TYPE_CHECKING:
    from . import Tensor

__all__ = ['conj', 'conj_blocks', 'consume_transpose',
           'flip_signature', 'flip_charges', 'switch_signature',
           'transpose', 'moveaxis', 'move_leg', 'diag',
           'add_leg', 'remove_leg', 'copy', 'clone', 'detach', 'to',
           'requires_grad_', 'grad', 'drop_leg_history', 'shallow_copy']


def shallow_copy(a) -> 'Tensor':
    r"""
    Return a shallow copy of the tensor.

    Returns
    -------
    yastn.Tensor
    """
    return a._replace()


def copy(a) -> 'Tensor':
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


def clone(a) -> 'Tensor':
    r"""
    Return a clone of the tensor preserving the autograd - resulting clone is a part
    of the computational graph. Data of the resulting tensor is independent
    from the original.
    """
    data = a.config.backend.clone(a._data)
    return a._replace(data=data)


def to(a, device=None, dtype=None, **kwargs) -> 'Tensor':
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
    if dtype in (None, a.yastn_dtype) and device in (None, a.device):
        return a
    data = a.config.backend.move_to(a._data, dtype=dtype, device=device, **kwargs)
    return a._replace(data=data)


def detach(a) -> 'Tensor':
    r"""
    Detach tensor from the computational graph returning a `view`.

    Data of the resulting tensor is a `view` of the original data.
    In case of NumPy backend, returns ``self``.

    .. warning::
        This operation does not preserve autograd on returned :class:`yastn.Tensor`.
    """
    data = a.config.backend.detach(a._data)
    return a._replace(data=data)


def detach_(a) -> 'Tensor':
    r"""
    Detach tensor from the computational graph, in place.
    """
    a.config.backend.detach_(a._data)


def grad(a) -> 'Tensor':
    """
    Calculate the gradient of tensor elements after .backward() is called on the scalar result.
    """
    data = a.config.backend.grad(a._data)
    return a._replace(data=data)


def requires_grad_(a, requires_grad=True) -> None:
    r"""
    Activate or deactivate recording of operations on the tensor for automatic differentiation.

    Parameters
    ----------
    requires_grad: bool
        If ``True``, activates autograd.
    """
    a.config.backend.requires_grad_(a._data, requires_grad=requires_grad)


def conj(a) -> 'Tensor':
    r"""
    Return conjugated tensor. In particular, change the sign of the signature `s` to `-s`,
    the total charge `n` to `-n`, and complex conjugate each block of the tensor.

    Follows the behavior of the :code:`backend.conj()` when it comes to creating a new copy of the data.
    """
    new_n = a.config.sym.add_charges(a.struct.n, new_signature=-1)
    new_legs = tuple(leg.conj() for leg in a.struct.legs)
    struct = a.struct.replace(legs=new_legs, n=new_n)
    hfs = tuple(hf.conj() for hf in a.hfs)
    data = a.config.backend.conj(a._data)
    return a._replace(hfs=hfs, struct=struct, data=data)


def conj_blocks(a) -> 'Tensor':
    """
    Complex-conjugate all blocks leaving symmetry structure (signature, blocks charge, and
    total charge) unchanged.

    Follows the behavior of the :code:`backend.conj()` when it comes to creating a new copy of the data.
    """
    data = a.config.backend.conj(a._data)
    return a._replace(data=data)


def flip_signature(a) -> 'Tensor':
    r"""
    Change the signature of the tensor, `s` to `-s` or equivalently
    reverse the direction of in- and out-going legs, and also the total charge
    of the tensor `n` to `-n`. Does not complex-conjugate the elements of the tensor.

    Creates a shallow copy of the data.
    """
    new_n = a.config.sym.add_charges(a.struct.n, new_signature=-1)
    new_legs = tuple(leg.conj() for leg in a.struct.legs)
    struct = a.struct.replace(legs=new_legs, n=new_n)
    hfs = tuple(hf.conj() for hf in a.hfs)
    return a._replace(hfs=hfs, struct=struct)


def flip_charges(a, axes=None) -> 'Tensor':
    r"""
    Flip signs of charges and signatures on specified legs.

    Flipping charges/signature of hard-fused legs is not supported.

    Parameters
    ----------
    axes: int | Sequence[int]
        index of the leg, or a group of legs.
        The default is ``None``, which flips all legs.
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
    uaxes = tuple(a.trans[ax] for ax in uaxes)

    new_legs = list(a.struct.legs)
    hfs_new = list(a.hfs)

    bl_all = get_blocks(a.config.sym, a.struct.replace(mask=None))
    t_flip = bl_all.t.copy()
    for ax in uaxes:
        if hfs_new[ax].is_fused():
            raise YastnError('Flipping charges of hard-fused leg is not supported.')
        hfs_new[ax] = hfs_new[ax].conj()
        leg = a.struct.legs[ax]
        new_legs[ax] = leg.conj_charges(a.config.sym)
        t_flip[:, ax, :] = a.config.sym.fuse(t_flip[:, (ax,), :], (leg.s,), -leg.s)

    if a.struct.mask.array is None:
        struct_new = a.struct.replace(legs=new_legs)
    else:
        inds_all = argsort_t(t_flip)
        mask_new = a.struct.mask.array[inds_all]
        struct_new = a.struct.replace(legs=new_legs, mask=mask_new)
        t_flip = t_flip[a.struct.mask.array]

    bl_old = get_blocks(a.config.sym, a.struct)
    bl_new = get_blocks(a.config.sym, struct_new)
    inds = argsort_t(t_flip)
    assert np.array_equal(t_flip[inds], bl_new.t), "Sanity check. Contact developers.."
    sln, slo = bl_new.slc, bl_old.slc[inds]
    meta = np.column_stack([sln, sln[:, 1] - sln[:, 0], slo, slo[:, 1] - slo[:, 0]])
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('Dn', np.int64, (1,)),
        ('slo', np.int64, (2,)),
        ('Do', np.int64, (1,))])
    meta = meta.view(meta_dt).reshape(-1)
    data = a.config.backend.embed_transpose(a._data, [0], meta, bl_new.size)  # used for embeding
    out = a._replace(struct=struct_new, data=data, hfs=hfs_new)
    return out


def switch_signature(a, axes: Union[Sequence[int],int,str] = ()) -> 'Tensor':
    r"""
    Flip signature (and hence also charges) on specified legs.
    This function supports flipping signature of hard-fused legs.

    Parameters
    ----------
    axes: int | Sequence[int] | str
        index of the leg, or a group of legs.
        If ``axes="all"``, all signatures are flipped.
    """
    from .. import eye
    if a.isdiag:
        raise YastnError('Cannot flip charges of a diagonal tensor. Use diag() first.')
    if type(axes) is str:
        if axes == "all":
            axes = tuple(range(a.ndim))
        else:
            raise YastnError("Invalid axes")
    if type(axes)==int: axes=[axes]
    if len(axes)==0: return a
    if not (all([type(x)==int for x in axes]) and len(set(axes))==len(axes)):
        raise YastnError("Invalid axes: all elements must be integers and no repeating axes are allowed.")
    def _conj_completion(leg):
        # new leg with sectors from both leg and leg.conj()
        # case leg is not fused:
        if not leg.is_fused():
            tDconj= np.array(leg.t, dtype=np.int64)
            tDconj= a.config.sym.fuse(tDconj.reshape(-1,1,leg.sym.NSYM), (leg.s,), -leg.s)
            tDconj= tuple(map(tuple, tDconj))
            tDs= dict(zip(tDconj, leg.D))
            return Leg(a.config.sym, -leg.s, t= tuple(tDs.keys()), D= tuple(tDs.values()))
        else:
            return leg_product(*tuple(_conj_completion(x) for x in leg.unfuse_leg()))
    symbols_1j= tuple(eye(a.config, legs=(a.get_legs(x).conj(), _conj_completion(a.get_legs(x))), isdiag=False) for x in axes)
    outi_a= [i+1 if i in axes else -(i+1) for i in range(len(a.get_legs()))] # shift by 1 to avoid 0,0 ambiguity
    contractedi= [[x+1,-(x+1)] for x in axes ]
    return ncon( (a,)+symbols_1j, [outi_a,]+contractedi )


def drop_leg_history(a, axes=None) -> 'Tensor':
    r"""
    Drops information about original structure of fused or blocked legs
    that have been combined into a selected tensor leg(s).

    Makes a shallow copy of tensor data.

    Parameters
    ----------
    axes: int | Sequence[int]
        index of the leg, or a group of legs.
        The default is :code:`None`, which drops information from all legs.
    """
    if axes is None:
        axes = tuple(range(a.ndim))
    else:
        try:
            axes = tuple(axes)
        except TypeError:
            axes = (axes,)
    uaxes, = _unpack_axes(a.mfs, axes)
    uaxes = tuple(a.trans[ax] for ax in uaxes)
    hfs = tuple(_Fusion(s=(a.struct.legs[n].s,)) if n in uaxes else a.hfs[n] for n in range(a.ndim_n))
    return a._replace(hfs=hfs)


def transpose(a, axes=None):
    r"""
    Transpose tensor by permuting the order of its legs (spaces).
    Do not copy tensor data.

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
    uaxes, = _unpack_axes(a.mfs, axes)
    mfs = tuple(a.mfs[ii] for ii in axes)
    trans = tuple(a.trans[ii] for ii in uaxes)
    return a._replace(trans=trans, mfs=mfs)


def consume_transpose(a) -> 'Tensor':
    r"""
    Enforce logical transformation done by Tensor.transpose()
    on Tensor.struct and reshufling Tensor.data
    """
    no_trans = tuple(range(a.ndim_n))
    if a.trans == no_trans:
        return a
    order = np.array(a.trans, dtype=np.int64)
    new_hfs = tuple(a.hfs[ii] for ii in a.trans)
    new_legs = tuple(a.struct.legs[ii] for ii in a.trans)

    if a.struct.mask.array is None:
        struct_new = a.struct.replace(legs=new_legs)
    else:
        bl_all = get_blocks(a.config.sym, a.struct.replace(mask=None))
        inds_all = argsort_t(bl_all.t[:, order, :])
        mask_new = a.struct.mask.array[inds_all]
        struct_new = a.struct.replace(legs=new_legs, mask=mask_new)

    bl_new = get_blocks(a.config.sym, struct_new)
    bl_old = get_blocks(a.config.sym, a.struct)
    inds = argsort_t(bl_old.t[:, order, :])
    meta = np.hstack([bl_new.slc, bl_new.D, bl_old.slc[inds], bl_old.D[inds]])
    ndim = len(new_legs)
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('Dn', np.int64, (ndim,)),
        ('slo', np.int64, (2,)),
        ('Do', np.int64, (ndim,))])
    meta = meta.view(meta_dt).reshape(-1)
    data = a._data if a.isdiag else a.config.backend.embed_transpose(a._data, a.trans, meta, bl_new.size)
    return a._replace(hfs=new_hfs, struct=struct_new, data=data, trans=no_trans)


def moveaxis(a, source, destination) -> 'Tensor':
    r"""
    Change the position of an axis (or a group of axes) of the tensor.
    This is a convenience function for subset of possible permutations. It
    computes the corresponding permutation and calls :meth:`yastn.transpose`.

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


def move_leg(a, source, destination) -> 'Tensor':
    r"""
    Change the position of an axis (or a group of axes) of the tensor.
    This is a convenience function for subset of possible permutations. It
    computes the corresponding permutation and calls :meth:`yastn.transpose`.

    TODO: remove

    Parameters
    ----------
    source, destination: int | Sequence[int]
    """
    return moveaxis(a, source, destination)


def add_leg(a, axis=-1, s=-1, t=None, leg=None) -> 'Tensor':
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
        charge carried by the new leg. The default is ``None``,
        which takes the total charge `n` of the original tensor resulting in a tensor with `n=0`.

    leg : Optional[Leg]
        It is possible to provide a new leg directly.
        It has to be of dimension one but can contain information about the fusion of other dimension-one legs.
        If provided, it overrides information in ``s`` and ``t``. The default is ``None``.
    """
    if a.isdiag:
        raise YastnError('Cannot add axis to a diagonal tensor.')

    if leg is not None:
        if len(leg.t) != 1 or leg.D[0] != 1:
            raise YastnError("Only the leg of dimension one can be added to the tensor.")
        if isinstance(leg, LegMeta):  # meta fused leg
            for ll in leg.legs[::-1]:
                a = a.add_leg(axis=axis, leg=ll)
            mfs = a.mfs[:axis] + (leg.mf,) + a.mfs[axis + len(leg.legs):]
            return a._replace(mfs=mfs)
        s = leg.s
        t = leg.t[0]
        hfsa = leg.hf
    else:
        hfsa = _Fusion(s=(s,))

    if s not in (-1, 1):
        raise YastnError('Signature of the new axis should be 1 or -1.')
    s = int(s)

    axis = axis % (a.ndim + 1)
    mfs = a.mfs[:axis] + ((1,),) + a.mfs[axis:]

    uaxis = sum(a.mfs[ii][0] for ii in range(axis))  # unpack mfs

    trans = list(a.trans)
    haxis = trans[uaxis] if uaxis < len(trans) else uaxis
    for k, v in enumerate(trans):
        if v >= haxis:
            trans[k] = v + 1
    trans = trans[:uaxis] + [haxis] + trans[uaxis:]

    nsym = a.config.sym.NSYM
    if t is None:
        t = a.config.sym.add_charges(a.struct.n, signatures=(-1,), new_signature=s)
    else:
        if (isinstance(t, int) and nsym != 1) or (hasattr(t, '__len__') and len(t) != nsym):
            raise YastnError('len(t) does not match the number of symmetry charges.')
        t = a.config.sym.add_charges(t, signatures=(s,), new_signature=s)

    newn = a.config.sym.add_charges(a.struct.n, t, signatures=(1, s))
    legs = a.struct.legs[:haxis] + (LegBasic(s=s, t=(t,), D=(1,)),) + a.struct.legs[haxis:]
    struct = a.struct.replace(legs=legs, n=newn)
    hfs = a.hfs[:haxis] + (hfsa,) + a.hfs[haxis:]
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, trans=trans)


def remove_leg(a, axis=-1) -> 'Tensor':
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
    mfs = a.mfs[:axis] + a.mfs[axis + 1:]
    remove = a.mfs[axis][0]
    uaxis = sum(a.mfs[ii][0] for ii in range(axis))  # unpack mfs

    trans = list(a.trans)

    for _ in range(remove):
        haxis = trans[uaxis]
        trans = trans[:uaxis] + trans[uaxis + 1:]
        for k, v in enumerate(trans):
            if v > haxis:
                trans[k] = v - 1

        leg = a.struct.legs[haxis]
        if len(leg.t) > 1 or (leg.D and leg.D[0] != 1):
            raise YastnError('Axis to be removed must have single charge of dimension one.')
        t = leg.t[0] if leg.t else a.config.sym.zero()

        new_n = a.config.sym.add_charges(a.struct.n, t, signatures=(-1, a.struct.legs[haxis].s), new_signature=-1)
        new_legs = a.struct.legs[:haxis] + a.struct.legs[haxis + 1:]
        struct = a.struct.replace(legs=new_legs, n=new_n)
        hfs = a.hfs[:haxis] + a.hfs[haxis + 1:]
        a = a._replace(mfs=mfs, hfs=hfs, struct=struct, trans=trans)
    return a


def diag(a) -> 'Tensor':
    """
    Select diagonal of 2D tensor and output it as a diagonal tensor, or vice versa.
    """
    bl = get_blocks(a.config.sym, a.struct)
    if not a.isdiag:  # isdiag=False -> isdiag=True
        if a.ndim_n != 2 or a.struct.legs[0].s == a.struct.legs[1].s:
            raise YastnError('Diagonal tensor requires 2 legs with opposite signatures.')
        if a.n != a.config.sym.zero():
            raise YastnError('Diagonal tensor requires zero tensor charge.')
        if any(mf != (1,) for mf in a.mfs) or any(hf.tree != (1,) for hf in a.hfs):
            raise YastnError('Diagonal tensor cannot have fused legs.')
        if np.any(bl.D[:, 0] != bl.D[:, 1]):
            raise YastnError('yastn.diag() allowed only for square blocks.')
        #     isdiag=True -> isdiag=False                    isdiag=False -> isdiag=True
    #
    new_legs = a.struct.legs
    if a.trans == (1, 0):  # sufficient for the transpose, to have consistent signature flow
        new_legs == new_legs[::-1]
    #
    struct_new = a.struct._replace(isdiag=not a.isdiag)
    bl_new = get_blocks(a.config.sym, struct_new)

    if a.isdiag:  # isdiag=True -> isdiag=False
        meta = np.hstack([bl_new.slc, bl.slc])
        meta_dt = np.dtype([
                ('sln', np.int64, (2,)),
                ('slo',  np.int64, (2,))])
        meta = meta.view(meta_dt).reshape(-1)
        data = a.config.backend.diag_1dto2d(a._data, meta, bl_new.size)
    else:  # isdiag=False -> isdiag=True
        meta = np.hstack([bl_new.slc, bl.slc, bl.D])
        meta_dt = np.dtype([
                ('sln', np.int64, (2,)),
                ('slo',  np.int64, (2,)),
                ('Do',  np.int64, (2,))])
        meta = meta.view(meta_dt).reshape(-1)
        data = a.config.backend.diag_2dto1d(a._data, meta, bl_new.size)
    return a._replace(struct=struct_new, data=data, trans=None)
