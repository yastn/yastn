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
from functools import lru_cache
from itertools import accumulate
from ._auxliary import _slc, _join_contiguous_slices
from ._tests import YastnError, _test_can_be_combined, _get_tD_legs, _test_axes_match
from ._merging import _mask_tensors_leg_union, _meta_mask, _mask_tensors_leg_union_new


__all__ = ['linear_combination', 'real', 'imag', 'sqrt', 'rsqrt', 'reciprocal', 'exp', 'bitwise_not', 'allclose']


def __add__(a, b) -> yastn.Tensor:
    """
    Add two tensors, use: :math:`a + b`.

    Signatures and total charges of two tensors should match.
    """
    (a, b), hfs = _pre_addition(a, b)
    metas, struct, slices = _meta_addition(((a.struct, a.slices), (b.struct, b.slices)), a.isdiag)
    data = a.config.backend.add((a._data, b._data), metas, struct.size)
    return a._replace(hfs=hfs, struct=struct, slices=slices, data=data)


def __sub__(a, b) -> yastn.Tensor:
    """
    Subtract two tensors, use: :math:`a - b`.

    Signatures and total charges of two tensors should match.
    """
    (a, b), hfs = _pre_addition(a, b)
    meta, struct, slices = _meta_addition(((a.struct, a.slices), (b.struct, b.slices)), a.isdiag)
    data = a.config.backend.sub(a._data, b._data, meta, struct.size)
    return a._replace(hfs=hfs, struct=struct, slices=slices, data=data)


def linear_combination(*tensors, amplitudes=None, **kwargs):
    r"""
    Linear combination of tensors with given amplitudes, :math:`\sum_i amplitudes[i] tensors[i]`.

    Parameters
    ----------
    tensors: Sequence[yastn.Tensor]
        Signatures and total charges of all tensors should match.

    amplitudes: None | Sequence[Number]
        If ``None``, all amplitudes are assumed to be one.
        Otherwise, The number of tensors and amplitudes should be the same.
        Individual amplitude can be ``None``, which gives the same result as ``1``
        but without an extra multiplication.
    """
    if amplitudes is not None:
        if len(tensors) != len(amplitudes):
            raise YastnError("Number of tensors and amplitudes do not match.")
        tensors = [v * amp if amp is not None else v for v, amp in zip(tensors, amplitudes)]

    tensors, hfs = _pre_addition(*tensors)
    datas = tuple((a.struct, a.slices) for a in tensors)
    a = tensors[0]
    metas, struct, slices = _meta_addition(datas, a.isdiag)
    datas = [v._data for v in tensors]
    data = a.config.backend.add(datas, metas, struct.size)
    return a._replace(hfs=hfs, struct=struct, slices=slices, data=data)


def _pre_addition(*tensors):
    """
    Test and prepare tensors before addition.
    """
    mask_needed = False
    a = tensors[0]
    for b in tensors[1:]:
        _test_can_be_combined(a, b)
        if a.struct.n != b.struct.n:
            raise YastnError('Tensor charges do not match.')
        if a.isdiag != b.isdiag:
            raise YastnError('Cannot add diagonal tensor to non-diagonal tensor.')
        mask_needed_ab, _ = _test_axes_match(a, b, sgn=1)
        mask_needed = mask_needed or mask_needed_ab

    if mask_needed:
        masks, masks_tD, hfs = _mask_tensors_leg_union_new(*tensors)
        nin = tuple(range(a.ndim_n))
        tensors = tuple(_embed_mask_axes(b, nin, mask, mask_tD) for b, mask, mask_tD in zip(tensors, masks, masks_tD))
    else:
        hfs = a.hfs

    return tensors, hfs


def _embed_mask_axes(a, naxes, masks, masks_tD):
    r""" Auxlliary function applying mask tensors to native legs. """
    for axis, mask, mask_tD in zip(naxes, masks, masks_tD):
        if mask is not None:
            mask_t = tuple(mask_tD.keys())
            mask_D = tuple(mask_tD.values())
            meta, struct, slices, axis, ndim = _meta_mask(a.struct, a.slices, a.isdiag, mask_t, mask_D, axis)
            data = a.config.backend.embed_mask(a._data, mask, meta, struct.size, axis, ndim)
            a = a._replace(struct=struct, slices=slices, data=data)
    return a


@lru_cache(maxsize=1024)
def _meta_addition(datas, isdiag):
    """ meta-information for backend and new tensor charges and dimensions. """

    if all(datas[0] == data for data in datas[1:]):
        Dsize = datas[0][0].size
        meta = (((0, Dsize), (0, Dsize)),)
        metas = tuple(meta for _ in range(len(datas)))
        return metas, *datas[0]

    check_structure = False
    struct_0, slices_0 = datas[0]
    temp = list((t, D, sl.Dp) for t, D, sl in zip(struct_0.t, struct_0.D, slices_0))

    for struct_1, slices_1 in datas[1:]:
        temp0 = temp
        temp1 = list((t, D, sl.Dp) for t, D, sl in zip(struct_1.t, struct_1.D, slices_1))
        i0, i1, temp = 0, 0, []
        while i0 < len(temp0) and i1 < len(temp1):
            t0, D0, Dp0 = temp0[i0]
            t1, D1, Dp1 = temp1[i1]
            if t0 == t1:
                if isdiag:
                    temp.append((t0, max(D0, D1), max(Dp0, Dp1)))
                else:
                    if D0 != D1:
                        raise YastnError('Bond dimensions do not match.')
                    temp.append((t0, D0, Dp0))
                i0 += 1
                i1 += 1
            elif t0 < t1:
                temp.append((t0, D0, Dp0))
                i0 += 1
                check_structure = True
            else:
                temp.append((t1, D1, Dp1))
                i1 += 1
                check_structure = True
        if i0 < len(temp0) or i1 < len(temp1):
            check_structure = True
        for t0, D0, Dp0 in temp0[i0:]:
            temp.append((t0, D0, Dp0))
        for t1, D1, Dp1 in temp1[i1:]:
            temp.append((t1, D1, Dp1))

    t_new, D_new, Dp_new = zip(*temp) if len(temp) > 0 else ((), (), ())
    slices_new = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(Dp_new), Dp_new, D_new))
    struct_new = struct_0._replace(t=t_new, D=D_new, size=sum(Dp_new))

    if check_structure:
        _get_tD_legs(struct_new)

    if isdiag:
        def update_meta(mtn, mto, slcn, slco):
            start = slcn.slcs[0][0]
            stop = start + slco.Dp
            mtn.append((start, stop))
            mto.append(slco.slcs[0])
    else:
        def update_meta(mtn, mto, slcn, slco):
            mtn.append(slcn.slcs[0])
            mto.append(slco.slcs[0])

    metas = []
    for struct, slices in datas:
        itn, islcn = iter(t_new), iter(slices_new)
        mtn, mto = [], []
        for to, slco in zip(struct.t, slices):
            tn, slcn = next(itn), next(islcn)
            while to != tn:
                tn, slcn = next(itn), next(islcn)
            update_meta(mtn, mto, slcn, slco)
        metas.append(_join_contiguous_slices(mtn, mto))

    return tuple(metas), struct_new, slices_new


def allclose(a, b, rtol=1e-13, atol=1e-13) -> bool:
    """
    Check if `a` and `b` are identical within a desired tolerance.
    To be :code:`True`, all tensors' blocks and merge history have to be identical.
    If this condition is satisfied, execute :code:`backend.allclose` function
    to compare tensors’ data.

    Note that if two tenors differ by zero blocks, the function returns :code:`False`.
    To resolve such differences, use :code:`(a - b).norm() < tol`

    Parameters
    ----------
    a, b: yastn.Tensor
        Tensor for comparison.

    rtol, atol: float
        Desired relative and absolute precision.
    """
    if a.struct != b.struct or a.slices != b.slices or a.hfs != b.hfs or a.mfs != b.mfs:
        return False
    return a.config.backend.allclose(a._data, b._data, rtol, atol)


def __lt__(a, number) -> yastn.Tensor[bool]:
    """
    Logical tensor with elements less-than a number (if it makes sense for backend data tensors),
    use: `mask = tensor < number`

    Intended for diagonal tensor to be applied as a truncation mask.
    """
    data = a._data < number
    return a._replace(data=data)


def __gt__(a, number) -> yastn.Tensor[bool]:
    """
    Logical tensor with elements greater-than a number (if it makes sense for backend data tensors),
    use: `mask = tensor > number`

    Intended for diagonal tensor to be applied as a truncation mask.
    """
    data = a._data > number
    return a._replace(data=data)


def __le__(a, number) -> yastn.Tensor[bool]:
    """
    Logical tensor with elements less-than-or-equal-to a number (if it makes sense for backend data tensors),
    use: `mask = tensor <= number`

    Intended for diagonal tensor to be applied as a truncation mask.
    """
    data = a._data <= number
    return a._replace(data=data)


def __ge__(a, number) -> yastn.Tensor[bool]:
    """
    Logical tensor with elements greater-than-or-equal-to a number (if it makes sense for backend data tensors),
    use: `mask = tensor >= number`

    Intended for diagonal tensor to be applied as a truncation mask.
    """
    data = a._data >= number
    return a._replace(data=data)


def __mul__(a, number) -> yastn.Tensor:
    """ Multiply tensor by a number, use: `number * tensor`. """
    data = a._data * number
    if a.config.backend.get_size(data) != a.struct.size:
        raise YastnError("Multiplication cannot change data size; broadcasting not supported.")
    return a._replace(data=data)


def __rmul__(a, number) -> yastn.Tensor:
    """ Multiply tensor by a number, use: `tensor * number`. """
    return a.__mul__(number)


def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """ This is to circumvent problems with `np.float64 * Mps`. """
    if ufunc.__name__ == 'multiply':
        lhs, rhs = inputs
        return rhs.__mul__(lhs)
    raise YastnError(f"Only np.float * Mps is supported; {ufunc.__name__} was called.")


def __neg__(a) -> yastn.Tensor:
    """ Multiply tensor by -1, use: `-tensor`. """
    return a.__mul__(-1)


def __pow__(a, exponent) -> yastn.Tensor:
    """ Element-wise exponent of tensor, use: `tensor ** exponent`. """
    data = a._data ** exponent
    if a.config.backend.get_size(data) != a.struct.size:
        raise YastnError("Exponent cannot change data size; broadcasting not supported.")
    return a._replace(data=data)


def __truediv__(a, number) -> yastn.Tensor:
    """ Divide tensor by a scalar, use: `tensor / number`. """
    data = a._data / number
    if a.config.backend.get_size(data) != a.struct.size:
        raise YastnError("truediv cannot change data size; broadcasting not supported.")

    return a._replace(data=data)


def __abs__(a) -> yastn.Tensor:
    r"""
    Return tensor with element-wise absolute values.

    Can be on called on tensor as ``abs(tensor)``.
    """
    data = a.config.backend.absolute(a._data)
    return a._replace(data=data)


def real(a) -> yastn.Tensor:
    r"""
    Return tensor with imaginary part set to zero.

    .. note::
        Follows the behavior of the :meth:`backend.real()`
        when it comes to creating a new copy of the data or handling datatype :code:`dtype`.
    """
    data = a.config.backend.real(a._data)
    return a._replace(data=data)


def imag(a) -> yastn.Tensor:
    r"""
    Return tensor with real part set to zero.

    .. note::
        Follows the behavior of the :meth:`backend.imag()`
        when it comes to creating a new copy of the data or handling datatype :code:`dtype`.
    """
    data = a.config.backend.imag(a._data)
    return a._replace(data=data)


def sqrt(a) -> yastn.Tensor:
    """ Return tensor after applying element-wise square root for each tensor element. """
    data = a.config.backend.sqrt(a._data)
    return a._replace(data=data)


def rsqrt(a, cutoff=0) -> yastn.Tensor:
    """
    Return element-wise operation `1/sqrt(tensor)`.

    The tensor elements with absolute value below the cutoff are set to zero.

    Parameters
    ----------
    cutoff: real scalar
        (element-wise) cutoff for inversion
    """
    data = a.config.backend.rsqrt(a._data, cutoff=cutoff)
    return a._replace(data=data)


def reciprocal(a, cutoff=0) -> yastn.Tensor:
    """
    Return element-wise operation `1/tensor`.

    The tensor elements with absolute value below the cutoff are set to zero.

    Parameters
    ----------
    cutoff: real scalar
        (element-wise) cutoff for inversion
    """
    data = a.config.backend.reciprocal(a._data, cutoff=cutoff)
    return a._replace(data=data)


def exp(a, step=1.) -> yastn.Tensor:
    r"""
    Return element-wise `\exp(step * tensor)`.

    .. note::
        This applies only to non-empty blocks of tensor
    """
    data = a.config.backend.exp(a._data, step)
    return a._replace(data=data)


def bitwise_not(a) -> yastn.Tensor[bool]:
    r"""
    Return tensor after applying bitwise not on each tensor element.

    .. note::
        Operation applies only to non-empty blocks of tensor with tensor data dtype
        that allows for bitwise operation, i.e. intended for
        masks used to truncate tensor legs.
    """
    data = a.config.backend.bitwise_not(a._data)
    return a._replace(data=data)
