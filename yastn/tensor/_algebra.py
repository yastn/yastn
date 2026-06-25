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

import numpy as np

from ._auxiliary import _slc, get_blocks, test_all_blocks, update_old_struct
from ._legs import legs_union
from ._merging import _embed_tensor
from ._tests import YastnError, _test_can_be_combined, _unpack_trans_test_axes_pair

__all__ = ['add', 'real', 'imag', 'sqrt', 'rsqrt', 'reciprocal', 'exp', 'bitwise_not', 'allclose']


def __add__(a, b) -> 'Tensor':
    """
    Add two tensors, use: :math:`a + b`.

    Signatures and total charges of two tensors should match.
    """
    (a, b), hfs = _pre_addition(a, b)
    metas, st_new = _meta_addition(a.config.sym, (a.legs, b.legs), a.n, a.isdiag)
    struct, slices = update_old_struct(a.struct, st_new)
    data = a.config.backend.add((a._data, b._data), metas, st_new.size)
    out = a._replace(hfs=hfs, struct=struct, slices=slices, data=data)
    return out

def __sub__(a, b) -> 'Tensor':
    """
    Subtract two tensors, use: :math:`a - b`.

    Signatures and total charges of two tensors should match.
    """
    (a, b), hfs = _pre_addition(a, b)
    metas, st_new = _meta_addition(a.config.sym, (a.legs, b.legs), a.n, a.isdiag)
    struct, slices = update_old_struct(a.struct, st_new)
    data = a.config.backend.sub(a._data, b._data, metas, st_new.size)
    out = a._replace(hfs=hfs, struct=struct, slices=slices, data=data)
    return out

def add(*tensors, amplitudes=None, **kwargs):
    r"""
    Linear combination of tensors with given amplitudes, :math:`\sum_i amplitudes[i] tensors[i]`.

    Parameters
    ----------
    tensors: Sequence[yastn.Tensor]
        Signatures and total charges of all tensors should match.

    amplitudes: None | Sequence[Number]
        If ``None``, all amplitudes are assumed to be one.
        Otherwise, the number of tensors and amplitudes should be the same.
        Individual amplitude can be ``None``, which gives the same result as ``1``
        but without an extra multiplication.
    """
    if amplitudes is not None:
        if len(tensors) != len(amplitudes):
            raise YastnError("Number of tensors and amplitudes do not match.")
        tensors = [v * amp if amp is not None else v for v, amp in zip(tensors, amplitudes)]

    if len(tensors) == 1:
        return tensors[0]

    tensors, hfs = _pre_addition(*tensors)
    legss = tuple(a.legs for a in tensors)
    a = tensors[0]
    metas, st_new = _meta_addition(a.config.sym, legss, a.n, a.isdiag)
    datas = [v._data for v in tensors]
    data = a.config.backend.add(datas, metas, st_new.size)
    struct, slices = update_old_struct(a.struct, st_new)
    out = a._replace(hfs=hfs, struct=struct, slices=slices, data=data)
    return out

def _pre_addition(*tensors):
    """
    Test and prepare tensors before addition.
    """
    for ten in tensors[1:]:
        _test_can_be_combined(tensors[0], ten)

    if len(set(ten.trans for ten in tensors)) > 1:  # if ten.trans differ
        tensors = [ten.consume_transpose() for ten in tensors]

    mask_needed = False
    a = tensors[0]
    for b in tensors[1:]:
        if a.struct.n != b.struct.n:
            raise YastnError('Tensor charges do not match.')
        if a.isdiag != b.isdiag:
            raise YastnError('Cannot add diagonal tensor to non-diagonal tensor.')
        mask_needed_ab, _ = _unpack_trans_test_axes_pair(a, b, sgn=1)
        mask_needed = mask_needed or mask_needed_ab

    if mask_needed:
        legss = [tensor.get_legs(native=True) for tensor in tensors]
        ulegs = {n: legs_union(*(legs[n] for legs in legss)) for n in range(a.ndim_n)}
        hfs = tuple(ulegs[n].hf for n in range(a.ndim_n))
        tensors = [_embed_tensor(tensor, legs, ulegs) for tensor, legs in zip(tensors, legss)]
    else:
        hfs = a.hfs

    return tensors, hfs


@lru_cache(maxsize=1024)
def _meta_addition(sym, legss, a_n, isdiag):
    """ meta-information for backend and new tensor charges and dimensions. """

    if all(legss[0] == legs for legs in legss[1:]):
        st_new = get_blocks(sym, legss[0], a_n, isdiag)
        Dsize = st_new.size
        meta = (((0, Dsize), (0, Dsize)),)
        metas = (meta, ) * len(legss)
        return metas, st_new

    ndim = len(legss[0])  # nlegs
    legs_new = []
    for n in range(ndim):
        leg = legss[0][n]
        for legs in legss[1:]:
            try:
                leg = leg.union(legs[n], isdiag=isdiag)
            except ValueError:
                raise YastnError('Bond dimensions of some charges do not match.')
        legs_new.append(leg)
    legs_new = tuple(legs_new)
    st_new = get_blocks(sym, legs_new, a_n, isdiag)

    metas = []
    for legs in legss:
        meta = []
        st_old = get_blocks(sym, legs, a_n, isdiag)
        meta, i, j, sn0, so0 = [], 0, 0, None, None
        if not isdiag:
            while i < st_old.nblocks and j < st_new.nblocks:
                if np.array_equal(st_old.t[i], st_new.t[j]):
                    if so0 is None:
                        sn0 = st_new.slc[j, 0]
                        so0 = st_old.slc[i, 0]
                    i += 1
                    j += 1
                else:
                    if so0 is not None:
                        sn1 = st_new.slc[j - 1, 1]
                        so1 = st_old.slc[i - 1, 1]
                        meta.append(((sn0, sn1), (so0, so1)))
                    sn0, so0 = None, None
                    j += 1
            if so0 is not None:
                sn1 = st_new.slc[j - 1, 1]
                so1 = st_old.slc[i - 1, 1]
                meta.append(((sn0, sn1), (so0, so1)))
        else:  # if isdiag: embed smaller dimensions into larger ones
            while i < st_old.nblocks and j < st_new.nblocks:
                if np.array_equal(st_old.t[i], st_new.t[j]):
                    sn0 = st_new.slc[j, 0]
                    so0 = st_old.slc[i, 0]
                    d = min(st_new.slc[j, 1] - st_new.slc[j, 0], st_old.slc[i, 1] - st_old.slc[i, 0])
                    meta.append(((sn0, sn0 + d), (so0, so0 + d)))
                    i += 1
                    j += 1
                else:
                    j += 1
        metas.append(tuple(meta))

    return tuple(metas), st_new


def allclose(a, b, rtol=1e-13, atol=1e-13) -> bool:
    """
    Check if `a` and `b` are identical within a desired tolerance.
    To be :code:`True`, all tensors' blocks and merge history have to be identical.
    If this condition is satisfied, execute :code:`backend.allclose` function
    to compare tensors’ data.

    Note that if two tensors differ by zero blocks, the function returns :code:`False`.
    To resolve such differences, use :code:`(a - b).norm() < tol`

    Parameters
    ----------
    a, b: yastn.Tensor
        Tensor for comparison.

    rtol, atol: float
        Desired relative and absolute precision.
    """
    if a.legs != b.legs or a.hfs != b.hfs or a.mfs != b.mfs or a.trans != b.trans:
        return False
    return a.config.backend.allclose(a._data, b._data, rtol, atol)


def __lt__(a, number) -> 'Tensor[bool]':
    """
    Logical tensor with elements less-than a number (if it makes sense for backend data tensors),
    use: `mask = tensor < number`

    Intended for diagonal tensor to be applied as a truncation mask.
    """
    data = a._data < number
    return a._replace(data=data)


def __gt__(a, number) -> 'Tensor[bool]':
    """
    Logical tensor with elements greater-than a number (if it makes sense for backend data tensors),
    use: `mask = tensor > number`

    Intended for diagonal tensor to be applied as a truncation mask.
    """
    data = a._data > number
    return a._replace(data=data)


def __le__(a, number) -> 'Tensor[bool]':
    """
    Logical tensor with elements less-than-or-equal-to a number (if it makes sense for backend data tensors),
    use: `mask = tensor <= number`

    Intended for diagonal tensor to be applied as a truncation mask.
    """
    data = a._data <= number
    return a._replace(data=data)


def __ge__(a, number) -> 'Tensor[bool]':
    """
    Logical tensor with elements greater-than-or-equal-to a number (if it makes sense for backend data tensors),
    use: `mask = tensor >= number`

    Intended for diagonal tensor to be applied as a truncation mask.
    """
    data = a._data >= number
    return a._replace(data=data)


def __mul__(a, number) -> 'Tensor':
    """ Multiply tensor by a number, use: `number * tensor`. """
    data = a._data * number
    if a.config.backend.get_size(data) != a.struct.size:
        raise YastnError("Multiplication cannot change data size; broadcasting not supported.")
    return a._replace(data=data)


def __rmul__(a, number) -> 'Tensor':
    """ Multiply tensor by a number, use: `tensor * number`. """
    return a.__mul__(number)


def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """ This is to circumvent problems with `np.float64 * Mps`. """
    if ufunc.__name__ == 'multiply':
        lhs, rhs = inputs
        return rhs.__mul__(lhs)
    raise YastnError(f"Only np.float * Mps is supported; {ufunc.__name__} was called.")


def __neg__(a) -> 'Tensor':
    """ Multiply tensor by -1, use: `-tensor`. """
    return a.__mul__(-1)


def __pow__(a, exponent) -> 'Tensor':
    """ Element-wise exponent of tensor, use: `tensor ** exponent`. """
    data = a._data ** exponent
    if a.config.backend.get_size(data) != a.struct.size:
        raise YastnError("Exponent cannot change data size; broadcasting not supported.")
    return a._replace(data=data)


def __truediv__(a, number) -> 'Tensor':
    """ Divide tensor by a scalar, use: `tensor / number`. """
    data = a._data / number
    if a.config.backend.get_size(data) != a.struct.size:
        raise YastnError("truediv cannot change data size; broadcasting not supported.")

    return a._replace(data=data)


def __abs__(a) -> 'Tensor':
    r"""
    Return tensor with element-wise absolute values.

    Can be called on tensor as ``abs(tensor)``.
    """
    data = a.config.backend.absolute(a._data)
    return a._replace(data=data)


def real(a) -> 'Tensor':
    r"""
    Return tensor with imaginary part set to zero.

    .. note::
        Follows the behavior of the :meth:`backend.real()`
        when it comes to creating a new copy of the data or handling datatype :code:`dtype`.
    """
    data = a.config.backend.real(a._data)
    return a._replace(data=data)


def imag(a) -> 'Tensor':
    r"""
    Return tensor with real part set to zero.

    .. note::
        Follows the behavior of the :meth:`backend.imag()`
        when it comes to creating a new copy of the data or handling datatype :code:`dtype`.
    """
    data = a.config.backend.imag(a._data)
    return a._replace(data=data)


def sqrt(a) -> 'Tensor':
    """ Return tensor after applying element-wise square root for each tensor element. """
    data = a.config.backend.sqrt(a._data)
    return a._replace(data=data)


def rsqrt(a, cutoff=0) -> 'Tensor':
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


def reciprocal(a, cutoff=0) -> 'Tensor':
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


def exp(a, step=1.) -> 'Tensor':
    r"""
    Return element-wise `\exp(step * tensor)`.

    .. note::
        This applies only to non-empty blocks of tensor
    """
    data = a.config.backend.exp(a._data, step)
    return a._replace(data=data)


def bitwise_not(a) -> 'Tensor[bool]':
    r"""
    Return tensor after applying bitwise not on each tensor element.

    .. note::
        Operation applies only to non-empty blocks of tensor with tensor data dtype
        that allows for bitwise operation, i.e. intended for
        masks used to truncate tensor legs.
    """
    data = a.config.backend.bitwise_not(a._data)
    return a._replace(data=data)
