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
"""Support of torch as a data structure used by yastn."""
from functools import lru_cache
from typing import Sequence, Union
import numpy as np
import torch
import tapp_torch
from .._cache_registry import register_cache
from .._profile import nsys_profile
from .backend_torch import *
from ._backend_torch_backwards import _project_grad_dtype


BACKEND_ID = "torch_cutensor"

from tapp_torch.ops import descriptor_cache_stats, plan_cache_stats


_PACK_CHUNK = 1 << 20


@register_cache(key='cutensor_nonabelian_packing')
@lru_cache(maxsize=128)
def _packing_parameters(slices, device):
    """Cache O(blocks) device metadata; never retain element-sized index maps."""
    lengths = tuple(stop - start for start, stop in slices)
    offset = 0
    shifts = []
    for (start, _), length in zip(slices, lengths):
        shifts.append(start - offset)
        offset += length
    return (torch.tensor(lengths, dtype=torch.int64, device=device).cumsum(0),
            torch.tensor(shifts, dtype=torch.int64, device=device), offset,
            all(shift == 0 for shift in shifts))


def _packing_indices(parameters, lo, hi):
    ends, shifts, _, _ = parameters
    positions = torch.arange(lo, hi, device=shifts.device)
    blocks = torch.searchsorted(ends, positions, right=True)
    return positions + shifts[blocks], blocks


@register_cache(key='cutensor_nonabelian_coefficients')
@lru_cache(maxsize=128)
def _coefficients(values, dtype, device):
    return torch.tensor(values, dtype=dtype, device=device)


class _Pack(torch.autograd.Function):
    """Recompute bounded gather maps in backward instead of saving full maps."""

    @staticmethod
    def forward(data, parameters, weights, dtype):
        size = parameters[2]
        result = torch.empty(size, dtype=dtype, device=data.device)
        for lo in range(0, size, _PACK_CHUNK):
            hi = min(lo + _PACK_CHUNK, size)
            indices, blocks = _packing_indices(parameters, lo, hi)
            values = data[indices].to(dtype=dtype)
            if weights is not None:
                values.mul_(weights[blocks])
            result[lo:hi] = values
            del indices, blocks, values
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        data, ctx.parameters, ctx.weights, _ = inputs
        ctx.input_size, ctx.input_dtype = data.numel(), data.dtype

    @staticmethod
    def backward(ctx, gradient):
        # Sum repeated slices at the promoted precision before projecting back.
        dtype = gradient.dtype if ctx.input_dtype.is_complex else gradient.real.dtype
        result = torch.zeros(ctx.input_size, dtype=dtype, device=gradient.device)
        for lo in range(0, gradient.numel(), _PACK_CHUNK):
            hi = min(lo + _PACK_CHUNK, gradient.numel())
            indices, blocks = _packing_indices(ctx.parameters, lo, hi)
            values = gradient[lo:hi]
            if ctx.weights is not None:
                values = values * ctx.weights[blocks]
            result.index_add_(0, indices, _project_grad_dtype(values, dtype))
            del indices, blocks, values
        return _project_grad_dtype(result, ctx.input_dtype), None, None, None


class _Unpack(torch.autograd.Function):
    @staticmethod
    def forward(data, parameters, size):
        result = data.new_zeros(size)
        for lo in range(0, data.numel(), _PACK_CHUNK):
            hi = min(lo + _PACK_CHUNK, data.numel())
            indices, blocks = _packing_indices(parameters, lo, hi)
            result.index_copy_(0, indices, data[lo:hi])
            del indices, blocks
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.parameters, _ = inputs

    @staticmethod
    def backward(ctx, gradient):
        return _Pack.apply(gradient, ctx.parameters, None, gradient.dtype), None, None


def _pack(data, slices, coefficients=(), dtype=None):
    dtype = data.dtype if dtype is None else dtype
    parameters = _packing_parameters(slices, data.device)
    weights = None
    if coefficients and any(value != 1 for value in coefficients):
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        weights = _coefficients(coefficients, real_dtype, data.device)
    if parameters[3] and weights is None:
        return data[:parameters[2]].to(dtype=dtype)
    return _Pack.apply(data, parameters, weights, dtype)


@nsys_profile
def tensordot_nonabelian(Adata, Bdata, plan):
    """One cuTENSOR call for all weighted products, with differentiable packing.

    Repeated source blocks are copied to disjoint packed slices: TAPP's backward
    writes one gradient per descriptor block, then gather's backward sums the
    contributions to the original data. Aliased descriptor offsets would lose
    those contributions. Packing uses reduced degeneracy blocks only.
    """
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    if not plan.coefficients:
        # Keep a zero derivative for both inputs, including empty tensors.
        return (Adata.new_zeros(plan.size, dtype=dtype)
                + Adata[:0].sum() + Bdata[:0].sum())
    packed_a = _pack(Adata, plan.slices_a, plan.coefficients, dtype)
    packed_b = _pack(Bdata, plan.slices_b, dtype=dtype)
    descriptors = [arg for descriptor in plan.descriptors for arg in descriptor]
    packed_c = tensordot_bs_v2(packed_a, packed_b, [1], [0],
                              *descriptors, [0, 1], list(plan.hashes))
    del packed_a, packed_b  # In inference, release packing before output embedding.
    parameters = _packing_parameters(plan.slices_c, Adata.device)
    _, _, size, contiguous = parameters
    if contiguous and size == plan.size:
        return packed_c
    # Some symmetry-allowed channels have no nonzero gluing contribution.
    # Do not expose uninitialized output gaps from the external allocator.
    return _Unpack.apply(packed_c, parameters, plan.size)


def tensordot_bs(Adata, Bdata, *args):
    dtype = torch.promote_types(Bdata.dtype, Adata.dtype)
    if Adata.dtype != dtype:
        Adata = Adata.to(dtype=dtype)
    if Bdata.dtype != dtype:
        Bdata = Bdata.to(dtype=dtype)
    return torch.ops.tapp_torch.tensordot_bs(Adata, Bdata, *args)

@nsys_profile
def tensordot_dense(Adata, Bdata, DA, DB, nin_a, nin_b, modes_out):
    dtype = torch.promote_types(Bdata.dtype, Adata.dtype)
    if Adata.dtype != dtype:
        Adata = Adata.to(dtype=dtype)
    if Bdata.dtype != dtype:
        Bdata = Bdata.to(dtype=dtype)
    Adata = Adata.reshape(DA)
    Bdata = Bdata.reshape(DB)
    return torch.ops.tapp_torch.tensordot(Adata, Bdata, nin_a, nin_b, modes_out).reshape(-1)


@nsys_profile
def tensordot_bs_v2(
    a: torch.Tensor, b: torch.Tensor,
    nin_a: Sequence[int], nin_b: Sequence[int],
    a_numSectionsPerMode : Sequence[int], 
    a_sectionExtents : Sequence[int], # flattened list of section extents for each mode of a
    a_coords : Union[torch.Tensor, np.ndarray], # 1D int64, flattened non-zero block coordinates
    a_strides: Union[torch.Tensor, np.ndarray], # 1D int64, flattened strides for each non-zero block
    a_offsets: Union[torch.Tensor, np.ndarray], # int64
    b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets,
    c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets,
    modes_out: Sequence[int], hashes_int64: Sequence[int]):

    # from_numpy is no-copy
    # contiguous is no-op if the tensor is already contiguous
    def _as_tensor(v):
        return torch.from_numpy(v).contiguous() if isinstance(v, np.ndarray) else v

    a_coords, a_strides, a_offsets = _as_tensor(a_coords), _as_tensor(a_strides), _as_tensor(a_offsets)
    b_coords, b_strides, b_offsets = _as_tensor(b_coords), _as_tensor(b_strides), _as_tensor(b_offsets)
    c_coords, c_strides, c_offsets = _as_tensor(c_coords), _as_tensor(c_strides), _as_tensor(c_offsets)

    dtype = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=dtype)
    b = b.to(dtype=dtype)

    # Signature 
    #
    # A: Tensor, B: Tensor,
    # contracted_modes_A: List[int], contracted_modes_B: List[int],
    # a_numSectionsPerMode: List[int], a_sectionExtents: List[int],
    # a_blocks: Tensor, a_strides: Tensor, a_offsets: Tensor,
    # b_numSectionsPerMode: List[int], b_sectionExtents: List[int],
    # b_blocks: Tensor, b_strides: Tensor, b_offsets: Tensor,
    # c_numSectionsPerMode: List[int], c_sectionExtents: List[int],
    # c_blocks: Tensor, c_strides: Tensor, c_offsets: Tensor,
    # modes_out: Optional[List[int]] = None, key_flags: int = 0
    #
    # with all but A,B Tensor arguments being CPU Tensors of int64
    #
    res = torch.ops.tapp_torch.tensordot_bs_v2(a, b, nin_a, nin_b,
        a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets,
        b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets,
        c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets,
        modes_out, hashes_int64)
    return res
