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
from math import prod
import os
import warnings
from typing import Sequence, Union
import numpy as np
import torch
import tapp_torch
from .._cache_registry import register_cache
from .._profile import nsys_profile
from .._memory_override import memory_override
from .backend_torch import *
from ._backend_torch_backwards import (_project_grad_dtype, _save_contraction_inputs,
                                       kernel_transpose_dot_sum, _transpose_dot_gradient)


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
        return _pack_adjoint(gradient, ctx.parameters, ctx.weights,
                             ctx.input_size, ctx.input_dtype), None, None, None


def _pack_adjoint(gradient, parameters, weights, input_size, input_dtype, out=None):
    # Sum repeated slices at the promoted precision before projecting back.
    dtype = gradient.dtype if input_dtype.is_complex else gradient.real.dtype
    result = torch.zeros(input_size, dtype=dtype, device=gradient.device) if out is None else out
    for lo in range(0, gradient.numel(), _PACK_CHUNK):
        hi = min(lo + _PACK_CHUNK, gradient.numel())
        indices, blocks = _packing_indices(parameters, lo, hi)
        values = gradient[lo:hi]
        if weights is not None:
            values = values * weights[blocks]
        result.index_add_(0, indices, _project_grad_dtype(values, dtype))
        del indices, blocks, values
    return _project_grad_dtype(result, input_dtype)


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
        # TAPP descriptors address flat storage with unit stride. A gradient
        # view (or expanded scalar gradient) need not have that physical layout.
        return data[:parameters[2]].to(dtype=dtype).contiguous()
    return _Pack.apply(data, parameters, weights, dtype)


class _RecouplePack(torch.autograd.Function):
    """Write basis changes and gluing weights directly into disjoint TAPP blocks."""

    @staticmethod
    def forward(data, slices, shapes, recoupling, coefficients, dtype):
        from ._recoupling import composed_recoupling_terms
        size = sum(stop - start for start, stop in slices)
        result = torch.zeros(size, dtype=dtype, device=data.device)
        offset = 0
        for i, ((start, stop), shape) in enumerate(zip(slices, shapes)):
            target = result[offset:offset + stop - start].view(shape)
            weight = coefficients[i] if coefficients else 1.
            for source, old_shape, coefficient in composed_recoupling_terms(recoupling, start):
                values = data[source:source + prod(old_shape)].view(old_shape).permute(recoupling[0])
                target.add_(values, alpha=float(weight * coefficient))
            offset += stop - start
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        data, ctx.slices, ctx.shapes, ctx.recoupling, ctx.coefficients, _ = inputs
        ctx.input_size, ctx.input_dtype = data.numel(), data.dtype

    @staticmethod
    def backward(ctx, gradient):
        return _recouple_pack_adjoint(gradient, ctx.slices, ctx.shapes, ctx.recoupling,
                                      ctx.coefficients, ctx.input_size, ctx.input_dtype), \
            None, None, None, None, None


def _recouple_pack_adjoint(gradient, slices, shapes, recoupling, coefficients, input_size, input_dtype, out=None):
    from ._recoupling import composed_recoupling_terms
    # Repeated packed blocks contribute to the same original source block.
    # Accumulate at promoted precision, projecting to the input dtype last.
    dtype = gradient.dtype if input_dtype.is_complex else gradient.real.dtype
    result = torch.zeros(input_size, dtype=dtype, device=gradient.device) if out is None else out
    offset = 0
    for i, ((start, stop), shape) in enumerate(zip(slices, shapes)):
        values = _project_grad_dtype(gradient[offset:offset + stop - start].reshape(shape), dtype)
        weight = coefficients[i] if coefficients else 1.
        for source, old_shape, coefficient in composed_recoupling_terms(recoupling, start):
            target = result[source:source + prod(old_shape)].view(old_shape).permute(recoupling[0])
            target.add_(values, alpha=float(weight * coefficient))
        offset += stop - start
    return _project_grad_dtype(result, input_dtype)


def _pack_operand(data, plan, side, recoupling, dtype):
    slices, shapes = (plan.slices_a, plan.shapes_a) if side == 0 else (plan.slices_b, plan.shapes_b)
    coefficients = plan.coefficients if side == 0 else ()
    if recoupling is None:
        return _pack(data, slices, coefficients, dtype)
    return _RecouplePack.apply(data, slices, shapes, recoupling, coefficients, dtype)


def _packed_product(a, b, plan, order, axes):
    descriptors = [arg for i in order for arg in plan.descriptors[i]]
    hashes = [value for i in order for value in plan.hashes[8 * i:8 * (i + 1)]]
    return tensordot_bs_v2(a, b, [axes[0]], [axes[1]], *descriptors, [0, 1], hashes)


def _packed_elements(plan):
    return sum(stop - start for slices in (plan.slices_a, plan.slices_b, plan.slices_c)
               for start, stop in slices)


def _pack_budget_divisor():
    """Read ``YASTN_CUTENSOR_PACK_DIVISOR`` (default 8); invalid values fall back to 8."""
    raw = os.environ.get('YASTN_CUTENSOR_PACK_DIVISOR', '8')
    try:
        value = int(raw)
        if value < 1:
            raise ValueError
    except (TypeError, ValueError):
        warnings.warn(f"Ignoring YASTN_CUTENSOR_PACK_DIVISOR={raw!r}: expected a positive integer.")
        return 8
    return value


def _nonabelian_pack_budget(data, plan, dtype):
    """Bound live packing and leave room for cuTENSOR workspace and the output."""
    setting = os.environ.get('YASTN_CUTENSOR_PACK_CHUNK_BYTES', str(1 << 30))
    try:
        budget = int(setting)
        if budget <= 0:
            raise ValueError
    except ValueError:
        raise ValueError(f'YASTN_CUTENSOR_PACK_CHUNK_BYTES={setting!r}: expected a positive integer.') from None
    itemsize = torch.empty((), dtype=dtype).element_size()
    if data.is_cuda and not memory_override() and _packed_elements(plan) * itemsize > (64 << 20):
        free, _ = torch.cuda.mem_get_info(data.device)
        reusable = torch.cuda.memory_reserved(data.device) - torch.cuda.memory_allocated(data.device)
        available = max(0, free + reusable - plan.size * itemsize)
        budget = min(budget, max(itemsize, available // _pack_budget_divisor()))
    # Quantize to keep dynamic free-memory fluctuations from filling the cache.
    elements = max(1, int(budget) // itemsize)
    return 1 << (elements.bit_length() - 1)


@register_cache(key='cutensor_nonabelian_streamed')
@lru_cache(maxsize=128)
def _streamed_metadata(products):
    """Reuse Torch's bounded block kernel when even one packed product is too large."""
    offsets, size = {}, 0
    for sl in sorted({product[0] for product in products}):
        offsets[sl] = slice(size, size + sl[1] - sl[0])
        size += sl[1] - sl[0]
    meta, ra, rb, ia, ib = [], [], [], {}, {}
    for sl, (m, k, n), sa, da, sb, db, coefficient in products:
        ka, kb = (sa, da, m, k), (sb, db, k, n)
        if ka not in ia:
            ia[ka] = len(ra)
            ra.append((slice(*sa), da, m, k))
        if kb not in ib:
            ib[kb] = len(rb)
            rb.append((slice(*sb), db, k, n))
        meta.append((offsets[sl], (m, n), ia[ka], ib[kb], coefficient))
    oa, ob = tuple(range(len(ra[0][1]))), tuple(range(len(rb[0][1])))
    return tuple(meta), tuple(ra), tuple(rb), oa, ob, size


def _allocation_failure(error):
    message = str(error)
    # TAPP's stable ABI replaces the CUDA OOM exception with this dispatcher
    # error. Match only its allocation call, never arbitrary dispatcher errors.
    return (isinstance(error, torch.OutOfMemoryError) or 'CUDA out of memory' in message
            or ('torch_call_dispatcher( "aten::empty", "memory_format"' in message
                and 'API call failed' in message))


def _packed_forward(a, b, plan, recouplings, dtype):
    packed_a = _pack_operand(a, plan, 0, recouplings[0], dtype)
    packed_b = _pack_operand(b, plan, 1, recouplings[1], dtype)
    return _packed_product(packed_a, packed_b, plan, (0, 1, 2), (1, 0))


def _forward_batch(a, b, plan, recouplings, dtype, budget):
    if _packed_elements(plan) <= budget:
        try:
            return _packed_forward(a, b, plan, recouplings, dtype)
        except RuntimeError as error:
            if not _allocation_failure(error):
                raise
            warnings.warn('cuTENSOR allocation failed; retrying this batch with streamed Torch blocks. '
                          + str(error), RuntimeWarning)
        # Leave the exception scope to release its traceback and packed inputs.
    meta, ra, rb, oa, ob, size = _streamed_metadata(plan.products)
    return kernel_transpose_dot_sum.apply(a, b, meta, ra, rb, oa, ob, size, *recouplings)


def _accumulate_packed(result, packed, slices):
    parameters = _packing_parameters(slices, result.device)
    for lo in range(0, packed.numel(), _PACK_CHUNK):
        hi = min(lo + _PACK_CHUNK, packed.numel())
        indices, blocks = _packing_indices(parameters, lo, hi)
        result.index_add_(0, indices, packed[lo:hi])
        del indices, blocks


def _packed_backward(gradient, other, plan, side, recoupling, dtype):
    packed_c = _pack(gradient, plan.slices_c, dtype=dtype)
    packed_other = _pack_operand(other, plan, 1 - side, recoupling, dtype)
    if side == 0:
        return _packed_product(packed_c, packed_other.conj(), plan, (2, 1, 0), (1, 1))
    return _packed_product(packed_other.conj(), packed_c, plan, (0, 2, 1), (0, 0))


class _NonabelianContraction(torch.autograd.Function):
    """Retain only original operands, rebuilding packing one derivative at a time."""

    @staticmethod
    def forward(a, b, plan, recouplings, batches, budget):
        dtype = torch.promote_types(a.dtype, b.dtype)
        if len(batches) == 1:
            packed = _forward_batch(a, b, plan, recouplings, dtype, budget)
            parameters = _packing_parameters(plan.slices_c, a.device)
            if parameters[3] and parameters[2] == plan.size:
                return packed
            return _Unpack.apply(packed, parameters, plan.size)
        result = torch.zeros(plan.size, dtype=dtype, device=a.device)
        for batch in batches:
            packed = _forward_batch(a, b, batch, recouplings, dtype, budget)
            _accumulate_packed(result, packed, batch.slices_c)
            del packed
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b, ctx.plan, ctx.recouplings, ctx.batches, ctx.budget = inputs
        _save_contraction_inputs(ctx, a, b)

    @staticmethod
    def backward(ctx, gradient):
        inputs = ctx.saved_tensors
        dtype = ctx.promoted_dtype
        gradients = []
        for side in (0, 1):
            if not ctx.needs_input_grad[side]:
                gradients.append(None)
                continue
            size, target_dtype = ctx.input_specs[side]
            work_dtype = dtype if target_dtype.is_complex else gradient.real.dtype
            result = torch.zeros(size, dtype=work_dtype, device=gradient.device)
            for batch in ctx.batches:
                packed = None
                if _packed_elements(batch) <= ctx.budget:
                    try:
                        packed = _packed_backward(gradient, inputs[1 - side], batch, side,
                                                  ctx.recouplings[1 - side], dtype)
                    except RuntimeError as error:
                        if not _allocation_failure(error):
                            raise
                        warnings.warn('cuTENSOR gradient allocation failed; retrying with streamed Torch blocks. '
                                      + str(error), RuntimeWarning)
                if packed is None:
                    meta, ra, rb, oa, ob, _ = _streamed_metadata(batch.products)
                    if side == 1:
                        ra, rb, oa, ob = rb, ra, ob, oa
                    packed_c = _pack(gradient, batch.slices_c, dtype=dtype)
                    partial = _transpose_dot_gradient(
                        packed_c, inputs[1 - side], meta, ra, rb, oa, ob,
                        (size, work_dtype), dtype, side, ctx.recouplings[side], ctx.recouplings[1 - side])
                    result.add_(partial)
                    del packed_c, partial
                    continue
                slices, shapes = (batch.slices_a, batch.shapes_a) if side == 0 else (batch.slices_b, batch.shapes_b)
                coefficients = batch.coefficients if side == 0 else ()
                if ctx.recouplings[side] is None:
                    parameters = _packing_parameters(slices, gradient.device)
                    weights = (_coefficients(coefficients, packed.real.dtype, gradient.device)
                               if coefficients and any(value != 1 for value in coefficients) else None)
                    _pack_adjoint(packed, parameters, weights, size, work_dtype, out=result)
                else:
                    _recouple_pack_adjoint(packed, slices, shapes, ctx.recouplings[side],
                                            coefficients, size, work_dtype, out=result)
                del packed
            gradients.append(_project_grad_dtype(result, target_dtype))
            del result
        return *gradients, None, None, None, None


@nsys_profile
def tensordot_nonabelian(Adata, Bdata, plan, steps_a=(), steps_b=()):
    """Contract weighted products in batches with bounded packed storage.

    Repeated source blocks are copied to disjoint packed slices: each cuTENSOR
    gradient call writes one gradient per descriptor block, then packing's adjoint sums the
    contributions to the original data. Aliased descriptor offsets would lose
    those contributions. Optional basis changes write directly into packing,
    without full recoupled inputs or saved intermediate values. Return None when
    composition is too costly so the caller can use sequential recoupling.
    """
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    if not plan.coefficients:
        # Keep a zero derivative for both inputs, including empty tensors.
        return (Adata.new_zeros(plan.size, dtype=dtype)
                + Adata[:0].sum() + Bdata[:0].sum())
    from ._recoupling import recoupling_sequence_plan, unfuse_sequence_plan
    recouplings = []
    for steps, slices, shapes in ((steps_a, plan.slices_a, plan.shapes_a),
                                  (steps_b, plan.slices_b, plan.shapes_b)):
        if steps:
            outputs = tuple((start, shape) for (start, _), shape in zip(slices, shapes))
            if unfuse_sequence_plan(tuple(('recouple', order, meta) for order, meta in steps), outputs) is None:
                return None
            recouplings.append(recoupling_sequence_plan(steps))
        else:
            recouplings.append(None)
    from ..tensor._contractions_cutensor import _nonabelian_cutensor_batches
    budget = _nonabelian_pack_budget(Adata, plan, dtype)
    batches = ((plan,) if _packed_elements(plan) <= budget else
               _nonabelian_cutensor_batches(plan.struct, plan.size, plan.products, budget))
    return _NonabelianContraction.apply(Adata, Bdata, plan, tuple(recouplings), batches, budget)


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
