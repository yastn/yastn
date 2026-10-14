"""Compact GPU plans for linear transformations of reduced fusion blocks."""
from functools import lru_cache
from math import prod

import torch

from .._cache_registry import register_cache


@register_cache(key='torch_recoupling_plan')
@lru_cache(maxsize=128)
def _recoupling_plan(order, meta, dtype, device):
    offsets, ends, sources, destinations, coefficients = [], [], [], [], []
    output_strides, input_strides = [], []
    offset = 0
    for destination, source, shape, coefficient in meta:
        count = prod(shape)
        offsets.append(offset)
        offset += count
        ends.append(offset)
        sources.append(source)
        destinations.append(destination)
        coefficients.append(coefficient)
        original = tuple(prod(shape[i + 1:]) for i in range(len(shape)))
        permuted = tuple(shape[i] for i in order)
        output_strides.append(tuple(prod(permuted[i + 1:]) for i in range(len(shape))))
        input_strides.append(tuple(original[i] for i in order))
    integers = torch.tensor([offsets, sources, destinations,
                             *zip(*output_strides), *zip(*input_strides)],
                            dtype=torch.int64, device=device)
    ends = torch.tensor(ends, dtype=torch.int64, device=device)
    coefficients = torch.tensor(coefficients, dtype=dtype, device=device)
    return integers, ends, coefficients, offset, len(order)


def _recoupling_indices(plan, lo, hi):
    integers, ends, coefficients, _, ndim = plan
    positions = torch.arange(lo, hi, device=integers.device)
    blocks = torch.searchsorted(ends, positions, right=True)
    local = positions - integers[0, blocks]
    destination = integers[2, blocks] + local
    source = integers[1, blocks]
    for axis in range(ndim):
        stride = integers[3 + axis, blocks]
        digit = torch.div(local, stride, rounding_mode='floor')
        local = local.remainder(stride)
        source = source + digit * integers[3 + ndim + axis, blocks]
    return source, destination, coefficients[blocks]


class _Recouple(torch.autograd.Function):
    @staticmethod
    def forward(data, plan, size):
        result = data.new_zeros(size)
        # Bound temporary element-sized maps; only O(terms * rank) is cached.
        for lo in range(0, plan[3], 1 << 20):
            source, destination, weights = _recoupling_indices(plan, lo, min(lo + (1 << 20), plan[3]))
            result.index_add_(0, destination, data[source] * weights)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        data, ctx.plan, _ = inputs
        ctx.input_size = data.numel()

    @staticmethod
    def backward(ctx, gradient):
        result = gradient.new_zeros(ctx.input_size)
        for lo in range(0, ctx.plan[3], 1 << 20):
            source, destination, weights = _recoupling_indices(ctx.plan, lo, min(lo + (1 << 20), ctx.plan[3]))
            result.index_add_(0, source, gradient[destination] * weights)
        return result, None, None


def recouple(data, order, meta, size):
    # Large blocks already amortize a kernel launch; indexing every element
    # there would add bandwidth and scratch-memory overhead. Batch small ones.
    small = tuple(record for record in meta if prod(record[2]) < (1 << 16)) if data.is_cuda else ()
    if len(small) >= 32:
        plan = _recoupling_plan(order, small, data.real.dtype, data.device)
        result = _Recouple.apply(data, plan, size)
        remaining = (record for record in meta if prod(record[2]) >= (1 << 16))
    else:
        result = data.new_zeros(size)
        remaining = meta
    for destination, source, shape, coefficient in remaining:
        count = prod(shape)
        block = data[source:source + count].reshape(shape).permute(order).reshape(-1)
        result[destination:destination + count] += coefficient * block
    return result + data[:0].sum() if not meta else result


@register_cache(key='torch_fusion_scale')
@lru_cache(maxsize=128)
def _scale_parameters(weights, lengths, inverse, dtype, device):
    values = torch.tensor(weights, dtype=dtype, device=device)
    if inverse:
        values = values.reciprocal()
    return values, torch.tensor(lengths, dtype=torch.int64, device=device)


def scale_blocks(data, weights, lengths, inverse):
    values, counts = _scale_parameters(weights, lengths, inverse, data.real.dtype, data.device)
    return data * torch.repeat_interleave(values, counts, output_size=data.numel())
