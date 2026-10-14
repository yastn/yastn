"""Packing for channel-aware cuTENSOR contractions, independent of TAPP imports."""
from functools import lru_cache

import torch

from .._cache_registry import register_cache


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
    return (torch.tensor(lengths, dtype=torch.int64, device=device),
            torch.tensor(shifts, dtype=torch.int64, device=device), offset,
            all(shift == 0 for shift in shifts))


def _slice_indices(lengths, shifts, size):
    # output_size avoids a GPU-to-host synchronization in repeat_interleave.
    return (torch.arange(size, device=shifts.device)
            + torch.repeat_interleave(shifts, lengths, output_size=size))


@register_cache(key='cutensor_nonabelian_coefficients')
@lru_cache(maxsize=128)
def _coefficients(values, dtype, device):
    return torch.tensor(values, dtype=dtype, device=device)


def _pack(data, slices, coefficients=()):
    lengths, shifts, size, contiguous = _packing_parameters(slices, data.device)
    packed = data[:size] if contiguous else data[_slice_indices(lengths, shifts, size)]
    if coefficients and any(value != 1 for value in coefficients):
        weights = _coefficients(coefficients, data.real.dtype, data.device)
        packed = packed * torch.repeat_interleave(weights, lengths, output_size=size)
    return packed


def contract(a, b, plan, tensordot_bs_v2):
    """One cuTENSOR call for all weighted products, with differentiable packing.

    Repeated source blocks are copied to disjoint packed slices: TAPP's backward
    writes one gradient per descriptor block, then gather's backward sums the
    contributions to the original data. Aliased descriptor offsets would lose
    those contributions. Packing uses reduced degeneracy blocks only.
    """
    dtype = torch.promote_types(a.dtype, b.dtype)
    a, b = a.to(dtype=dtype), b.to(dtype=dtype)
    if not plan.coefficients:
        # Keep a zero derivative for both inputs, including empty tensors.
        return a.new_zeros(plan.size) + a[:0].sum() + b[:0].sum()
    packed_a = _pack(a, plan.slices_a, plan.coefficients)
    packed_b = _pack(b, plan.slices_b)
    descriptors = [arg for descriptor in plan.descriptors for arg in descriptor]
    packed_c = tensordot_bs_v2(packed_a, packed_b, [1], [0],
                              *descriptors, [0, 1], list(plan.hashes))
    lengths, shifts, size, contiguous = _packing_parameters(plan.slices_c, a.device)
    if contiguous and size == plan.size:
        return packed_c
    # Some symmetry-allowed channels have no nonzero gluing contribution.
    # Do not expose uninitialized output gaps from the external allocator.
    return a.new_zeros(plan.size).index_copy(0, _slice_indices(lengths, shifts, size), packed_c)
