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
import os
import warnings
from itertools import groupby
from functools import lru_cache, reduce
from math import prod
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from torch.utils.checkpoint import checkpoint as _checkpoint

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True) # to avoid issues with ''fork' and sharing CUDA tensors in some environments

from .._cache_registry import register_cache
from .._memory_override import memory_override
from .linalg.torch_eig_sym import SYMEIG


def _linalg_backend_name(backend):
    """ The string accepted by :func:`torch.backends.cuda.preferred_linalg_library`. """
    name = str(backend).lower()
    for option in ('magma', 'cusolver', 'default'):
        if option in name:
            return option
    return 'default'


def _with_cusolver_retry(solve, x):
    r"""Run a cuSOLVER-backed decomposition, recovering from a library failure.

    cuSOLVER raises ``CUSOLVER_STATUS_INTERNAL_ERROR`` on matrices it cannot handle -- long
    fPEPS evolution runs on Hopper abort once their gauge-fixing metric becomes rank
    deficient and extremely ill conditioned (rank 326/696 and cond 7e17 in one observed
    case), while LAPACK and MAGMA solve the very same block. The call is retried on a fresh
    (16-byte aligned) copy, then with MAGMA, and finally on the CPU, each a different
    implementation than the failing cuSOLVER path.

    MAGMA is selected only around this call and restored immediately afterwards: leaving the
    process-wide preference on MAGMA breaks the SVD calls that pass ``driver=``, which only
    cuSOLVER supports. ``solve`` must accept one tensor and return a tensor or a tuple of
    tensors.
    """
    try:
        return solve(x)
    except RuntimeError as exc:
        if 'cusolver' not in str(exc).lower():
            raise
        try:
            return solve(x.clone())
        except RuntimeError:
            if not x.is_cuda:
                raise
        previous = torch.backends.cuda.preferred_linalg_library()
        try:
            torch.backends.cuda.preferred_linalg_library('magma')
            return solve(x)
        except RuntimeError:
            pass
        finally:
            torch.backends.cuda.preferred_linalg_library(_linalg_backend_name(previous))
        result = solve(x.cpu())
        if isinstance(result, tuple):
            return type(result)(v.to(x.device) if torch.is_tensor(v) else v for v in result)
        return result.to(x.device)

from ._backend_torch_backwards import kernel_svd, kernel_svds_scipy
from ._backend_torch_backwards import kernel_dot, kernel_transpose_dot_sum, kernel_negate_blocks
from ._backend_torch_backwards import kernel_apply_mask, kernel_embed_mask
from ._backend_torch_backwards import kernel_embed_transpose, kernel_transpose_and_merge, kernel_unmerge
from ._backend_torch_backwards import kernel_transpose_and_merge_scaled
from ._backend_torch_backwards import kernel_transpose_and_merge_scatter
from . import _backend_torch_backwards as _torch_backwards
from ._backend_torch_backwards import kernel_unmerge_scatter
from ._backend_torch_backwards import kernel_transpose_and_merge_hybrid, kernel_unmerge_hybrid
from ._backend_torch_backwards import kernel_embed_slices


__all__= ['DTYPE', 'get_dtype', 'get_yastn_dtype',
    'nvtx', 'cuda_is_available', 'get_device', 'is_cpu_device', 'move_to',
    'grad', 'requires_grad_', 'requires_grad', 'detach', 'detach_', 'clone', 'copy', 'checkpoint',
    'random_seed', 'randint',
    'to_numpy', 'get_shape', 'get_size', 'diag_create', 'diag_get', 'real', 'is_complex', 'conj',
    'imag', 'max_abs', 'maximum', 'norm_matrix', 'delete', 'insert',
    'expm', 'first_element', 'item', 'sum_elements', 'norm', 'entropy',
    'zeros', 'ones', 'rand', 'to_tensor', 'to_mask', 'square_matrix_from_dict',
    'trace', 'rsqrt', 'reciprocal', 'exp', 'sqrt', 'absolute', 'clip', 'permute_dims',
    'fix_svd_signs', 'svdvals', 'svd_lowrank', 'svd', 'svd_randomized', 'svds_scipy', 'nonzero_blocks',
    'eigh', 'qr', 'pinv', 'eig', 'eigh_lowrank', 'eigvals',
    'argsort', 'argsort_which', 'argmax', 'flip', 'allclose',
    'add', 'sub', 'apply_mask', 'vdot', 'diag_1dto2d', 'diag_2dto1d',
    'dot', 'dot_diag', 'transpose_dot_sum', 'recouple_dot_sum',
    'merge_to_dense', 'merge_super_blocks', 'is_independent',
    'apply_mask', 'embed_mask', 'embed_slices',
    'embed_transpose', 'transpose_and_merge', 'transpose_and_merge_scaled', 'unmerge',
    'negate_blocks', 'gather_slices', 'bitwise_not', 'recouple', 'recouple_sequence', 'recouple_and_merge', 'recouple_and_unfuse', 'recouple_and_fuse', 'scale_blocks', 'scale_blocks_inplace',
    'unique', 'searchsorted', 'isin', 'nonzero', 'cumsum', 'concatenate',
    'repeat', 'where', 'zeros_like', 'full_like', 'arange']


torch.random.seed()
BACKEND_ID = "torch"
DTYPE = {'float32': torch.float32,
         'float64': torch.float64,
         'complex64': torch.complex64,
         'complex128': torch.complex128,
         'int32': torch.int32,
         'int64': torch.int64,
         'bool': torch.bool}


def cuda_is_available():
    return torch.cuda.is_available()


# Reduced fusion-block transformations share this backend with Abelian kernels.
# Cache only structural metadata; tensor values remain in the autograd graph.
@register_cache(key='torch_recoupling_plan')
@lru_cache(maxsize=128)
def _recoupling_plan(order, meta, dtype, device):
    """Cache offsets and permutation strides in O(terms * rank) device storage."""
    ndim = len(order)
    # One C-level unzip of the records instead of four python-level passes: this runs once per
    # distinct (order, meta) per step, and the generator-driven np.fromiter calls showed up as
    # ~1.4 s of a D=8 step.  Values and layout are unchanged (np.asarray casts like fromiter).
    dst_t, src_t, shapes_t, coef_t = zip(*meta)
    shapes = np.asarray(shapes_t, dtype=np.int64)
    counts = shapes.prod(axis=1)
    ends = np.cumsum(counts)
    offsets = ends - counts
    sources = np.asarray(src_t, dtype=np.int64)
    destinations = np.asarray(dst_t, dtype=np.int64)
    coefficients = np.asarray(coef_t, dtype=np.float64)
    # C-order strides, vectorized: strides[:, i] = prod(shape[:, i + 1:]).
    def c_order_strides(shape):
        reversed_cumprod = np.cumprod(shape[:, ::-1], axis=1)[:, ::-1]
        strides = np.empty_like(shape)
        strides[:, :-1] = reversed_cumprod[:, 1:]
        strides[:, -1] = 1
        return strides
    original = c_order_strides(shapes)
    permuted = shapes[:, list(order)]
    output_strides = c_order_strides(permuted)
    input_strides = original[:, list(order)]
    integers = np.vstack((offsets, sources, destinations, output_strides.T, input_strides.T))
    integers = torch.from_numpy(np.ascontiguousarray(integers)).to(device)
    offset = int(ends[-1]) if len(ends) else 0
    ends = torch.from_numpy(ends).to(device)
    coefficients = torch.from_numpy(coefficients).to(dtype=dtype, device=device)
    return integers, ends, coefficients, offset, ndim


def _recoupling_indices(plan, lo, hi):
    integers, ends, coefficients, _, ndim = plan
    # Locate each flattened contribution, then undo its output-axis permutation.
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
        # Fusion coefficients are real, so the adjoint swaps gather/scatter maps.
        result = gradient.new_zeros(ctx.input_size)
        for lo in range(0, ctx.plan[3], 1 << 20):
            source, destination, weights = _recoupling_indices(ctx.plan, lo, min(lo + (1 << 20), ctx.plan[3]))
            result.index_add_(0, source, gradient[destination] * weights)
        return result, None, None


class _RecoupleSequence(torch.autograd.Function):
    @staticmethod
    def forward(data, plan, size, adjoint):
        from ._recoupling import composed_recoupling_blocks
        order = plan[0]
        result = data.new_zeros(size)

        def accumulate(record):
            destination, source, shape, coefficient = record
            count = prod(shape)
            permuted = tuple(shape[i] for i in order)
            if adjoint:
                target = result[source:source + count].view(shape).permute(order)
                values = data[destination:destination + count].view(permuted)
            else:
                target = result[destination:destination + count].view(permuted)
                values = data[source:source + count].view(shape).permute(order)
            target.add_(values, alpha=coefficient)

        def flush(records):
            if len(records) < 32:
                for record in records:
                    accumulate(record)
                return
            # Do not cache device maps for the streamed global product. Their
            # lifetime and metadata size are bounded by this small batch.
            batch = _recoupling_plan.__wrapped__(order, tuple(records), data.real.dtype, data.device)
            for lo in range(0, batch[3], 1 << 20):
                source, destination, weights = _recoupling_indices(batch, lo, min(lo + (1 << 20), batch[3]))
                if adjoint:
                    source, destination = destination, source
                result.index_add_(0, destination, data[source] * weights)

        pending = []
        for destination, terms in composed_recoupling_blocks(plan):
            for source, shape, coefficient in terms:
                record = (destination, source, shape, coefficient)
                if prod(shape) < (1 << 16):
                    pending.append(record)
                    if len(pending) == 4096:
                        flush(pending)
                        pending.clear()
                else:
                    # Large blocks stream through native strided accumulation.
                    accumulate(record)
        flush(pending)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        data, ctx.plan, _, ctx.adjoint = inputs
        ctx.input_size = data.numel()

    @staticmethod
    def backward(ctx, gradient):
        result = _RecoupleSequence.apply(gradient, ctx.plan, ctx.input_size, not ctx.adjoint)
        return result, None, None, None


def recouple_sequence(data, steps, size):
    from ._recoupling import recoupling_sequence_plan
    return _RecoupleSequence.apply(data, recoupling_sequence_plan(steps), size, False)


class _RecoupleMerge(torch.autograd.Function):
    """Write a hard fusion and its surrounding basis changes into one buffer."""

    @staticmethod
    def forward(data, before, groups, after, size, adjoint):
        from ._recoupling import composed_recoupling_blocks, composed_recoupling_terms
        result = data.new_zeros(size)
        if after is None:
            blocks = ((offset, ((offset, records[0][0][1], 1.),))
                      for offset, records in groups.items())
        else:
            blocks = composed_recoupling_blocks(after)
        for destination, terms in blocks:
            for middle, shape, right in terms:
                order = after[0] if after is not None else tuple(range(len(shape)))
                # Undo the final permutation as a view of the destination.
                # Splitting its fused axes below also uses views only.
                output = data if adjoint else result
                merged = output[destination:destination + prod(shape)].view(
                    tuple(shape[i] for i in order)).permute(tuple(np.argsort(order)))
                for record, weight in groups.get(middle, ()):
                    _, _, source_slice, source_shape, subslices, _ = record
                    target = merged[subslices].view(source_shape)
                    incoming = (composed_recoupling_terms(before, source_slice.start)
                                if before is not None else
                                ((source_slice.start, source_shape, 1.),))
                    for source, old_shape, left in incoming:
                        input_data = result if adjoint else data
                        values = input_data[source:source + prod(old_shape)].view(old_shape)
                        if before is not None:
                            values = values.permute(before[0])
                        coefficient = float(left * weight * right)
                        if adjoint:
                            values.add_(target, alpha=coefficient)
                        else:
                            target.add_(values, alpha=coefficient)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        data, ctx.before, ctx.groups, ctx.after, _, ctx.adjoint = inputs
        ctx.input_size = data.numel()

    @staticmethod
    def backward(ctx, gradient):
        result = _RecoupleMerge.apply(gradient, ctx.before, ctx.groups, ctx.after,
                                      ctx.input_size, not ctx.adjoint)
        return result, None, None, None, None, None


def recouple_and_merge(data, before, meta, after, size, weights, lengths, adjoint=False):
    """Fuse a canonical prefix between permutations without full intermediates.

    Metadata and coefficients are composed per block. Both forward and its
    adjoint allocate only their output; all block reads and writes are views.
    """
    from ._recoupling import recoupling_sequence_plan, adjoint_recoupling_steps
    if adjoint:
        # The caller supplies unmerge's forward permutations and reciprocal
        # weights. Execute it as the adjoint of a merge between transposed
        # permutations; reduced recoupling matrices need not be orthogonal.
        before, after = adjoint_recoupling_steps(after), adjoint_recoupling_steps(before)
    before = recoupling_sequence_plan(before) if before else None
    after = recoupling_sequence_plan(after) if after else None
    source_weights = {}
    start = 0
    for weight, length in zip(weights, lengths):
        source_weights[start] = weight
        start += int(length)
    groups = {}
    for record in meta:
        groups.setdefault(record[0].start, []).append((record, source_weights[record[2].start]))
    return _RecoupleMerge.apply(data, before, groups, after, size, adjoint)


class _RecoupleUnfuse(torch.autograd.Function):
    """Apply multiple unfusions through source views and one output buffer."""

    @staticmethod
    def forward(data, plan, outputs, size, adjoint):
        from ._recoupling import unfuse_sequence_blocks
        result = data.new_zeros(size)
        source = result if adjoint else data
        destination = data if adjoint else result
        stride = source.stride(0)
        for offset, shape, terms in unfuse_sequence_blocks(plan, outputs):
            block = destination[offset:offset + prod(shape)].view(shape)
            for start, strides, coefficient in terms:
                values = source.as_strided(shape, tuple(s * stride for s in strides),
                                           source.storage_offset() + start * stride)
                if adjoint:
                    values.add_(block, alpha=coefficient)
                else:
                    block.add_(values, alpha=coefficient)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        data, ctx.plan, ctx.outputs, _, ctx.adjoint = inputs
        ctx.input_size = data.numel()

    @staticmethod
    def backward(ctx, gradient):
        result = _RecoupleUnfuse.apply(gradient, ctx.plan, ctx.outputs,
                                       ctx.input_size, not ctx.adjoint)
        return result, None, None, None, None


def recouple_and_unfuse(data, steps, outputs, size):
    """Return the composed output, or None when composition is too costly."""
    from ._recoupling import unfuse_sequence_plan
    plan = unfuse_sequence_plan(steps, outputs)
    if plan is None:
        return None
    return _RecoupleUnfuse.apply(data, plan, outputs, size, False)


def recouple_and_fuse(data, adjoint_steps, inputs, size):
    """Apply a fusion as the adjoint of its unmerge and transpose sequence."""
    from ._recoupling import unfuse_sequence_plan
    plan = unfuse_sequence_plan(adjoint_steps, inputs)
    if plan is None:
        return None
    return _RecoupleUnfuse.apply(data, plan, inputs, size, True)


def recouple(data, order, meta, size):
    """Apply cached fusion-tree coefficients to permuted reduced blocks."""
    # Batch the small blocks into one index_add_ plan; the large ones already amortize a launch
    # and would only pay for a wide index.  Measured on CPU (D=8 t-J, 4 threads) the plan is worth
    # ~2.8x on a warm step (110.5 s -> 39.4 s) with bit-identical truncation errors, so it is no
    # longer restricted to CUDA -- the scratch stays bounded by the small blocks either way.
    small = tuple(record for record in meta if prod(record[2]) < (1 << 16))
    if len(small) >= 32:
        plan = _recoupling_plan(order, small, data.real.dtype, data.device)
        result = _Recouple.apply(data, plan, size)
        remaining = (record for record in meta if prod(record[2]) >= (1 << 16))
    else:
        result = data.new_zeros(size)
        remaining = meta
    for destination, source, shape, coefficient in remaining:
        count = prod(shape)
        # Write the permuted source directly into the destination view. The old
        # path materialized the permuted copy and its scaled temporary, which
        # is the last big allocation at large chi.
        perm_shape = tuple(shape[i] for i in order)
        target = result[destination:destination + count].view(perm_shape)
        values = data[source:source + count].view(shape).permute(order)
        if isinstance(coefficient, complex):
            target.add_(values * coefficient)
        else:
            target.add_(values, alpha=coefficient)
    return result + data[:0].sum() if not meta else result


@register_cache(key='torch_fusion_scale')
@lru_cache(maxsize=128)
def _scale_parameters(weights, lengths, inverse, dtype, device):
    values = torch.tensor(weights, dtype=dtype, device=device)
    if inverse:
        values = values.reciprocal()
    return values, torch.tensor(lengths, dtype=torch.int64, device=device)


def scale_blocks(data, weights, lengths, inverse=False):
    """Scale reduced blocks by irrep weights (or their inverses)."""
    # Explicit output_size avoids a device-to-host synchronization on CUDA.
    values, counts = _scale_parameters(weights, lengths, inverse, data.real.dtype, data.device)
    return data * torch.repeat_interleave(values, counts, output_size=data.numel())


def scale_blocks_inplace(data, weights, lengths, inverse=False):
    """Scale an owned 1D buffer block by block, without any temporary.

    Only for buffers the caller will not reuse (e.g. the fresh output of
    ``unmerge``); this removes the full-size expanded-weights buffer of
    ``scale_blocks``.
    """
    if data.numel() == 0 or len(lengths) == 0:
        return data
    values, _ = _scale_parameters(weights, lengths, inverse, data.real.dtype, data.device)
    start = 0
    for value, length in zip(values.unbind(), lengths):
        stop = start + int(length)
        data[start:stop].mul_(value)
        start = stop
    return data


def transpose_and_merge_scaled(data, order, meta, size, weights, lengths):
    """Merge with one real weight per source block, without a scaled source buffer.

    Use strided block writes on CPU and CUDA. This bounds scratch independently
    of block size, at the cost of one multiply launch per metadata record.
    """
    values, _ = _scale_parameters(weights, lengths, False, data.real.dtype, data.device)
    source_blocks = {}
    start = 0
    for i, length in enumerate(lengths):
        stop = start + int(length)
        source_blocks[(start, stop)] = i
        start = stop
    # Merge records can reorder or omit source blocks (trimmed contraction legs).
    indices = tuple(source_blocks[(slo.start, slo.stop)] for _, _, slo, *_ in meta)
    return kernel_transpose_and_merge_scaled.apply(data, order, meta, size, values, indices)


def get_dtype(t):
    return t.dtype


def get_yastn_dtype(t):
    dtypes = [k for k, v in DTYPE.items() if t.dtype == v]
    return dtypes[0] if len(dtypes) == 1 else 'unknown'


def is_complex(x):
    return x.is_complex()


def get_device(x)->str:
    return str(x.device)


def is_cpu_device(device) -> bool:
    """ Whether ``device`` refers to a host (CPU) device. Unset device (None/'') counts as CPU. """
    if not device:
        return True
    return torch.device(device).type == 'cpu'


def random_seed(seed):
    torch.random.manual_seed(seed)


def grad(x):
    return x.grad


####################################
#     single tensor operations     #
####################################


def detach(x):
    return x.detach()


def detach_(x):
    x.detach_()


def clone(x):
    return x.clone()


def copy(x):
    return x.detach().clone()


def to_numpy(x):
    return x if isinstance(x, (int, float, complex)) else x.numpy(force=True).copy()


def get_shape(x):
    return x.size()


def get_size(x):
    return x.numel()


def diag_create(x, p=0):
    return torch.diag(x, diagonal=p)


def diag_get(x):
    return torch.diag(x)


def real(x):
    return torch.real(x)


def imag(x):
    return torch.imag(x) if torch.is_complex(x) else 0 * x


def max_abs(x):
    return x.abs().max() if x.numel() > 0 else torch.tensor(0, device=x.device)


def maximum(input, output):
    return torch.maximum(input, output)


def norm_matrix(x):
    return torch.linalg.norm(x)


def delete(x, sl):
    return torch.cat([x[:sl[0]], x[sl[1]:]])


def insert(x, start, values):
    return torch.cat([x[:start], values, x[start:]])


def expm(x):
    return torch.matrix_exp(x)


def permute_dims(x, old_shape, axes):
    axes, old_shape = tuple(axes), tuple(old_shape)
    return torch.permute(x.reshape(old_shape), axes)


#########################
#    output numbers     #
#########################


def first_element(x):
    return x.view(-1)[0]


def item(x):
    return x.item()


def sum_elements(data):
    """ sum of all elements of all tensors in A """
    return data.sum().reshape(1)


def norm(data, p):
    """ 'fro' for Frobenious; 'inf' for max(abs(A)) """
    if p == "fro":
        return data.norm()
    return data.abs().max() if len(data) > 0 else torch.tensor(0.)


def entropy(data, alpha, tol):
    """ von Neuman or Renyi entropy from svd's"""
    Snorm = torch.sum(data) if len(data) > 0 else 0.
    if Snorm.abs() > 0:
        data = data / Snorm
        data = data[data.abs() > tol]
        if alpha == 1:
            return -1 * torch.sum(data * torch.log2(data))
        return torch.log2(torch.sum(data ** alpha)) / (1 - alpha)
    return torch.tensor(0.)


##########################
#     setting values     #
##########################


def zeros(D, dtype='float64', device='cpu'):
    return torch.zeros(D, dtype=DTYPE[dtype], device=device)


def ones(D, dtype='float64', device='cpu'):
    return torch.ones(D, dtype=DTYPE[dtype], device=device)


def rand(D, dtype='float64', distribution=(0, 1), device='cpu'):
    if dtype == 'bool':
        return torch.rand(D, dtype=DTYPE['float64'], device=device) > 0.5
    if distribution == 'normal':
        return torch.randn(D, dtype=DTYPE[dtype], device=device)
    ds = 1 if ('float' in dtype or 'bool' in dtype) else 1 + 1j
    out = torch.rand(D, dtype=DTYPE[dtype], device=device)
    return out if distribution == (0, 1) else (distribution[1] - distribution[0]) * out + distribution[0] * ds


def randint(low, high):
    return torch.randint(low, high, (1,))[0]


def to_tensor(val, Ds=None, dtype='float64', device='cpu'):
    T = torch.as_tensor(val, dtype=DTYPE[dtype], device=device)
    return T if Ds is None else T.reshape(Ds).contiguous()


@torch.no_grad()
def to_mask(val):
    return torch.as_tensor(val).nonzero().ravel()


def square_matrix_from_dict(H, D=None, **kwargs):
    dtype = reduce(torch.promote_types, (a.dtype for a in H.values()))
    device = next(iter(H.values())).device
    T = torch.zeros((D, D), dtype=dtype, device=device)
    for (i, j), v in H.items():
        if i < D and j < D:
            T[i, j] = v
    return T


def requires_grad_(data, requires_grad=True):
    data.requires_grad_(requires_grad)


def requires_grad(data):
    return data.requires_grad


def move_to(data, *args, **kwargs):
    dtype = kwargs.pop("dtype", None)
    if dtype is not None:
        kwargs["dtype"] = DTYPE[dtype]
    return data.to(*args, **kwargs)


def conj(data):
    return data.conj()


def trace(data, order, meta, Dsize):
    newdata = torch.zeros(Dsize, dtype=data.dtype, device=data.device)
    for sln, slo, Do, Drsh, coef in meta:
        tmp_sln = data[slo].reshape(Do).permute(order).reshape(Drsh)
        newdata[sln] = newdata[sln] + coef * torch.sum(torch.diagonal(tmp_sln, dim1=0, dim2=1), dim=-1)
    return newdata


def rsqrt(data, cutoff=0):
    res = torch.zeros_like(data)
    ind = data.abs() > cutoff
    res[ind] = data[ind].rsqrt()
    return res


def reciprocal(data, cutoff=0):
    res = torch.zeros_like(data)
    ind = data.abs() > cutoff
    res[ind] = 1. / data[ind]
    return res


def exp(data, step):
    return torch.exp(step * data)


def sqrt(data):
    return torch.sqrt(data)


def absolute(data):
    return torch.abs(data)


def bitwise_not(data):
    return torch.bitwise_not(data)


def clip(data, a_min=None, a_max=None):
    return torch.clamp(data, min=a_min, max=a_max)

##############################################################
#   low-level integer/index array primitives (numpy-aligned) #
#   backend-agnostic building blocks for the GPU meta path   #
##############################################################


def unique(x, return_inverse=False, return_counts=False):
    return torch.unique(x, return_inverse=return_inverse, return_counts=return_counts)


def searchsorted(a, v):
    return torch.searchsorted(a, v)


def isin(a, b):
    return torch.isin(a, b)


def nonzero(x):
    """ numpy-aligned: tuple of index arrays, one per dimension. """
    return torch.nonzero(x, as_tuple=True)


def cumsum(x, axis=0):
    return torch.cumsum(x, dim=axis)


def concatenate(seq, axis=0):
    return torch.cat(tuple(seq), dim=axis)


def repeat(x, repeats):
    """ numpy-aligned np.repeat: element-wise repeat (torch.repeat_interleave). """
    return torch.repeat_interleave(x, repeats)


def where(condition, x, y):
    return torch.where(condition, x, y)


def zeros_like(x):
    return torch.zeros_like(x)


def full_like(x, fill_value):
    return torch.full_like(x, fill_value)


def arange(*args, device='cpu'):
    return torch.arange(*args, device=device)


def svd_lowrank(data, meta, sizes, **kwargs):
    return svds_scipy(data, meta, sizes, solver='arpack', **kwargs)


def nonzero_blocks(data, slcs):
    """ Per-slice flags: whether data[slice] has any nonzero element. """
    return tuple(bool(data[slice(*sl)].any()) for sl in slcs)


def dtype_to_complex(data):
    """ type promotion to complex. """
    tmp = 1j * torch.tensor(1, dtype=data.dtype)
    return tmp.dtype


def svd(data, meta, sizes, fullrank_uv=False, ad_decomp_reg=1.0e-12, diagnostics=None, driver=None, **kwargs):
    return kernel_svd.apply(data, meta, sizes, fullrank_uv, ad_decomp_reg, driver, diagnostics)


def svdvals(data, meta, sizeS, **kwargss):
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    Sdata = torch.zeros((sizeS,), dtype=real_dtype, device=data.device)
    
    _loc_svd_vals= lambda block: torch.linalg.svdvals(block.cpu()).to(block.device) if block.device.type == 'mps' else _with_cusolver_retry(torch.linalg.svdvals, block)
    
    for (slo, Do, _, _, slS, _, _) in meta:
        Sdata[slS] = _loc_svd_vals(data[slo].view(Do))
    return Sdata


def svd_randomized(data, meta, sizes, q=None, niter=3, **kwargs):
    """
    Computes the SVD of block-sparse matrix using randomized SVD within each block.
    See torch.svd_lowrank for details.
    """
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    Udata = torch.zeros((sizes[0],), dtype=data.dtype, device=data.device)
    Sdata = torch.zeros((sizes[1],), dtype=real_dtype, device=data.device)
    Vdata = torch.zeros((sizes[2],), dtype=data.dtype, device=data.device)
    for slo, Do, slU, DU, slS, slV, DV in meta:
        k = slS.stop - slS.start
        q = max(k, min(2 * k, *Do))
        U, S, V = _with_cusolver_retry(lambda block: torch.svd_lowrank(block, q=q, niter=niter), data[slo].view(Do))
        Udata[slU].reshape(DU)[:] = U[:,:k]
        Sdata[slS] = S[:k]
        Vdata[slV].reshape(DV)[:] = V[:,:k].t().conj()
    return Udata, Sdata, Vdata


def svds_scipy(data, meta, sizes, thresh=0.1, solver='arpack', **kwargs):
    return kernel_svds_scipy.apply(data, meta, sizes, thresh, solver, **kwargs)


def fix_svd_signs(Udata, Vhdata, meta):
    Ud = torch.empty_like(Udata)
    Vhd = torch.empty_like(Vhdata)
    Uamp = (abs(Udata) * (2**40)).to(dtype=torch.int64)
    for (_, _, slU, DU, _, slV, DV) in meta:
        Utemp = Udata[slU].reshape(DU)
        Vtemp = Vhdata[slV].reshape(DV)
        Utemp_amp = Uamp[slU].reshape(DU)
        ii = torch.argmax(Utemp_amp, dim=0, keepdims=True)
        phase = torch.take_along_dim(Utemp, ii, dim=0)
        phase = phase / abs(phase)
        Ud[slU].reshape(DU)[:] = Utemp * phase.conj().reshape(1, -1)
        Vhd[slV].reshape(DV)[:] = Vtemp * phase.reshape(-1, 1)
    return Ud, Vhd


def eigh(data, meta=None, sizes=(1, 1), order_by_magnitude=False, ad_decomp_reg=1.0e-12):
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    Sdata = torch.zeros((sizes[0],), dtype=real_dtype, device=data.device)
    Udata = torch.zeros((sizes[1],), dtype=data.dtype, device=data.device)
    if meta is not None:
        if order_by_magnitude:
            reg = torch.as_tensor(ad_decomp_reg, dtype=real_dtype, device=data.device)
            f = lambda x: SYMEIG.apply(x, reg)
        else:
            def f(x):
                if x.device.type == 'mps':
                    S, U = torch.linalg.eigh(x.cpu())
                    return S.to(x.device), U.to(x.device)
                return _with_cusolver_retry(torch.linalg.eigh, x)
        for slo, Do, slU, DU, slS in meta:
            S, U = f(data[slo].view(Do))
            Sdata[slS] = S
            Udata[slU].view(DU)[:] = U
        return Sdata, Udata
    return _with_cusolver_retry(torch.linalg.eigh, data)  # S, U


def eig(data, meta=None, sizes=(1, 1), biorth_tol=None, **kwargs):
    if meta is None:
        return torch.linalg.eig(data)  # S, U
    # NOTE torch.linalg.eig returns right eigenvectors U only, i.e. M U = diag(S) U
    #
    # Assume worst case ?
    dtype = dtype_to_complex(data)
    Udata = torch.empty((sizes[0],), dtype=dtype, device=data.device)
    Sdata = torch.empty((sizes[1],), dtype=dtype, device=data.device)
    Vdata = torch.empty((sizes[2],), dtype=dtype, device=data.device)
    for slo, Do, slU, DU, slS, slV, DV in meta:
        S, U = torch.linalg.eig(data[slo].reshape(Do))
        #
        # in general diag(U.H @ U) = 1 but not U.H @ U = I, i.e. right eigenvectors are not orthogonal
        #
        # The solutions satisfy
        # M @ U / U = S (as cols)
        # V.H @ M / V.H = S (as rows)
        #
        # Search for left eigenvectors V (rows) via biorthogonality condition V.H @ U = I
        Id = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)
        try:
            V = torch.linalg.solve(U.conj().T, Id, left=True, out=None)
            V = V.conj().T
        except Exception as e:
            raise ValueError("Biorthonormalization of left/right eigenvector pairs failed.") from e

        eps = torch.finfo(U.dtype).eps
        err = torch.abs(torch.sum(V.T * U, axis=0) - 1).max()
        if err > 8 * eps:
            # V <- V + (I - V @ U) @ V; two matmuls, error is squared so a single step suffices (Newton-Schulz refinement)
            V = V + (Id - V @ U) @ V
            err = torch.abs(torch.sum(V.T * U, axis=0) - 1).max()

        # The attainable residual is O(eps * cond(U)) -- no amount of refinement beats it.
        # ||U||_1 ||V||_1 upper-bounds cond(U) in O(n^2), reusing the inverse computed above.
        # NOTE keep this per-block; assigning to biorth_tol would judge later blocks by the first block's bound.
        tol = biorth_tol if biorth_tol is not None else \
              10 * eps * torch.linalg.matrix_norm(U, 1) * torch.linalg.matrix_norm(V, 1)
        if err > tol:
            raise ValueError("Biorthonormalization of left/right eigenvector pairs failed: residual "
                            f"{err.item():.3e} exceeds tolerance {float(tol):.3e}. The matrix of right "
                            "eigenvectors is numerically singular (defective or nearly-defective input).")

        s_order= argsort_which(S, which=kwargs.get('which', 'LM'))
        Udata[slU].reshape(DU)[:] = U[:,s_order]
        Sdata[slS] = S[s_order]
        Vdata[slV].reshape(DV)[:] = V[s_order,:]
    return Udata, Sdata, Vdata


def eigh_lowrank(data, meta, sizes, thresh=None, **kwargs):
    # TODO user-defined threshold
    # TODO cupyx implementation of eigsh for GPU
    import numpy as np
    import scipy
    from .backend_np import argsort_which as np_argsort_which

    _which_map= {'LM': 'LM', 'SM': 'SM', 'LR': 'LA',  'SR': 'SA'}
    _which = kwargs.get('which', 'LM')
    data_device= data.device
    data = data.cpu().numpy() if data.is_cuda else data.numpy()
    real_dtype = data.real.dtype if np.iscomplexobj(data) else data.dtype
    Sdata = np.zeros((sizes[0],), dtype=real_dtype)
    Udata = np.zeros((sizes[1],), dtype=data.dtype)
    for slo, Do, slU, DU, slS in meta:
        k = slS.stop - slS.start
        n = Do[0]
        block = data[slo].reshape(Do)
        if k < n - 1 and n * n > 5000:
            try:
                S, U = scipy.sparse.linalg.eigsh(block, k=k, which=_which_map[_which],
                    M=None, sigma=None, v0=None, ncv=None, maxiter=None, tol=0,
                    return_eigenvectors=kwargs.get("return_eigenvectors", True),
                    Minv=None, OPinv=None, mode='normal',) #rng=None)
            except scipy.sparse.linalg.ArpackError as e:
                raise e
                # S, U = scipy.linalg.eigh(block)
        else:
            S, U = scipy.linalg.eigh(block) # always returns result sorted in ascending order ('SA')
        # sort in case of non-default order
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
        if not (_which in ['SR'] and  kwargs.get("return_eigenvectors", True)):
            arg_b = np_argsort_which(S, _which)  # S, U are numpy here
            S,U = S[arg_b[:k]], U[:,arg_b[:k]]
        else:
            S,U = S[:k], U[:,:k]
        Sdata[slS] = S
        Udata[slU].reshape(DU)[:] = U
    Sdata = torch.from_numpy(Sdata).to(device=data_device)
    Udata = torch.from_numpy(Udata).to(device=data_device)
    return Sdata, Udata


def eigvals(data, meta, sizeS, **kwargs):
    dtype = dtype_to_complex(data)
    Sdata = torch.empty((sizeS,), dtype=dtype, device=data.device)
    for slo, Do, _, _, slS, _, _ in meta:
        S = torch.linalg.eigvals(data[slo].reshape(Do))
        Sdata[slS] = S[argsort_which(S, which=kwargs.get('which', 'LM'))]
    return Sdata


def qr(data, meta, sizes, compute_q=True):
    # R-only QR has no derivative in Torch; retain the differentiable reduced
    # decomposition when R contributes to a graph, without assembling Qdata.
    mode = 'reduced' if compute_q or (torch.is_grad_enabled() and data.requires_grad) else 'r'
    Qdata = torch.zeros((sizes[0],), dtype=data.dtype, device=data.device) if compute_q else None
    Rdata = torch.zeros((sizes[1],), dtype=data.dtype, device=data.device)

    def _loc_qr(block):
        if block.device.type == 'mps':
            Q, R = torch.linalg.qr(block.cpu(), mode=mode)
            return Q.to(block.device), R.to(block.device)
        return _with_cusolver_retry(lambda x: torch.linalg.qr(x, mode=mode), block)
    
    for slo, Do, slQ, DQ, slR, DR in meta:
        Q, R = _loc_qr(data[slo].view(Do))
        sR = torch.sign(real(R.diag()))
        sR[sR == 0] = 1
        if compute_q:
            Qdata[slQ].view(DQ)[:] = Q * sR  # positive diag of R
        Rdata[slR].view(DR)[:] = sR.reshape([-1, 1]) * R
        del Q, R, sR
    return Qdata, Rdata


def pinv(A, rcond=None, hermitian=False, out=None, atol=None, rtol=None):
    if A.device.type == 'mps':
        result = torch.linalg.pinv(
            A.cpu(), atol=atol,
            rtol=rtol if rtol is not None else rcond,
            hermitian=hermitian,
        ).to(A.device)
        if out is not None:
            out.copy_(result)
            return out
        return result
    return torch.linalg.pinv(A, atol=atol, rtol=rtol if not rtol is None else rcond, hermitian=hermitian, out=out)


@torch.no_grad()
def flip(data):
    return torch.flip(data.ravel(), dims=(0,))


@torch.no_grad()
def argmax(data):
    return torch.argmax(data)


@torch.no_grad()
def argsort(data):
    return torch.argsort(data)


@torch.no_grad()
def argsort_which(val, which):
    if which == 'LR':
        return (-real(val)).argsort()
    if which == 'LM':
        return (-abs(val)).argsort()
    if which == 'SM':
        return abs(val).argsort()
    #if which == 'SR':
    return (real(val)).argsort()


################################
#     two dicts operations     #
################################


def allclose(Adata, Bdata, rtol, atol):
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    return torch.allclose(Adata.to(dtype=dtype), Bdata.to(dtype=dtype), rtol=rtol, atol=atol)


def add(datas, metas, Dsize):
    dtype = reduce(torch.promote_types, (data.dtype for data in datas))
    newdata = torch.zeros(Dsize, dtype=dtype, device=datas[0].device)
    for data, meta in zip(datas, metas):
        for sl_c, sl_a in meta:
            newdata[sl_c] = newdata[sl_c] + data[sl_a]
    return newdata


def sub(Adata, Bdata, metas, Dsize):
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    newdata = torch.zeros(Dsize, dtype=dtype, device=Adata.device)
    for sl_c, sl_a in metas[0]:
        newdata[sl_c] = newdata[sl_c] + Adata[sl_a]
    for sl_c, sl_b in metas[1]:
        newdata[sl_c] = newdata[sl_c] - Bdata[sl_b]
    return newdata


def vdot(Adata, Bdata, meta):
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    if dtype != Adata.dtype:
        Adata = Adata.to(dtype=dtype)
    if dtype != Bdata.dtype:
        Bdata = Bdata.to(dtype=dtype)
    tmp = torch.empty(len(meta), dtype=dtype, device=Adata.device)
    for ii, (sla, slb) in enumerate(meta):
        tmp[ii] = torch.dot(Adata[sla], Bdata[slb])
    return torch.sum(tmp)


def dot(Adata, Bdata, meta_dot, Dsize):
    return kernel_dot.apply(Adata, Bdata, meta_dot, Dsize)


def transpose_dot_sum(Adata, Bdata, meta_dot, Areshape, Breshape, Aorder, Border, Dsize):
    return kernel_transpose_dot_sum.apply(Adata, Bdata, meta_dot, Areshape, Breshape, Aorder, Border, Dsize)


def recouple_dot_sum(Adata, Bdata, meta_dot, Areshape, Breshape, Asteps, Bsteps, Dsize):
    """Recouple only the blocks requested by the contraction's bounded cache."""
    from ._recoupling import recoupling_sequence_plan, unfuse_sequence_plan
    plans = []
    for steps, reshape in ((Asteps, Areshape), (Bsteps, Breshape)):
        if steps:
            outputs = tuple((sl.start, shape) for sl, shape, _, _ in reshape)
            if unfuse_sequence_plan(tuple(('recouple', order, meta) for order, meta in steps), outputs) is None:
                return None
            plans.append(recoupling_sequence_plan(steps))
        else:
            plans.append(None)
    order_a = tuple(range(len(Areshape[0][1]))) if Areshape else ()
    order_b = tuple(range(len(Breshape[0][1]))) if Breshape else ()
    return kernel_transpose_dot_sum.apply(Adata, Bdata, meta_dot, Areshape, Breshape,
                                           order_a, order_b, Dsize, *plans)


def dot_diag(Adata, Bdata, meta, Dsize, axis, a_ndim):
    dim = [1] * a_ndim
    dim[axis] = -1
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    newdata = torch.empty(Dsize, dtype=dtype, device=Adata.device)
    for sln, slb, Db, sla in meta:
        newdata[sln].reshape(Db)[:] = Adata[sla].reshape(dim) * Bdata[slb].reshape(Db)
    return newdata


def negate_blocks(Adata, slices):
    return kernel_negate_blocks.apply(Adata, slices)


def gather_slices(Adata, slices):
    return torch.cat([Adata[slice(*slc)] for slc in slices])


#####################################################
#     block merging, truncations and un-merging     #
#####################################################


_MASK_TILE = 1 << 20


@torch.no_grad()
def _mask_parameters(mask, meta, axis, device, small_is_out):
    """Store block and selected-axis metadata, never an element-sized map."""
    if small_is_out:
        small, large, starts = meta['Dn'], meta['Da'], meta['sla'][:, 0]
        small_slices, large_slices = meta['sln'], meta['sla']
    else:
        small, large, starts = meta['Da'], meta['Dn'], meta['sln'][:, 0]
        small_slices, large_slices = meta['sla'], meta['sln']
    counts = small.prod(axis=1)
    ends = counts.cumsum()
    offsets = ends - counts
    size = int(ends[-1]) if len(ends) else 0
    if device.type == 'cpu' and counts.max(initial=0) <= _MASK_TILE:
        # Direct axis operations avoid per-element integer arithmetic on CPU.
        # Each block is bounded by the same tile limit as the vectorized path.
        records = tuple((slice(*ss), slice(*ls), tuple(ds), tuple(dl),
                         torch.as_tensor(mask[tuple(t)], dtype=torch.int64, device=device))
                        for ss, ls, ds, dl, t in zip(small_slices, large_slices, small, large, meta['tm']))
        return None, None, None, size, records, axis
    limit = max(size, int((starts + large.prod(axis=1)).max(initial=0)))
    index_dtype = torch.int32 if limit < 2 ** 31 else torch.int64
    # Share the axis indices of blocks with the same charge.
    selected, mask_offsets, offset = [], {}, 0
    for charge in dict.fromkeys(tuple(t) for t in meta['tm']):
        values = torch.as_tensor(mask[charge], dtype=index_dtype, device=device)
        mask_offsets[charge] = offset
        offset += values.numel()
        selected.append(values)
    mask_data = torch.cat(selected) if selected else torch.empty(0, dtype=torch.int64, device=device)
    integers = np.asarray((offsets, starts, small[:, axis], large[:, axis],
                           small[:, axis + 1:].prod(axis=1),
                           [mask_offsets[tuple(t)] for t in meta['tm']]), dtype=np.int64)
    return (torch.as_tensor(integers, dtype=index_dtype, device=device),
            torch.as_tensor(ends, dtype=index_dtype, device=device), mask_data, size, None, axis)


def _mask_indices(parameters, lo, hi):
    integers, ends, mask, _, _, _ = parameters
    positions = torch.arange(lo, hi, dtype=integers.dtype, device=integers.device)
    blocks = torch.searchsorted(ends, positions, right=True, out_int32=integers.dtype == torch.int32)
    local = positions - integers[0, blocks]
    right = integers[4, blocks]
    q = torch.div(local, right, rounding_mode='floor')
    width = integers[2, blocks]
    left = torch.div(q, width, rounding_mode='floor')
    selected = mask[integers[5, blocks] + q.remainder(width)]
    indices = integers[1, blocks] + (left * integers[3, blocks] + selected) * right + local.remainder(right)
    return indices.to(torch.int64)


class _MaskGather(torch.autograd.Function):
    @staticmethod
    def forward(data, parameters):
        size = parameters[3]
        result = data.new_empty(size)
        if parameters[4] is not None:
            for ss, ls, ds, dl, indices in parameters[4]:
                torch.index_select(data[ls].view(dl), parameters[5], indices, out=result[ss].view(ds))
            return result
        for lo in range(0, size, _MASK_TILE):
            hi = min(lo + _MASK_TILE, size)
            indices = _mask_indices(parameters, lo, hi)
            result[lo:hi] = data.index_select(0, indices)
            del indices
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        data, ctx.parameters = inputs
        ctx.input_size = data.numel()

    @staticmethod
    def backward(ctx, gradient):
        return _MaskScatter.apply(gradient, ctx.parameters, ctx.input_size), None


class _MaskScatter(torch.autograd.Function):
    @staticmethod
    def forward(data, parameters, size):
        result = data.new_zeros(size)
        if parameters[4] is not None:
            for ss, ls, ds, dl, indices in parameters[4]:
                result[ls].view(dl).index_add_(parameters[5], indices, data[ss].view(ds))
            return result
        for lo in range(0, data.numel(), _MASK_TILE):
            hi = min(lo + _MASK_TILE, data.numel())
            indices = _mask_indices(parameters, lo, hi)
            result.index_add_(0, indices, data[lo:hi])
            del indices
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.parameters, _ = inputs

    @staticmethod
    def backward(ctx, gradient):
        return _MaskGather.apply(gradient, ctx.parameters), None, None


def apply_mask(Adata, mask, meta, Dsize, axis, a_ndim):
    parameters = _mask_parameters(mask, meta, axis, Adata.device, small_is_out=True)
    return _MaskGather.apply(Adata, parameters)


def embed_mask(Adata, mask, meta, Dsize, axis, a_ndim):
    parameters = _mask_parameters(mask, meta, axis, Adata.device, small_is_out=False)
    return _MaskScatter.apply(Adata, parameters, Dsize)


def embed_transpose(data, axes, meta_transpose, size):
   return kernel_embed_transpose.apply(data, axes, meta_transpose, size)


def embed_slices(data, meta, size):
   return kernel_embed_slices.apply(data, meta, size)


def _fuse_scatter_chunk():
    r"""
    Read env ``YASTN_FUSE_SCATTER_CHUNK`` controlling the GPU ``transpose_and_merge`` path:
    unset -> ``None`` (scatter, adaptive tiles); ``0`` -> force the per-block loop even on GPU;
    a positive int -> scatter with that tile size. Invalid values warn and are ignored (``None``).
    """
    env = os.environ.get('YASTN_FUSE_SCATTER_CHUNK')
    if env in [None, ""]:
        # Old, larger-memory path: force the per-block loop (the pre-scatter
        # baseline) when the global override is enabled.
        return 0 if memory_override() else None
    try:
        val = int(env)
    except ValueError:
        val = -1
    if val < 0:
        warnings.warn(f"Ignoring YASTN_FUSE_SCATTER_CHUNK={env!r}: expected a non-negative integer.")
        return None
    return val   # 0 => force loop; positive => scatter tile size


def _fuse_scatter_thresh():
    r"""
    Read env ``YASTN_FUSE_SCATTER_THRESH`` -- the per-block element-count threshold for the hybrid
    GPU fuse/unfuse path: blocks with >= THRESH elements go through the per-block loop (bandwidth-
    optimal, no index build), smaller blocks through a compact scatter/gather. Unset -> 2**16
    (~saturates the A100/H100 memory interface; where the loop launch cost and index-build cost
    cross over). Invalid values warn and fall back to the default.
    """
    env = os.environ.get('YASTN_FUSE_SCATTER_THRESH')
    if env in [None, ""]:
        return 1 << 16
    try:
        val = int(env)
    except ValueError:
        val = -1
    if val < 0:
        warnings.warn(f"Ignoring YASTN_FUSE_SCATTER_THRESH={env!r}: expected a non-negative integer.")
        return 1 << 16
    return val


def transpose_and_merge(data, order, meta_mrg, size):
    r"""
    Transpose-and-merge source blocks into the fused 1D buffer. On CPU uses the per-block loop
    kernel. On GPU uses a hybrid: large blocks (>= ``YASTN_FUSE_SCATTER_THRESH`` elements) go through
    the loop, small blocks through tiled compact scatter; if all blocks are small it is a pure
    tiled scatter, if all are large a pure loop. ``YASTN_FUSE_SCATTER_CHUNK=0`` forces the
    loop; a positive value sets the scatter tile size. Partial coverage is handled without a sink
    on the hybrid path (compact indices cover only real blocks).
    """
    if data.is_cuda:
        chunk = _fuse_scatter_chunk()
        if chunk != 0 and len(meta_mrg) > 0:   # 0 forces the loop even on GPU; empty meta is always a loop
            thr = _fuse_scatter_thresh()
            meta_small = [m for m in meta_mrg if m[2].stop - m[2].start < thr]   # slo = m[2]
            if not meta_small:                 # all large -> per-block loop (no index build at all)
                return kernel_transpose_and_merge.apply(data, order, meta_mrg, size)
            if len(meta_small) == len(meta_mrg):
                params = _torch_backwards.device_merge_parameters(order, tuple(meta_mrg), data.numel(), data.device)
                tile = chunk or _torch_backwards.adaptive_index_tile(
                    data.device, data.numel(), size * data.element_size())
                return kernel_transpose_and_merge_scatter.apply(data, params, size, tile)
            meta_large = [m for m in meta_mrg if m[2].stop - m[2].start >= thr]
            params = _torch_backwards.device_merge_parameters(order, tuple(meta_small), data.numel(), data.device)
            tile = chunk or _torch_backwards.adaptive_index_tile(
                data.device, params['compact_size'], size * data.element_size())
            return kernel_transpose_and_merge_hybrid.apply(data, order, meta_large, params, size, tile)
    return kernel_transpose_and_merge.apply(data, order, meta_mrg, size)


def unmerge(data, meta, size):
    r"""
    Unfuse (split) fused blocks into the 1D buffer. On CPU uses the per-block loop kernel. On GPU
    uses the same hybrid split as :func:`transpose_and_merge`, keyed on the unfused (dest) block
    size: large blocks loop, small blocks go through a compact gather.
    """
    if data.is_cuda:
        chunk = _fuse_scatter_chunk()
        if chunk != 0 and len(meta) > 0:   # 0 forces the loop even on GPU; empty meta is always a loop
            thr = _fuse_scatter_thresh()
            meta_small = [m for m in meta if m[0].stop - m[0].start < thr]   # sln = m[0] (dest)
            if not meta_small:                 # all large -> per-block loop (no index build at all)
                return kernel_unmerge.apply(data, meta, size)
            ndimo = len(meta[0][1])        # len(Dn)
            # unmerge is the order=identity, source/dest-swapped case of transpose_and_merge; relabel
            # meta (sln,Dn,slo,Do,sslo) -> merge meta so _build_dest maps each DEST position to its
            # SOURCE flat index (gather_idx).
            merge_meta = [(slo, Do, sln, Dn, sslo, Dn) for (sln, Dn, slo, Do, sslo) in meta]
            host_params = _torch_backwards.pack_transpose_and_merge_params(tuple(range(ndimo)), tuple(merge_meta), size)
            if not host_params['contiguous']:   # dense dest expected; else fall back to loop
                return kernel_unmerge.apply(data, meta, size)
            if len(meta_small) == len(meta):
                params = _torch_backwards.device_merge_parameters(tuple(range(ndimo)), tuple(merge_meta), size, data.device)
                tile = chunk or _torch_backwards.adaptive_index_tile(
                    data.device, size, size * data.element_size())
                return kernel_unmerge_scatter.apply(data, params, size, tile)
            meta_large = [m for m in meta if m[0].stop - m[0].start >= thr]
            merge_small = tuple((slo, Do, sln, Dn, sslo, Dn) for sln, Dn, slo, Do, sslo in meta_small)
            params = _torch_backwards.device_merge_parameters(tuple(range(ndimo)), merge_small, size, data.device)
            tile = chunk or _torch_backwards.adaptive_index_tile(
                data.device, params['compact_size'], size * data.element_size())
            return kernel_unmerge_hybrid.apply(data, meta_large, params, size, tile)
    return kernel_unmerge.apply(data, meta, size)


def merge_to_dense(data, Dtot, meta):
    newdata = torch.zeros(Dtot, dtype=data.dtype, device=data.device)
    for sl, Dss in meta:
        newdata[tuple(slice(*Ds) for Ds in Dss)] = data[slice(*sl)].reshape(tuple(Ds[1] - Ds[0] for Ds in Dss))
    return newdata.ravel()


def merge_super_blocks(pos_tens, meta, size):
    dtype = reduce(torch.promote_types, (a._data.dtype for a in pos_tens.values()))
    device = next(iter(pos_tens.values()))._data.device
    newdata = torch.zeros(size, dtype=dtype, device=device)
    for sln, Dn, pa, slo, Do, Dslcs in meta:
        Do, Dn = tuple(Do), tuple(Dn)
        slcs = tuple(slice(*x) for x in Dslcs)
        newdata[slice(*sln)].reshape(Dn)[slcs] = pos_tens[pa]._data[slice(*slo)].reshape(Do)
    return newdata


def diag_1dto2d(data, meta, Dsize):
    newdata = torch.zeros(Dsize, dtype=data.dtype, device=data.device)
    for sln, slo in meta:
        newdata[sln] = torch.diag(data[slo]).ravel()
    return newdata


def diag_2dto1d(data, meta, Dsize):
    newdata = torch.zeros(Dsize, dtype=data.dtype, device=data.device)
    for sln, slo, Do in meta:
        torch.diag(data[slo].reshape(tuple(Do)), out=newdata[sln])
    return newdata


# functionals
def checkpoint(f, *args, **kwargs):
    return _checkpoint(f, *args,
                       use_reentrant=kwargs.pop('use_reentrant',None),
                       determinism_check=kwargs.pop('determinism_check', 'default'),
                       debug=kwargs.pop('debug', False),
                       **kwargs)


#############
#   tests   #
#############


def is_independent(x, y):
    """
    check if two arrays are identical, or share the same view.
    """
    return (x is not y) and (x.numel() == 0 or x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr())
