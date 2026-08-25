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
from types import SimpleNamespace
from typing import Sequence
from functools import lru_cache
import numpy as np
import torch

from .._profile import nsys_profile

from .linalg.torch_svd_gesdd import SVDGESDD
from .linalg.torch_svds_scipy import SVDS_SCIPY
# from .linalg.torch_eig_arnoldi import SYMARNOLDI, SYMARNOLDI_2C


def _project_grad_dtype(grad, target_dtype):
    """Cast an accumulated gradient back to the dtype of its input operand.

    Backward passes promote to a common dtype (e.g. real+complex -> complex);
    the gradient of a lower-dtype input must be projected back to it, taking the
    real part when the input is real but the promoted gradient is complex.
    """
    if grad.dtype == target_dtype:
        return grad
    if grad.is_complex() and not torch.empty((), dtype=target_dtype).is_complex():
        grad = grad.real
    return grad.to(dtype=target_dtype)


class kernel_svd(torch.autograd.Function):
    @staticmethod
    def forward(data_in, meta, sizes, fullrank_uv=False, ad_decomp_reg=1.0e-12, diagnostics=None):
        real_dtype = data_in.real.dtype if data_in.is_complex() else data_in.dtype
        Udata = torch.empty((sizes[0],), dtype=data_in.dtype, device=data_in.device)
        Sdata = torch.empty((sizes[1],), dtype=real_dtype, device=data_in.device)
        Vhdata = torch.empty((sizes[2],), dtype=data_in.dtype, device=data_in.device)
        reg = torch.as_tensor(ad_decomp_reg, dtype=real_dtype, device=data_in.device)
        for slo, Do, slU, DU, slS, slV, DV in meta:
            U, S, Vh = SVDGESDD.forward(data_in[slo].view(Do), reg, fullrank_uv, diagnostics)
            Udata[slU].reshape(DU)[:] = U
            Sdata[slS] = S
            Vhdata[slV].reshape(DV)[:] = Vh
        return Udata, Sdata, Vhdata

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        data_in, meta, _, _, ad_decomp_reg, diagnostics = inputs
        reg = torch.as_tensor(ad_decomp_reg, dtype=data_in.real.dtype, device=data_in.device)
        Udata, Sdata, Vhdata = output
        ctx.save_for_backward(Udata, Sdata, Vhdata, reg)
        ctx.meta = meta
        ctx.size_in = data_in.numel()
        ctx.diagnostics = diagnostics

    @staticmethod
    def backward(ctx, Udata_b, Sdata_b, Vhdata_b):
        Udata, Sdata, Vhdata, reg = ctx.saved_tensors
        Smax = Sdata.max()
        data_b= torch.zeros((ctx.size_in,), dtype=Udata.dtype, device=Udata.device)
        for slo, Do, slU, DU, slS, slV, DV in ctx.meta:
            loc_ctx = SimpleNamespace(diagnostics=ctx.diagnostics,
                saved_tensors = (Udata[slU].view(DU), Sdata[slS], Vhdata[slV].view(DV), reg, Smax))
            data_b[slo].view(Do)[:], _, _, _ = SVDGESDD.backward(loc_ctx, Udata_b[slU].view(DU), Sdata_b[slS], Vhdata_b[slV].view(DV))
        return data_b, None, None, None, None, None


class kernel_svds_scipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data_in, meta, sizes, thresh, solver, **kwargs):
        real_dtype = data_in.real.dtype if data_in.is_complex() else data_in.dtype
        Udata = torch.empty((sizes[0],), dtype=data_in.dtype, device=data_in.device)
        Sdata = torch.empty((sizes[1],), dtype=real_dtype, device=data_in.device)
        Vhdata = torch.empty((sizes[2],), dtype=data_in.dtype, device=data_in.device)
        for slo, Do, slU, DU, slS, slV, DV in meta:
            k = slS.stop - slS.start
            U, S, V = SVDS_SCIPY.apply(data_in[slo].view(Do), k, thresh, solver, **kwargs)
            Udata[slU].reshape(DU)[:] = U
            Sdata[slS] = S
            Vhdata[slV].reshape(DV)[:] = V

        ctx.save_for_backward(Udata, Sdata, Vhdata)
        ctx.meta = meta
        ctx.size_in = data_in.numel()
        return Udata, Sdata, Vhdata

    @staticmethod
    def backward(ctx, Udata_b, Sdata_b, Vhdata_b):
        raise Exception("backward not implemented")
        Udata, Sdata, Vhdata = ctx.saved_tensors
        data_b= torch.zeros((data_size,), dtype=Udata.dtype, device=Udata.device)
        return None,None,None,None,None,None


class kernel_dot(torch.autograd.Function):
    @staticmethod
    def forward(data_A, data_B, meta, size_C):
        dtype = torch.promote_types(data_A.dtype, data_B.dtype)
        if dtype != data_A.dtype:
            data_A = data_A.to(dtype=dtype)
        if dtype != data_B.dtype:
            data_B = data_B.to(dtype=dtype)
        newdata = torch.zeros((size_C,), dtype=dtype, device=data_A.device)
        for slc, Dc, sla, Da, slb, Db in meta:
            newdata[slc].view(Dc)[:] = data_A[sla].view(Da) @ data_B[slb].view(Db)
        return newdata

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_A, data_B, meta, _ = inputs
        ctx.save_for_backward(data_A, data_B)
        ctx.meta= meta

    @staticmethod
    def backward(ctx, data_C_b):
        # adjoint of block-sparse matrix-matrix multiplication A.B = C
        # A_b = C_b.B^T ; B_b = A^T . C_b
        data_A, data_B = ctx.saved_tensors
        data_A_dtype, data_B_dtype = data_A.dtype, data_B.dtype
        dtype = torch.promote_types(data_A.dtype, data_B.dtype)
        data_A_b = torch.zeros_like(data_A, dtype=dtype)
        data_B_b = torch.zeros_like(data_B, dtype=dtype)
        if dtype != data_A.dtype:
            data_A = data_A.to(dtype=dtype)
        if dtype != data_B.dtype:
            data_B = data_B.to(dtype=dtype)
        for slc, Dc, sla, Da, slb, Db in ctx.meta:
            Ab = data_A_b[sla].view(Da)
            Bb = data_B_b[slb].view(Db)
            Cb = data_C_b[slc].view(Dc)
            B = data_B[slb].view(Db)
            A = data_A[sla].view(Da)
            Ab += Cb @ B.adjoint()  #  += is for fuse_contracted
            Bb += A.adjoint() @ Cb
        # project gradients back to each input's dtype (real+complex mixes)
        data_A_b = _project_grad_dtype(data_A_b, data_A_dtype)
        data_B_b = _project_grad_dtype(data_B_b, data_B_dtype)
        return data_A_b, data_B_b, None, None


class kernel_transpose_dot_sum(torch.autograd.Function):
    @staticmethod
    def forward(data_A, data_B, meta, reshape_A, reshape_B, order_A, order_B, size_C):
        dtype = torch.promote_types(data_A.dtype, data_B.dtype)
        if dtype != data_A.dtype:
            data_A = data_A.to(dtype=dtype)
        if dtype != data_B.dtype:
            data_B = data_B.to(dtype=dtype)
        data_C = torch.zeros((size_C,), dtype=dtype, device=data_A.device)
        At = {ii: data_A[slo].view(Do).permute(order_A).reshape(Dl, Dr) for ii, (slo, Do, Dl, Dr) in enumerate(reshape_A)}
        Bt = {ii: data_B[slo].view(Do).permute(order_B).reshape(Dl, Dr) for ii, (slo, Do, Dl, Dr) in enumerate(reshape_B)}
        for sln, Dn, ta, tb in meta:
            data_C[sln].view(Dn)[:] += At[ta] @ Bt[tb]
        return data_C

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_A, data_B, meta, reshape_A, reshape_B, order_A, order_B, _ = inputs
        ctx.save_for_backward(data_A, data_B)
        ctx.meta = meta
        ctx.reshape_A = reshape_A
        ctx.reshape_B = reshape_B
        ctx.order_A = order_A
        ctx.order_B = order_B

    @staticmethod
    def backward(ctx, data_C_b):
        # adjoint of block-sparse matrix-matrix multiplication A . B = C
        # A_b = C_b . B^T ; B_b = A^T . C_b
        data_A, data_B = ctx.saved_tensors
        data_A_dtype, data_B_dtype = data_A.dtype, data_B.dtype
        promoted_dtype = torch.promote_types(torch.promote_types(data_A_dtype, data_B_dtype), data_C_b.dtype)
        inv_order_A = tuple(np.argsort(ctx.order_A))
        inv_order_B = tuple(np.argsort(ctx.order_B))

        if promoted_dtype != data_A.dtype:
            data_A = data_A.to(dtype=promoted_dtype)
        if promoted_dtype != data_B.dtype:
            data_B = data_B.to(dtype=promoted_dtype)
        if promoted_dtype != data_C_b.dtype:
            data_C_b = data_C_b.to(dtype=promoted_dtype)

        At = {ii: data_A[slo].view(Do).permute(ctx.order_A).reshape(Dl, Dr) for ii, (slo, Do, Dl, Dr) in enumerate(ctx.reshape_A)}
        Bt = {ii: data_B[slo].view(Do).permute(ctx.order_B).reshape(Dl, Dr) for ii, (slo, Do, Dl, Dr) in enumerate(ctx.reshape_B)}
        At_b = {ii: torch.zeros_like(v) for ii, v in At.items()}
        Bt_b = {ii: torch.zeros_like(v) for ii, v in Bt.items()}

        for sln, Dn, ta, tb in ctx.meta:
            tmp = data_C_b[sln].view(Dn)
            At_b[ta] += tmp @ Bt[tb].adjoint()
            Bt_b[tb] += At[ta].adjoint() @ tmp

        # Accumulate gradients
        # Build gradient tensors using scatter (vmap compatible)
        def build_grad(blocks_grad, reshape_info, order, inv_order, size_in, dtype, device):
            indices_list = []
            values_list = []
            for grad, (sl, Di, _, _) in zip(blocks_grad.values(), reshape_info):
                inv_Di = tuple(Di[n] for n in order)
                values = grad.reshape(inv_Di).permute(inv_order).contiguous().reshape(-1)
                indices = torch.arange(sl[0], sl[1], dtype=torch.long, device=device)
                indices_list.append(indices)
                values_list.append(values)

            if indices_list:
                all_idx = torch.cat(indices_list)
                all_val = torch.cat(values_list)
                return torch.zeros((size_in,), dtype=dtype, device=device).scatter(0, all_idx, all_val)
            return torch.zeros((size_in,), dtype=dtype, device=device)

        data_A_b = build_grad(At_b, ctx.reshape_A, ctx.order_A, inv_order_A, data_A.numel(), promoted_dtype, data_A.device)
        data_B_b = build_grad(Bt_b, ctx.reshape_B, ctx.order_B, inv_order_B, data_B.numel(), promoted_dtype, data_B.device)
        # project gradients back to each input's dtype (real+complex mixes)
        data_A_b = _project_grad_dtype(data_A_b, data_A_dtype)
        data_B_b = _project_grad_dtype(data_B_b, data_B_dtype)

        return data_A_b, data_B_b, None, None, None, None, None, None


class kernel_negate_blocks(torch.autograd.Function):

    @staticmethod
    def _sign(slices : Sequence[Sequence[int]], n, device):
        # 2 * (int8 * n) + bool * n mem cost
        #
        sl = torch.as_tensor(slices, dtype=torch.long, device=device)  # (N, 2)
        delta = torch.zeros(n + 1, dtype=torch.int8, device=device)
        delta.index_put_((sl[:, 0],), torch.tensor(1, dtype=torch.int8, device=device), accumulate=True)
        delta.index_put_((sl[:, 1],), torch.tensor(-1, dtype=torch.int8, device=device), accumulate=True)
        delta.cumsum_(0)              # in-place cumulative sum, shape (n+1,)
        delta.mul_(-2).add_(1)        # +1 / -1, shape (n,)
        return delta[:n]

    @staticmethod
    def forward(data_in, slices):
        return data_in * kernel_negate_blocks._sign(slices, data_in.numel(), data_in.device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, slices = inputs
        ctx.slices = slices

    @staticmethod
    def backward(ctx, data_out_b):
        return data_out_b * kernel_negate_blocks._sign(ctx.slices, data_out_b.numel(), data_out_b.device), None


class kernel_apply_mask(torch.autograd.Function):
    @staticmethod
    def forward(data_in, mask, meta, size_out, axis, ndim):
        slc0 = (slice(None),) * axis
        slc2 = (slice(None),) * (ndim - (axis + 1))
        data_out = torch.zeros((size_out,), dtype=data_in.dtype, device=data_in.device)
        for sln, Dn, sla, Da, tm in meta:
            slcs = slc0 + (mask[tuple(tm)],) + slc2
            data_out[slice(*sln)].view(tuple(Dn))[:] = data_in[slice(*sla)].view(tuple(Da))[slcs]
        return data_out

    def setup_context(ctx, inputs, output):
        data_in, mask, meta, _, axis, ndim = inputs
        ctx.mask = mask
        ctx.meta = meta
        ctx.axis = axis
        ctx.ndim = ndim
        ctx.size_in = data_in.numel()

    @staticmethod
    def backward(ctx, data_out_b):
        mask = ctx.mask
        slc0 = (slice(None),) * ctx.axis
        slc2 = (slice(None),) * (ctx.ndim - (ctx.axis + 1))
        data_in_b = torch.zeros((ctx.size_in,), dtype=data_out_b.dtype, device=data_out_b.device)
        for sln, Dn, sla, Da, tm in ctx.meta:
            slcs = slc0 + (mask[tuple(tm)],) + slc2
            data_in_b[slice(*sla)].view(tuple(Da))[slcs] = data_out_b[slice(*sln)].view(tuple(Dn))
        return data_in_b, None, None, None, None, None


class kernel_embed_mask(torch.autograd.Function):
    @staticmethod
    def forward(data_in, mask, meta, size_out, axis, ndim):
        slc0 = (slice(None),) * axis
        slc2 = (slice(None),) * (ndim - (axis + 1))
        data_out = torch.zeros((size_out,), dtype=data_in.dtype, device=data_in.device)
        for sln, Dn, sla, Da, tm in meta:
            slcs = slc0 + (mask[tm],) + slc2
            data_out[sln].view(Dn)[slcs] = data_in[sla].view(Da)
        return data_out

    def setup_context(ctx, inputs, output):
        data_in, mask, meta, _, axis, ndim = inputs
        ctx.mask = mask
        ctx.meta = meta
        ctx.axis = axis
        ctx.ndim = ndim
        ctx.size_in = data_in.numel()

    @staticmethod
    def backward(ctx, data_out_b):
        mask = ctx.mask
        slc0 = (slice(None),) * ctx.axis
        slc2 = (slice(None),) * (ctx.ndim - (ctx.axis + 1))
        data_in_b = torch.zeros((ctx.size_in,), dtype=data_out_b.dtype, device=data_out_b.device)
        for sln, Dn, sla, Da, tm in ctx.meta:
            slcs = slc0 + (mask[tm],) + slc2
            data_in_b[sla].view(Da)[:] = data_out_b[sln].view(Dn)[slcs]
        return data_in_b, None, None, None, None, None


class kernel_embed_transpose(torch.autograd.Function):
    @staticmethod
    def forward(data_in, order, meta, size_out):
        data_out = torch.zeros((size_out,), dtype=data_in.dtype, device=data_in.device)
        for sln, Dn, slo, Do in meta:
            data_out[sln].view(Dn)[:] = data_in[slo].view(Do).permute(order)
        return data_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_in, order, meta, _ = inputs
        ctx.order = order
        ctx.meta = meta
        ctx.size_in = data_in.numel()

    @staticmethod
    def backward(ctx, data_out_b):
        inv_order = tuple(np.argsort(ctx.order))
        data_in_b = torch.zeros((ctx.size_in,), dtype=data_out_b.dtype, device=data_out_b.device)
        for sln, Dn, slo, Do in ctx.meta:
            data_in_b[slo].view(Do)[:] = data_out_b[sln].view(Dn).permute(inv_order)
        return data_in_b, None, None, None


class kernel_embed_slices(torch.autograd.Function):
    @staticmethod
    def forward(data_in, meta, size_out):
        data_out = torch.zeros((size_out,), dtype=data_in.dtype, device=data_in.device)
        for sln, slo in meta:
            data_out[sln] = data_in[slo]
        return data_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_in, meta, _ = inputs
        ctx.meta = meta
        ctx.size_in = data_in.numel()

    @staticmethod
    def backward(ctx, data_out_b):
        data_in_b = torch.zeros((ctx.size_in,), dtype=data_out_b.dtype, device=data_out_b.device)
        for sln, slo in ctx.meta:
            data_in_b[slo] = data_out_b[sln]
        return data_in_b, None, None


class kernel_transpose_and_merge(torch.autograd.Function):
    @staticmethod
    def forward(data_in, order, meta, size_out):
        data_out = torch.zeros((size_out,), dtype=data_in.dtype, device=data_in.device)
        for sln, Dn, slo, Do, ssln, Dns in meta:
            data_out[sln].reshape(Dn)[ssln] = data_in[slo].reshape(Do).permute(order).reshape(Dns)
        return data_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_in, order, meta, _ = inputs
        ctx.order = order
        ctx.meta = meta
        ctx.size_in = data_in.numel()

    @staticmethod
    def backward(ctx, data_out_b):
        inv_order = tuple(np.argsort(ctx.order))
        data_in_b = torch.zeros((ctx.size_in,), dtype=data_out_b.dtype, device=data_out_b.device)
        for sln, Dn, slo, Do, ssln, _ in ctx.meta:
            inv_Do = tuple(Do[n] for n in ctx.order)
            data_in_b[slo].view(Do)[:] = data_out_b[sln].view(Dn)[ssln].view(inv_Do).permute(inv_order)
        return data_in_b, None, None, None


def _strides1(D):
    r"""Row-major inner strides: out[d] = prod(D[d+1:]); out[-1] == 1."""
    out = [1] * len(D)
    acc = 1
    for d in range(len(D) - 1, -1, -1):
        out[d] = acc
        acc *= D[d]
    return out


@lru_cache(maxsize=1024)
@nsys_profile
def pack_transpose_and_merge_params(order, meta, size_in):
    r"""
    Precompute small per-block parameter tensors for the scatter/index-map variant of
    ``transpose_and_merge``. ``order`` is the single global permutation shared by all blocks,
    so each source element's destination flat index is pure integer arithmetic.

    Blocks are sorted by their source start so that ``searchsorted`` maps a global source
    position to its block. ``params['contiguous']`` records whether the source blocks tile
    ``[0, size_in)`` exactly. When they do not (e.g. the sub-legs ``struct_sub`` path of
    ``_fuse_blocks``, which fuses only a subset of source blocks), some source positions belong
    to no block; :class:`kernel_transpose_and_merge_scatter` then routes those to a sentinel sink.
    """
    if not meta:
        return None
    order = tuple(order)
    inv_order = tuple(int(x) for x in np.argsort(order))
    ndimo = len(order)
    ndimn = len(meta[0][1])

    rows = []
    for sln, Dn, slo, Do, ssln, Dns in meta:
        sDo1 = _strides1(Do)
        sP1 = _strides1(tuple(Do[o] for o in order))          # strides of permuted source shape
        wperm = [sP1[inv_order[d]] for d in range(ndimo)]     # weight of X[d] in i_perm
        rows.append((slo.start, slo.stop, list(Do), wperm, sDo1,
                     list(Dns), _strides1(Dns), [s.start for s in ssln], _strides1(Dn), sln.start))
    rows.sort(key=lambda r: r[0])

    covered, contiguous = 0, True                             # do the source blocks tile [0, size_in)?
    for r in rows:
        if r[0] != covered:
            contiguous = False
            break
        covered = r[1]
    contiguous = contiguous and covered == size_in

    # one packed [nblocks, K] int64 table (columns: slo_start, slo_stop, Do, wperm, sDo1, Dns,
    # sDns1, ssln_start, sDn1, sln_start) so _build_dest does a *single* per-element gather instead
    # of ~9; K = 3 + 3*ndimo + 4*ndimn.
    packed = torch.tensor(
        [[r[0], r[1], *r[2], *r[3], *r[4], *r[5], *r[6], *r[7], *r[8], r[9]] for r in rows],
        dtype=torch.int64, device='cpu')
    return {'ndimo': ndimo, 'ndimn': ndimn, 'contiguous': contiguous,
            'slo_start': packed[:, 0].contiguous(),   # separate 1D copy for searchsorted
            'packed': packed}


@nsys_profile
def _build_dest(params, g, sentinel=None):
    r"""
    Vectorized destination flat index for arbitrary source positions ``g`` (int64 tensor).

    With ``sentinel is None`` every position in ``g`` is assumed to land in a block (contiguous
    coverage, or ``g`` restricted to real block positions as in the compact/hybrid path). With an
    int ``sentinel`` source positions that fall in a gap between blocks are routed to ``sentinel``
    (the caller's sink slot) rather than a wrong block.
    """
    ndimo, ndimn = params['ndimo'], params['ndimn']
    packed = params['packed']
    b = torch.searchsorted(params['slo_start'], g, right=True) - 1
    bc = b.clamp_min(0) if sentinel is not None else b        # keep gather indices valid for gaps

    # ``g`` is block-sorted, so ``bc`` has exactly one consecutive run per block. Expand each per-block
    # column to ``[len(g)]`` with ``repeat_interleave`` one at a time (pipelined) instead of gathering
    # the whole ``[len(g), K]`` at once -- peak working set is ~6 [len(g)] vectors, not K.
    blocks, counts = torch.unique_consecutive(bc, return_counts=True)
    col = lambda i: torch.repeat_interleave(packed[blocks, i], counts)   # column i -> [len(g)]

    o = 2
    l = g - col(0)                                           # slo_start
    i_perm = torch.zeros_like(g)                             # decode Do, permute, reshape(Dns)
    for d in range(ndimo):
        i_perm += ((l // col(o + 2*ndimo + d)) % col(o + d)) * col(o + ndimo + d)   # sDo1, Do, wperm

    o2 = o + 3 * ndimo
    i_dest = col(packed.shape[1] - 1).clone()               # sln_start; reshape(Dns) -> +ssln -> encode Dn
    for e in range(ndimn):
        i_dest += ((i_perm // col(o2 + ndimn + e)) % col(o2 + e) + col(o2 + 2*ndimn + e)) * col(o2 + 3*ndimn + e)

    if sentinel is not None:                                 # gap elements -> sink
        i_dest = torch.where((b >= 0) & (g < col(1)), i_dest, i_dest.new_full((), sentinel))   # slo_stop
    return i_dest


def _build_dest_tiled(params, g, tile, sentinel=None):
    r"""
    Tile ``g`` into ``<= tile``-sized contiguous chunks and build the destination index per chunk,
    bounding the transient working set of :func:`_build_dest` to ``~tile`` regardless of ``len(g)``.
    ``g`` is block-sorted, so each chunk is still block-sorted (whole blocks, or clipped edge runs).
    """
    n = g.numel()
    if not tile or n <= tile:
        return _build_dest(params, g, sentinel)
    out = torch.empty_like(g)
    for lo in range(0, n, tile):
        hi = min(lo + tile, n)
        out[lo:hi] = _build_dest(params, g[lo:hi], sentinel)
    return out


def build_source_to_dest(params, lo, hi, device, sentinel=None):
    r"""Destination flat index for the contiguous source range ``[lo, hi)`` (int64, on ``device``)."""
    g = torch.arange(lo, hi, dtype=torch.int64, device=device)
    return _build_dest(params, g, sentinel)


def _concat_ranges(starts, stops, device):
    r"""
    Concatenate the integer ranges ``[starts[i], stops[i])`` into one int64 tensor (vectorized).
    ``starts``/``stops`` are int64 tensors on ``device``. Used to enumerate the flat positions of a
    subset of blocks (the small blocks in the hybrid path) without a per-block Python loop.
    """
    lens = stops - starts
    total = int(lens.sum())
    if total == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    out_off = torch.cumsum(lens, 0) - lens                    # start offset of each block in the output
    blk = torch.repeat_interleave(torch.arange(starts.numel(), device=device), lens)
    return starts[blk] + (torch.arange(total, device=device) - out_off[blk])


class kernel_transpose_and_merge_scatter(torch.autograd.Function):
    r"""
    Index-map variant of :class:`kernel_transpose_and_merge`. Collapses the per-block permute-copy
    loop into a single ``scatter_`` driven by a source->destination index map built on the fly on
    the GPU (uncached). ``chunk`` tiles the source range so the transient int64 index buffer is
    bounded by ``chunk`` rather than ``size_in`` (``None`` => single tile over the whole array).

    When ``params['contiguous']`` is ``False`` (partial coverage, e.g. the sub-legs contraction
    path) source positions with no destination are routed to a one-element sink appended to the
    output; the sink is dropped on return and contributes zero gradient.
    """
    @staticmethod
    @nsys_profile("kernel_transpose_and_merge_scatter")
    def forward(data_in, params, size_out, chunk):
        size_in, dev = data_in.numel(), data_in.device
        contiguous = params['contiguous']
        sentinel = None if contiguous else size_out           # gaps -> sink slot at index size_out
        data_out = torch.zeros((size_out + (0 if contiguous else 1),),
                               dtype=data_in.dtype, device=dev)
        step = size_in if not chunk else chunk
        for lo in range(0, size_in, step):
            hi = min(lo + step, size_in)
            s2d = build_source_to_dest(params, lo, hi, dev, sentinel)
            data_out.scatter_(0, s2d, data_in[lo:hi])
        return data_out if contiguous else data_out[:size_out]   # slice is a view, no copy

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_in, params, size_out, chunk = inputs
        ctx.params = params
        ctx.chunk = chunk
        ctx.size_in = data_in.numel()
        ctx.size_out = size_out

    @staticmethod
    def backward(ctx, data_out_b):
        params, chunk, size_in, dev = ctx.params, ctx.chunk, ctx.size_in, data_out_b.device
        contiguous, size_out = params['contiguous'], ctx.size_out
        sentinel = None if contiguous else size_out
        data_in_b = torch.empty((size_in,), dtype=data_out_b.dtype, device=dev)
        step = size_in if not chunk else chunk
        for lo in range(0, size_in, step):
            hi = min(lo + step, size_in)
            s2d = build_source_to_dest(params, lo, hi, dev, sentinel)
            if contiguous:
                data_in_b[lo:hi] = data_out_b[s2d]           # adjoint of a permutation = gather
            else:                                            # clamp sentinel in-range, zero gap grads
                g = data_out_b[s2d.clamp_max(size_out - 1)]
                data_in_b[lo:hi] = g.masked_fill_(s2d >= size_out, 0)
        return data_in_b, None, None, None


class kernel_unmerge(torch.autograd.Function):
    @staticmethod
    def forward(data_in, meta, size_out):
        data_out = torch.zeros((size_out,), dtype=data_in.dtype, device=data_in.device)
        for sln, Dn, slo, Do, sslo in meta:
            data_out[sln].view(Dn)[:] = data_in[slo].view(tuple(Do))[sslo]
        return data_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_in, meta, _ = inputs
        ctx.meta = meta
        ctx.size_in = data_in.numel()

    @staticmethod
    def backward(ctx, data_out_b):
        data_in_b = torch.zeros((ctx.size_in,), dtype=data_out_b.dtype, device=data_out_b.device)
        for sln, Dn, slo, Do, sslo in ctx.meta:
            data_in_b[slo].view(Do)[sslo] = data_out_b[sln].view(Dn)
        return data_in_b, None, None


class kernel_unmerge_scatter(torch.autograd.Function):
    r"""
    Index-map variant of :class:`kernel_unmerge`. Unmerge is a pure dest->source gather (no permute,
    dense dest), so a single ``data_in[gather_idx]`` replaces the per-block sub-block-copy loop. The
    index is built on the fly on the GPU (uncached) by reusing :func:`build_source_to_dest` with a
    meta relabeled to the ``order = identity`` case (see ``backend_torch.unmerge``). ``chunk`` tiles
    the dest range so the transient int64 index buffer is bounded by ``chunk`` rather than ``size_out``.
    """
    @staticmethod
    def forward(data_in, params, size_out, chunk):
        data_out = torch.empty((size_out,), dtype=data_in.dtype, device=data_in.device)
        dev = data_in.device
        step = size_out if not chunk else chunk
        for lo in range(0, size_out, step):
            hi = min(lo + step, size_out)
            gidx = build_source_to_dest(params, lo, hi, dev)   # dest [lo,hi) -> source flat index
            data_out[lo:hi] = data_in[gidx]
        return data_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_in, params, _, chunk = inputs
        ctx.params = params
        ctx.chunk = chunk
        ctx.size_in = data_in.numel()

    @staticmethod
    def backward(ctx, data_out_b):
        params, chunk, dev = ctx.params, ctx.chunk, data_out_b.device
        size_out = data_out_b.numel()
        data_in_b = torch.zeros((ctx.size_in,), dtype=data_out_b.dtype, device=dev)
        step = size_out if not chunk else chunk
        for lo in range(0, size_out, step):
            hi = min(lo + step, size_out)
            gidx = build_source_to_dest(params, lo, hi, dev)   # adjoint of a gather = scatter-add
            data_in_b.scatter_add_(0, gidx, data_out_b[lo:hi])
        return data_in_b, None, None, None


class kernel_transpose_and_merge_hybrid(torch.autograd.Function):
    r"""
    Hybrid of :class:`kernel_transpose_and_merge` (loop) and its scatter variant. Large blocks go
    through the per-block permute-copy loop (bandwidth-optimal, no index build); the many small
    blocks go through a single compact scatter over precomputed ``src_small``/``dst_small`` (built
    only over the small blocks, so no sentinel and only small-block data is touched). Large and small
    blocks write disjoint regions of the output.
    """
    @staticmethod
    def forward(data_in, order, meta_large, src_small, dst_small, size_out):
        data_out = torch.zeros((size_out,), dtype=data_in.dtype, device=data_in.device)
        for sln, Dn, slo, Do, ssln, Dns in meta_large:
            data_out[sln].reshape(Dn)[ssln] = data_in[slo].reshape(Do).permute(order).reshape(Dns)
        if src_small.numel() > 0:
            data_out.scatter_(0, dst_small, data_in[src_small])
        return data_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_in, order, meta_large, src_small, dst_small, _ = inputs
        ctx.order = order
        ctx.meta_large = meta_large
        ctx.save_for_backward(src_small, dst_small)
        ctx.size_in = data_in.numel()

    @staticmethod
    def backward(ctx, data_out_b):
        order = ctx.order
        inv_order = tuple(np.argsort(order))
        src_small, dst_small = ctx.saved_tensors
        data_in_b = torch.zeros((ctx.size_in,), dtype=data_out_b.dtype, device=data_out_b.device)
        for sln, Dn, slo, Do, ssln, _ in ctx.meta_large:
            inv_Do = tuple(Do[n] for n in order)
            data_in_b[slo].view(Do)[:] = data_out_b[sln].view(Dn)[ssln].view(inv_Do).permute(inv_order)
        if src_small.numel() > 0:
            data_in_b[src_small] = data_out_b[dst_small]     # adjoint of the compact scatter = gather
        return data_in_b, None, None, None, None, None


class kernel_unmerge_hybrid(torch.autograd.Function):
    r"""
    Hybrid of :class:`kernel_unmerge` (loop) and its gather variant. Large blocks go through the
    per-block sub-block-copy loop; the many small blocks go through a single compact gather over
    precomputed ``dst_small`` (dest positions) / ``gidx_small`` (their source flat indices).
    """
    @staticmethod
    def forward(data_in, meta_large, dst_small, gidx_small, size_out):
        data_out = torch.zeros((size_out,), dtype=data_in.dtype, device=data_in.device)
        for sln, Dn, slo, Do, sslo in meta_large:
            data_out[sln].view(Dn)[:] = data_in[slo].view(tuple(Do))[sslo]
        if dst_small.numel() > 0:
            data_out[dst_small] = data_in[gidx_small]
        return data_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        data_in, meta_large, dst_small, gidx_small, _ = inputs
        ctx.meta_large = meta_large
        ctx.save_for_backward(dst_small, gidx_small)
        ctx.size_in = data_in.numel()

    @staticmethod
    def backward(ctx, data_out_b):
        dst_small, gidx_small = ctx.saved_tensors
        data_in_b = torch.zeros((ctx.size_in,), dtype=data_out_b.dtype, device=data_out_b.device)
        for sln, Dn, slo, Do, sslo in ctx.meta_large:
            data_in_b[slo].view(Do)[sslo] = data_out_b[sln].view(Dn)
        if dst_small.numel() > 0:                            # adjoint of the compact gather = scatter-add
            data_in_b.scatter_add_(0, gidx_small, data_out_b[dst_small])
        return data_in_b, None, None, None, None
