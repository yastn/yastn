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
import numpy as np
import torch

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
