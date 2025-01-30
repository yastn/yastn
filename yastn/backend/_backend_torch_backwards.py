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
from itertools import groupby
from types import SimpleNamespace
import numpy as np
import torch

from .linalg.torch_svd_gesdd import SVDGESDD
from .linalg.torch_svd_arnoldi import SVDARNOLDI
# from .linalg.torch_eig_arnoldi import SYMARNOLDI, SYMARNOLDI_2C


class kernel_svd(torch.autograd.Function):

    @staticmethod
    def forward(data, meta, sizes, fullrank_uv=False, ad_decomp_reg=1.0e-12, diagnostics=None):
        real_dtype = data.real.dtype if data.is_complex() else data.dtype
        Udata = torch.empty((sizes[0],), dtype=data.dtype, device=data.device)
        Sdata = torch.empty((sizes[1],), dtype=real_dtype, device=data.device)
        Vhdata = torch.empty((sizes[2],), dtype=data.dtype, device=data.device)
        reg = torch.as_tensor(ad_decomp_reg, dtype=real_dtype, device=data.device)
        for (sl, D, slU, DU, slS, slV, DV) in meta:
            U, S, Vh = SVDGESDD.forward(data[slice(*sl)].view(D), reg, fullrank_uv, diagnostics)
            Udata[slice(*slU)].reshape(DU)[:] = U
            Sdata[slice(*slS)] = S
            Vhdata[slice(*slV)].reshape(DV)[:] = Vh
        return Udata, Sdata, Vhdata

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        data, meta, sizes, _, ad_decomp_reg,diagnostics= inputs
        reg= torch.as_tensor(ad_decomp_reg, dtype=data.real.dtype, device=data.device)
        Udata, Sdata, Vhdata= output
        ctx.save_for_backward(Udata, Sdata, Vhdata, reg)
        ctx.meta_svd= meta
        ctx.data_size= data.numel()
        ctx.diagnostics= diagnostics

    @staticmethod
    def backward(ctx, Udata_b, Sdata_b, Vhdata_b):
        Udata, Sdata, Vhdata, reg= ctx.saved_tensors
        meta= ctx.meta_svd
        diagnostics= ctx.diagnostics
        data_size= ctx.data_size
        data_b= torch.zeros(data_size, dtype=Udata.dtype, device=Udata.device)
        for (sl, D, slU, DU, slS, slV, DV) in meta:
            loc_ctx= SimpleNamespace(diagnostics=diagnostics,
                saved_tensors=(Udata[slice(*slU)].view(DU),Sdata[slice(*slS)],Vhdata[slice(*slV)].view(DV),reg))
            data_b[slice(*sl)].view(D)[:],_,_,_ = SVDGESDD.backward(loc_ctx,\
                Udata_b[slice(*slU)].view(DU),Sdata_b[slice(*slS)],Vhdata_b[slice(*slV)].view(DV))
        return data_b, None, None, None, None, None


class kernel_svd_arnoldi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, meta, sizes):
        real_dtype = data.real.dtype if data.is_complex() else data.dtype
        Udata = torch.empty((sizes[0],), dtype=data.dtype, device=data.device)
        Sdata = torch.empty((sizes[1],), dtype=real_dtype, device=data.device)
        Vhdata = torch.empty((sizes[2],), dtype=data.dtype, device=data.device)
        for (sl, D, slU, DU, slS, slV, DV) in meta:
            k = slS[1] - slS[0]
            U, S, V = SVDARNOLDI.apply(data[slice(*sl)].view(D), k)
            Udata[slice(*slU)].reshape(DU)[:] = U
            Sdata[slice(*slS)] = S
            Vhdata[slice(*slV)].reshape(DV)[:] = V

        ctx.save_for_backward(Udata, Sdata, Vhdata)
        ctx.meta_svd= meta
        ctx.data_size= data.numel()
        return Udata, Sdata, Vhdata

    @staticmethod
    def backward(ctx, Udata_b, Sdata_b, Vhdata_b):
        raise Exception("backward not implemented")
        Udata, Sdata, Vhdata = ctx.saved_tensors
        meta= ctx.meta_svd
        data_size= ctx.data_size
        data_b= torch.zeros(data_size, dtype=Udata.dtype, device=Udata.device)
        return None,None,None,None,None,None


class kernel_dot(torch.autograd.Function):
    @staticmethod
    def forward(Adata, Bdata, meta_dot, Dsize):
        dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
        if dtype != Adata.dtype:
            Adata = Adata.to(dtype=dtype)
        if dtype != Bdata.dtype:
            Bdata = Bdata.to(dtype=dtype)
        newdata = torch.zeros(Dsize, dtype=dtype, device=Adata.device)
        for (slc, Dc, sla, Da, slb, Db) in meta_dot:
            newdata[slice(*slc)].view(Dc)[:] = Adata[slice(*sla)].view(Da) @ Bdata[slice(*slb)].view(Db)
        return newdata

    @staticmethod
    def setup_context(ctx, inputs, output):
        Adata, Bdata, meta_dot, Dsize = inputs
        ctx.save_for_backward(Adata, Bdata)
        ctx.meta_dot= meta_dot

    @staticmethod
    def backward(ctx, Cdata_b):
        # adjoint of block-sparse matrix-matrix multiplication A.B = C
        # A_b = C_b.B^T ; B_b = A^T . C_b
        Adata, Bdata= ctx.saved_tensors
        meta_dot= ctx.meta_dot
        dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
        Adata_b = torch.zeros_like(Adata, dtype=dtype)
        Bdata_b = torch.zeros_like(Bdata, dtype=dtype)
        if dtype != Adata.dtype:
            Adata = Adata.to(dtype=dtype)
        if dtype != Bdata.dtype:
            Bdata = Bdata.to(dtype=dtype)
        for (slc, Dc, sla, Da, slb, Db) in meta_dot:
            Ab = Adata_b[slice(*sla)].view(Da)
            Bb = Bdata_b[slice(*slb)].view(Db)
            Cb = Cdata_b[slice(*slc)].view(Dc)
            B = Bdata[slice(*slb)].view(Db)
            A = Adata[slice(*sla)].view(Da)
            Ab += Cb @ B.adjoint()  #  += is for fuse_contracted
            Bb += A.adjoint() @ Cb
        return Adata_b, Bdata_b, None, None


class kernel_transpose_dot_sum(torch.autograd.Function):
    @staticmethod
    def forward(Adata, Bdata, meta_dot, Areshape, Breshape, Aorder, Border, Dsize):
        dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
        if dtype != Adata.dtype:
            Adata = Adata.to(dtype=dtype)
        if dtype != Bdata.dtype:
            Bdata = Bdata.to(dtype=dtype)
        Cdata = torch.zeros(Dsize, dtype=dtype, device=Adata.device)
        At = tuple(Adata[slice(*sl)].view(Di).permute(Aorder).reshape(Dl, Dr) for sl, Di, Dl, Dr in Areshape)
        Bt = tuple(Bdata[slice(*sl)].view(Di).permute(Border).reshape(Dl, Dr) for sl, Di, Dl, Dr in Breshape)

        for (sl, Dslc, list_tab) in meta_dot:
            tmp = Cdata[slice(*sl)].view(Dslc)
            for ta, tb in list_tab:
                tmp[:] += At[ta] @ Bt[tb]
        return Cdata

    @staticmethod
    def setup_context(ctx, inputs, output):
        Adata, Bdata, meta_dot, Areshape, Breshape, Aorder, Border, Dsize = inputs
        ctx.save_for_backward(Adata, Bdata)
        ctx.meta_dot = meta_dot
        ctx.Areshape = Areshape
        ctx.Breshape = Breshape
        ctx.Aorder = Aorder
        ctx.Border = Border

    @staticmethod
    def backward(ctx, Cdata_b):
        # adjoint of block-sparse matrix-matrix multiplication A . B = C
        # A_b = C_b . B^T ; B_b = A^T . C_b
        Adata, Bdata = ctx.saved_tensors
        meta_dot = ctx.meta_dot
        Areshape = ctx.Areshape
        Breshape = ctx.Breshape
        Aorder = ctx.Aorder
        Border = ctx.Border
        inv_Aorder = tuple(np.argsort(Aorder))
        inv_Border = tuple(np.argsort(Border))

        At = tuple(Adata[slice(*sl)].view(Di).permute(Aorder).reshape(Dl, Dr) for sl, Di, Dl, Dr in Areshape)
        Bt = tuple(Bdata[slice(*sl)].view(Di).permute(Border).reshape(Dl, Dr) for sl, Di, Dl, Dr in Breshape)
        At_b = [torch.zeros_like(v) for v in At]
        Bt_b = [torch.zeros_like(v) for v in Bt]

        for (sl, Dslc, list_tab) in meta_dot:
            tmp = Cdata_b[slice(*sl)].view(Dslc)
            for ta, tb in list_tab:
                At_b[ta] += tmp @ Bt[tb].adjoint()
                Bt_b[tb] += At[ta].adjoint() @ tmp

        Adata_b = torch.zeros_like(Adata)
        for v, (sl, Di, _, _) in zip(At_b, Areshape):
            inv_Di = tuple(Di[n] for n in Aorder)
            Adata_b[slice(*sl)].reshape(Di)[:] = v.reshape(inv_Di).permute(inv_Aorder)

        Bdata_b = torch.zeros_like(Bdata)
        for v, (sl, Di, _, _) in zip(Bt_b, Breshape):
            inv_Di = tuple(Di[n] for n in Border)
            Bdata_b[slice(*sl)].reshape(Di)[:] = v.reshape(inv_Di).permute(inv_Border)

        return Adata_b, Bdata_b, None, None, None, None, None, None


class kernel_negate_blocks(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Adata, slices):
        ctx.slices = slices
        newdata = Adata.clone()
        for slc in slices:
            newdata[slice(*slc)] *= -1
        return newdata

    @staticmethod
    def backward(ctx, Cdata_b):
        slices = ctx.slices
        Adata_b = Cdata_b.clone()
        for slc in slices:
            Adata_b[slice(*slc)] *= -1
        return Adata_b, None


class kernel_apply_mask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Adata, mask, meta, Dsize, axis, ndim):
        ctx.mask = mask
        ctx.meta = meta
        ctx.axis = axis
        ctx.ndim = ndim
        ctx.size_Adata = Adata.numel()

        slc0 = (slice(None),) * axis
        slc2 = (slice(None),) * (ndim - (axis + 1))
        Cdata = torch.empty(Dsize, dtype=Adata.dtype, device=Adata.device)
        for sln, Dn, sla, Da, tm in meta:
            slcs = slc0 + (mask[tm],) + slc2
            Cdata[slice(*sln)].view(Dn)[:] = Adata[slice(*sla)].view(Da)[slcs]
        return Cdata

    @staticmethod
    def backward(ctx, Cdata_b):
        mask = ctx.mask
        slc0 = (slice(None),) * ctx.axis
        slc2 = (slice(None),) * (ctx.ndim - (ctx.axis + 1))
        Adata_b = torch.zeros(ctx.size_Adata, dtype=Cdata_b.dtype, device=Cdata_b.device)
        for sln, Dn, sla, Da, tm in ctx.meta:
            slcs = slc0 + (mask[tm],) + slc2
            Adata_b[slice(*sla)].view(Da)[slcs] = Cdata_b[slice(*sln)].view(Dn)
        return Adata_b, None, None, None, None, None, None


class kernel_embed_mask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Adata, mask, meta, Dsize, axis, ndim):
        ctx.mask = mask
        ctx.meta = meta
        ctx.axis = axis
        ctx.ndim = ndim
        ctx.size_Adata = Adata.numel()

        slc0 = (slice(None),) * axis
        slc2 = (slice(None),) * (ndim - (axis + 1))
        Cdata = torch.zeros(Dsize, dtype=Adata.dtype, device=Adata.device)
        for sln, Dn, sla, Da, tm in meta:
            slcs = slc0 + (mask[tm],) + slc2
            Cdata[slice(*sln)].view(Dn)[slcs] = Adata[slice(*sla)].view(Da)
        return Cdata

    @staticmethod
    def backward(ctx, Cdata_b):
        mask = ctx.mask
        slc0 = (slice(None),) * ctx.axis
        slc2 = (slice(None),) * (ctx.ndim - (ctx.axis + 1))
        Adata_b = torch.zeros(ctx.size_Adata, dtype=Cdata_b.dtype, device=Cdata_b.device)
        for sln, Dn, sla, Da, tm in ctx.meta:
            slcs = slc0 + (mask[tm],) + slc2
            Adata_b[slice(*sla)].view(Da)[:] = Cdata_b[slice(*sln)].view(Dn)[slcs]
        return Adata_b, None, None, None, None, None, None


class kernel_transpose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, axes, meta_transpose):
        ctx.axes = axes
        ctx.meta_transpose = meta_transpose

        newdata = torch.zeros_like(data)
        for sln, Dn, slo, Do in meta_transpose:
            newdata[slice(*sln)].view(Dn)[:] = data[slice(*slo)].view(Do).permute(axes)
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        axes = ctx.axes
        inv_axes = tuple(np.argsort(axes))
        meta_transpose = ctx.meta_transpose

        newdata_b = torch.zeros_like(data_b)
        for sln, Dn, slo, Do in meta_transpose:
            newdata_b[slice(*slo)].view(Do)[:] = data_b[slice(*sln)].view(Dn).permute(inv_axes)
        return newdata_b, None, None


class kernel_transpose_and_merge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, order, meta_new, meta_mrg, Dsize):
        ctx.order = order
        ctx.meta_new = meta_new
        ctx.meta_mrg = meta_mrg
        ctx.D_source = data.numel()

        newdata = torch.zeros(Dsize, dtype=data.dtype, device=data.device)
        for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
            assert tn == t1
            temp = newdata[slice(*sln)].reshape(Dn)
            for (_, slo, Do, Dslc, Drsh) in gr:
                slcs = tuple(slice(*x) for x in Dslc)
                temp[slcs] = data[slice(*slo)].reshape(Do).permute(order).reshape(Drsh)
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        order = ctx.order
        inv_order= tuple(np.argsort(order))
        meta_new = ctx.meta_new
        meta_mrg = ctx.meta_mrg
        D_source = ctx.D_source

        newdata_b = torch.zeros((D_source,), dtype=data_b.dtype, device=data_b.device)
        for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
            assert tn == t1
            tmp_b = data_b[slice(*sln)].view(Dn)
            for (_, slo, Do, Dslc, _) in gr:
                slcs = tuple(slice(*x) for x in Dslc)
                inv_Do = tuple(Do[n] for n in order)
                newdata_b[slice(*slo)].reshape(Do)[:] = tmp_b[slcs].reshape(inv_Do).permute(inv_order)
        return newdata_b, None, None, None, None


class kernel_unmerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, meta):
        Dsize = data.size()
        ctx.meta = meta
        ctx.fwd_data_size = Dsize
        newdata = torch.empty(Dsize, dtype=data.dtype, device=data.device)
        for sln, Dn, slo, Do, sub_slc in meta:
            slcs = tuple(slice(*x) for x in sub_slc)
            newdata[slice(*sln)].view(Dn)[:] = data[slice(*slo)].view(Do)[slcs]
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        meta = ctx.meta
        fwd_data_size = ctx.fwd_data_size
        # no zero blocks should be introduces here
        newdata_b = torch.empty(fwd_data_size, dtype=data_b.dtype, device=data_b.device)
        for sln, Dn, slo, Do, sub_slc in meta:
            slcs = tuple(slice(*x) for x in sub_slc)
            newdata_b[slice(*slo)].view(Do)[slcs] = data_b[slice(*sln)].view(Dn)
        return newdata_b, None, None, None
