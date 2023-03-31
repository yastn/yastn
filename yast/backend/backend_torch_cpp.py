"""Support of torch as a data structure used by yast."""
from itertools import groupby
import numpy as np
import torch

from .backend_torch import *
from .backend_torch import transpose

import fused_transpose_merge_1d

BACKEND_ID = "torch_cpp"

def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge_p2p_v3.apply(data, order, meta_new, meta_mrg, Dsize)


class kernel_transpose_and_merge_plain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, order, meta_new, meta_mrg, Dsize):
        ctx.order = order
        ctx.meta_new = meta_new
        ctx.meta_mrg = meta_mrg
        ctx.D_source = data.numel()

        # meta_new -> list of [(tn, Dn, sln), ...] where
        #             tn -> effective charge for block in fused tensor
        #             Dn -> effective shape of block tn in fused tensor
        #             sln -> slice specifying the location of serialized tn block in 1d data of fused tensor
        #
        # meta_mrg -> t1 is effective charge of source block after fusion. I.e. t1==tn, means, that
        #             this source block will belong to destination block tn
        #          -> gr: tuple holding description of source data
        #                 slo -> specifies the location of source block in 1d data
        #                 Do  -> shape of the source block
        #                 Dscl-> list of slice data which specifies the location of the "transformed"
        #                        source block in the destination block tn
        #                 Drsh-> the shape of the "transformed" source block in the destination block tn
        #
        jobs= [ (tn,Dn,sln,t1,list(gr)) for (tn,Dn,sln),(t1,gr) in \
            zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])) ]
        # print(f"rank(dest): {len(jobs[0][0])} src: {data.size()} dest: {Dsize}")
        # for job in jobs:
        #     print(f"{job[0]} {job[1]} {len(job[4])}")
        newdata= fused_transpose_merge_1d.tm_forward_plain(data, order, jobs, Dsize)
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


class kernel_transpose_and_merge_plain_omp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, order, meta_new, meta_mrg, Dsize):
        ctx.order = order
        ctx.meta_new = meta_new
        ctx.meta_mrg = meta_mrg
        ctx.D_source = data.numel()

        # meta_new -> list of [(tn, Dn, sln), ...] where
        #             tn -> effective charge for block in fused tensor
        #             Dn -> effective shape of block tn in fused tensor
        #             sln -> slice specifying the location of serialized tn block in 1d data of fused tensor
        #
        # meta_mrg -> t1 is effective charge of source block after fusion. I.e. t1==tn, means, that
        #             this source block will belong to destination block tn
        #          -> gr: tuple holding description of source data
        #                 slo -> specifies the location of source block in 1d data
        #                 Do  -> shape of the source block
        #                 Dscl-> list of slice data which specifies the location of the "transformed"
        #                        source block in the destination block tn
        #                 Drsh-> the shape of the "transformed" source block in the destination block tn
        #
        jobs= [ (tn,Dn,sln,t1,list(gr)) for (tn,Dn,sln),(t1,gr) in \
            zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])) ]
        newdata= fused_transpose_merge_1d.tm_forward_plain_omp(data, order, jobs, Dsize)
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


class kernel_transpose_and_merge_p2p_v2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, order, meta_new, meta_mrg, Dsize):
        ctx.order = order
        ctx.meta_new = meta_new
        ctx.meta_mrg = meta_mrg
        ctx.D_source = data.numel()

        # meta_new -> list of [(tn, Dn, sln), ...] where
        #             tn -> effective charge for block in fused tensor
        #             Dn -> effective shape of block tn in fused tensor
        #             sln -> slice specifying the location of serialized tn block in 1d data of fused tensor
        #
        # meta_mrg -> t1 is effective charge of source block after fusion. I.e. t1==tn, means, that
        #             this source block will belong to destination block tn
        #          -> gr: tuple holding description of source data
        #                 slo -> specifies the location of source block in 1d data
        #                 Do  -> shape of the source block
        #                 Dscl-> list of slice data which specifies the location of the "transformed"
        #                        source block in the destination block tn
        #                 Drsh-> the shape of the "transformed" source block in the destination block tn
        #
        jobs= [ (tn,Dn,sln,t1,list(gr)) for (tn,Dn,sln),(t1,gr) in \
            zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])) ]
        # for row in jobs:
        #     print(f"{row[0]} {len(row[4])}")
        source_inds, dest_inds= fused_transpose_merge_1d.map_source_to_dest_plain_omp_v2(order,meta_new,meta_mrg)
        newdata= fused_transpose_merge_1d.forward_p2p_v2(data, source_inds, dest_inds, Dsize)
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


class kernel_transpose_and_merge_p2p_v3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, order, meta_new, meta_mrg, Dsize):
        # ctx.order = order
        # ctx.meta_new = meta_new
        # ctx.meta_mrg = meta_mrg
        ctx.D_source = data.numel()

        # meta_new -> list of [(tn, Dn, sln), ...] where
        #             tn -> effective charge for block in fused tensor
        #             Dn -> effective shape of block tn in fused tensor
        #             sln -> slice specifying the location of serialized tn block in 1d data of fused tensor
        #
        # meta_mrg -> t1 is effective charge of source block after fusion. I.e. t1==tn, means, that
        #             this source block will belong to destination block tn
        #          -> gr: tuple holding description of source data
        #                 slo -> specifies the location of source block in 1d data
        #                 Do  -> shape of the source block
        #                 Dscl-> list of slice data which specifies the location of the "transformed"
        #                        source block in the destination block tn
        #                 Drsh-> the shape of the "transformed" source block in the destination block tn
        #
        source_inds, dest_inds= fused_transpose_merge_1d.map_source_to_dest_v3(data, order, meta_new, meta_mrg)
        ctx.save_for_backward(source_inds, dest_inds)
        newdata= torch.zeros(Dsize,dtype=data.dtype, device=data.device,
            requires_grad=data.requires_grad)
        newdata[dest_inds]= data[source_inds]
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        # order = ctx.order
        # inv_order= tuple(np.argsort(order))
        # meta_new = ctx.meta_new
        # meta_mrg = ctx.meta_mrg
        D_source = ctx.D_source
        source_inds, dest_inds = ctx.saved_tensors

        newdata_b = torch.zeros((D_source,), dtype=data_b.dtype, device=data_b.device)
        # for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        #     assert tn == t1
        #     tmp_b = data_b[slice(*sln)].view(Dn)
        #     for (_, slo, Do, Dslc, _) in gr:
        #         slcs = tuple(slice(*x) for x in Dslc)
        #         inv_Do = tuple(Do[n] for n in order)
        #         newdata_b[slice(*slo)].reshape(Do)[:] = tmp_b[slcs].reshape(inv_Do).permute(inv_order)
        newdata_b[source_inds]= data_b[dest_inds]
        return newdata_b, None, None, None, None


def unmerge(data, meta):
    return kernel_unmerge_ptp.apply(data, meta)

class kernel_unmerge_ptp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, meta):
        Dsize = data.size()
        # ctx.meta = meta
        ctx.fwd_data_size = Dsize
        # sln -> slice in dest tensor, specifying location of unfused block
        # Dn  -> shape of the unfused block
        # slo -> slice in source tensor, specifying location of t_effective(fused) block
        # Do  -> shape of the fused block with t_eff
        # sub_slc -> sub-block within block Do
        source_inds= fused_transpose_merge_1d.map_source_to_dest_unmerge(Dsize[0], meta)
        ctx.save_for_backward(source_inds)

        newdata= data[source_inds]
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        # meta = ctx.meta
        fwd_data_size = ctx.fwd_data_size
        # no zero blocks should be introduces here
        newdata_b = torch.empty(fwd_data_size, dtype=data_b.dtype, device=data_b.device)
        # for sln, Dn, slo, Do, sub_slc in meta:
        #     slcs = tuple(slice(*x) for x in sub_slc)
        #     newdata_b[slice(*slo)].view(Do)[slcs] = data_b[slice(*sln)].view(Dn)
        source_inds = ctx.saved_tensors
        newdata_b[source_inds]= data_b
        return newdata_b, None, None, None
