import torch
from itertools import groupby
import merge_to_matrix_cpp_1d

def randR(D, device='cpu', dtype=torch.float64):
    return 2 * torch.rand(D, device=device, dtype=dtype) - 1

def _source_to_dest_v1(data, order, meta_new, meta_mrg):
    jobs= [ (tn,Dn,sln,t1,list(gr)) for (tn,Dn,sln),(t1,gr) in \
            zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])) ]
    return merge_to_matrix_cpp_1d.map_source_to_dest_v1(data, order, jobs)

def _source_to_dest_v2(data, order, meta_new, meta_mrg):
    jobs= [ (tn,Dn,sln,t1,list(gr)) for (tn,Dn,sln),(t1,gr) in \
            zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])) ]
    return merge_to_matrix_cpp_1d.map_source_to_dest_v2(data, order, jobs)

def _source_to_dest_plain(data, order, meta_new, meta_mrg):
    jobs= [ (tn,Dn,sln,t1,list(gr)) for (tn,Dn,sln),(t1,gr) in \
            zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])) ]
    return merge_to_matrix_cpp_1d.map_source_to_dest_plain(data, order, jobs)

def _source_to_dest_plain_omp(data, order, meta_new, meta_mrg):
    jobs= [ (tn,Dn,sln,t1,list(gr)) for (tn,Dn,sln),(t1,gr) in \
            zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])) ]
    return merge_to_matrix_cpp_1d.map_source_to_dest_plain_omp(data, order, jobs)

def _source_to_dest_plain_omp_v2(data, order, meta_new, meta_mrg):
    return merge_to_matrix_cpp_1d.map_source_to_dest_plain_omp_v2(data, order, meta_new, meta_mrg)

def _source_to_dest_plain_omp_v3(data, order, meta_new, meta_mrg):
    return merge_to_matrix_cpp_1d.map_source_to_dest_plain_omp_v3(data, order, meta_new, meta_mrg)


def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge.apply(data, order, meta_new, meta_mrg, Dsize)

class kernel_transpose_and_merge(torch.autograd.Function):
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

        newdata= merge_to_matrix_cpp_1d.forward_plain(data, order, jobs, Dsize)
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


def transpose_and_merge_omp(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge_omp.apply(data, order, meta_new, meta_mrg, Dsize)

class kernel_transpose_and_merge_omp(torch.autograd.Function):
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

        newdata= merge_to_matrix_cpp_1d.forward_plain_omp(data, order, jobs, Dsize)
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



def transpose_and_merge_ptp(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge_ptp.apply(data, order, meta_new, meta_mrg, Dsize)

class kernel_transpose_and_merge_ptp(torch.autograd.Function):
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
        source_to_dest= _source_to_dest_plain(data, order, meta_new, meta_mrg)
        newdata= merge_to_matrix_cpp_1d.forward_ptp(data, source_to_dest, Dsize)
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


def transpose_and_merge_ptp_omp(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge_ptp_omp.apply(data, order, meta_new, meta_mrg, Dsize)

class kernel_transpose_and_merge_ptp_omp(torch.autograd.Function):
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
        source_to_dest= _source_to_dest_plain_omp(data, order, meta_new, meta_mrg)
        newdata= merge_to_matrix_cpp_1d.forward_ptp(data, source_to_dest, Dsize)
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


def transpose_and_merge_ptp_omp_v2(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge_ptp_omp_v2.apply(data, order, meta_new, meta_mrg, Dsize)

class kernel_transpose_and_merge_ptp_omp_v2(torch.autograd.Function):
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
        source_inds, dest_inds= _source_to_dest_plain_omp_v3(data, order, meta_new, meta_mrg)
        newdata= merge_to_matrix_cpp_1d.forward_ptp_v2(data, source_inds, dest_inds, Dsize)
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

