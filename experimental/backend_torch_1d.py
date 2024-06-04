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
import torch
import numpy as np
from itertools import groupby

def set_num_threads(num_threads):
    torch.set_num_threads(num_threads)

def randR(D, device='cpu'):
    return 2 * torch.rand(D, device=device) - 1


def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge.apply(data, order, meta_new, \
        meta_mrg, Dsize)

class kernel_transpose_and_merge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, order, meta_new, meta_mrg, Dsize):
        ctx.order = order
        ctx.meta_new = meta_new
        ctx.meta_mrg = meta_mrg
        ctx.D_source = data.numel()

        # Dsize - total size of fused representation (might include some zero-blocks)
        newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)

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


from math import prod
def get_strides(D):
    # D - Sequence(int)
    #
    # returns
    # Sequence(int) - row-major strides, starting with total number of elements
    #
    # v. a
    res= [1]
    for d in reversed(D):
        res += [res[-1]*d]
    return list(reversed(res))
    #
    # v. b
    # return [ prod(D[i:]) for i in range(len(D)+1) ]

def get_indices(i,strides):
    # i - integer index of element in flattened array
    # strides - Sequence(int) - strides wrt. to which compute indices
    #
    # returns
    # Sequence(int) - indices in tensors specified by strides
    return [ (i%strides[d])//strides[d+1] for d in range(len(strides)-1) ]

def index_1d(X,strides):
    # X - Sequence(int) - indices of element wrt. to strides
    # strides - Sequence(int) - strides
    #
    # returns
    # int - index of element in underlying 1d row-major array
    return sum([X[d]*strides[d+1] for d in range(len(X))])

def i_source_to_dest(i,offset_source,strides_Do,order,strides_perm,strides_reshaped,\
    Dslc,strides_Dn,offset_dest):
    # 0) subtract offset
    i=i-offset_source
    # i) map from source 1d-array (specified by slice slo) to tuple X=(x_0,...,x_n)
    #    specifying location inside Do shape
    X= get_indices(i, strides_Do)
    # ii) permute
    #
    X_perm= [X[d] for d in order]
    i_perm= index_1d(X_perm, strides_perm)
    # iii) indices in (permute+reshape)d source
    #
    X_reshaped= get_indices(i_perm,strides_reshaped)
    # iv) indices in reshaped destination
    #
    X_dest_block= [ X_reshaped[d]+Dslc[d][0] for d in range(len(Dslc)) ]
    i_dest_block= index_1d(X_dest_block, strides_Dn)
    i_dest= i_dest_block + offset_dest
    return i_dest

def _source_to_dest_v1(data, order, meta_new, meta_mrg):
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
    source_to_dest= torch.arange(data.numel(), dtype=torch.int64, device='cpu')
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        strides_Dn= get_strides(Dn)
        for (_, slo, Do, Dslc, Drsh) in gr:
            # prelim)
            # get strides of shape Do, strides of shape permute(Do; order)
            strides_Do= get_strides(Do)
            strides_perm= get_strides([Do[d] for d in order])
            strides_reshaped= get_strides(Drsh)

            # source_to_dest[slice(*slo)]= torch.as_tensor([i_source_to_dest(i) for i in range(slo[0],slo[1])], \
            #     dtype=torch.int64, device='cpu')
            source_to_dest[slice(*slo)]= torch.as_tensor([i_source_to_dest(i,slo[0],strides_Do,order,\
                strides_perm,strides_reshaped,Dslc,strides_Dn,sln[0]) for i in range(slo[0],slo[1])], dtype=torch.int64, device='cpu')
            # for i in range(slo[0],slo[1]):
            #     source_to_dest[i]= i_source_to_dest(i)
            # source_to_dest[slice(*slo)].apply_(i)
    return source_to_dest

def _source_to_dest_v2(data, order, meta_new, meta_mrg):
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
    source_to_dest= torch.arange(data.numel(), dtype=torch.int64, device='cpu')
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        strides_Dn= get_strides(Dn)
        for (_, slo, Do, Dslc, Drsh) in gr:
            # prelim)
            # get strides of shape Do, strides of shape permute(Do; order)
            strides_Do= get_strides(Do)
            strides_perm= get_strides([Do[d] for d in order])
            strides_reshaped= get_strides(Drsh)

            def local_i_source_to_dest(i):
                # 0) subtract offset
                i=i-slo[0]
                # i) map from source 1d-array (specified by slice slo) to tuple X=(x_0,...,x_n)
                #    specifying location inside Do shape
                X= get_indices(i, strides_Do)
                # ii) permute
                #
                X_perm= [X[d] for d in order]
                i_perm= index_1d(X_perm, strides_perm)
                # iii) indices in (permute+reshape)d source
                #
                X_reshaped= get_indices(i_perm,strides_reshaped)
                # iv) indices in reshaped destination
                #
                X_dest_block= [ X_reshaped[d]+Dslc[d][0] for d in range(len(Dslc)) ]
                i_dest_block= index_1d(X_dest_block, strides_Dn)
                i_dest= i_dest_block + sln[0]
                return i_dest

            # source_to_dest[slice(*slo)]= torch.as_tensor([i_source_to_dest(i) for i in range(slo[0],slo[1])], \
            #     dtype=torch.int64, device='cpu')
            source_to_dest[slice(*slo)]= torch.as_tensor(\
                [local_i_source_to_dest(i) for i in range(slo[0],slo[1])], \
                dtype=torch.int64, device='cpu')
            # for i in range(slo[0],slo[1]):
            #     source_to_dest[i]= i_source_to_dest(i)
            # source_to_dest[slice(*slo)].apply_(i)
    return source_to_dest

def _source_to_dest_np(data, order, meta_new, meta_mrg):
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
    inv_order= tuple(np.argsort(order))
    source_to_dest = np.empty((data.numel(),), dtype=np.int64)
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        assert tn == t1
        tmp_b = np.arange(sln[0], sln[1], dtype=np.int64).reshape(Dn)
        for (_, slo, Do, Dslc, _) in gr:
            slcs = tuple(slice(*x) for x in Dslc)
            inv_Do = tuple(Do[n] for n in order)
            source_to_dest[slice(*slo)].reshape(Do)[:] = tmp_b[slcs].reshape(inv_Do).transpose(inv_order)
    return source_to_dest


def transpose_and_merge_ptp(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge_ptp.apply(data, order, meta_new, \
        meta_mrg, Dsize)

class kernel_transpose_and_merge_ptp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, order, meta_new, meta_mrg, Dsize):
        ctx.order = order
        ctx.meta_new = meta_new
        ctx.meta_mrg = meta_mrg
        ctx.D_source = data.numel()

        # Dsize - total size of fused representation (might include some zero-blocks)
        newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
        source_to_dest= _source_to_dest_np(data,order,meta_new,meta_mrg)
        newdata.scatter_(0,torch.as_tensor(source_to_dest),data)

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
        return newdata_b, None, None, None, None, None
