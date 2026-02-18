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
from itertools import accumulate, groupby
import operator
from typing import Sequence, Union
import numpy as np
import torch
from .backend_torch import *
import tapp_torch

BACKEND_ID = "torch_cpp"


def _meta_tapp_torch_tensordot_bs(
        NSYM: int,
        a_struct_t : Sequence[Sequence[int]], # non-zero blocks of a indexed via charges (for product groups, the charges are flattened)
        a_slices,   # slices of a, i.e. the locations of the non-zero blocks in the 1D tensor a
        a_t_per_mode : Union[Sequence[Sequence[int]], Sequence[Sequence[Sequence[int]]]] , # list of charges for each leg of a (for product groups, the charges are not flattened)
        a_D_per_mode : Sequence[Sequence[int]],   # list of Ds for each leg of a
        nout_a : Sequence[int], nin_a : Sequence[int], # outgoing and contracted legs of a
        b_struct_t : Sequence[Sequence[int]], # non-zero blocks of a indexed via charges (for product groups, the charges are flattened),
        b_slices,   # slices of a, i.e. the locations of the non-zero blocks in the 1D tensor b
        b_t_per_mode : Union[Sequence[Sequence[int]], Sequence[Sequence[Sequence[int]]]],   # list of charges for each leg of b
        b_D_per_mode : Sequence[Sequence[int]],   # list of Ds for each leg of b
        nout_b, nin_b,
        c_struct_t, c_slices,  # non-zero blocks of c indexed via charges
        profile=False):
    r"""
    Prepare metadata for block-sparse tensordot.

    Returns:
        meta_dot: metadata for cutensor block-sparse tensordot
    """
    # I. Merge sectors for contracted modes
    #
    # Sectors on contracted modes must share extents. Hence, we need to fill sectors (and their extents)
    # which are present in only one of the operands.
    # pre-process _t_per_mode such that each mode (formally) contains all t's
    def _merge_t_v1(a_idx, b_idx):
        # Merge two sorted lists in O(n+m) time
        a_list = a_t_per_mode[a_idx]
        b_list = b_t_per_mode[b_idx]
        result = []
        i, j = 0, 0
        while i < len(a_list) and j < len(b_list):
            if a_list[i] < b_list[j]:
                result.append(a_list[i])
                i += 1
            elif a_list[i] > b_list[j]:
                result.append(b_list[j])
                j += 1
            else:
                result.append(a_list[i])
                i += 1
                j += 1
        result.extend(a_list[i:])
        result.extend(b_list[j:])
        return result

    filled_a_t_per_mode= list(a_t_per_mode)
    filled_b_t_per_mode= list(b_t_per_mode)

    for ia,ib in zip(nin_a,nin_b):
        if filled_a_t_per_mode[ia]==filled_b_t_per_mode[ib]:
            continue
        merged_t= _merge_t_v1(ia,ib)
        filled_a_t_per_mode[ia]= merged_t
        filled_b_t_per_mode[ib]= merged_t
    filled_c_t_per_mode= [ a_t_per_mode[i] for i in nout_a ] + [ b_t_per_mode[i] for i in nout_b ]
    if profile: torch.cuda.nvtx.mark("kernel_tensordot_bs _merge_t done")
    
    filled_a_D_per_mode= [None]*len(nout_a+nin_a)
    filled_b_D_per_mode= [None]*len(nout_b+nin_b)
    filled_c_D_per_mode= [None]*len(nout_a+nout_b)
    for i_c,i in enumerate(nout_a): # fill extents outgoing legs of a
        filled_c_D_per_mode[i_c]= filled_a_D_per_mode[i]= a_D_per_mode[i]
    for i_c,i in enumerate(nout_b): # fill extents outgoing legs of b
        filled_c_D_per_mode[i_c+len(nout_a)]= filled_b_D_per_mode[i]= b_D_per_mode[i]
    for i_a,i_b in zip(nin_a,nin_b): # fill extents of union of sectors present in contracted legs of a and b
        _tmp_a= dict(zip(a_t_per_mode[i_a],a_D_per_mode[i_a]))
        _tmp_b= dict(zip(b_t_per_mode[i_b],b_D_per_mode[i_b]))
        _tmp= _tmp_a | _tmp_b
        filled_a_D_per_mode[i_a]= [ _tmp[t] for t in filled_a_t_per_mode[i_a] ]
        filled_b_D_per_mode[i_b]= [ _tmp[t] for t in filled_b_t_per_mode[i_b] ]
    if profile: torch.cuda.nvtx.mark("kernel_tensordot_bs filled_*_D_per_mode done")

    # II. Domain-agnostic indexing of block coordinates
    #
    # The blocks are indexed by their charges. However, cutensor expects non-negative integers as block coordinates. 
    # Here we convert the charge-based coordinates to unsigned-int block coordinates. 
    # The conversion is done in two steps:
    #   1) For each mode, we shift the coordinates such that the minimum coordinate becomes zero.
    #   2) For each mode, we raise the coordinates by the base equal to the number of unique coordinates in that mode.
    #
    # Linearization of block coordinates
    # NOTE: Block index can have value at most equal to number of extents in the respective mode - 1
    def normalize_ts(filled_t_per_mode):
        res= []
        for t_mode in filled_t_per_mode:
            tm= np.asarray(t_mode)
            floor= np.min(tm, axis=0)

            idx2i= np.empty( np.max(tm,axis=0)-floor+1, dtype=np.int64 )
            for i, idx in enumerate(tm):
                idx2i[tuple(idx - floor)]= i

            res.append( (floor, list(accumulate( np.max(tm, axis=0)-floor+1, operator.mul )), idx2i) )
        return res

    def _blocksparse_coords_v3(struct_t, filled_t_per_mode):
        ts= np.array(struct_t).reshape( len(struct_t), len(filled_t_per_mode), max(1,NSYM) ) # NSYM=0 is treated as NSYM=1 with 0 sector charge only
        n= normalize_ts(filled_t_per_mode)
        ts-= np.stack([f[0] for f in n])                # shift to zero-based [:,...] -= [...] broadcast over :
        # ts[...,1:]*= np.stack([f[1] for f in n])[:,:-1] # raise by base
        # ts= np.sum(ts,axis=-1).tolist()              # compute linearized indices
        B= np.empty( ts.shape[:2], dtype=np.int64 )
        for mode in range(len(filled_t_per_mode)):
            B[:,mode]= n[mode][2][ tuple( ts[:,mode,i] for i in range(max(1,NSYM)) ) ]

        return B.reshape(-1).tolist()

    # For compliance with PyTorch Custom OPS API
    # lists of lists of indices flattened to 1D (which is expected format for cutensor's block-sparse API)
    a_coords = _blocksparse_coords_v3(a_struct_t, filled_a_t_per_mode)
    b_coords = _blocksparse_coords_v3(b_struct_t, filled_b_t_per_mode)
    c_coords = _blocksparse_coords_v3(c_struct_t, filled_c_t_per_mode)
    if profile: torch.cuda.nvtx.mark("kernel_tensordot_bs _blocksparse_coords done")

    # III. Compute offsets and strides for each block
    #
    # The offsets and strides are needed to map the block-sparse coordinates to the respective locations in the 1D tensors a, b, c. 
    # 
    def _offsets_and_strides(slices):
        S= np.empty( (len(slices),len(slices[0].D)), dtype=np.int64 )
        S[:,-1]=1
        Ds= np.asarray( tuple(b.D for b in slices) )
        np.cumprod( Ds[:, -1:0:-1], axis=-1, out= S[:,:len(slices[0].D)-1][:,::-1])
        return tuple(s.slcs[0][0] for s in slices), S.reshape(-1).tolist()

    a_offsets, a_strides= _offsets_and_strides(a_slices)
    b_offsets, b_strides= _offsets_and_strides(b_slices)
    c_offsets, c_strides= _offsets_and_strides(c_slices)
    if profile: torch.cuda.nvtx.mark("kernel_tensordot_bs strides_and_offsets done")

    def _convert_extents(filled_D_per_mode):
        numSectionsPerMode= list([len(m) for m in filled_D_per_mode])
        sectionExtents_flat= sum([ list(m) for m in filled_D_per_mode ], start=[])
        return numSectionsPerMode, sectionExtents_flat

    a_numSectionsPerMode, a_sectionExtents= _convert_extents(filled_a_D_per_mode)
    b_numSectionsPerMode, b_sectionExtents= _convert_extents(filled_b_D_per_mode)
    c_numSectionsPerMode, c_sectionExtents= _convert_extents(filled_c_D_per_mode)
    if profile: torch.cuda.nvtx.mark("kernel_tensordot_bs _pad_and_convert done")

    return a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets, \
        b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets, \
        c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets


def kernel_tensordot_bs(
        a: torch.Tensor, b: torch.Tensor,
        NSYM: int,
        a_struct_t : Sequence[Sequence[int]], # non-zero blocks of a indexed via charges (for product groups, the charges are flattened)
        a_slices,   # slices of a, i.e. the locations of the non-zero blocks in the 1D tensor a
        a_t_per_mode : Union[Sequence[Sequence[int]], Sequence[Sequence[Sequence[int]]]] , # list of charges for each leg of a (for product groups, the charges are not flattened)
        a_D_per_mode : Sequence[Sequence[int]],   # list of Ds for each leg of a
        nout_a : Sequence[int], nin_a : Sequence[int], # outgoing and contracted legs of a
        b_struct_t : Sequence[Sequence[int]], # non-zero blocks of a indexed via charges (for product groups, the charges are flattened),
        b_slices,   # slices of a, i.e. the locations of the non-zero blocks in the 1D tensor b
        b_t_per_mode : Union[Sequence[Sequence[int]], Sequence[Sequence[Sequence[int]]]],   # list of charges for each leg of b
        b_D_per_mode : Sequence[Sequence[int]],   # list of Ds for each leg of b
        nout_b, nin_b,
        c_size, c_struct_t, c_slices,   # non-zero blocks of c indexed via charges
        profile=False
    ):
    dtype = torch.promote_types(a.dtype, b.dtype)
    if c_size==0:
        return torch.zeros(c_size, dtype=dtype, device=a.device)

    if profile: torch.cuda.nvtx.range_push("_meta_tensordot_bs")
    meta= _meta_tapp_torch_tensordot_bs(NSYM,
        a_struct_t, a_slices, a_t_per_mode, a_D_per_mode,
        nout_a, nin_a,
        b_struct_t, b_slices, b_t_per_mode, b_D_per_mode,
        nout_b, nin_b,
        c_struct_t, c_slices, profile)
    if profile: torch.cuda.nvtx.range_pop()

    if profile: torch.cuda.nvtx.range_push(f"ops.tensordot_bs")
    # TODO type promotion should be handled by cutensor
    if a.dtype != dtype:
        a = a.to(dtype=dtype)
    if b.dtype != dtype:
        b = b.to(dtype=dtype)

    def _remap_nout_(nout,offset):
        r_nout= [None]*len(nout)
        for n,i in enumerate(np.argsort(nout)):
            r_nout[i]= n + offset
        return r_nout

    modes_out= _remap_nout_(nout_a,0) + _remap_nout_(nout_b,len(nout_a))

    res= torch.ops.tapp_torch.tensordot_bs(a, b, nin_a, nin_b,
        *meta,
        modes_out)

    if profile: torch.cuda.nvtx.range_pop()
    return res
