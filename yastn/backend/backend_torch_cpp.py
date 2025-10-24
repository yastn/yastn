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
from typing import Sequence, Union
import numpy as np
import torch

from .backend_torch import *
import cublocksparse

BACKEND_ID = "torch_cpp"

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
        c_size, c_struct_t,   # non-zero blocks of c indexed via charges
        c_slices
    ):
    dtype = torch.promote_types(a.dtype, a.dtype)
    if c_size==0:
        return torch.zeros(c_size, dtype=dtype, device=a.device)

    # Sectors on contracted modes must share extents. Hence, we need to fill sectors (and their extents) 
    # which are present in only one of the operands.
    # pre-process _t_per_mode such that each mode (formally) contains all t's
    
    # Ts= set()
    # for l in [a_t_per_mode[i] for i in nin_a]+[b_t_per_mode[j] for j in nin_b]:
    #     Ts= Ts | set(l)
    # Ts= sorted(list(Ts))
    _merge_t = lambda x,y: sorted(list( set(a_t_per_mode[x]) | set( b_t_per_mode[y] )))

    # pre-process _D_per_mode
    filled_a_t_per_mode= [ a_t_per_mode[i] if i in nout_a else _merge_t(i,nin_b[nin_a.index(i)]) for i in range(len(nout_a+nin_a)) ]
    filled_b_t_per_mode= [ b_t_per_mode[i] if i in nout_b else _merge_t(nin_a[nin_b.index(i)],i) for i in range(len(nout_b+nin_b)) ]
    filled_c_t_per_mode= [ a_t_per_mode[i] for i in nout_a ] + [ b_t_per_mode[i] for i in nout_b ]
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

    # TODO for NSYM>1 extra nesting
    def _blocksparse_coords(struct_t, t_per_mode):
        if NSYM>0:
            return [ [ t_per_mode[i].index( row[i*NSYM:(i+1)*NSYM] ) for i in range(len(t_per_mode)) ] \
                for row in struct_t ]
        return [ [ t_per_mode[i].index( (row[i],) ) for i in range(len(t_per_mode)) ] \
                for row in struct_t ]


    # c_t_per_mode, c_D_per_mode can be obtained from per_mode info of a and b
    c_t_per_mode= [ a_t_per_mode[i] for i in nout_a ] + [ b_t_per_mode[i] for i in nout_b ]
    c_D_per_mode= [ a_D_per_mode[i] for i in nout_a ] + [ b_D_per_mode[i] for i in nout_b ]

    # for compliance with PyTorch Custom OPS API
    # a) as int64 tensors
    # a_coords = torch.tensor(_blocksparse_coords(a_struct_t, a_t_per_mode), dtype=torch.int64, device=device)
    # b_coords = torch.tensor(_blocksparse_coords(b_struct_t, b_t_per_mode), dtype=torch.int64, device=device)
    # c_coords = torch.tensor(_blocksparse_coords(c_struct_t, c_t_per_mode), dtype=torch.int64, device=device)
    # b) as lists of lists of indices flattened to 1D (which is expected format for cutensor's block-sparse API)
    a_coords = sum(_blocksparse_coords(a_struct_t, filled_a_t_per_mode), start=[])
    b_coords = sum(_blocksparse_coords(b_struct_t, filled_b_t_per_mode), start=[])
    c_coords = sum(_blocksparse_coords(c_struct_t, filled_c_t_per_mode), start=[])

    def _pad_and_convert(D_per_mode):
        max_len = max(len(sublist) for sublist in D_per_mode)
        padded_D_per_mode = [list(sublist) + [-1] * (max_len - len(sublist))
                            for sublist in D_per_mode]
        return torch.tensor(padded_D_per_mode, dtype=torch.int64, device='cpu')
    T_a_D_per_mode= _pad_and_convert(filled_a_D_per_mode)
    T_b_D_per_mode= _pad_and_convert(filled_b_D_per_mode)
    T_c_D_per_mode= _pad_and_convert(filled_c_D_per_mode)

    # offsets & strides
    _get_strides= lambda D: list(np.cumprod(D[::-1])[::-1][1:])+[1,]
    a_offsets, a_strides= zip(*((s.slcs[0][0], _get_strides(s.D)) for s in a_slices))
    b_offsets, b_strides= zip(*((s.slcs[0][0], _get_strides(s.D)) for s in b_slices))
    c_offsets, c_strides= zip(*((s.slcs[0][0], _get_strides(s.D)) for s in c_slices))
    # serialize to 1D
    a_strides= sum(a_strides, start=[])
    b_strides= sum(b_strides, start=[])
    c_strides= sum(c_strides, start=[])

    # import pdb; pdb.set_trace()  # DEBUG
    return cublocksparse.ops.tensordot_bs(
            a, b, 
            a_coords, a_offsets, a_strides, T_a_D_per_mode, nout_a, nin_a,
            b_coords, b_offsets, b_strides, T_b_D_per_mode, nout_b, nin_b,
            c_size, c_coords, c_offsets, c_strides, T_c_D_per_mode
        )
