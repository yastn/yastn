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
import numpy as np
import torch

from .backend_torch import *

# import os
# import sys
# from cuda_blocksparse import some_feature
# # from .torch_cpp_ext._backend_torch_transpose_and_merge import *
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../experimental/cuda_blocksparse'))

import cublocksparse

BACKEND_ID = "torch_cpp"

def kernel_tensordot_bs(
        a, b, 
        a_struct_t,     # non-zero blocks of a indexed via charges
        a_t_per_mode,   # list of charges for each leg of a
        a_D_per_mode,   # list of Ds for each leg of a
        nout_a, nin_a,
        b_struct_t, 
        b_t_per_mode,   # list of charges for each leg of b
        b_D_per_mode,   # list of Ds for each leg of b
        nout_b, nin_b,
        c_size, c_struct_t   # non-zero blocks of c indexed via charges
    ):
    # pre-process _t_per_mode such that each mode (formally) contains all t's
    Ts= set()
    for l in a_t_per_mode+b_t_per_mode:
        Ts= Ts | set(l)
    Ts= sorted(list(Ts))
    
    # pre-process _D_per_mode
    # However, sectors on contracted modes must share extents.
    filled_a_t_per_mode= [ tuple(Ts) ]*len(nout_a+nin_a) 
    filled_b_t_per_mode= [ tuple(Ts) ]*len(nout_b+nin_b)
    filled_c_t_per_mode= [ tuple(Ts) ]*len(nout_a+nout_b)
    filled_a_D_per_mode= [None]*len(nout_a+nin_a)
    filled_b_D_per_mode= [None]*len(nout_b+nin_b)
    filled_c_D_per_mode= [None]*len(nout_a+nout_b)
    for i_c,i in enumerate(nout_a): # fill extends for charges originally not in outgoing legs by zero
        _tmp= dict(zip(a_t_per_mode[i],a_D_per_mode[i]))
        filled_a_D_per_mode[i]= [ _tmp[t] if t in _tmp.keys() else 1 for t in filled_a_t_per_mode[i] ]
        filled_c_D_per_mode[i_c]= filled_a_D_per_mode[i]
    for i_c,i in enumerate(nout_b): # fill extends for charges originally not in outgoing legs by zero
        _tmp= dict(zip(b_t_per_mode[i],b_D_per_mode[i]))
        filled_b_D_per_mode[i]= [ _tmp[t] if t in _tmp.keys() else 1 for t in filled_b_t_per_mode[i] ]
        filled_c_D_per_mode[i_c+len(nout_a)]= filled_b_D_per_mode[i]
    for i_a,i_b in zip(nin_a,nin_b):
        _tmp_a= dict(zip(a_t_per_mode[i_a],a_D_per_mode[i_a]))
        _tmp_b= dict(zip(b_t_per_mode[i_b],b_D_per_mode[i_b]))
        _tmp= _tmp_a | _tmp_b
        filled_a_D_per_mode[i_a]= [ _tmp[t] if t in _tmp.keys() else 1 for t in filled_a_t_per_mode[i_a] ]
        filled_b_D_per_mode[i_b]= [ _tmp[t] if t in _tmp.keys() else 1 for t in filled_b_t_per_mode[i_b] ]

    # TODO for NSYM>1 extra nesting
    def _blocksparse_coords(struct_t, t_per_mode):
        return [ [ t_per_mode[i].index( (c,) ) for i,c in enumerate(row) ] for row in struct_t ]

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

    return cublocksparse.ops.tensordot_bs(
            a, b, 
            a_coords, T_a_D_per_mode, nout_a, nin_a,
            b_coords, T_b_D_per_mode, nout_b, nin_b,
            c_size, c_coords, T_c_D_per_mode
        )

def _ORIG_kernel_tensordot_bs(
        a, b, 
        a_struct_t,     # non-zero blocks of a indexed via charges
        a_t_per_mode,   # list of charges for each leg of a
        a_D_per_mode,   # list of Ds for each leg of a
        nout_a, nin_a,
        b_struct_t, 
        b_t_per_mode,   # list of charges for each leg of b
        b_D_per_mode,   # list of Ds for each leg of b
        nout_b, nin_b,
        c_size, c_struct_t   # non-zero blocks of a indexed via charges
    ):
    # TODO for NSYM>1 extra nesting
    def _blocksparse_coords(struct_t, t_per_mode):
        return [ [ t_per_mode[i].index( (c,) ) for i,c in enumerate(row) ] for row in struct_t ]

    # c_t_per_mode, c_D_per_mode can be obtained from per_mode info of a and b
    c_t_per_mode= [ a_t_per_mode[i] for i in nout_a ] + [ b_t_per_mode[i] for i in nout_b ]
    c_D_per_mode= [ a_D_per_mode[i] for i in nout_a ] + [ b_D_per_mode[i] for i in nout_b ]

    # for compliance with PyTorch Custom OPS API
    # a) as int64 tensors
    # a_coords = torch.tensor(_blocksparse_coords(a_struct_t, a_t_per_mode), dtype=torch.int64, device=device)
    # b_coords = torch.tensor(_blocksparse_coords(b_struct_t, b_t_per_mode), dtype=torch.int64, device=device)
    # c_coords = torch.tensor(_blocksparse_coords(c_struct_t, c_t_per_mode), dtype=torch.int64, device=device)
    # b) as lists of lists of indices flattened to 1D (which is expected format for cutensor's block-sparse API)
    a_coords = sum(_blocksparse_coords(a_struct_t, a_t_per_mode), start=[])
    b_coords = sum(_blocksparse_coords(b_struct_t, b_t_per_mode), start=[])
    c_coords = sum(_blocksparse_coords(c_struct_t, c_t_per_mode), start=[])

    def _pad_and_convert(D_per_mode):
        max_len = max(len(sublist) for sublist in D_per_mode)
        padded_D_per_mode = [list(sublist) + [-1] * (max_len - len(sublist))
                            for sublist in D_per_mode]
        return torch.tensor(padded_D_per_mode, dtype=torch.int64, device='cpu')
    T_a_D_per_mode= _pad_and_convert(a_D_per_mode)
    T_b_D_per_mode= _pad_and_convert(b_D_per_mode)
    T_c_D_per_mode= _pad_and_convert(c_D_per_mode)

    return cublocksparse.ops.tensordot_bs(
            a, b, 
            a_coords, T_a_D_per_mode, nout_a, nin_a,
            b_coords, T_b_D_per_mode, nout_b, nin_b,
            c_size, c_coords, T_c_D_per_mode
        )

# def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
#    return kernel_transpose_and_merge_p2p_v3.apply(data, order, meta_new, meta_mrg, Dsize)

# def unmerge(data, meta):
#    return kernel_unmerge_ptp.apply(data, meta)

