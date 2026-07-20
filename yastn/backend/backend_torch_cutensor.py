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
from typing import Sequence, Union
import numpy as np
import torch
import tapp_torch
from .._profile import nsys_profile
from .backend_torch import *


BACKEND_ID = "torch_cutensor"


def tensordot_bs(Adata, Bdata, *args):
    dtype = torch.promote_types(Bdata.dtype, Adata.dtype)
    if Adata.dtype != dtype:
        Adata = Adata.to(dtype=dtype)
    if Bdata.dtype != dtype:
        Bdata = Bdata.to(dtype=dtype)
    return torch.ops.tapp_torch.tensordot_bs(Adata, Bdata, *args)

@nsys_profile
def tensordot_dense(Adata, Bdata, DA, DB, nin_a, nin_b, modes_out):
    dtype = torch.promote_types(Bdata.dtype, Adata.dtype)
    if Adata.dtype != dtype:
        Adata = Adata.to(dtype=dtype)
    if Bdata.dtype != dtype:
        Bdata = Bdata.to(dtype=dtype)
    Adata = Adata.reshape(DA)
    Bdata = Bdata.reshape(DB)
    return torch.ops.tapp_torch.tensordot(Adata, Bdata, nin_a, nin_b, modes_out).reshape(-1)


@nsys_profile
def tensordot_bs_v2(
    a: torch.Tensor, b: torch.Tensor,
    nin_a: Sequence[int], nin_b: Sequence[int],
    a_numSectionsPerMode : Sequence[int], 
    a_sectionExtents : Sequence[int], # flattened list of section extents for each mode of a
    a_coords : Union[torch.Tensor, np.ndarray], # 1D int64, flattened non-zero block coordinates
    a_strides: Union[torch.Tensor, np.ndarray], # 1D int64, flattened strides for each non-zero block
    a_offsets: Union[torch.Tensor, np.ndarray], # int64
    b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets,
    c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets,
    modes_out: Sequence[int]):

    # from_numpy is no-copy
    # contiguous is no-op if the tensor is already contiguous
    def _as_tensor(v):
        return torch.from_numpy(v).contiguous() if isinstance(v, np.ndarray) else v

    a_coords, a_strides, a_offsets = _as_tensor(a_coords), _as_tensor(a_strides), _as_tensor(a_offsets)
    b_coords, b_strides, b_offsets = _as_tensor(b_coords), _as_tensor(b_strides), _as_tensor(b_offsets)
    c_coords, c_strides, c_offsets = _as_tensor(c_coords), _as_tensor(c_strides), _as_tensor(c_offsets)

    dtype = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=dtype)
    b = b.to(dtype=dtype)

    # Signature 
    #
    # A: Tensor, B: Tensor,
    # contracted_modes_A: List[int], contracted_modes_B: List[int],
    # a_numSectionsPerMode: List[int], a_sectionExtents: List[int],
    # a_blocks: Tensor, a_strides: Tensor, a_offsets: Tensor,
    # b_numSectionsPerMode: List[int], b_sectionExtents: List[int],
    # b_blocks: Tensor, b_strides: Tensor, b_offsets: Tensor,
    # c_numSectionsPerMode: List[int], c_sectionExtents: List[int],
    # c_blocks: Tensor, c_strides: Tensor, c_offsets: Tensor,
    # modes_out: Optional[List[int]] = None
    #
    # with all but A,B Tensor arguments being CPU Tensors of int64
    #
    res = torch.ops.tapp_torch.tensordot_bs_v2(a, b, nin_a, nin_b,
        a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets,
        b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets,
        c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets,
        modes_out)
    return res