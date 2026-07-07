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
import torch
from .backend_torch import *

BACKEND_ID = "torch_cpp"


def tensordot_bs(Adata, Bdata, *args):
    dtype = torch.promote_types(Bdata.dtype, Adata.dtype)
    if Adata.dtype != dtype:
        Adata = Adata.to(dtype=dtype)
    if Bdata.dtype != dtype:
        Bdata = Bdata.to(dtype=dtype)
    return torch.ops.tapp_torch.tensordot_bs(Adata, Bdata, *args)


def tensordot_dense(Adata, Bdata, DA, DB, nin_a, nin_b, modes_out):
    dtype = torch.promote_types(Bdata.dtype, Adata.dtype)
    if Adata.dtype != dtype:
        Adata = Adata.to(dtype=dtype)
    if Bdata.dtype != dtype:
        Bdata = Bdata.to(dtype=dtype)
    Adata = Adata.reshape(DA)
    Bdata = Bdata.reshape(DB)
    return torch.ops.tapp_torch.tensordot(Adata, Bdata, nin_a, nin_b, modes_out).ravel()
