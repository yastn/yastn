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
import numpy as np
import pytest
import yastn
from yastn.tensor._auxliary import _config
try:
    from .configs import config_U1xU1_fermionic
except ImportError:
    from configs import config_U1xU1_fermionic

tol = 1e-12  #pylint: disable=invalid-name


@pytest.mark.skipif(config_U1xU1_fermionic.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_bug_conj_transpose():
    loc_c = config_U1xU1_fermionic
    loc_c = loc_c if isinstance(loc_c, _config) else _config(**{a: getattr(loc_c, a) for a in _config._fields if hasattr(loc_c, a)})
    loc_c = loc_c._replace(default_dtype="complex128", fermionic=False)
    t1_data = np.random.randn(2157759) + 1.0j * np.random.randn(2157759)
    R = yastn.load_from_dict(loc_c,\
        {'_d': t1_data,'s': (-1, -1), 'n': (0, 0), 't': ((-8, 0, 8, 0), (-7, -3, 7, 3), (-7, 3, 7, -3), (-6, -6, 6, 6), (-6, 0, 6, 0), (-6, 6, 6, -6), (-5, -9, 5, 9), (-5, -3, 5, 3), (-5, 3, 5, -3), (-5, 9, 5, -9), (-4, -12, 4, 12), (-4, -6, 4, 6), (-4, 0, 4, 0), (-4, 6, 4, -6), (-4, 12, 4, -12), (-3, -9, 3, 9), (-3, -3, 3, 3), (-3, 3, 3, -3), (-3, 9, 3, -9), (-2, -12, 2, 12), (-2, -6, 2, 6), (-2, 0, 2, 0), (-2, 6, 2, -6), (-2, 12, 2, -12), (-1, -9, 1, 9), (-1, -3, 1, 3), (-1, 3, 1, -3), (-1, 9, 1, -9), (0, -12, 0, 12), (0, -6, 0, 6), (0, 0, 0, 0), (0, 6, 0, -6), (0, 12, 0, -12), (1, -9, -1, 9), (1, -3, -1, 3), (1, 3, -1, -3), (1, 9, -1, -9), (2, -12, -2, 12), (2, -6, -2, 6), (2, 0, -2, 0), (2, 6, -2, -6), (2, 12, -2, -12), (3, -9, -3, 9), (3, -3, -3, 3), (3, 3, -3, -3), (3, 9, -3, -9), (4, -12, -4, 12), (4, -6, -4, 6), (4, 0, -4, 0), (4, 6, -4, -6), (4, 12, -4, -12), (5, -9, -5, 9), (5, -3, -5, 3), (5, 3, -5, -3), (5, 9, -5, -9), (6, -6, -6, 6), (6, 0, -6, 0), (6, 6, -6, -6), (7, -3, -7, 3), (7, 3, -7, -3), (8, 0, -8, 0)), 'D': ((1, 1), (4, 4), (4, 4), (6, 6), (24, 24), (6, 6), (4, 4), (52, 52), (52, 52), (4, 4), (1, 1), (52, 52), (160, 160), (52, 52), (1, 1), (24, 24), (228, 228), (228, 228), (24, 24), (4, 4), (160, 160), (456, 456), (160, 160), (4, 4), (52, 52), (456, 456), (456, 456), (52, 52), (6, 6), (228, 228), (639, 639), (228, 228), (6, 6), (52, 52), (456, 456), (456, 456), (52, 52), (4, 4), (160, 160), (456, 456), (160, 160), (4, 4), (24, 24), (228, 228), (228, 228), (24, 24), (1, 1), (52, 52), (160, 160), (52, 52), (1, 1), (4, 4), (52, 52), (52, 52), (4, 4), (6, 6), (24, 24), (6, 6), (4, 4), (4, 4), (1, 1)), 'isdiag': False, 'mfs': ((1,), (1,)), 'hfs': [{'tree': (4, 2, 1, 1, 2, 1, 1), 'op': 'ppnnpnn', 's': (-1, -1, -1, 1, -1, -1, 1), 't': (((-4, 0), (-3, -3), (-3, 3), (-2, -6), (-2, 0), (-2, 6), (-1, -3), (-1, 3), (0, -6), (0, 0), (0, 6), (1, -3), (1, 3), (2, -6), (2, 0), (2, 6), (3, -3), (3, 3), (4, 0)), ((-2, 0), (-1, -3), (-1, 3), (0, 0), (1, -3), (1, 3), (2, 0)), ((-2, 0), (-1, -3), (-1, 3), (0, 0), (1, -3), (1, 3), (2, 0)), ((-4, 0), (-3, -3), (-3, 3), (-2, -6), (-2, 0), (-2, 6), (-1, -3), (-1, 3), (0, -6), (0, 0), (0, 6), (1, -3), (1, 3), (2, -6), (2, 0), (2, 6), (3, -3), (3, 3), (4, 0)), ((-2, 0), (-1, -3), (-1, 3), (0, 0), (1, -3), (1, 3), (2, 0)), ((-2, 0), (-1, -3), (-1, 3), (0, 0), (1, -3), (1, 3), (2, 0))), 'D': ((1, 2, 2, 1, 8, 1, 8, 8, 2, 15, 2, 8, 8, 1, 8, 1, 2, 2, 1), (1, 1, 1, 3, 1, 1, 1), (1, 1, 1, 3, 1, 1, 1), (1, 2, 2, 1, 8, 1, 8, 8, 2, 15, 2, 8, 8, 1, 8, 1, 2, 2, 1), (1, 1, 1, 3, 1, 1, 1), (1, 1, 1, 3, 1, 1, 1))}, {'tree': (4, 2, 1, 1, 2, 1, 1), 'op': 'ppnnpnn', 's': (-1, -1, -1, 1, -1, -1, 1), 't': (((-4, 0), (-3, -3), (-3, 3), (-2, -6), (-2, 0), (-2, 6), (-1, -3), (-1, 3), (0, -6), (0, 0), (0, 6), (1, -3), (1, 3), (2, -6), (2, 0), (2, 6), (3, -3), (3, 3), (4, 0)), ((-2, 0), (-1, -3), (-1, 3), (0, 0), (1, -3), (1, 3), (2, 0)), ((-2, 0), (-1, -3), (-1, 3), (0, 0), (1, -3), (1, 3), (2, 0)), ((-4, 0), (-3, -3), (-3, 3), (-2, -6), (-2, 0), (-2, 6), (-1, -3), (-1, 3), (0, -6), (0, 0), (0, 6), (1, -3), (1, 3), (2, -6), (2, 0), (2, 6), (3, -3), (3, 3), (4, 0)), ((-2, 0), (-1, -3), (-1, 3), (0, 0), (1, -3), (1, 3), (2, 0)), ((-2, 0), (-1, -3), (-1, 3), (0, 0), (1, -3), (1, 3), (2, 0))), 'D': ((1, 2, 2, 1, 8, 1, 8, 8, 2, 15, 2, 8, 8, 1, 8, 1, 2, 2, 1), (1, 1, 1, 3, 1, 1, 1), (1, 1, 1, 3, 1, 1, 1), (1, 2, 2, 1, 8, 1, 8, 8, 2, 15, 2, 8, 8, 1, 8, 1, 2, 2, 1), (1, 1, 1, 3, 1, 1, 1), (1, 1, 1, 3, 1, 1, 1))}], 'SYM_ID': 'U(1)xU(1)'})

    # conj & transpose
    X= R.transpose((1,0)).conj()
    Y= R.conj().transpose((1,0))
    assert (X-Y).norm() < tol

@pytest.mark.skipif(config_U1xU1_fermionic.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
@pytest.mark.parametrize("block_D", [50, 160])
def test_bug_conj_transpose_torch(block_D):
    import torch

    block_Ds= (block_D,block_D)
    block_size= block_D*block_D

    R= torch.rand(2000000,dtype=torch.complex128)-(0.5+0.5j)
    D0= torch.zeros_like(R)
    D1= torch.zeros_like(R)

    # order: transpose then conj
    block=slice(10,10+block_size)
    D0[block].view(block_Ds)[:]= R[block].view(block_Ds).permute(1,0)
    Y= D0.conj()

    # order: conj then transpose
    X= R.conj()
    D1[block].view(block_Ds)[:]= X[block].view(block_Ds).permute(1,0)

    assert torch.allclose(D1[block].view(block_Ds)[:5,:5],Y[block].view(block_Ds)[:5,:5],\
        rtol=1e-05, atol=1e-08)

@pytest.mark.skipif(config_U1xU1_fermionic.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
@pytest.mark.parametrize("block_D", [50,160])
def test_bug_conj_transpose_torch_conjphys(block_D):
    import torch

    block_Ds= (block_D,block_D)
    block_size= block_D*block_D

    R= torch.rand(2000000,dtype=torch.complex128)-(0.5+0.5j)
    D0= torch.zeros_like(R)
    D1= torch.zeros_like(R)

    # order: transpose then conj
    block=slice(10,10+block_size)
    D0[block].view(block_Ds)[:]= R[block].view(block_Ds).permute(1,0)
    Y= D0.conj()

    # order: conj then transpose
    X= R.conj_physical()
    D1[block].view(block_Ds)[:]= X[block].view(block_Ds).permute(1,0)

    assert torch.allclose(D1[block].view(block_Ds)[:5,:5],Y[block].view(block_Ds)[:5,:5],\
        rtol=1e-05, atol=1e-08)

if __name__ == '__main__':
    test_bug_conj_transpose()
    test_bug_conj_transpose_torch()
    test_bug_conj_transpose_torch_conjphys()
