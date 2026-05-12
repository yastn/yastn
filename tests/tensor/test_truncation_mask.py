
# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
""" yastn.truncation_mask() """
import pytest
import yastn


def test_svd_multiplets(config_kwargs):

    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    S = yastn.Tensor(config_U1, isdiag=True)
    #
    # fixing singular values for testing
    v00 = [1, 1, 0.1001, 0.1000, 0.1000, 0.0999, 0.001001, 0.001000] + [0] * 16
    S.set_block(ts=(0, 0), Ds=24, val=v00)

    v11 = [1, 1, 0.1001, 0.1000, 0.0999, 0.001000, 0.000999] + [0] * 10
    S.set_block(ts=(1, 1), Ds=17, val=v11)
    S.set_block(ts=(-1, -1), Ds=17, val=v11)

    v22 = [1, 1, 0.1001, 0.1000, 0.001000, 0]
    S.set_block(ts=(2, 2), Ds=6, val=v22)
    S.set_block(ts=(-2, -2), Ds=6, val=v22)

    Smask = yastn.truncation_mask(S, tol=0.0001, D_block=7, D_total=30)
    assert yastn.trace(Smask).item() == 29

    Smask = yastn.truncation_mask(S, tol=0.0001, D_total=30, eps_multiplet=0.001)
    assert yastn.trace(Smask).item() == 24

    Smask = yastn.truncation_mask(S, tol=0.0001, largest_gap=True)
    assert yastn.trace(Smask).item() == 32

    Smask = yastn.truncation_mask(S, D_total=17, largest_gap=True)
    assert yastn.trace(Smask).item() == 24

    Smask = yastn.truncation_mask(S, D_total=0)
    assert yastn.trace(Smask).item() == 0



if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
