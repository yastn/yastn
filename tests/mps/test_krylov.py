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
""" test tdvp """
import pytest
import numpy as np
import time
import yastn.tn.mps as mps
import yastn
try:
    from .configs import config_dense as cfg
except ImportError:
    from configs import config_dense as cfg
# pytest modifies cfg to inject different backends and devices during tests


def test_krylov():
    """ krylov in mps """
    #
    N = 10  # Consider a system of 10 sites
    #
    # Load spin-1/2 operators
    #
    opts_config = {'backend': cfg.backend,
                   'default_device': cfg.default_device}
    # pytest uses config to inject various backends and devices for testing
    #
    ops = yastn.operators.Spin12(sym='dense', **opts_config)
    ops.random_seed(seed=0)
    #
    # Hterm-s to generate H = -sum_i X_i X_{i+1} - g * Z_i
    #
    I = mps.product_mpo(ops.I(), N)  # identity MPO
    termsXX = [mps.Hterm(-1, [i, (i + 1) % N], [ops.x(), ops.x()]) for i in range(N)]
    HXX = mps.generate_mpo(I, termsXX)
    termsZ = [mps.Hterm(-1, [i], [ops.z()]) for i in range(N)]
    HZ = mps.generate_mpo(I, termsZ)
    #
    H = HXX + HZ
    psi = mps.random_mps(I, D_total=4)

    # yastn.expmv(lambda x: H @ x, psi, 0.1)



if __name__ == "__main__":
    test_krylov()