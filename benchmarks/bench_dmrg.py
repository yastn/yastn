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
""" benchmark dmrg in Heisenberg model """
import numpy as np
import time
import yastn.tn.mps as mps
import yastn



def dmrg_Heisenberg(config=None, tol=1e-6):
    """
    Initialize random MPS of dense tensors and
    runs a few sweeps of DMRG with the Hamiltonian of XX model.
    """
    N = 100
    J = 1
    D = 16
    ops = yastn.operators.Spin1(sym='U1')
    #
    sp, sm, sz = ops.sp(), ops.sm(), ops.sz()
    #
    I = mps.product_mpo(ops.I(), N)
    terms = [mps.Hterm(J, [n, n+1], [sz, sz]) for n in range(N - 1)]
    terms += [mps.Hterm(J / 2, [n, n+1], [sp, sm]) for n in range(N - 1)]
    terms += [mps.Hterm(J / 2, [n, n+1], [sm, sp]) for n in range(N - 1)]
    H = mps.generate_mpo(I, terms)
    #
    psi = mps.random_mps(I, n=(0,), D_total=D, dtype='float64')
    #
    keep_time = time.time()
    opts_svd = {"D_total": D}
    info = mps.dmrg_(psi, H, method='2site',
                     Schmidt_tol=1e-5, max_sweeps=32,
                     opts_svd=opts_svd)
    wall_time = time.time() - keep_time
    print(info)
    print(f"DMRG time = {wall_time:0.2f} s.")
    print(f"Energy = {info.energy:0.5f}")

if __name__ == "__main__":
    dmrg_Heisenberg()
