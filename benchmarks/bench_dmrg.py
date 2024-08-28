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
""" benchmark DMRG in Heisenberg spin-1 model """
import argparse
import time
import yastn.tn.mps as mps
import yastn


def dmrg_Heisenberg(args):
    """
    Initialize MPS in a Neel state and run a few sweeps of
    DMRG 2-site with the Heisenberg spin-1 Hamiltonian.
    """
    N = 100
    if args.backend == 'np':
        import yastn.backend.backend_np as backend
    elif args.backend == 'torch':
        import yastn.backend.backend_torch as backend
    ops = yastn.operators.Spin1(sym=args.sym, backend=backend, default_device=args.device)
    #
    sp, sm, sz = ops.sp(), ops.sm(), ops.sz()
    #
    # Hamiltonian MPO
    #
    I = mps.product_mpo(ops.I(), N)
    terms = [mps.Hterm(1, [n, n+1], [sz, sz]) for n in range(N - 1)]
    terms += [mps.Hterm(1 / 2, [n, n+1], [sp, sm]) for n in range(N - 1)]
    terms += [mps.Hterm(1 / 2, [n, n+1], [sm, sp]) for n in range(N - 1)]
    H = mps.generate_mpo(I, terms)
    #
    # Initial product state
    #
    psi = mps.product_mps([ops.vec_z(1), ops.vec_z(-1)], N)
    #
    # setting up DMRG parameters
    #
    Ds = [10, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
    rep = 2  # we will make 2 sweeps per D in Ds
    opts_svd = {"D_total": Ds[0]}
    dmrg = mps.dmrg_(psi, H, method='2site', iterator_step=1, max_sweeps=rep * len(Ds), opts_svd=opts_svd)
    #
    # execute dmrg generator
    #
    ref_time = time.time()
    ref_time_global = ref_time
    for info in dmrg:
        wall_time = time.time() - ref_time
        print(f"Sweep={info.sweeps:02d}; Energy={info.energy:4.12f}; D={max(psi.get_bond_dimensions()):4d}; time={wall_time:3.1f}")
        ref_time = time.time()
        if info.sweeps % rep == 0 and info.sweeps // rep < len(Ds):
            opts_svd["D_total"] = Ds[info.sweeps // rep]  # update D used by DMRG

        if ref_time - ref_time_global > args.max_seconds:
            print(f"Maximal simulation time reached after {ref_time - ref_time_global:0.1f}")
            break

    print("Cache info")
    for x in yastn.get_cache_info().items():
        print(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sym", type=str, default='U1', choices=['Z3', 'dense', 'U1'])
    parser.add_argument("-backend", type=str, default='np', choices=['np', 'torch'])
    parser.add_argument("-device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("-max_seconds", type=int, default=3600)
    args = parser.parse_args()

    dmrg_Heisenberg(args)
