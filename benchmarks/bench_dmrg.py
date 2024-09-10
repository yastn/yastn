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

def dmrg_Heisenberg(args):
    """
    Initialize MPS in a Neel state and run a few sweeps of
    DMRG 2-site with the Heisenberg spin-1 Hamiltonian.
    """
    import time
    import yastn.tn.mps as mps
    import yastn

    N = 100
    if args.backend == 'np':
        import yastn.backend.backend_np as backend
    elif args.backend == 'torch':
        import yastn.backend.backend_torch as backend
    #
    ops = yastn.operators.Spin1(sym=args.sym, backend=backend, default_device=args.device)
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
    # DMRG parameters
    #
    Ds = {0: 10, 2: 32, 4: 64, 6: 128, 8: 256, 10: 384, 12: 512, 14: 768, 16: 1024, 18: 1536, 20: 2048}  # sweep no.: dimension
    opts_svd = {"D_total": Ds[0], 'svd_on_cpu': args.svd_on_cpu} #, 'tol': 1e-14}
    opts_eigs = {'hermitian': True, 'ncv': 3, 'which': 'SR'}  # default opts_eigs in dmrg_; provided here to show them explicitly
    dmrg = mps.dmrg_(psi, H, method='2site', iterator_step=1, max_sweeps=22, opts_svd=opts_svd, opts_eigs=opts_eigs)
    #
    # execute dmrg generator
    #
    ref_time_sweep = ref_time_total = time.time()
    for info in dmrg:
        wall_time = time.time() - ref_time_sweep
        print(f"Sweep={info.sweeps:02d}; Energy={info.energy:4.12f}; D={max(psi.get_bond_dimensions()):4d}; time={wall_time:3.1f}")
        ref_time_sweep  = time.time()
        if info.sweeps in Ds:
            opts_svd["D_total"] = Ds[info.sweeps]  # update D_total used by DMRG

        total_time = time.time() - ref_time_total
        if total_time  > args.max_seconds:
            print(f"Maximal simulation time reached after {total_time:0.1f}")
            break

    # print("Cache info")  # auxiliary information from lru_cache
    # for x in yastn.get_cache_info().items():
    #     print(x)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-sym", type=str, default='U1', choices=['Z3', 'dense', 'U1'])
    parser.add_argument("-backend", type=str, default='np', choices=['np', 'torch'])
    parser.add_argument("-device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("-svd_on_cpu", dest='svd_on_cpu', action='store_true')
    parser.add_argument("-max_seconds", type=int, default=3600)
    parser.add_argument("-num_threads", type=str, default='none', choices=['none'] + [str(n) for n in range(1, 33)])
    args = parser.parse_args()

    if args.num_threads != 'none':
        import os
        os.environ["OMP_NUM_THREADS"] = args.num_threads
        os.environ["OPENBLAS_NUM_THREADS"] = args.num_threads
        os.environ["MKL_NUM_THREADS"] = args.num_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = args.num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = args.num_threads

    dmrg_Heisenberg(args)
