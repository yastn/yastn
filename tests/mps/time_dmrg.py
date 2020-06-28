import yamps.mps as mps
import yamps.ops.ops_full as ops_full
import yamps.ops.ops_Z2 as ops_Z2
import numpy as np
import time


def run_dmrg_2_site(psi, H, sweeps=20, Dmax=128):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    opts_svd = {'D_block': Dmax, 'D_total': Dmax}
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_2site_group(psi, H, env=env, dtype='float64', opts_svd=opts_svd)
        Eng = env.measure()
        print(Eng)
    return Eng


def time_full_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 32
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)
    Dmax = 128

    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    t0 = time.time()
    Eng = run_dmrg_2_site(psi, H)
    t1 = time.time()
    print('Energy = ', Eng, ' time = ', t1 - t0, ' s.')


def time_Z2_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 32
    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0)
    Dmax = 64

    psi = ops_Z2.mps_random(N=N, Dmax=Dmax, total_parity=0)
    psi.canonize_sweep(to='first')
    t0 = time.time()
    Eng = run_dmrg_2_site(psi, H)
    t1 = time.time()
    print('Energy = ', Eng, ' time = ', t1 - t0, ' s.')


if __name__ == "__main__":
    # pass
    # time_full_dmrg()
    time_Z2_dmrg()
