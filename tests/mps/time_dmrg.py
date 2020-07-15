import mps
import ops_full
import ops_Z2
import ops_U1
import numpy as np
import time


def run_dmrg_2_site(psi, H, sweeps=20, Dmax=128):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    opts_svd = {'D_total': Dmax}
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_2site(psi, H, env=env, dtype='float64', opts_svd=opts_svd)
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
    Dmax = 128

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax / 2, total_parity=0)
    psi.canonize_sweep(to='first')
    t0 = time.time()
    Eng = run_dmrg_2_site(psi, H)
    t1 = time.time()
    print('Energy = ', Eng, ' time = ', t1 - t0, ' s.')


def time_U1_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 32
    H = ops_U1.mpo_XX_model(N=N, t=1, mu=0)
    Dmax = 46

    psi = ops_U1.mps_random(N=N, Dblocks=[1, 2, 4, 8, 16, 8, 4, 2, 1], total_charge=16)

    psi.canonize_sweep(to='first')
    t0 = time.time()
    Eng = run_dmrg_2_site(psi, H)
    t1 = time.time()
    print('Energy = ', Eng, ' time = ', t1 - t0, ' s.')
    for n in range(N):
        psi.A[n].show_properties()


if __name__ == "__main__":
    # pass
    # time_full_dmrg()
    time_U1_dmrg()