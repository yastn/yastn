import yamps.mps as mps
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
    env = mps.dmrg.dmrg_sweep_1site(psi, H, env=env)
    t0 = time.time()
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_2site(psi, H, env=env, opts_svd=opts_svd)
    print(sweeps,' sweeps with 2-site dmrg in', time.time() - t0, 's.')
    print('Energy = ', env.measure())
    print('MPS bond dimensions: ', psi.get_D())


def time_full_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 32
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)
    Dmax = 128
    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2).canonize_sweep(to='first')

    print('*** Dense ***')
    run_dmrg_2_site(psi, H, Dmax=Dmax)


def time_Z2_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 32
    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0)
    Dmax = 128
    psi = ops_Z2.mps_random(N=N, Dblock=Dmax / 2, total_parity=0).canonize_sweep(to='first')

    print('*** Z2 ***')
    run_dmrg_2_site(psi, H, Dmax=Dmax)
    

def time_U1_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 32
    Dmax = 128
    H = ops_U1.mpo_XX_model(N=N, t=1, mu=0)
    psi = ops_U1.mps_random(N=N, Dblocks=[Dmax/8, 3*Dmax/8, Dmax/2, 3*Dmax/8, Dmax/8], total_charge=16).canonize_sweep(to='first')
    psi.canonize_sweep(to='first')

    print('*** U1 ***')
    run_dmrg_2_site(psi, H, Dmax=Dmax)

if __name__ == "__main__":
    time_full_dmrg()
    time_Z2_dmrg()
    time_U1_dmrg()
