import yamps.mps as mps
import yamps.mps.tdvp as tdvp
import yamps.ops.ops_full as ops_full
import yamps.ops.ops_Z2 as ops_Z2
import yamps.ops.ops_U1 as ops_U1
import numpy as np
import pytest


def run_tdvp_1site(psi, H, dt=1., sweeps=10, opts=None, Eng_gs=0):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    for _ in range(sweeps):
        env = tdvp.tdvp_sweep_1site(psi, H, dt=1., dtype='float64', hermitian=True, fermionic=False, opts_svd=opts)
        Eng = env.measure()
        print('1site: Energy err = ', Eng+Eng_gs)
    return Eng


def test_full_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    D_total = 32
    dt = 1j
    sweeps = 20
    opts_svd = {'tol': 1e-6, 'D_total': D_total}
    Eng_gs = -4.758770483143633

    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)    

    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    Eng = run_tdvp_1site(psi, H, dt = dt, sweeps = sweeps, opts=opts_svd, Eng_gs=Eng_gs)


if __name__ == "__main__":
    # pass
    test_full_tdvp()
