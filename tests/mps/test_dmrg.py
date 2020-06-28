import yamps.mps as mps
import yamps.ops.ops_full as ops_full
import yamps.ops.ops_Z2 as ops_Z2
import numpy as np
import pytest


def run_dmrg_1site(psi, H, Etarget, sweeps=10):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_1site(psi, H, env=env, dtype='float64')
    Eng = env.measure()
    assert pytest.approx(Eng) == Etarget
    return Eng


def run_dmrg_2site(psi, H, Etarget, sweeps=10):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    opts_svd = {'tol': 1e-8, 'D_total': 32}
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_2site(psi, H, env=env, dtype='float64', opts_svd=opts_svd)
    Eng = env.measure()
    assert pytest.approx(Eng) == Etarget
    return Eng


def run_dmrg_2site_group(psi, H, Etarget, sweeps=10):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    opts_svd = {'tol': 1e-8, 'D_total': 32}
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_2site_group(psi, H, env=env, dtype='float64', opts_svd=opts_svd)
    Eng = env.measure()
    assert pytest.approx(Eng) == Etarget
    return Eng


def test_full_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    Eng_gs = -4.758770483143633
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1site(psi, H, Eng_gs)
    print('Energy = ', Eng)

    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site(psi, H, Eng_gs)
    print('Energy = ', Eng)

    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site_group(psi, H, Eng_gs)
    print('Energy = ', Eng)


def test_Z2_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 8
    Eng_parity0 = -4.758770483143633
    Eng_parity1 = -4.411474127809773

    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1site(psi, H, Eng_parity0)
    print('Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=1)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1site(psi, H, Eng_parity1)
    print('Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site(psi, H, Eng_parity0)
    print('Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=1)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site(psi, H, Eng_parity1)
    print('Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site_group(psi, H, Eng_parity0)
    print('Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=1)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site_group(psi, H, Eng_parity1)
    print('Energy = ', Eng)


if __name__ == "__main__":
    # pass
    test_full_dmrg()
    test_Z2_dmrg()
