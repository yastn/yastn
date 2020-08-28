import yamps.mps as mps
import ops_full as ops_full
import ops_Z2 as ops_Z2
import ops_U1 as ops_U1
import numpy as np
import pytest


def run_tdvp_1site(psi, H, dt, sweeps,  Eng_gs, opts=None):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None

    for _ in range(sweeps):
        env = mps.tdvp.tdvp_sweep_1site(
            psi, H, env=env, dt=dt, hermitian=True, opts_svd=opts)
        norm = mps.measure.measure_overlap(psi, psi)
        Eng = env.measure()/norm
        print('1site: Energy err = ', Eng-Eng_gs, ' Eg = ', Eng)
    assert 1. - abs(Eng/Eng_gs) < .1
    return Eng


def run_tdvp_2site(psi, H, dt, sweeps,  Eng_gs, opts=None):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None

    for _ in range(sweeps):
        env = mps.tdvp.tdvp_sweep_2site(
            psi, H, env=env, dt=dt, hermitian=True, opts_svd=opts)
        norm = mps.measure.measure_overlap(psi, psi)
        Eng = env.measure()/norm
        print('2site: Energy err = ', Eng-Eng_gs, ' Eg = ', Eng)
    assert 1. - abs(Eng/Eng_gs) < .1
    return Eng


def run_tdvp_2site_group(psi, H, dt, sweeps,  Eng_gs, opts=None):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None

    for _ in range(sweeps):
        env = mps.tdvp.tdvp_sweep_2site_group(
            psi, H, env=env, dt=dt, hermitian=True, opts_svd=opts)
        norm = mps.measure.measure_overlap(psi, psi)
        Eng = env.measure()/norm
        print('2site_group: Energy err = ', Eng-Eng_gs, ' Eg = ', Eng)
    assert 1. - abs(Eng/Eng_gs) < .1
    return Eng


def test_full_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 3
    D_total = 8
    dt = -.25
    sweeps = 15
    opts_svd = {'tol': 1e-6, 'D_total': D_total}
    Eng_gs = -1.4142135610633282

    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps,
                         opts=opts_svd, Eng_gs=Eng_gs)

    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site(psi, H, dt=dt, sweeps=sweeps,
                         opts=opts_svd, Eng_gs=Eng_gs)

    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site_group(
        psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_gs)


def test_Z2_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 3
    D_total = 8
    dt = -.25
    sweeps = 15
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    Eng_parity0 = -1.4142135623730947
    Eng_parity1 = -1.4142135623730938

    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_Z2.mps_random(N=N, Dblock=D_total, total_parity=0)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps,
                         opts=opts_svd, Eng_gs=Eng_parity0)

    psi = ops_Z2.mps_random(N=N, Dblock=D_total, total_parity=1)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps,
                         opts=opts_svd, Eng_gs=Eng_parity1)

    psi = ops_Z2.mps_random(N=N, Dblock=D_total, total_parity=0)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site(psi, H, dt=dt, sweeps=sweeps,
                         opts=opts_svd, Eng_gs=Eng_parity0)

    psi = ops_Z2.mps_random(N=N, Dblock=D_total, total_parity=0)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site_group(
        psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_parity0)


def test_U1_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 4
    D_total = 8
    dt = -.25
    sweeps = 15
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    Eng_ch1 = -1.6180339886438018
    Eng_ch2 = -2.2360679246823536

    H = ops_U1.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_U1.mps_random(N=N, Dblocks=[2, 4, 2], total_charge=1)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_ch1)

    psi = ops_U1.mps_random(N=N, Dblocks=[2, 4, 2], total_charge=2)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_ch2)

    psi = ops_U1.mps_random(N=N, Dblocks=[2, 4, 2], total_charge=1)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_ch1)

    psi = ops_U1.mps_random(N=N, Dblocks=[2, 4, 2], total_charge=1)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site_group(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_ch1)


def test_OBC_tdvp():
    """
    Check tdvp_OBC with measuring additional expectation values
    """
    N = 8
    D_total = 8
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)
    M = None

    Eng_gs = -4.758770483143633

    dt = -.125
    tmax = dt*1.

    version = '1site'
    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.tdvp.tdvp_OBC(
        psi=psi, tmax=tmax, dt=dt, H=H, M=M, version=version, opts_svd=opts_svd)
    print('1site: Energy - Eref= ', E-Eng_gs)

    version = '2site'
    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.tdvp.tdvp_OBC(
        psi=psi, tmax=tmax, dt=dt, H=H, M=M, version=version, opts_svd=opts_svd)
    print('2site: Energy - Eref= ', E-Eng_gs)

    version = '2site_group'
    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.tdvp.tdvp_OBC(
        psi=psi, tmax=tmax, dt=dt, H=H, M=M, version=version, opts_svd=opts_svd)
    print('2site_group: Energy - Eref= ', E-Eng_gs)


if __name__ == "__main__":
    # pass
    test_full_tdvp()
    test_Z2_tdvp()
    test_U1_tdvp()
    test_OBC_tdvp()
