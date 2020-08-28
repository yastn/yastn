import yamps.mps as mps
import ops_full
import ops_Z2
import ops_U1
import numpy as np
import pytest


def run_dmrg_0site(psi, H, Etarget, sweeps=10):
    """
    Run a faw sweeps of dmrg_0site_sweep. Returns energy
    """
    env = None
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_0site(psi, H, env=env)
    Eng = env.measure().real
    assert pytest.approx(Eng) == Etarget
    return Eng


def run_dmrg_1site(psi, H, Etarget, sweeps=10):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_1site(psi, H, env=env)
    Eng = env.measure().real
    assert pytest.approx(Eng) == Etarget
    return Eng


def run_dmrg_2site(psi, H, Etarget, sweeps=10, D_total=32):
    """
    Run a faw sweeps of dmrg_2site_sweep. Returns energy
    """
    env = None
    opts_svd = {'tol': 1e-8, 'D_total': D_total}
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_2site(psi, H, env=env, opts_svd=opts_svd)
    Eng = env.measure().real
    assert pytest.approx(Eng) == Etarget
    return Eng


def run_dmrg_2site_group(psi, H, Etarget, sweeps=10, D_total=32):
    """
    Run a faw sweeps of dmrg_2site_group_sweep. Returns energy
    """
    env = None
    opts_svd = {'tol': 1e-8, 'D_total': D_total}
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_2site_group(
            psi, H, env=env, opts_svd=opts_svd)
    Eng = env.measure().real
    assert pytest.approx(Eng) == Etarget
    return Eng


def test_full_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    Eng_gs = -4.758770483143633
    D_total = 32
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_0site(psi, H, Eng_gs)
    print('0site: Energy = ', Eng)

    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1site(psi, H, Eng_gs)
    print('1site: Energy = ', Eng)

    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site(psi, H, Eng_gs, D_total=D_total)
    print('2site: Energy = ', Eng)

    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site_group(psi, H, Eng_gs, D_total=D_total)
    print('2site group: Energy = ', Eng)


def test_Z2_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 8
    D_total = 32
    Eng_parity0 = -4.758770483143633
    Eng_parity1 = -4.411474127809773

    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_0site(psi, H, Eng_parity0)
    print('0site: Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=1)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_0site(psi, H, Eng_parity1)
    print('0site: Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1site(psi, H, Eng_parity0)
    print('1site: Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=1)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1site(psi, H, Eng_parity1)
    print('1site: Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site(psi, H, Eng_parity0, D_total=D_total)
    print('2site: Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=1)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site(psi, H, Eng_parity1, D_total=D_total)
    print('2site: Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site_group(psi, H, Eng_parity0, D_total=D_total)
    print('2site group: Energy = ', Eng)

    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=1)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site_group(psi, H, Eng_parity1, D_total=D_total)
    print('2site group: Energy = ', Eng)


def test_U1_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 8
    D_total = 32
    Eng_parity0 = -4.758770483143633
    Eng_parity1 = -4.411474127809773

    H = ops_U1.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_U1.mps_random(N=N, Dblocks=[8, 16, 8], total_charge=3)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site(psi, H, Eng_parity1, D_total=D_total)
    print('2site: Energy = ', Eng)

    psi = ops_U1.mps_random(N=N, Dblocks=[8, 16, 8], total_charge=4)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site(psi, H, Eng_parity0, D_total=D_total)
    print('2site: Energy = ', Eng)

    psi = ops_U1.mps_random(N=N, Dblocks=[8, 16, 8], total_charge=3)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_2site_group(psi, H, Eng_parity1, D_total=D_total)
    print('2site group: Energy = ', Eng)


def test_OBC_dmrg():
    """
    Check dmrg_OBC with measuring additional expectation values
    """
    N = 8
    D_total = 8
    cutoff_sweep = 1
    cutoff_dE = 1e-9
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)

    Eng_gs = -4.758770483143633

    version = '0site'
    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.dmrg.dmrg_OBC(psi=psi, H=H, env=None, version=version, cutoff_sweep=cutoff_sweep,
                                cutoff_dE=cutoff_dE, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=opts_svd)
    print('0site: Energy - Eref= ', E-Eng_gs)

    version = '1site'

    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)
    _, E, _ = mps.dmrg.dmrg_OBC(psi=psi, H=H, env=None, version=version, cutoff_sweep=cutoff_sweep,
                                cutoff_dE=cutoff_dE, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=opts_svd)
    print('1site: Energy - Eref= ', E-Eng_gs)

    version = '2site'
    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.dmrg.dmrg_OBC(psi=psi, H=H, env=None, version=version, cutoff_sweep=cutoff_sweep,
                                cutoff_dE=cutoff_dE, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=opts_svd)
    print('2site: Energy - Eref= ', E-Eng_gs)

    version = '2site_group'
    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.dmrg.dmrg_OBC(psi=psi, H=H, env=None, version=version, cutoff_sweep=cutoff_sweep,
                                cutoff_dE=cutoff_dE, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=opts_svd)
    print('2site_group: Energy - Eref= ', E-Eng_gs)


if __name__ == "__main__":
    # pass
    test_full_dmrg()
    test_Z2_dmrg()
    test_U1_dmrg()
    test_OBC_dmrg()
