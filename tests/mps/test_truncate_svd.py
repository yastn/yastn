import yamps.mps as mps
import ops_full
import ops_Z2
import ops_U1
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
    # assert pytest.approx(Eng) == Etarget
    return Eng


def test_truncate_svd_full():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    Eng_gs = -4.758770483143633
    D_total = 32
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1site(psi, H, Eng_gs)

    psi2 = psi.copy()
    discarded = psi2.sweep_truncate(to='last', opts={'D_total': 8})
    print('energy befor truncation :', Eng)
    print('energy after truncation :', mps.measure_mpo(psi2, H, psi2))
    print('overlap after truncation : ', mps.measure_overlap(psi, psi2))
    print('max discarded : ', discarded)


def test_truncate_svd_Z2():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 8
    D_total = 32
    Eng_parity0 = -4.758770483143633
    Eng_parity1 = -4.411474127809773

    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0)
    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=1)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1site(psi, H, Eng_parity1)

    psi2 = psi.copy()
    discarded = psi2.sweep_truncate(to='last', opts={'D_total': 8})
    print('energy befor truncation :', Eng)
    print('energy after truncation :', mps.measure_mpo(psi2, H, psi2))
    print('overlap after truncation : ', mps.measure_overlap(psi, psi2))
    print('max discarded : ', discarded)


if __name__ == "__main__":
    # pass
    test_truncate_svd_full()
    test_truncate_svd_Z2()
