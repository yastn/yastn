import numpy as np

import yamps.mps as mps
import yamps.ops.ops_full as ops_full
import yamps.ops.ops_Z2 as ops_Z2


def run_dmrg_1_site(psi, H, sweeps=10):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_1site(psi, H, env=env, dtype='float64')
    return env.measure()


def test_full_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    Eng_gs = -4.758770483143633
    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)

    Eng = run_dmrg_1_site(psi, H)
    print('Energy = ', Eng)
    assert(np.allclose(Eng_gs, Eng))


def test_Z2_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 8
    Eng_parity0 = -4.758770483143633
    Eng_parity1 = -4.411474127809773

    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0)

    psi = ops_Z2.mps_random(N=N, Dmax=32, total_parity=0)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1_site(psi, H)
    print('Energy = ', Eng)
    assert(np.allclose(Eng_parity0, Eng))

    psi = ops_Z2.mps_random(N=N, Dmax=32, total_parity=1)
    psi.canonize_sweep(to='first')
    Eng = run_dmrg_1_site(psi, H)
    print('Energy = ', Eng)
    assert(np.allclose(Eng_parity1, Eng))


if __name__ == "__main__":
    pass
    test_full_dmrg()
    test_Z2_dmrg()
