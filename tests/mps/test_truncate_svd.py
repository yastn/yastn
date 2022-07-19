""" truncation of mps """
import yamps
import generate_random
import generate_automatic
try:
    from .configs import config_dense, config_dense_fermionic
    from .configs import config_U1, config_U1_fermionic
    from .configs import config_Z2, config_Z2_fermionic
except ImportError:
    from configs import config_dense, config_dense_fermionic
    from configs import config_U1, config_U1_fermionic
    from configs import config_Z2, config_Z2_fermionic



def run_dmrg_1site(psi, H, sweeps=10):
    """ Run a few sweeps of dmrg_1site_sweep. Returns energy. """
    env = None
    for _ in range(sweeps):
        env = yamps.dmrg_sweep_1site(psi, H, env=env)
    return env.measure()


def run_truncation(psi, H, Egs, sweeps=2):
    psi2 = psi.copy()
    discarded = psi2.truncate_sweep(to='last', opts={'D_total': 4})

    ov_t = yamps.measure_overlap(psi, psi2)
    Eng_t = yamps.measure_mpo(psi2, H, psi2)
    assert 1 > ov_t > 0.99
    assert Egs < Eng_t < Egs * 0.99

    psi2.canonize_sweep(to='first')
    env = None
    for _ in range(sweeps):
        env = yamps.variational_sweep_1site(psi2, psi_target=psi, env=env)

    ov_v = yamps.measure_overlap(psi, psi2)
    Eng_v = yamps.measure_mpo(psi2, H, psi2)
    assert psi2.get_bond_dimensions() == [1, 2, 4, 4, 4, 4, 4, 2, 1]
    assert 1 > ov_v > ov_t
    assert Egs < Eng_v < Eng_t


def test_truncate_svd_full():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    Eng_gs = -4.758770483143633
    D_total = 8
    H = generate_automatic.mpo_XX_model(config_dense_fermionic, N=N, t=1, mu=0)

    psi =generate_random.mps_random(config_dense_fermionic, N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    run_dmrg_1site(psi, H)
    run_truncation(psi, H, Eng_gs)


def test_truncate_svd_Z2():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 8
    D_total = 8
    Eng_parity0 = -4.758770483143633
    Eng_parity1 = -4.411474127809773

    H = generate_automatic.mpo_XX_model(config_Z2_fermionic, N=N, t=1, mu=0)

    psi = generate_random.mps_random(config_Z2_fermionic, N=N, Dblock=D_total/2, total_parity=1)
    psi.canonize_sweep(to='first')
    run_dmrg_1site(psi, H)
    run_truncation(psi, H, Eng_parity1)

    psi = generate_random.mps_random(config_Z2_fermionic, N=N, Dblock=D_total/2, total_parity=0)
    psi.canonize_sweep(to='first')
    run_dmrg_1site(psi, H)
    run_truncation(psi, H, Eng_parity0)


if __name__ == "__main__":
    test_truncate_svd_full()
    test_truncate_svd_Z2()
