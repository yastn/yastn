""" truncation of mps """
import yamps
import yast
try:
    from . import generate_by_hand
except ImportError:
    import generate_by_hand



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
    assert psi2.get_bond_dimensions() == (1, 2, 4, 4, 4, 4, 4, 2, 1)
    assert 1 > ov_v > ov_t
    assert Egs < Eng_v < Eng_t


def test_truncate_svd_full():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    Eng_gs = -4.758770483143633
    D_total = 8

    operators = yast.operators.Spin12(sym='dense')
    generate = yamps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    H = generate_by_hand.mpo_XX_model(generate.config, N=N, t=1, mu=0)
    psi = generate.random_mps(D_total=D_total).canonize_sweep(to='first')
    run_dmrg_1site(psi, H)
    run_truncation(psi, H, Eng_gs)


def test_truncate_svd_Z2():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 8
    D_total = 8
    Eng_parity = {0: -4.758770483143633, 1: -4.411474127809773}

    operators = yast.operators.Spin12(sym='Z2')
    generate = yamps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    H = generate_by_hand.mpo_XX_model(generate.config, N=N, t=1, mu=0)

    for parity in (0, 1):
        psi = generate.random_mps(D_total=D_total, n=parity)
        psi.canonize_sweep(to='first')
        run_dmrg_1site(psi, H)
        run_truncation(psi, H, Eng_parity[parity])


if __name__ == "__main__":
    test_truncate_svd_full()
    test_truncate_svd_Z2()
