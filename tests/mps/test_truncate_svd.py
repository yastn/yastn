""" truncation of mps """
import yamps
import yast


def run_dmrg_1site(psi, H, sweeps=10):
    """ Run a few sweeps of dmrg_1site_sweep. Returns energy. """
    env = None
    for _ in range(sweeps):
        env = yamps.dmrg_sweep_1site(psi, H, env=env)
    return env.measure()


def run_multiply_svd(psi, H, Egs, sweeps=1):
    Hpsi = yamps.multiply_svd(H, psi, opts={'D_total': 6})

    Eng_t = yamps.measure_overlap(Hpsi, psi)
    assert Egs < Eng_t < Egs * 0.995

    Hnorm = yamps.measure_overlap(Hpsi, Hpsi) ** 0.5

    for _ in range(sweeps):
        yamps.variational_sweep_1site(Hpsi, psi_target=psi, op=H)
        Eng_new = yamps.measure_overlap(Hpsi, psi) * Hnorm
        assert Egs < Eng_new < Eng_t
        Eng_t = Eng_new


def run_truncation(psi, H, Egs, sweeps=2):
    psi2 = psi.copy()
    discarded = psi2.truncate_sweep(to='last', opts={'D_total': 4})

    ov_t = yamps.measure_overlap(psi, psi2).item()
    Eng_t = yamps.measure_mpo(psi2, H, psi2).item()
    assert 1 > abs(ov_t) > 0.99
    assert Egs < Eng_t.real < Egs * 0.99

    psi2.canonize_sweep(to='first')
    env = None
    for _ in range(sweeps):
        env = yamps.variational_sweep_1site(psi2, psi_target=psi, env=env)

    ov_v = yamps.measure_overlap(psi, psi2).item()
    Eng_v = yamps.measure_mpo(psi2, H, psi2).item()
    assert psi2.get_bond_dimensions() == (1, 2, 4, 4, 4, 4, 4, 2, 1)
    assert 1 > abs(ov_v) > abs(ov_t)
    assert Egs < Eng_v.real < Eng_t.real


def test_truncate_svd_dense():
    """
    Initialize random mps of dense tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    Eng_gs = -4.758770483143633
    D_total = 8

    operators = yast.operators.Spin12(sym='dense')
    generate = yamps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    parameters = {"t": lambda j: 1.0, "mu": lambda j: 0, "range1": range(N), "range2": range(N-1)}
    H_str = "\sum_{j \in range2} t ( sp_{j} sm_{j+1} + sp_{j+1} sm_{j} ) + \sum_{j\in range1} mu sp_{j} sm_{j}"
    H = generate.mpo(H_str, parameters)
    psi = generate.random_mps(D_total=D_total).canonize_sweep(to='first')
    run_dmrg_1site(psi, H)
    run_truncation(psi, H, Eng_gs)


def test_truncate_svd_Z2():
    """
    Initialize random mps of dense tensors and checks canonization
    """
    N = 8
    D_total = 8
    Eng_parity = {0: -4.758770483143633, 1: -4.411474127809773}

    operators = yast.operators.Spin12(sym='Z2')
    generate = yamps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    parameters = {"t": lambda j: 1.0, "mu": lambda j: 0, "range1": range(N), "range2": range(N-1)}
    H_str = "\sum_{j \in range2} t ( sp_{j} sm_{j+1} + sp_{j+1} sm_{j} ) + \sum_{j\in range1} mu sp_{j} sm_{j}"
    H = generate.mpo(H_str, parameters)

    for parity in (0, 1):
        psi = generate.random_mps(D_total=D_total, n=parity)
        psi.canonize_sweep(to='first')
        run_dmrg_1site(psi, H)
        run_truncation(psi, H, Eng_parity[parity])
        run_multiply_svd(psi, H, Eng_parity[parity])


if __name__ == "__main__":
    test_truncate_svd_dense()
    test_truncate_svd_Z2()
