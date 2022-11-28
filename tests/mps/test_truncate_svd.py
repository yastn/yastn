""" truncation of mps """
import pytest
import yast
import yast.tn.mps as mps
try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense as cfg


def run_zipper(psi, H, Egs):
    Hpsi = mps.zipper(H, psi, opts={'D_total': 6})

    Eng_t = mps.measure_overlap(Hpsi, psi)
    assert Egs < Eng_t < Egs * 0.995

    Hnorm = mps.measure_overlap(Hpsi, Hpsi) ** 0.5

    for out in mps.variational_(Hpsi, H, psi, iterator_step=1, max_sweeps=1):
        Eng_new = mps.vdot(Hpsi, psi) * Hnorm
        assert Egs < Eng_new < Eng_t
        Eng_t = Eng_new

def run_truncation(psi, H, Egs, sweeps=2):
    psi2 = psi.copy()
    discarded = psi2.truncate_(to='last', opts={'D_total': 4})

    ov_t = mps.measure_overlap(psi, psi2).item()
    Eng_t = mps.measure_mpo(psi2, H, psi2).item()
    assert 1 > abs(ov_t) > 0.99
    assert Egs < Eng_t.real < Egs * 0.99

    out = mps.variational_(psi2, psi, max_sweeps=5)
    ov_v = mps.measure_overlap(psi, psi2).item()
    Eng_v = mps.measure_mpo(psi2, H, psi2).item()
    assert all(dp <= do for dp, do in zip(psi2.get_bond_dimensions(), (1, 2, 4, 4, 4, 4, 4, 2, 1)))
    assert 1 > abs(ov_v) > abs(ov_t)
    assert Egs < Eng_v.real < Eng_t.real
    assert pytest.approx(out.overlap.item(), rel=1e-12) == ov_v

def test_truncate_svd_dense():
    """
    Initialize random mps of dense tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    Eng_gs = -4.758770483143633
    D_total = 8

    operators = yast.operators.Spin12(sym='dense', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    parameters = {"t": lambda j: 1.0, "mu": lambda j: 0, "range1": range(N), "range2": range(N-1)}
    H_str = "\sum_{j \in range2} t ( sp_{j} sm_{j+1} + sp_{j+1} sm_{j} ) + \sum_{j\in range1} mu sp_{j} sm_{j}"
    H = generate.mpo(H_str, parameters)
    psi = generate.random_mps(D_total=D_total)
    mps.dmrg_(psi, H, max_sweeps=10, Schmidt_tol=1e-8)
    run_truncation(psi, H, Eng_gs)


def test_truncate_svd_Z2():
    """
    Initialize random mps of dense tensors and checks canonization
    """
    N = 8
    D_total = 10
    Eng_parity = {0: -4.758770483143633, 1: -4.411474127809773}

    operators = yast.operators.Spin12(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    parameters = {"t": lambda j: 1.0, "mu": lambda j: 0, "range1": range(N), "range2": range(N-1)}
    H_str = "\sum_{j \in range2} t ( sp_{j} sm_{j+1} + sp_{j+1} sm_{j} ) + \sum_{j\in range1} mu sp_{j} sm_{j}"
    H = generate.mpo(H_str, parameters)

    for parity in (0, 1):
        psi = generate.random_mps(D_total=D_total, n=parity)
        mps.dmrg_(psi, H, max_sweeps=10, Schmidt_tol=1e-8)
        run_truncation(psi, H, Eng_parity[parity])
        run_zipper(psi, H, Eng_parity[parity])


if __name__ == "__main__":
    test_truncate_svd_dense()
    test_truncate_svd_Z2()
