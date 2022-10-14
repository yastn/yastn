import pytest
import yast
import yamps

try:
    from . import generate_by_hand
except ImportError:
    import generate_by_hand

tol = 1e-12

def test_generator_mps():
    N = 10
    D_total = 16
    bds = (1,) + (D_total,) * (N - 1) + (1,)

    for sym, nn in (('Z2', (0,)), ('Z2', (1,)), ('U1', (N // 2,))):
        operators = yast.operators.SpinlessFermions(sym=sym)
        generate = yamps.Generator(N, operators)
        I = generate.I()
        assert pytest.approx(yamps.measure_overlap(I, I).item(), rel=tol) == 2 ** N
        O = I @ I + (-1 * I)
        assert pytest.approx(yamps.measure_overlap(O, O).item(), abs=tol) == 0
        psi = generate.random_mps(D_total=D_total, n = nn)
        assert psi[psi.last].get_legs(axis=psi.right[0]).t == (nn,)
        assert psi[psi.first].get_legs(axis=psi.left[0]).t == ((0,) * len(nn),)
        bds = psi.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])


def test_generator_mpo():
    N = 5
    t = 1
    mu = 0.2
    operators = yast.operators.SpinlessFermions(sym='Z2')
    generate = yamps.Generator(N, operators)
    generate.random_seed(seed=0)
    parameters = {"t": lambda j: t, "mu": lambda j: mu, "range1": range(N), "range2": range(1, N-1)}
    # is CORRECT // full model hopping:
    H_str = "\sum_{j \in range2} t ( cp_{j} c_{j+1} + cp_{j+1} c_{j} ) + \sum_{j\in range1} mu cp_{j} c_{j} + ( cp_{0} c_{1} + 1*cp_{1} c_{0} )*t "
    H_ref = generate_by_hand.mpo_XX_model(generate.config, N=N, t=t, mu=mu)
    # is CORRECT // only hopping: H_str, H_ref = "\sum_{j \in range2} t ( cp_{j} c_{j+1} + cp_{j+1} c_{j} )", generate_by_hand.mpo_XX_model(config_Z2_fermionic, N=N, t=t, mu=0)
    # is CORRECT // no hopping: H_str, H_ref = "+ \sum_{j \in range1} mu cp_{j} c_{j}", generate_by_hand.mpo_XX_model(config_Z2_fermionic, N=N, t=0, mu=0.2)
    H = generate.mpo(H_str, parameters)
    psi = generate.random_mps(D_total=8, n=0) + generate.random_mps( D_total=8, n=1)
    x_ref = yamps.measure_mpo(psi, H_ref, psi).item()
    x = yamps.measure_mpo(psi, H, psi).item()
    assert abs(x_ref - x) < tol

    psi.canonize_sweep(to='first')
    psi.canonize_sweep(to='last')
    x_ref = yamps.measure_mpo(psi, H_ref, psi).item()
    x = yamps.measure_mpo(psi, H, psi).item()
    assert abs(x_ref - x) < tol

if __name__ == "__main__":
    test_generator_mps()
    test_generator_mpo()
