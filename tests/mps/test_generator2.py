import numpy as np
import pytest
import yast
import yamps

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
        assert psi.A[psi.last].get_legs(axis=psi.right[0]).t == (nn,)
        assert psi.A[psi.first].get_legs(axis=psi.left[0]).t == ((0,) * len(nn),)
        bds = psi.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])


def test_generator_mpo():
    N = 10
    t = 1
    mu = 0.2
    operators = yast.operators.SpinlessFermions(sym='Z2')
    generate = yamps.Generator(N, operators)
    parameters = {"t": t, "mu": mu}
    H_str = "\sum_{j=0}^{"+str(N-1)+"} mu*cp_{j}.c_{j} + \sum_{j=0}^{"+str(N-2)+"} cp_{j}.c_{j+1} + \sum_{j=0}^{"+str(N-2)+"} t*cp_{j+1}.c_{j}"
    H = generate.mpo(H_str, parameters)



if __name__ == "__main__":
    # test_generator_mps()
    test_generator_mpo()
