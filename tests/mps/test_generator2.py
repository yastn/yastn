from math import gamma
from os import remove
from tracemalloc import get_tracemalloc_memory
import numpy as np
import pytest
import yast
import yamps

try:
    from . import generate_random, generate_by_hand
    from .configs import config_dense, config_dense_fermionic
    from .configs import config_U1, config_U1_fermionic
    from .configs import config_Z2, config_Z2_fermionic
except ImportError:
    import generate_random, generate_by_hand
    from configs import config_dense, config_dense_fermionic
    from configs import config_U1, config_U1_fermionic
    from configs import config_Z2, config_Z2_fermionic
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
    conf = config_Z2_fermionic
    generate_random.random_seed(conf, seed=0)
    N = 5
    t = 1
    mu = 0.2
    operators = yast.operators.SpinlessFermions(sym='Z2', backend=conf.backend)
    generate = yamps.Generator(N, operators)
    parameters = {"t": lambda j: t, "mu": lambda j: mu, "range1": range(N), "range2": range(1, N-1)}
    # is CORRECT // full model hopping:
    H_str, H_ref = r"\sum_{j \in range2} t ( cp_{j} c_{j+1} + cp_{j+1} c_{j} ) "\
        r"+ \sum_{j\in range1} mu cp_{j} c_{j} + ( cp_{0} c_{1} + 1*cp_{1} c_{0} )*t ", \
        generate_by_hand.mpo_XX_model(config_Z2_fermionic, N=N, t=t, mu=mu)
    # is CORRECT // only hopping: H_str, H_ref = "\sum_{j \in range2} t ( cp_{j} c_{j+1} + cp_{j+1} c_{j} )", generate_by_hand.mpo_XX_model(config_Z2_fermionic, N=N, t=t, mu=0)
    # is CORRECT // no hopping: H_str, H_ref = "+ \sum_{j \in range1} mu cp_{j} c_{j}", generate_by_hand.mpo_XX_model(config_Z2_fermionic, N=N, t=0, mu=0.2)
    H = generate.mpo(H_str, parameters)
    psi = generate_random.mps_random(conf, N=N, Dblock=4, total_parity=0, dtype='float64') +\
        generate_random.mps_random(conf, N=N, Dblock=4, total_parity=1, dtype='float64')
    psi.canonize_sweep(to='first')
    x_ref = yamps.measure_mpo(psi, H_ref, psi)
    x = yamps.measure_mpo(psi, H, psi)
    assert (x_ref - x).real < 1e-15

if __name__ == "__main__":
    # test_generator_mps()
    test_generator_mpo()
