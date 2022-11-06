import pytest
import yast
import yast.tn.mps as mps
try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense as cfg

tol = 1e-12


def test_generator_mps():
    N = 10
    D_total = 16

    for sym, nn in (('Z2', (0,)), ('Z2', (1,)), ('U1', (N // 2,))):
        operators = yast.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        generate = mps.Generator(N, operators)

        I = generate.I()
        all(I[site].s == (-1, 1, 1, -1) for site in I.sweep())

        assert pytest.approx(mps.measure_overlap(I, I).item(), rel=tol) == 2 ** N
        O = I @ I + (-1 * I)
        assert pytest.approx(mps.measure_overlap(O, O).item(), abs=tol) == 0
        n0 = (0,) * len(nn)

        states = (generate.random_mps(D_total=D_total, n = nn), generate.random_mpo(D_total=D_total))
        for psi, ref_s, ref_n in zip(states, ((-1, 1, 1), (-1, 1, 1, -1)), (nn, n0)):
            vlf, vll = psi.virtual_leg('first'), psi.virtual_leg('last')
            assert vlf.t == (ref_n,) and vlf.D == (1,)
            assert vll.t == (n0,) and vlf.D == (1,)
            all(psi[site].s == ref_s for site in psi.sweep())
            bds = psi.get_bond_dimensions()
            assert bds[0] == bds[-1] == 1 and len(bds) == N + 1
            assert all(bd > D_total/2 for bd in bds[2:-2])


if __name__ == "__main__":
    test_generator_mps()
