""" basic procedures of single mps """
import pytest
import yastn
import yastn.tn.mps as mps
try:
    from .configs import config_dense as cfg
except ImportError:
    from configs import config_dense as cfg
# pytest modifies cfg to inject different backends and divices during tests


def test_canonize(config=cfg, tol=1e-12):
    """ Initialize random mps and checks canonization. """
    opts_config = {} if config is None else \
                {'backend': config.backend, 'default_device': config.default_device}
    N = 16

    ops = yastn.operators.Spin1(sym='Z3', **opts_config)
    I = mps.product_mpo(ops.I(), N=N)
    for n in (0, 1, 2):
        psi = mps.random_mps(I, n=n, D_total=16)
        check_canonize(psi, tol)
    H = mps.random_mpo(I, D_total=8, dtype='complex128')
    check_canonize(H, tol)

    ops = yastn.operators.Spin12(sym='dense', **opts_config)
    psi = mps.random_mps(I, D_total=16, dtype='complex128')
    check_canonize(psi, tol)
    H = mps.random_mpo(I, D_total=8)
    check_canonize(H, tol)

    with pytest.raises(yastn.YastnError):
        psi.orthogonalize_site(4, to="center")
        # "to" should be in "first" or "last"
    with pytest.raises(yastn.YastnError):
        psi.orthogonalize_site(4, to="last")
        psi.orthogonalize_site(5, to="last")
        # Only one central block is allowed. Attach the existing central block before orthogonalizing site.


def check_canonize(psi, tol):
    """ Canonize mps to left and right, running tests if it is canonical. """
    ref_s = (-1, 1, 1) if psi.nr_phys == 1 else (-1, 1, 1, -1)
    norm = psi.norm()
    for to in ('last', 'first'):
        psi.canonize_(to=to, normalize=False)
        assert psi.is_canonical(to=to, tol=tol)
        assert all(psi[site].s == ref_s for site in psi.sweep())
        assert psi.pC is None
        assert len(psi.A) == len(psi)
        assert pytest.approx(psi.factor, rel=tol) == norm
        assert pytest.approx(mps.vdot(psi, psi), rel=tol) == norm ** 2

    for to in ('last', 'first'):
        phi = psi.shallow_copy()
        phi.canonize_(to=to)
        assert abs(phi.factor - 1) < tol
        assert abs(mps.vdot(phi, phi) - 1) < tol


if __name__ == "__main__":
    test_canonize()
