""" basic procedures of single mps """
import numpy as np
import yast
import yast.tn.mps as mps
try:
    from .configs import config_dense as cfg
    # pytest is modifying cfg to inject different backends and divices during tests
except ImportError:
    from configs import config_dense as cfg

tol = 1e-12


def check_copy(psi1, psi2):
    """ Test if two mps-s have the same tensors (values). """
    for n in psi1.sweep():
        assert np.allclose(psi1[n].to_numpy(), psi2[n].to_numpy())
    assert psi1.A is not psi2.A
    assert psi1 is not psi2


def test_copy():
    """ Initialize random mps of full tensors and checks copying. """
    operators = yast.operators.Spin1(sym='Z3', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=16, operators=operators)

    psi = generate.random_mps(n=(1,), D_total=16)
    phi = psi.copy()
    check_copy(psi, phi)

    psi = generate.random_mpo(D_total=16)
    phi = psi.clone()
    check_copy(psi, phi)


if __name__ == "__main__":
    test_copy()
