""" basic procedures of single mps """
import numpy as np
try:
    from . import generate_random, generate_by_hand, generate_automatic
    from .configs import config_dense, config_dense_fermionic
    from .configs import config_U1, config_U1_fermionic
    from .configs import config_Z2, config_Z2_fermionic
except ImportError:
    import generate_random, generate_by_hand, generate_automatic
    from configs import config_dense, config_dense_fermionic
    from configs import config_U1, config_U1_fermionic
    from configs import config_Z2, config_Z2_fermionic



tol = 1e-12


def check_copy(psi1, psi2):
    """ Test if two mps-s have the same tensors (velues). """
    for n in psi1.sweep():
        assert np.allclose(psi1.A[n].to_numpy(), psi2.A[n].to_numpy())


def test_full_copy():
    """ Initialize random mps of full tensors and checks copying. """
    psi = generate_random.mps_random(config_dense, N=16, Dmax=15, d=2)
    phi = psi.copy()
    check_copy(psi, phi)

    psi = generate_random.mps_random(config_dense, N=16, Dmax=19, d=[2, 3])
    phi = psi.copy()
    check_copy(psi, phi)

    psi = generate_random.mpo_random(config_dense, N=16, Dmax=25, d=[2, 3], d_out=[2, 1])
    phi = psi.copy()
    check_copy(psi, phi)


def test_Z2_copy():
    """ Initialize random mps of full tensors and checks copying. """
    psi = generate_random.mps_random(config_Z2, N=16, Dblock=25, total_parity=0)
    phi = psi.copy()
    check_copy(psi, phi)

    psi = generate_random.mps_random(config_Z2, N=16, Dblock=25, total_parity=1)
    phi = psi.copy()
    check_copy(psi, phi)


if __name__ == "__main__":
    test_full_copy()
    test_Z2_copy()
