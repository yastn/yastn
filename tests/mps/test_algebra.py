""" examples for addition of the Mps-s """
import pytest
import yamps
import generate_random
import generate_automatic
try:
    from .configs import config_dense, config_dense_fermionic
    from .configs import config_U1, config_U1_fermionic
    from .configs import config_Z2, config_Z2_fermionic
except ImportError:
    from configs import config_dense, config_dense_fermionic
    from configs import config_U1, config_U1_fermionic
    from configs import config_Z2, config_Z2_fermionic



tol = 1e-6


def check_add(psi0, psi1):
    """ test yamps.add using overlaps"""
    out1 = yamps.add(psi0, psi1, amplitudes=[1., 2.])
    o1 = yamps.measure_overlap(out1, out1)
    p0 = yamps.measure_overlap(psi0, psi0)
    p1 = yamps.measure_overlap(psi1, psi1)
    p01 = yamps.measure_overlap(psi0, psi1)
    p10 = yamps.measure_overlap(psi1, psi0)
    assert abs(o1 - p0 - 4 * p1 - 2 * p01 - 2 * p10) < tol


def check_add_mul(psi0, psi1):
    """ test __add__ and __mul__ by a number using overlaps"""
    out1 = (1.0 * psi0) + (2.0 * psi1)
    o1 = yamps.measure_overlap(out1, out1)
    p0 = yamps.measure_overlap(psi0, psi0)
    p1 = yamps.measure_overlap(psi1, psi1)
    p01 = yamps.measure_overlap(psi0, psi1)
    p10 = yamps.measure_overlap(psi1, psi0)
    assert abs(o1 - p0 - 4 * p1 - 2 * p01 - 2 * p10) < tol


def test_addition():
    """create two Mps-s and add them to each other"""
    psi0 = generate_random.mps_random(config_dense, N=8, Dmax=15, d=1)
    psi1 = generate_random.mps_random(config_dense, N=8, Dmax=19, d=1)
    check_add(psi0, psi1)
    check_add_mul(psi0, psi1)

    psi0 = generate_random.mps_random(config_Z2, N=8, Dblock=8, total_parity=0)
    psi1 = generate_random.mps_random(config_Z2, N=8, Dblock=12, total_parity=0)
    check_add(psi0, psi1)
    check_add_mul(psi0, psi1)


def test_multiply():
    """ Calculate ground state and checks yamps.multiply() and __mul__()and yamps.add() within eigen-condition."""
    generate_random.random_seed(config_U1_fermionic, seed=0)
    N = 7
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    Eng = -3.427339492125848
    total_charge = 3
    H = generate_automatic.mpo_XX_model(config_U1_fermionic, N=N, t=1, mu=0.2)

    psi = generate_random.mps_random(config_U1_fermionic, N=N, Dblocks=[1, 2, 1], total_charge=total_charge).canonize_sweep(to='first')
    env = yamps.dmrg(psi, H, version='2site', max_sweeps=20, opts_svd=opts_svd)

    assert pytest.approx(env.measure().item(), rel=tol) == Eng

    Hpsi = yamps.multiply(H, psi)
    assert pytest.approx(yamps.measure_overlap(Hpsi, Hpsi).item(), rel=tol) == Eng ** 2
    p0 = yamps.add(Hpsi, psi, amplitudes=[1, -Eng])
    assert yamps.measure_overlap(p0, p0) < tol  # == 0.
    p0 = yamps.add(Hpsi * -1, Eng * psi)
    assert yamps.measure_overlap(p0, p0) < tol  # == 0.


    Hpsi = H @ psi
    assert pytest.approx(yamps.measure_overlap(Hpsi, Hpsi).item(), rel=tol) == Eng ** 2
    p0 = yamps.add(Hpsi, psi, amplitudes=[1, -Eng])
    assert yamps.measure_overlap(p0, p0) < tol  # == 0.
    p0 = yamps.add(Hpsi * -1, Eng * psi)
    assert yamps.measure_overlap(p0, p0) < tol  # == 0.


if __name__ == "__main__":
    test_addition()
    test_multiply()
