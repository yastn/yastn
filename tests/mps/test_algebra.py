""" examples for addition of the Mps-s """
import pytest
import yamps
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


def test_multiplication():
    """ Calculate ground state and checks yamps.multiply() and __mul__()and yamps.add() within eigen-condition."""
    #
    # This test presents a multiplication as a part of DMRG study. 
    # We use multiplication to get expectation values from a state.
    # Knowing exact solution we will compare it to the value we obtain.
    N = 7
    Eng = -3.427339492125848
    #
    # The Hamiltonian is obtained with automatic generator (see source file).
    #
    H = generate_automatic.mpo_XX_model(config_U1_fermionic, N=N, t=1, mu=0.2)
    #
    # To standardize this test we will fix a seed for random MPS we use
    #
    generate_random.random_seed(config_U1_fermionic, seed=0)
    #
    # In this example we use yast.Tensor's with U(1) symmetry. 
    #
    total_charge = 3
    psi = generate_random.mps_random(config_U1_fermionic, N=N, Dblocks=[1, 2, 1], total_charge=total_charge)
    #
    # You always have to start with MPS in right canonical form.
    #
    psi.canonize_sweep(to='first')
    #
    # We set truncation for DMRG and runt the algorithm in '2site' version
    #
    opts_svd = {'tol': 1e-8, 'D_total': 8} 
    env = yamps.dmrg(psi, H, version='2site', max_sweeps=20, opts_svd=opts_svd)
    #
    # Test if we obtained exact solution for the energy?:
    #
    assert pytest.approx(env.measure().item(), rel=tol) == Eng
    #
    # If the code didn't break then we should get a ground state. 
    # Now we calculate the variation of energy <H^2>-<H>^2=<(H-Eng)^2> to check if DMRG converged properly to tol.
    # We have two equivalent ways to do that:
    #
    # case 1/
    Hpsi = yamps.multiply(H, psi)
    #
    # use yamps.measure_overlap to get variation
    #
    p0 = -1 * Hpsi + Eng * psi
    assert yamps.measure_overlap(p0, p0) < tol
    #
    # case 2/
    Hpsi = H @ psi
    p0 = -1 * Hpsi + Eng * psi
    assert yamps.measure_overlap(p0, p0) < tol


if __name__ == "__main__":
    test_addition()
    test_multiplication()
