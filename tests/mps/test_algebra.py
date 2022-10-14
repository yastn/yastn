""" examples for addition of the Mps-s """
import pytest
import yamps
import yast


tol = 1e-6


def check_add(psi0, psi1):
    """ test yamps.add using overlaps"""
    out1 = yamps.add(psi0, psi1, amplitudes=[1., 2.])
    out2 = (1.0 * psi0) + (2.0 * psi1)
    p0 = yamps.measure_overlap(psi0, psi0)
    p1 = yamps.measure_overlap(psi1, psi1)
    p01 = yamps.measure_overlap(psi0, psi1)
    p10 = yamps.measure_overlap(psi1, psi0)
    for out in (out1, out2):
        o1 = yamps.measure_overlap(out, out)
        assert abs(o1 - p0 - 4 * p1 - 2 * p01 - 2 * p10) < tol


def test_addition():
    """create two Mps-s and add them to each other"""
    operators = yast.operators.SpinfulFermions(sym='U1xU1')
    generate = yamps.Generator(N=9, operators=operators)

    psi0 = generate.random_mps(D_total=15, n=(3, 5))
    psi1 = generate.random_mps(D_total=19, n=(3, 5))
    check_add(psi0, psi1)

    psi0 = generate.random_mpo(D_total=12)
    psi1 = generate.random_mpo(D_total=11)
    check_add(psi0, psi1)


def test_multiplication():
    """ Calculate ground state and checks yamps.multiply() and __mul__()and yamps.add() within eigen-condition."""
    #
    # This test presents a multiplication as a part of DMRG study. 
    # We use multiplication to get expectation values from a state.
    # Knowing exact solution we will compare it to the value we obtain.
    N = 7
    Eng = -3.427339492125848
    #
    operators = yast.operators.SpinlessFermions(sym='U1')
    generate = yamps.Generator(N=N, operators=operators)
    #
    # The Hamiltonian is obtained with automatic generator (see source file).
    #
    parameters = {"t": lambda j: 1.0, "mu": lambda j: 0.2, "range1": range(N), "range2": range(N-1)}
    H_str = "\sum_{j \in range2} t ( cp_{j} c_{j+1} + cp_{j+1} c_{j} ) + \sum_{j\in range1} mu cp_{j} c_{j}"
    H = generate.mpo(H_str, parameters)
    #
    # To standardize this test we will fix a seed for random MPS we use
    #
    generate.random_seed(seed=0)
    #
    # In this example we use yast.Tensor's with U(1) symmetry. 
    #
    total_charge = 3
    psi = generate.random_mps(D_total=5, n=total_charge)
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
