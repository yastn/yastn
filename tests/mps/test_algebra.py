""" examples for addition of the Mps-s """
import pytest
import yast.tn.mps as mps
import yast
try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense as cfg

tol = 1e-6


def check_add(psi0, psi1):
    """ test mps.add using overlaps"""
    out1 = mps.add(psi0, psi1, amplitudes=[1., 2.])
    out2 = (1.0 * psi0) + (2.0 * psi1)
    p0 = mps.measure_overlap(psi0, psi0)
    p1 = mps.measure_overlap(psi1, psi1)
    p01 = mps.measure_overlap(psi0, psi1)
    p10 = mps.measure_overlap(psi1, psi0)
    for out in (out1, out2):
        o1 = mps.measure_overlap(out, out)
        assert abs(o1 - p0 - 4 * p1 - 2 * p01 - 2 * p10) < tol


def test_addition():
    """create two Mps-s and add them to each other"""
    operators = yast.operators.SpinfulFermions(sym='U1xU1', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=9, operators=operators)

    psi0 = generate.random_mps(D_total=15, n=(3, 5))
    psi1 = generate.random_mps(D_total=19, n=(3, 5))
    check_add(psi0, psi1)

    psi0 = generate.random_mpo(D_total=12)
    psi1 = generate.random_mpo(D_total=11)
    check_add(psi0, psi1)


def test_multiplication():
    """ Calculate ground state and checks mps.multiply() and __mul__()and mps.add() within eigen-condition."""
    #
    # This test presents a multiplication as a part of DMRG study. 
    # We use multiplication to get expectation values from a state.
    # Knowing exact solution we will compare it to the value we obtain.
    N = 7
    Eng = -3.427339492125848
    #
    operators = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=N, operators=operators)
    #
    # The Hamiltonian is obtained with automatic generator (see source file).
    #
    parameters = {"t": 1.0, "mu": 0.2, "rangeN": range(N), "rangeNN": zip(range(N-1),range(1,N))}
    H_str = "\sum_{i,j \in rangeNN} t ( cp_{i} c_{j} + cp_{j} c_{i} ) + \sum_{j\in rangeN} mu cp_{j} c_{j}"
    H = generate.mpo_from_latex(H_str, parameters)
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
    env = mps.dmrg(psi, H, version='2site', max_sweeps=20, opts_svd=opts_svd)
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
    Hpsi = mps.multiply(H, psi)
    #
    # use mps.measure_overlap to get variation
    #
    p0 = -1 * Hpsi + Eng * psi
    assert mps.measure_overlap(p0, p0) < tol
    #
    # case 2/
    Hpsi = H @ psi
    p0 = -1 * Hpsi + Eng * psi
    assert mps.measure_overlap(p0, p0) < tol


if __name__ == "__main__":
    test_addition()
    test_multiplication()
