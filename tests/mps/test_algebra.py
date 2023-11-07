""" examples for addition of the Mps-s """
import pytest
import yastn.tn.mps as mps
import yastn
try:
    from .configs import config_dense as cfg
    # pytest modifies cfg to inject different backends and devices during tests
except ImportError:
    from configs import config_dense as cfg


def check_add(psi0, psi1, tol):
    """ series of test of mps.add performed on provided psi0 and psi1"""
    out1 = mps.add(psi0, psi1, amplitudes=[1., 2.])
    out2 = (1.0 * psi0) + (2.0 * psi1)
    p0 = mps.measure_overlap(psi0, psi0)
    p1 = mps.measure_overlap(psi1, psi1)
    p01 = mps.measure_overlap(psi0, psi1)
    p10 = mps.measure_overlap(psi1, psi0)
    for out in (out1, out2):
        o1 = mps.measure_overlap(out, out)
        assert abs(o1 - p0 - 4 * p1 - 2 * p01 - 2 * p10) < tol


def test_addition_basic(tol=1e-12):
    import yastn.tn.mps as mps
    import yastn

    # Define random MPS's without any symmetry
    #
    config_dense = yastn.make_config()
    psi0 = mps.random_dense_mps(N=8, D=5, d=2)
    psi1 = mps.random_dense_mps(N=8, D=3, d=2)

    # We want to calculate: res = psi0 + 2 * psi1. There is a couple of ways:
    # A/
    resA = mps.add(psi0, 2.0 * psi1)

    # B/
    resB = mps.add(psi0, psi1, amplitudes=[1.0, 2.0])

    # C/
    resC = psi0 + 2.0 * psi1

    nresA, nresB, nresC = resA.norm(), resB.norm(), resC.norm()
    assert abs(mps.vdot(resA, resB) / (nresA * nresB) - 1) < tol
    assert abs(mps.vdot(resA, resC) / (nresA * nresC) - 1) < tol
    assert (x.get_bond_dimensions == (1, 8, 8, 8, 8, 8, 8, 8, 1) for x in (resA, resB, resC))


def test_addition(tol=1e-8):
    """Create two Mps-s and add them to each other."""

    ops = yastn.operators.SpinfulFermions(sym='U1xU1', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=9, operators=ops)

    psi0 = generate.random_mps(D_total=15, n=(3, 5))
    psi1 = generate.random_mps(D_total=19, n=(3, 5))
    check_add(psi0, psi1, tol)

    psi0 = generate.random_mpo(D_total=12)
    psi1 = generate.random_mpo(D_total=11)
    check_add(psi0, psi1, tol)


@pytest.mark.parametrize("kwargs", [{'config': cfg}])
def test_multiplication_example_gs(kwargs):
    multiplication_example_gs(**kwargs)

def multiplication_example_gs(config=None, tol=1e-12):
    """
    Calculate ground state and tests, within eigen-condition,
    functions mps.multiply, __mul__, mps.zipper and mps.add

    This test presents multiplication as part of the DMRG study.
    We use multiplication to get expectation values from a state.
    Knowing the exact solution, we will compare it to the value we obtain.
    """
    N = 7
    Eng = -3.427339492125
    t, mu = 1.0, 0.2
    #
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject backend and device for testing
    ops = yastn.operators.SpinlessFermions(sym='U1', **opts_config)
    #
    # The Hamiltonian is obtained using Hterm.
    #
    I = mps.product_mpo(ops.I(), N)  # identity MPO
    terms = []
    for i in range(N - 1):
        terms.append(mps.Hterm(t, (i, i+1), (ops.cp(), ops.c())))
        terms.append(mps.Hterm(t, (i+1, i), (ops.cp(), ops.c())))
    for i in range(N):
        terms.append(mps.Hterm(mu, [i], [ops.cp() @ ops.c()]))
    H = mps.generate_mpo(I, terms)
    #
    # To standardize this test we will fix a seed for random MPS we use.
    #
    ops.random_seed(seed=0)
    #
    # In this example we use yastn.Tensor with U(1) symmetry.
    #
    total_charge = 3
    psi = mps.random_mps(I, D_total=8, n=total_charge)
    #
    # We set truncation for DMRG and run the algorithm in '2site' version.
    #
    opts_svd = {'D_total': 8}
    opts_dmrg = {'max_sweeps': 20, 'Schmidt_tol': 1e-14}
    out = mps.dmrg_(psi, H, method='2site', **opts_dmrg, opts_svd=opts_svd)
    out = mps.dmrg_(psi, H, method='1site', **opts_dmrg)
    #
    # Test if we obtained the exact solution for energy?:
    #
    assert abs(out.energy - Eng) < tol
    #
    # We should have the ground state.
    # Now we calculate the variation of energy <H^2>-<H>^2=<(H-Eng)^2>
    # to check if DMRG converged properly to tol.
    # There are a few ways to do that:
    #
    # case 1/
    # Find approximate Mps representing H @ psi. This is done in two steps.
    # First, zipper function gives a good first approximation
    # though a series of svd.
    #
    Hpsi = mps.zipper(H, psi, opts={"D_total": 8})
    #
    # Second, compression is optimizing the overlap
    # between approximation and a product H @ psi.
    #
    mps.compression_(Hpsi, (H, psi), method='2site', max_sweeps=5,
                     Schmidt_tol=1e-6, opts_svd={"D_total": 8},
                     normalize=False)
    #
    # Use mps.norm() to get variation.
    # Norm canonizes a copy of the state and, close to zero,
    # is more precise than direct contraction using vdot.
    #
    p0 = Eng * psi - Hpsi
    assert p0.norm() < tol
    #
    # case 2/
    # Here H @ psi is calculated exactly with resulting
    # bond dimension being a product of bond dimensions of H and psi.
    #
    Hpsi = mps.multiply(H, psi)
    p0 = Eng * psi - Hpsi
    assert p0.norm() < tol
    #
    # Equivalently we can call.
    Hpsi = H @ psi
    p0 = Eng * psi - Hpsi
    assert p0.norm() < tol


if __name__ == "__main__":
    test_addition()
    test_addition_basic()
    multiplication_example_gs()
