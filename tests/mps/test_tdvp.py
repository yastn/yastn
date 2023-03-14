""" mps.tdvp_ """
import logging
import pytest
import yast
import yast.tn.mps as mps
try:
    from .configs import config_dense as cfg
    # pytest modifies cfg to inject different backends and devices during tests
except ImportError:
    from configs import config_dense as cfg

tol=1e-8


def run_tdvp_imag(psi, H, time, steps, Eng_gs, opts_svd=None):
    """ Run a few sweeps in imaginary time of tdvp_1site_sweep. """
    # In order to converge faster we cheat and bring random MPS closer to a ground state using DMRG-1site.
    # After cheat-move we have MPS of the energy Eng_old but we want to bring it closer to 
    # a ground state by imaginary time evolution.
    #
    out = mps.dmrg_(psi, H, method='2site', opts_svd=opts_svd, max_sweeps=2)
    Eng_old = out.energy.item()
    #
    # We set parameters for exponentiation in TDVP giving the information on the operator 
    # and setting what is desired Krylow dimension opts_expmv['ncv'] and truncation of Arnoldin algorithm opts_expmv['tol'].
    #
    opts_expmv = {'hermitian': True, 'ncv': 5, 'tol': 1e-8}
    #
    # We do evolve the MPS by dt and repeat it sweeps-time to observe the convergence.
    #
    for method in ('2site', '12site', '1site'):
        dt = time / steps
        times = [n * dt for n in range(steps + 1)]
        for out in mps.tdvp_(psi, H, times=times, dt=dt, u=1, method=method, opts_expmv=opts_expmv, opts_svd=opts_svd):
            Eng = mps.vdot(psi, H, psi)
            logging.info("%s tdvp  t = %0.3f  energy = %0.8f", method, out.tf, Eng)
            print(out)
            assert (Eng - Eng_old).real < tol
            Eng_old = Eng
    logging.info("%s tdvp; Energy: %0.8f / %0.8f", method, Eng, Eng_gs)
    #
    # Finally we can check if we obtained a ground state:
    #
    assert pytest.approx(Eng, rel=1e-3) == Eng_gs
    return psi


def test_dense_tdvp():
    """
    Initialize random mps of dense tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    # Knowing the exact solution for a ground state energy we can compare 
    # it to our DMRG result.
    #
    N = 7
    Eng_gs = -3.427339492125848
    #
    # The Hamiltonian is obtained with automatic generator (see source file).
    #
    operators = yast.operators.Spin12(sym='dense', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=N, operators=operators)
    parameters = {"t": 1.0, "mu": 0.2, "rangeN": range(N), "rangeNN": zip(range(N-1),range(1,N))}
    H_str = "\sum_{i,j \in rangeNN} t ( sp_{i} sm_{j} + sp_{j} sm_{i} ) + \sum_{j\in rangeN} mu sp_{j} sm_{j}"
    H = generate.mpo_from_latex(H_str, parameters)
    #
    # To standardize this test we will fix a seed for random MPS we use
    #
    generate.random_seed(seed=0)
    #
    # In this example we use yast.Tensor's with no symmetry imposed. 
    #
    logging.info(' Tensor : dense ')
    #
    # Set options for truncation for '2site' method of mps.tdvp.
    #
    D_total = 6
    opts_svd = {'tol': 1e-6, 'D_total': D_total}
    #
    # We define how long the imaginary time evolution is run. 
    # In this example we run mps.tdvp sweeps-times and each run evolves 
    # initial state by a time dt, such that we obtain: 
    # psi(t+dt) = exp( dt * H) @ psi(t)
    #
    time = 2.
    steps = 20
    #
    # Finally run TDVP starting:
    psi = generate.random_mps(D_total=D_total, sigma=2)
    #
    # Single run can be done using:
    #
    # env = mps.tdvp_(psi, H, env=env, dt=dt, method=method, opts_expmv=opts_expmv, opts_svd=opts_svd)
    #
    # To explain how we iterate over sweeps we create a subfunction run_tdvp_imag.
    # This is not necessary but we do it for the sake of clarity.
    #
    run_tdvp_imag(psi, H, time, steps, Eng_gs=Eng_gs, opts_svd=opts_svd)


def test_Z2_tdvp():
    """
    Initialize random mps of Z2 tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 7
    D_total = 6
    time = 2.
    steps = 30
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    logging.info(' Tensor : Z2 ')

    operators = yast.operators.SpinlessFermions(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    Eng_gs = {0: -3.227339492125848, 1: -3.427339492125848}
    parameters = {"t": 1.0, "mu": 0.2, "rangeN": range(N), "rangeNN": zip(range(N-1),range(1,N))}
    H_str = "\sum_{i,j \in rangeNN} t ( cp_{i} c_{j} + cp_{j} c_{i} ) + \sum_{j\in rangeN} mu cp_{j} c_{j}"
    H = generate.mpo_from_latex(H_str, parameters)

    for parity in (0, 1):
        psi = generate.random_mps(D_total=D_total, n=parity, sigma=2)
        psi.canonize_(to='first')
        run_tdvp_imag(psi, H, time, steps, Eng_gs=Eng_gs[parity], opts_svd=opts_svd)


def test_U1_tdvp():
    """
    Initialize random mps of U(1) tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 7
    D_total = 6
    time = 2.
    steps = 20
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    operators = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    logging.info(' Tensor : U1 ')

    Eng_gs = {2: -2.861972627395668, 3: -3.427339492125848, 4:-3.227339492125848}
    parameters = {"t": 1.0, "mu": 0.2, "rangeN": range(N), "rangeNN": zip(range(N-1),range(1,N))}
    H_str = "\sum_{i,j \in rangeNN} t ( cp_{i} c_{j} + cp_{j} c_{i} ) + \sum_{j\in rangeN} mu cp_{j} c_{j}"
    H = generate.mpo_from_latex(H_str, parameters)

    for charge in Eng_gs.keys():
        psi = generate.random_mps(D_total=D_total, n=charge, sigma=1)
        psi.canonize_(to='first')
        run_tdvp_imag(psi, H, time, steps, Eng_gs=Eng_gs[charge], opts_svd=opts_svd)


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    test_dense_tdvp()
    test_Z2_tdvp()
    test_U1_tdvp()
