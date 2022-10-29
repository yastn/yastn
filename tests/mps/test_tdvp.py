""" yamps.tdvp """
import logging
import pytest
import yamps
import yast


tol=1e-8


def run_tdvp_imag(psi, H, dt, Eng_gs, sweeps, version='1site', opts_svd=None):
    """ Run a faw sweeps in imaginary time of tdvp_1site_sweep. """
    # In order to converge faster we cheat and bring random MPS closer to a ground state using DMRG-1site.
    # After cheat-move we have MPS of the energy Eng_old but we want to bring it closer to 
    # a ground state by imaginary time evolution.
    #
    env = yamps.dmrg_sweep_1site(psi, H)
    Eng_old = env.measure()
    #
    # We set parameters for exponentiation in TDVP giving the information on the operator 
    # and setting what is desired Krylow dimension opts_expmv['ncv'] and truncation of Arnoldin algorithm opts_expmv['tol'].
    #
    opts_expmv = {'hermitian': True, 'ncv': 5, 'tol': 1e-8}
    #
    # We do evolve the MPS by dt and repeat it sweeps-time to observe the convergence.
    #
    for _ in range(sweeps):
        #
        # We run TDVP evolution
        #
        env = yamps.tdvp(psi, H, env=env, dt=dt, version=version, opts_expmv=opts_expmv, opts_svd=opts_svd)
        #
        # After the evolution the energy is:
        #
        Eng = env.measure()
        #
        # We check how much it changes comparing to energy before TDVP. 
        # If the state is not converged we do another TDVP step.
        assert (Eng - Eng_old).real < tol
        Eng_old = Eng
    logging.info("%s tdvp; Energy: %0.8f / %0.8f", version, Eng, Eng_gs)
    #
    # Finally we can check if we obtained a ground state:
    #
    assert pytest.approx(Eng.item(), rel=2e-2) == Eng_gs
    return psi


def test_dense_tdvp():
    """
    Initialize random mps of dense tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    # Knowing the exact solution for a ground state energy we can compare 
    # it to our DMRG result.
    #
    N = 5
    Eng_gs = -2.232050807568877
    #
    # The Hamiltonian is obtained with automatic generator (see source file).
    #
    operators = yast.operators.Spin12(sym='dense')
    generate = yamps.Generator(N=N, operators=operators)
    parameters = {"t": lambda j: 1.0, "mu": lambda j: 0.25, "range1": range(N), "range2": range(N-1)}
    H_str = "\sum_{j \in range2} t ( sp_{j} sm_{j+1} + sp_{j+1} sm_{j} ) + \sum_{j\in range1} mu sp_{j} sm_{j}"
    H = generate.mpo(H_str, parameters)
    #
    # To standardize this test we will fix a seed for random MPS we use
    #
    generate.random_seed(seed=0)
    #
    # In this example we use yast.Tensor's with no symmetry imposed. 
    #
    logging.info(' Tensor : dense ')
    #
    # Set options for truncation for '2site' version of yamps.tdvp.
    #
    D_total = 4
    opts_svd = {'tol': 1e-6, 'D_total': D_total}
    #
    # We define how long the imaginary time evolution is run. 
    # In this example we run yamps.tdvp sweeps-times and each run evolves 
    # initial state by a time dt, such that we obtain: 
    # psi(t+dt) = exp( dt * H) @ psi(t)
    #
    dt = -.25
    sweeps = 10
    #
    # Finally run TDVP starting:
    for version in ('1site', '2site'):
        psi = generate.random_mps(D_total=D_total)
        #
        # The initial guess has to be prepared in right canonical form!
        #
        psi.canonize_sweep(to='first')
        #
        # Single run can be done using:
        #
        # env = yamps.tdvp(psi, H, env=env, dt=dt, version=version, opts_expmv=opts_expmv, opts_svd=opts_svd)
        #
        # To explain how we iterate over sweeps we create a subfunction run_tdvp_imag.
        # This is not necessary but we do it for the sake of clarity.
        #
        run_tdvp_imag(psi, H, dt=dt, Eng_gs=Eng_gs, sweeps=sweeps, version=version, opts_svd=opts_svd)


def test_Z2_tdvp():
    """
    Initialize random mps of Z2 tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 5
    D_total = 6
    dt = -.25
    sweeps = 10
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    logging.info(' Tensor : Z2 ')

    operators = yast.operators.SpinlessFermions(sym='Z2')
    generate = yamps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    Eng_gs = {0: -2.232050807568877, 1: -1.982050807568877}
    parameters = {"t": lambda j: 1.0, "mu": lambda j: 0.25, "range1": range(N), "range2": range(N-1)}
    H_str = "\sum_{j \in range2} t ( cp_{j} c_{j+1} + cp_{j+1} c_{j} ) + \sum_{j\in range1} mu cp_{j} c_{j}"
    H = generate.mpo(H_str, parameters)

    for parity in (0, 1):
        psi = generate.random_mps(D_total=D_total, n=parity)
        psi.canonize_sweep(to='first')
        for version in ('2site', '1site'):
            run_tdvp_imag(psi, H, dt=dt, Eng_gs=Eng_gs[parity], sweeps=sweeps, version=version, opts_svd=opts_svd)


def test_U1_tdvp():
    """
    Initialize random mps of U(1) tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 5
    D_total = 5
    dt = -.25
    sweeps = 10
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    operators = yast.operators.SpinlessFermions(sym='U1')
    generate = yamps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)

    logging.info(' Tensor : U1 ')

    Eng_gs = {1: -1.482050807568877, 2: -2.232050807568877}
    parameters = {"t": lambda j: 1.0, "mu": lambda j: 0.25, "range1": range(N), "range2": range(N-1)}
    H_str = "\sum_{j \in range2} t ( cp_{j} c_{j+1} + cp_{j+1} c_{j} ) + \sum_{j\in range1} mu cp_{j} c_{j}"
    H = generate.mpo(H_str, parameters)

    for charge in (1, 2):
        psi = generate.random_mps(D_total=D_total, n=charge)
        psi.canonize_sweep(to='first')
        for version in ('2site', '12site', '1site'):
            run_tdvp_imag(psi, H, dt=dt, Eng_gs=Eng_gs[charge], sweeps=sweeps, version=version, opts_svd=opts_svd)


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    test_dense_tdvp()
    test_Z2_tdvp()
    test_U1_tdvp()
