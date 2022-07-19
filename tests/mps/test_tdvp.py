""" yamps.tdvp """
import logging
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

tol=1e-8


def run_tdvp_imag(psi, H, dt, Eng_gs, sweeps, version='1site', opts_svd=None):
    """ Run a faw sweeps in imaginary time of tdvp_1site_sweep. """
    env = yamps.dmrg_sweep_1site(psi, H)
    opts_expmv = {'hermitian': True, 'ncv': 5, 'tol': 1e-8}
    Eng_old = env.measure()
    for _ in range(sweeps):
        env = yamps.tdvp(psi, H, env=env, dt=dt, version=version, opts_expmv=opts_expmv, opts_svd=opts_svd)
        Eng = env.measure()
        assert Eng < Eng_old + tol
        Eng_old = Eng
    logging.info("%s tdvp; Energy: %0.8f / %0.8f", version, Eng, Eng_gs)
    assert pytest.approx(Eng.item(), rel=2e-2) == Eng_gs
    return psi


def test_full_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    generate_random.random_seed(config_dense_fermionic, seed=0)
    N = 5
    dt = -.25
    sweeps = 10
    D_total = 4
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    logging.info(' Tensor : dense ')

    Eng_gs = -2.232050807568877
    H = generate_automatic.mpo_XX_model(config_dense_fermionic, N=N, t=1, mu=0.25)

    for version in ('1site', '2site'):
        psi = generate_random.mps_random(config_dense_fermionic, N=N, Dmax=D_total, d=2).canonize_sweep(to='first')
        run_tdvp_imag(psi, H, dt=dt, Eng_gs=Eng_gs, sweeps=sweeps, version=version, opts_svd=opts_svd)


def test_Z2_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    generate_random.random_seed(config_Z2_fermionic, seed=0)
    N = 5
    D_total = 4
    dt = -.25
    sweeps = 10
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    logging.info(' Tensor : Z2 ')

    Eng_gs = {0: -2.232050807568877, 1: -1.982050807568877}
    H = generate_automatic.mpo_XX_model(config_Z2_fermionic, N=N, t=1, mu=0.25)

    for parity in (0, 1):
        for version in ('1site', '2site'):
            psi = generate_random.mps_random(config_Z2_fermionic, N=N, Dblock=D_total/2, total_parity=parity)
            psi.canonize_sweep(to='first')
            run_tdvp_imag(psi, H, dt=dt, Eng_gs=Eng_gs[parity], sweeps=sweeps, version=version, opts_svd=opts_svd)


def test_U1_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    generate_random.random_seed(config_U1_fermionic, seed=0)
    N = 5
    D_total = 4
    dt = -.25
    sweeps = 10
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    logging.info(' Tensor : U1 ')

    Eng_gs = {1: -1.482050807568877, 2: -2.232050807568877}
    H = generate_automatic.mpo_XX_model(config_U1_fermionic, N=N, t=1, mu=0.25)

    for charge in (1, 2):
        for version in ('1site', '2site', '12site'):
            psi = generate_random.mps_random(config_U1_fermionic, N=N, Dblocks=[1, 2, 1], total_charge=charge)
            psi.canonize_sweep(to='first')
            run_tdvp_imag(psi, H, dt=dt, Eng_gs=Eng_gs[charge], sweeps=sweeps, version=version, opts_svd=opts_svd)


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    test_full_tdvp()
    test_Z2_tdvp()
    test_U1_tdvp()
