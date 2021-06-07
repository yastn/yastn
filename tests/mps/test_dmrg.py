
import ops_full
import ops_Z2
import ops_U1
import yamps
import pytest
import logging


def run_dmrg_1site(psi, H, occ, E_target, occ_target, sweeps=10):
    """ Run a faw sweeps of dmrg_1site_sweep. Returns energy. """
    project = []
    for ii, Eng_ii in enumerate(E_target):
        psi2 = psi.copy()
        env = None
        for _ in range(sweeps):
            env = yamps.dmrg_sweep_1site(psi2, H, env=env, project=project)
        EE, Eocc = env.measure(), yamps.measure_mpo(psi2, occ, psi2)
        logging.info("1-site; Energy: %0.8f / %0.8f   Occupation: %0.8f / %0.8f", EE, E_target[ii], Eocc, occ_target[ii])
        assert pytest.approx(EE, rel=1e-6) == Eng_ii
        assert pytest.approx(Eocc, rel=1e-4) == occ_target[ii]
        project.append(psi2)
    return project[0]


def run_dmrg_2site(psi, H, occ, E_target, occ_target, sweeps=10, D_total=32):
    """ Run a faw sweeps of dmrg_2site_sweep. Returns energy. """
    opts_svd = {'tol': 1e-8, 'D_total': D_total}
    project = []
    for ii, Eng_ii in enumerate(E_target):
        psi2 = psi.copy()
        env = None
        for _ in range(sweeps):
            env = yamps.dmrg_sweep_2site(psi2, H, env=env, opts_svd=opts_svd, project=project)
        EE, Eocc = env.measure(), yamps.measure_mpo(psi2, occ, psi2)
        logging.info("2-site; Energy: %0.8f / %0.8f   Occupation: %0.8f / %0.8f", EE, E_target[ii], Eocc, occ_target[ii])
        assert pytest.approx(EE, rel=1e-6) == Eng_ii
        assert pytest.approx(Eocc, rel=1e-4) == occ_target[ii]
        project.append(psi2)
    return project[0]


def test_full_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 7
    Eng_gs = [-3.427339492125848, -3.227339492125848, -2.8619726273956685]
    Occ_gs = [3, 4, 2]

    H = ops_full.mpo_XX_model(N=N, t=1, mu=0.2)
    occ = ops_full.mpo_occupation(N=N)
    Dmax = 8

    logging.info(' Tensor : dense ')

    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_1site(psi, H, occ, Eng_gs, Occ_gs)

    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_2site(psi, H, occ, Eng_gs, Occ_gs, D_total=Dmax)


def test_Z2_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 7
    Dmax = 8
    Occ_gs0 = [4, 2, 4]
    Eng_parity0 = [-3.227339492125848, -2.8619726273956685, -2.461972627395668]
    Occ_gs1 = [3, 3, 5]
    Eng_parity1 = [-3.427339492125848, -2.6619726273956683, -2.261972627395668]

    logging.info(' Tensor : Z2 ')

    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0.2)
    occ = ops_Z2.mpo_occupation(N=N)

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=1, dtype='float64')
    psi.canonize_sweep(to='first')
    psi = run_dmrg_1site(psi, H, occ, Eng_parity1, Occ_gs1)

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=1, dtype='float64')
    psi.canonize_sweep(to='first')
    psi = run_dmrg_2site(psi, H, occ, Eng_parity1, Occ_gs1, D_total=Dmax)

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=0, dtype='float64')
    psi.canonize_sweep(to='first')
    psi = run_dmrg_1site(psi, H, occ, Eng_parity0, Occ_gs0)

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=0, dtype='float64')
    psi.canonize_sweep(to='first')
    psi = run_dmrg_2site(psi, H, occ, Eng_parity0, Occ_gs0, D_total=Dmax)


def test_U1_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 7
    Dmax = 8
    Eng_sectors = {2: [-2.861972627395668, -2.213125929752753, -1.7795804271032745],
                   3: [-3.427339492125848, -2.6619726273956683, -2.0131259297527526],
                   4: [-3.227339492125848, -2.461972627395668, -1.8131259297527529]}

    H = ops_U1.mpo_XX_model(N=N, t=1, mu=0.2)
    occ = ops_U1.mpo_occupation(N=N)

    logging.info(' Tensor : U1 ')

    for total_occ, E_target in Eng_sectors.items():
        psi = ops_U1.mps_random(N=N, Dblocks=[1, 2, 1], total_charge=total_occ)
        psi.canonize_sweep(to='first')
        occ_target = [total_occ] * len(E_target)
        psi = run_dmrg_2site(psi, H, occ, E_target, occ_target, D_total=Dmax)


def test_dmrg():
    """
    Check dmrg with measuring additional expectation values
    """
    N = 7
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0.2)
    Eng_gs = -3.427339492125848

    Dmax = 8
    opts_svd = {'tol': 1e-6, 'D_total': Dmax}

    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    env = yamps.dmrg(psi=psi, H=H, env=None, max_sweeps=10, version='1site', tol_dE=1e-10, opts_svd=opts_svd)
    print('1site: Energy - Eref= ', env.measure()-Eng_gs)

    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    env = yamps.dmrg(psi=psi, H=H, env=None, max_sweeps=10, version='2site', tol_dE=1e-10, opts_svd=opts_svd)
    print('1site: Energy - Eref= ', env.measure()-Eng_gs)


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    test_full_dmrg()
    test_Z2_dmrg()
    test_U1_dmrg()
    test_dmrg()
