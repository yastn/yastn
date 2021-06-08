
import logging
import pytest
import ops_full
import ops_Z2
import ops_U1
import yamps

tol = 1e-6

def run_dmrg(psi, H, occ, E_target, occ_target, version='1site', opts_svd=None):
    """ Run a faw sweeps of dmrg_1site_sweep. Returns energy. """
    project = []
    for ii, Eng_ii in enumerate(E_target):
        psi2 = psi.copy()
        env, info = yamps.dmrg(psi2, H, project=project, version=version,
                        converge='energy', atol=tol/10, max_sweeps=20, opts_svd=opts_svd, return_info=True)
        EE, Eocc = env.measure(), yamps.measure_mpo(psi2, occ, psi2)
        logging.info("%s dmrg; Energy: %0.8f / %0.8f   Occupation: %0.8f / %0.8f",
                        version, EE, E_target[ii], Eocc, occ_target[ii])
        logging.info(" Convergence info: %s", info)
        assert pytest.approx(EE, rel=tol) == Eng_ii
        assert pytest.approx(Eocc, rel=10 * tol) == occ_target[ii]
        project.append(psi2)
    return project[0]


def test_full_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 7
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    logging.info(' Tensor : dense ')

    Eng_gs = [-3.427339492125848, -3.227339492125848, -2.8619726273956685]
    Occ_gs = [3, 4, 2]
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0.2)
    occ = ops_full.mpo_occupation(N=N)

    for version in ('1site', '2site'):
        psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2).canonize_sweep(to='first')
        psi = run_dmrg(psi, H, occ, Eng_gs, Occ_gs, version=version, opts_svd=opts_svd)


def test_Z2_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 7
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    logging.info(' Tensor : Z2 ')

    Occ_target = {0: [4, 2, 4], 1: [3, 3, 5]}
    Eng_target = {0: [-3.227339492125848, -2.8619726273956685, -2.461972627395668],
                  1: [-3.427339492125848, -2.6619726273956683, -2.261972627395668]}
    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0.2)
    occ = ops_Z2.mpo_occupation(N=N)

    for parity in (0, 1):
        for version in ('1site', '2site'):
            psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=parity, dtype='float64')
            psi.canonize_sweep(to='first')
            psi = run_dmrg(psi, H, occ, Eng_target[parity], Occ_target[parity], version=version, opts_svd=opts_svd)


def test_U1_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 7
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    logging.info(' Tensor : U1 ')

    Eng_sectors = {2: [-2.861972627395668, -2.213125929752753, -1.7795804271032745],
                   3: [-3.427339492125848, -2.661972627395668, -2.0131259297527526],
                   4: [-3.227339492125848, -2.461972627395668, -1.8131259297527529]}
    H = ops_U1.mpo_XX_model(N=N, t=1, mu=0.2)
    occ = ops_U1.mpo_occupation(N=N)

    for total_occ, E_target in Eng_sectors.items():
        psi = ops_U1.mps_random(N=N, Dblocks=[1, 2, 1], total_charge=total_occ).canonize_sweep(to='first')
        occ_target = [total_occ] * len(E_target)
        psi = run_dmrg(psi, H, occ, E_target, occ_target, version='2site', opts_svd=opts_svd)


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    test_full_dmrg()
    test_Z2_dmrg()
    test_U1_dmrg()
