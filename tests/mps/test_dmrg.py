""" dmrg tested on XX model. """
import logging
import pytest
import yamps
import generate_random
import generate_by_hand
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

def run_dmrg(psi, H, occ, E_target, occ_target, version='1site', opts_svd=None):
    """ Run a faw sweeps of dmrg_1site_sweep. Returns energy. """
    # To obtain states above ground state be project lower-lying states. 
    # The list project keeps all lower lying states which we obtain iteratively in this code.
    #
    project = []
    for ii, Eng_ii in enumerate(E_target):
        #
        # In the loop we will find a state and then check its total occupation number and energy. 
        #
        # We copy initial random MPS psi. This creates an independent MPS which can be used as initial 
        # guess for DMRG.
        #
        psi2 = psi.copy()
        #
        # We set up dmrg to converge according to energy.
        #
        env, info = yamps.dmrg(psi2, H, project=project, version=version,
                        converge='energy', atol=tol/10, max_sweeps=20, opts_svd=opts_svd, return_info=True)
        #
        # The energy can be extracted from env we generated.
        # Occupation number has to be calcuted with measure_mpo.
        #
        EE, Eocc = env.measure(), yamps.measure_mpo(psi2, occ, psi2)
        #
        # Print the result:
        #
        logging.info("%s dmrg; Energy: %0.8f / %0.8f   Occupation: %0.8f / %0.8f",
                        version, EE, E_target[ii], Eocc, occ_target[ii])
        logging.info(" Convergence info: %s", info)
        #
        # Check if DMRG reached exact energy:
        #
        assert pytest.approx(EE.item(), rel=tol) == Eng_ii
        #
        # and if occupation is as we expected:
        #
        assert pytest.approx(Eocc.item(), rel=10 * tol) == occ_target[ii]
        #
        # The loop allows to find MPS states with increasing energy. 
        # We append the list of lower-lying states to target next excited state.
        #
        project.append(psi2)
    return project[0]

def test_dense_dmrg():
    """
    Initialize random mps of dense tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    # Knowing exact solution we can compare it DMRG result.
    # In this test we will conside sectors of different occupation.
    #
    N = 7
    Eng_gs = [-3.427339492125848, -3.227339492125848, -2.8619726273956685]
    Occ_gs = [3, 4, 2]
    #
    # The Hamiltonian is obtained with automatic generator (see source file).
    #
    H = generate_automatic.mpo_XX_model(config_dense_fermionic, N=N, t=1, mu=0.2)
    #
    # and MPO to measure occupation:
    #
    occ = generate_automatic.mpo_occupation(config_dense_fermionic, N=N)
    #
    # To standardize this test we will fix a seed for random MPS we use
    #
    generate_random.random_seed(config_dense_fermionic, seed=0)
    #
    # In this example we use yast.Tensor's with no symmetry imposed. 
    #
    logging.info(' Tensor : dense ')
    #
    # Set options for truncation for '2site' version of yamps.dmrg.
    #
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}
    #
    # Finally run DMRG starting from random MPS psi:
    #
    for version in ('1site', '2site'):
        psi = generate_random.mps_random(config_dense_fermionic, N=N, Dmax=Dmax, d=2)
        #
        # The initial guess has to be prepared in right canonical form!
        #
        psi.canonize_sweep(to='first')
        #
        # Single run for a ground state can be done using:
        #
        # env, info = yamps.dmrg(psi, H, version=version, converge='energy', atol=tol/10, max_sweeps=20, opts_svd=opts_svd, return_info=True)
        #
        # To explain how to target some sectors for occupation we create a subfunction run_dmrg
        # This is not necessary but we do it for the sake of clarity.
        #
        psi = run_dmrg(psi, H, occ, Eng_gs, Occ_gs, version=version, opts_svd=opts_svd)


def test_Z2_dmrg():
    """
    Initialize random mps of Z2 tensors and checks canonization
    """
    generate_random.random_seed(config_Z2_fermionic, seed=0)
    N = 7
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    logging.info(' Tensor : Z2 ')

    Occ_target = {0: [4, 2, 4], 1: [3, 3, 5]}
    Eng_target = {0: [-3.227339492125848, -2.8619726273956685, -2.461972627395668],
                  1: [-3.427339492125848, -2.6619726273956683, -2.261972627395668]}
    H = generate_automatic.mpo_XX_model(config_Z2_fermionic, N=N, t=1, mu=0.2)
    occ = generate_automatic.mpo_occupation(config_Z2_fermionic, N=N)

    for parity in (0, 1):
        for version in ('1site', '2site'):
            psi = generate_random.mps_random(config_Z2_fermionic, N=N, Dblock=Dmax/2, total_parity=parity, dtype='float64')
            psi.canonize_sweep(to='first')
            psi = run_dmrg(psi, H, occ, Eng_target[parity], Occ_target[parity], version=version, opts_svd=opts_svd)


def test_U1_dmrg():
    """
    Initialize random mps of U(1) tensors and checks canonization
    """
    generate_random.random_seed(config_U1_fermionic, seed=0)
    N = 7
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    logging.info(' Tensor : U1 ')

    Eng_sectors = {2: [-2.861972627395668, -2.213125929752753, -1.7795804271032745],
                   3: [-3.427339492125848, -2.661972627395668, -2.0131259297527526],
                   4: [-3.227339492125848, -2.461972627395668, -1.8131259297527529]}
    H = generate_automatic.mpo_XX_model(config_U1_fermionic, N=N, t=1, mu=0.2)
    occ = generate_automatic.mpo_occupation(config_U1_fermionic, N=N)

    for total_occ, E_target in Eng_sectors.items():
        psi = generate_random.mps_random(config_U1_fermionic, N=N, Dblocks=[1, 2, 1], total_charge=total_occ).canonize_sweep(to='first')
        occ_target = [total_occ] * len(E_target)
        psi = run_dmrg(psi, H, occ, E_target, occ_target, version='2site', opts_svd=opts_svd)


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    test_dense_dmrg()
    test_Z2_dmrg()
    test_U1_dmrg()
