""" dmrg tested on XX model. """
import logging
import pytest
import yastn.tn.mps as mps
import yastn
try:
    from .configs import config_dense as cfg
    # pytest modifies cfg to inject different backends and devices during tests
except ImportError:
    from configs import config_dense as cfg

tol = 1e-5

def run_dmrg(phi, H, occ, E_target, occ_target, opts_svd=None):
    r"""
    Run mps.dmrg_ to find the ground state and a few low-energy states of the Hamiltonian H.
    Verify resulting energies against known reference solutions.
    """
    # 
    # DMRG can look for solutions in a space orthogonal to provided MPSs.
    # We start with empty list, project = [], and keep adding to it previously found eigenstates.
    # This allows us to find the ground state and a few consequative lowest-energy eigenstates.
    #
    project = []
    for reference_energy, reference_occupation in zip(E_target, occ_target):
        #
        # We find a state and check that its energy and total occupation matches the expected reference values.
        #
        # We copy initial random MPS psi, as yastn.dmrg_ modifies provided input state in place.
        #
        psi = phi.copy()
        #
        # We set up dmrg_ to terminate iterations when energy is converged within some tolerance.
        #
        out = mps.dmrg_(psi, H, project=project, method='2site',
                        energy_tol=tol / 10, max_sweeps=20, opts_svd=opts_svd)
        #
        # Output of _dmrg is a nametuple with basic information about the run, including the final energy.
        # Occupation number has to be calcuted using measure_mpo.
        #
        energy = out.energy
        occupation = mps.measure_mpo(psi, occ, psi)
        #
        # Print the result:
        #
        logging.info(" 2site dmrg; Energy: %0.8f / %0.8f   Occupation: %0.8f / %0.8f", 
                        energy, reference_energy, occupation, reference_occupation)
        assert pytest.approx(energy.item(), rel=tol) == reference_energy
        assert pytest.approx(occupation.item(), rel=tol) == reference_occupation
        #
        # We run further iterations with 1site dmrg scheme and stricter convergence criterion.
        #
        out = mps.dmrg_(psi, H, project=project, method='1site',
                        Schmidt_tol=tol / 10, max_sweeps=20)
        
        energy = mps.measure_mpo(psi, H, psi)
        occupation = mps.measure_mpo(psi, occ, psi)        
        logging.info(" 1site dmrg; Energy: %0.8f / %0.8f   Occupation: %0.8f / %0.8f",
                        energy, reference_energy, occupation, reference_occupation)
        # test that energy outputed by dmrg is correct
        assert pytest.approx(energy.item(), rel=tol) == reference_energy
        assert pytest.approx(occupation.item(), rel=tol) == reference_occupation
        assert pytest.approx(energy.item(), rel=1e-14) == out.energy.item()
        #
        # Finally, we add the found state psi to the list of states to be projected out in the next step of the loop.
        #
        project.append(psi)
    return project[0]


def test_dense_dmrg():
    """
    Initialize random mps of dense tensors and runs a few sweeps of dmrg with Hamiltonian of XX model.
    """
    # Knowing exact solution we can compare it DMRG result.
    # In this test we will consider sectors of different occupation.
    #
    # In this example we use yastn.Tensor's with no symmetry imposed. 
    #
    logging.info(' Tensor : dense ')
    #
    # The Hamiltonian of N = 7 sites is obtained with automatic generator (see source file).
    #
    N = 7
    #
    operators = yastn.operators.Spin12(sym='dense', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=N, operators=operators)
    parameters = {"t": 1.0, "mu": 0.2, "rangeN": range(N), "rangeNN": zip(range(N-1),range(1,N))}
    H_str = "\sum_{i,j \in rangeNN} t ( sp_{i} sm_{j} + sp_{j} sm_{i} ) + \sum_{j\in rangeN} mu sp_{j} sm_{j}"
    H = generate.mpo_from_latex(H_str, parameters)
    #
    # and MPO to measure occupation:
    #
    occ = generate.mpo_from_latex("\sum_{j\in rangeN} sp_{j} sm_{j}", {"rangeN": range(N)})
    #
    # Known energies and occupations of low energy eigen-states.
    #
    Eng_gs = [-3.427339492125848, -3.227339492125848, -2.8619726273956685]
    Occ_gs = [3, 4, 2]
    #
    # To standardize this test we will fix a seed for random MPS we use.
    #
    generate.random_seed(seed=0)
    #
    # Set options for truncation for '2site' method of mps._dmrg.
    #
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}
    #
    # Finally run DMRG starting from random MPS psi:
    #
    psi = generate.random_mps(D_total=Dmax)
    #
    # Single run for a ground state can be done using:
    #
    # mps._dmrg(psi, H, method=method,energy_tol=tol/10, max_sweeps=20, opts_svd=opts_svd)
    #
    # To explain how to target some sectors for occupation we create a subfunction run_dmrg
    # This is not necessary but we do it for the sake of clarity and tests.
    #
    psi = run_dmrg(psi, H, occ, Eng_gs, Occ_gs, opts_svd=opts_svd)
    #
    # _dmrg can be executed as generator to monitor state between dmrg sweeps 
    # This is done by providing `iterator_step`.
    #
    psi = generate.random_mps(D_total=Dmax)
    for step in mps.dmrg_(psi, H, method='1site', max_sweeps=20, iterator_step=2):
        assert step.sweeps % 2 == 0  # stop every second step as iteration_step=2
        Eocc = mps.measure_mpo(psi, occ, psi)  # measure occupation
        if abs(Eocc.item() - Occ_gs[0]) < tol:
            break  # here break if close to known result.
    assert step.sweeps < 20  # dmrg was stopped based on external condition, not max_sweeps


def test_Z2_dmrg():
    """
    Initialize random mps of Z2 tensors and tests mps._dmrg
    """
    logging.info(' Tensor : Z2 ')
    operators = yastn.operators.SpinlessFermions(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=7, operators=operators)
    generate.random_seed(seed=0)
    N, Dmax  = 7, 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    logging.info(' Tensor : Z2 ')

    Occ_target = {0: [4, 2, 4],
                  1: [3, 3, 5]}
    Eng_target = {0: [-3.227339492125848, -2.8619726273956685, -2.461972627395668],
                  1: [-3.427339492125848, -2.6619726273956683, -2.261972627395668]}
    parameters = {"t": 1.0, "mu": 0.2, "rangeN": range(N), "rangeNN": zip(range(N-1),range(1,N))}
    H_str = "\sum_{i,j \in rangeNN} t ( cp_{i} c_{j} + cp_{j} c_{i} ) + \sum_{j\in rangeN} mu cp_{j} c_{j}"
    H = generate.mpo_from_latex(H_str, parameters)
    occ = generate.mpo_from_latex("\sum_{j\in rangeN} cp_{j} c_{j}", {"rangeN": range(N)})
    
    for parity in (0, 1):
        psi = generate.random_mps(D_total=Dmax, n=parity)
        # run_dmrg starts with 2-site method to update bond dimension for small tests with random distribution of bond dimensions.
        psi = run_dmrg(psi, H, occ, Eng_target[parity], Occ_target[parity], opts_svd=opts_svd)


def test_U1_dmrg():
    """
    Initialize random mps of U(1) tensors and tests _dmrg against known results.
    """
    logging.info(' Tensor : U1 ')
    operators = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=7, operators=operators)
    generate.random_seed(seed=0)

    Eng_sectors = {2: [-2.861972627395668, -2.213125929752753, -1.7795804271032745],
                   3: [-3.427339492125848, -2.661972627395668, -2.0131259297527526],
                   4: [-3.227339492125848, -2.461972627395668, -1.8131259297527529]}

    N = 7
    parameters = {"t": 1.0, "mu": 0.2, "rangeN": range(N), "rangeNN": zip(range(N-1),range(1,N))}
    H_str = "\sum_{i,j \in rangeNN} t ( cp_{i} c_{j} + cp_{j} c_{i} ) + \sum_{j\in rangeN} mu cp_{j} c_{j}"
    H = generate.mpo_from_latex(H_str, parameters)
    occ = generate.mpo_from_latex("\sum_{j\in rangeN} cp_{j} c_{j}", {"rangeN": range(N)})

    Eng_sectors = {2: [-2.861972627395668, -2.213125929752753, -1.7795804271032745],
                   3: [-3.427339492125848, -2.661972627395668, -2.0131259297527526],
                   4: [-3.227339492125848, -2.461972627395668, -1.8131259297527529]}

    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    for total_occ, E_target in Eng_sectors.items():
        psi = generate.random_mps(D_total=Dmax, n=total_occ)
        occ_target = [total_occ] * len(E_target)
        psi = run_dmrg(psi, H, occ, E_target, occ_target, opts_svd=opts_svd)


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    test_dense_dmrg()
    test_Z2_dmrg()
    test_U1_dmrg()
