import yast.mps as mps
import ops_full
import ops_Z2
import ops_U1
import numpy as np
import pytest


def run_dmrg_0site(psi, H, occ, Etarget, occ_target, sweeps=10):
    """ Run a faw sweeps of dmrg_0site_sweep. Returns energy. """
    mps.dmrg.dmrg_sweep_1site(psi, H, env=None)
    env = None
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_0site(psi, H, env=env)
    assert pytest.approx(env.measure(), rel=1e-4) == Etarget  # This seems to be coverging slowly
    assert pytest.approx(mps.measure_mpo(psi, occ, psi), rel=1e-4) == occ_target  # This seems to be coverging slowly
    return psi


def run_dmrg_1site(psi, H, occ, Etarget, occ_target, sweeps=5):
    """ Run a faw sweeps of dmrg_1site_sweep. Returns energy. """
    env = None
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_1site(psi, H, env=env)
    assert pytest.approx(env.measure(), rel=1e-6) == Etarget
    assert pytest.approx(mps.measure_mpo(psi, occ, psi), rel=1e-4) == occ_target  # This seems to be coverging slowly
    return psi


def run_dmrg_2site(psi, H, occ, Etarget, occ_target, sweeps=5, D_total=32):
    """ Run a faw sweeps of dmrg_2site_sweep. Returns energy. """
    env = None
    opts_svd = {'tol': 1e-8, 'D_total': D_total}
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_2site(psi, H, env=env, opts_svd=opts_svd)
    assert pytest.approx(env.measure(), rel=1e-6) == Etarget
    assert pytest.approx(mps.measure_mpo(psi, occ, psi), rel=1e-4) == occ_target  # This seems to be coverging slowly
    return psi


def run_dmrg_2site_group(psi, H, occ, Etarget, occ_target, sweeps=5, D_total=32):
    """ Run a faw sweeps of dmrg_2site_group_sweep. Returns energy. """
    env = None
    opts_svd = {'tol': 1e-8, 'D_total': D_total}
    for _ in range(sweeps):
        env = mps.dmrg.dmrg_sweep_2site_group(psi, H, env=env, opts_svd=opts_svd)
    assert pytest.approx(env.measure(), rel=1e-6) == Etarget
    assert pytest.approx(mps.measure_mpo(psi, occ, psi), rel=1e-4) == occ_target  # This seems to be coverging slowly
    return psi


def test_full_dmrg():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 7
    Eng_gs = -3.427339492125848
    Occ_gs = 3
    print('Expected   : Energy =', Eng_gs, ' Occupation = ', Occ_gs)
    
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0.2)
    occ = ops_full.mpo_occupation(N=N)
    Dmax = 8

    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_0site(psi, H, occ, Eng_gs, Occ_gs)
    print('0site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_1site(psi, H, occ, Eng_gs, Occ_gs)
    print('1site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_2site(psi, H, occ, Eng_gs, Occ_gs, D_total=Dmax)
    print('2site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_2site_group(psi, H, occ, Eng_gs, Occ_gs, D_total=Dmax)
    print('2site group: Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))


def test_Z2_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 7
    Dmax = 8
    Occ_gs0 = 4
    Occ_gs1 = 3
    Eng_parity0 = -3.227339492125848
    Eng_parity1 = -3.427339492125848

    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0.2)
    occ = ops_Z2.mpo_occupation(N=N)

    print('Expected p=0: Energy =', Eng_parity0, ' Occupation = ', Occ_gs0)

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=0)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_0site(psi, H, occ, Eng_parity0, Occ_gs0)
    print('0site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=1)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_1site(psi, H, occ, Eng_parity1, Occ_gs1)
    print('1site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=1)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_2site(psi, H, occ, Eng_parity1, Occ_gs1, D_total=Dmax)
    print('2site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=1)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_2site_group(psi, H, occ, Eng_parity1, Occ_gs1, D_total=Dmax)
    print('2site group: Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    print('Expected p=1: Energy =', Eng_parity1, ' Occupation = ', Occ_gs1)

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=1)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_0site(psi, H, occ, Eng_parity1, Occ_gs1)
    print('0site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=0)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_1site(psi, H, occ, Eng_parity0, Occ_gs0)
    print('1site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=0)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_2site(psi, H, occ, Eng_parity0, Occ_gs0, D_total=Dmax)
    print('2site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))

    psi = ops_Z2.mps_random(N=N, Dblock=Dmax/2, total_parity=0)
    psi.canonize_sweep(to='first')
    psi = run_dmrg_2site_group(psi, H, occ, Eng_parity0, Occ_gs0, D_total=Dmax)
    print('2site group: Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))


def test_U1_dmrg():
    """
    Initialize random mps of full tensors and checks canonization
    """
    N = 7
    Dmax = 8 
    Eng_sectors = {2: -2.861972627395668, 3: -3.427339492125848, 4: -3.227339492125848}
    
    H = ops_U1.mpo_XX_model(N=N, t=1, mu=0.2)
    occ = ops_U1.mpo_occupation(N=N)

    for tcharge, Eng_gs in Eng_sectors.items():
        psi = ops_U1.mps_random(N=N, Dblocks=[1, 2, 1], total_charge=tcharge)
        psi.canonize_sweep(to='first')
        psi = run_dmrg_2site(psi, H, occ, Eng_gs, tcharge, D_total=Dmax)
        print('2site      : Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))
        
        psi = ops_U1.mps_random(N=N, Dblocks=[1, 2, 1], total_charge=tcharge)
        psi.canonize_sweep(to='first')
        psi = run_dmrg_2site_group(psi, H, occ, Eng_gs, tcharge, D_total=Dmax)
        print('2site group: Energy =', mps.measure_mpo(psi, H, psi), ' Occupation =', mps.measure_mpo(psi, occ, psi))


def test_OBC_dmrg():
    """
    Check dmrg_OBC with measuring additional expectation values
    """
    N = 7
    H = ops_full.mpo_XX_model(N=N, t=1, mu=0.2)
    Eng_gs = -3.427339492125848

    Dmax = 8
    cutoff_sweep = 1
    cutoff_dE = 1e-9
    opts_svd = {'tol': 1e-6, 'D_total': Dmax}

    version = '0site'
    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.dmrg.dmrg_OBC(psi=psi, H=H, env=None, version=version, cutoff_sweep=cutoff_sweep,
                                cutoff_dE=cutoff_dE, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=opts_svd)
    print('0site: Energy - Eref= ', E-Eng_gs)

    version = '1site'
    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.dmrg.dmrg_OBC(psi=psi, H=H, env=None, version=version, cutoff_sweep=cutoff_sweep,
                                cutoff_dE=cutoff_dE, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=opts_svd)
    print('1site: Energy - Eref= ', E-Eng_gs)

    version = '2site'
    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.dmrg.dmrg_OBC(psi=psi, H=H, env=None, version=version, cutoff_sweep=cutoff_sweep,
                                cutoff_dE=cutoff_dE, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=opts_svd)
    print('2site: Energy - Eref= ', E-Eng_gs)

    version = '2site_group'
    psi = ops_full.mps_random(N=N, Dmax=Dmax, d=2)
    psi.canonize_sweep(to='first')
    _, E, _ = mps.dmrg.dmrg_OBC(psi=psi, H=H, env=None, version=version, cutoff_sweep=cutoff_sweep,
                                cutoff_dE=cutoff_dE, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=opts_svd)
    print('2site_group: Energy - Eref= ', E-Eng_gs)


if __name__ == "__main__":
    test_full_dmrg()
    test_Z2_dmrg()
    test_U1_dmrg()
    test_OBC_dmrg()
