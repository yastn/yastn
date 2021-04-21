import yast.mps as mps
import ops_full as ops_full
import ops_Z2 as ops_Z2
import ops_U1 as ops_U1
import numpy as np
import pytest

tol=1e-8

def run_tdvp_1site(psi, H, dt, sweeps,  Eng_gs, opts=None):
    """ Run a faw sweeps in imaginary time of tdvp_1site_sweep. """
    env = mps.dmrg.dmrg_sweep_1site(psi, H, env=None)
    Eng_old = env.measure()
    for _ in range(sweeps):
        env = mps.tdvp.tdvp_sweep_1site(psi, H, env=env, dt=dt, hermitian=True, opts_svd=opts)
        Eng = env.measure()
        assert Eng < Eng_old + tol
        Eng_old = Eng
    print('Eng =', Eng, ' Egs =', Eng_gs)
    assert pytest.approx(Eng, rel=1e-1) == Eng_gs
    return psi


def run_tdvp_2site(psi, H, dt, sweeps,  Eng_gs, opts=None):
    """ Run a faw sweeps in imaginary time of tdvp_2site_sweep. """
    env = mps.dmrg.dmrg_sweep_1site(psi, H, env=None)
    Eng_old = env.measure()
    for _ in range(sweeps):
        env = mps.tdvp.tdvp_sweep_2site(psi, H, env=env, dt=dt, hermitian=True, opts_svd=opts)
        Eng = env.measure()
        assert Eng < Eng_old + tol
        Eng_old = Eng
    print('Eng =', Eng, ' Egs =', Eng_gs)
    assert pytest.approx(Eng, rel=1e-1) == Eng_gs
    return psi


def run_tdvp_2site_group(psi, H, dt, sweeps,  Eng_gs, opts=None):
    """ Run a faw sweeps in imaginary time of tdvp_2site_group. """
    env = mps.dmrg.dmrg_sweep_1site(psi, H, env=None)
    Eng_old = env.measure()
    for _ in range(sweeps):
        env = mps.tdvp.tdvp_sweep_2site_group(psi, H, env=env, dt=dt, hermitian=True, opts_svd=opts)
        Eng = env.measure()
        assert Eng < Eng_old + tol
        Eng_old = Eng
    print('Eng =', Eng, ' Egs =', Eng_gs)
    assert pytest.approx(Eng, rel=1e-1) == Eng_gs
    return psi

def test_full_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 5
    D_total = 4
    dt = -.25
    sweeps = 3
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    H = ops_full.mpo_XX_model(N=N, t=1, mu=0.25)
    Eng_gs = -2.232050807568877

    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    psi = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_gs)

    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    psi = run_tdvp_2site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_gs)

    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first')
    psi = run_tdvp_2site_group(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_gs)


def test_Z2_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 5
    D_total = 4
    dt = -.25
    sweeps = 3
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    Eng_parity0 = -2.232050807568877
    Eng_parity1 = -1.982050807568877

    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0.25)

    psi = ops_Z2.mps_random(N=N, Dblock=D_total/2, total_parity=0)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_parity0)

    psi = ops_Z2.mps_random(N=N, Dblock=D_total/2, total_parity=1)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_parity1)

    psi = ops_Z2.mps_random(N=N, Dblock=D_total/2, total_parity=0)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_parity0)

    psi = ops_Z2.mps_random(N=N, Dblock=D_total/2, total_parity=0)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site_group(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_parity0)


def test_U1_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 5
    D_total = 4
    dt = -.25
    sweeps = 3
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    Eng_ch1 = -1.482050807568877
    Eng_ch2 = -2.232050807568877

    H = ops_U1.mpo_XX_model(N=N, t=1, mu=0.25)

    psi = ops_U1.mps_random(N=N, Dblocks=[1, 2, 1], total_charge=1)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_ch1)

    psi = ops_U1.mps_random(N=N, Dblocks=[1, 2, 1], total_charge=2)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_1site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_ch2)

    psi = ops_U1.mps_random(N=N, Dblocks=[1, 2, 1], total_charge=1)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_ch1)

    psi = ops_U1.mps_random(N=N, Dblocks=[1, 2, 1], total_charge=1)
    psi.canonize_sweep(to='first')
    _ = run_tdvp_2site_group(psi, H, dt=dt, sweeps=sweeps, opts=opts_svd, Eng_gs=Eng_ch1)


def test_OBC_tdvp():
    """
    Check tdvp_OBC with measuring additional expectation values
    """
    N = 5
    D_total = 4
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    H = ops_full.mpo_XX_model(N=N, t=1, mu=0.25)
    M = None

    Eng_gs = -2.232050807568877
    dt = -.25
    tmax = dt*2.

    version = '1site'
    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2).canonize_sweep(to='first')
    mps.dmrg.dmrg_sweep_1site(psi, H, env=None)
    _, E, _ = mps.tdvp.tdvp_OBC(psi=psi, tmax=tmax, dt=dt, H=H, M=M, version=version, opts_svd=opts_svd)
    print('1site: Energy =', E, ' Eref= ', Eng_gs)
    assert pytest.approx(E, rel=1e-1) == Eng_gs

    version = '2site'
    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2).canonize_sweep(to='first')
    mps.dmrg.dmrg_sweep_1site(psi, H, env=None)
    _, E, _ = mps.tdvp.tdvp_OBC(psi=psi, tmax=tmax, dt=dt, H=H, M=M, version=version, opts_svd=opts_svd)
    print('2site: Energy =', E, ' Eref= ', Eng_gs)
    assert pytest.approx(E, rel=1e-1) == Eng_gs
 
    version = '2site_group'
    psi = ops_full.mps_random(N=N, Dmax=D_total, d=2).canonize_sweep(to='first')
    mps.dmrg.dmrg_sweep_1site(psi, H, env=None)
    _, E, _ = mps.tdvp.tdvp_OBC(psi=psi, tmax=tmax, dt=dt, H=H, M=M, version=version, opts_svd=opts_svd)
    print('2site_group: Energy =', E, ' Eref= ', Eng_gs)
    assert pytest.approx(E, rel=1e-1) == Eng_gs

if __name__ == "__main__":
    test_full_tdvp()
    test_Z2_tdvp()
    test_U1_tdvp()
    test_OBC_tdvp()
