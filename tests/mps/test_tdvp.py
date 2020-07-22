import yamps.mps as mps
import yamps.mps.tdvp as tdvp
import yamps.ops.ops_full as ops_full
import yamps.ops.ops_Z2 as ops_Z2
import yamps.ops.ops_U1 as ops_U1
import numpy as np
import pytest


def run_tdvp_1site(psi, H, dt=1., sweeps=10, opts=None, Eng_gs=0):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    
    for _ in range(sweeps):
        env = tdvp.tdvp_sweep_1site(psi, H, env=env, dt=dt, dtype='float64', hermitian=True, fermionic=False, opts_svd=opts)
        Eng = env.measure()
        print('1site: Energy err = ', Eng-Eng_gs, ' Eg = ', Eng, ' norma = ', psi.norma)
    return Eng

def run_tdvp_2site(psi, H, dt=1., sweeps=10, opts=None, Eng_gs=0):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    
    for _ in range(sweeps):
        env = tdvp.tdvp_sweep_2site(psi, H, env=env, dt=dt, dtype='float64', hermitian=True, fermionic=False, opts_svd=opts)
        Eng = env.measure()
        print('2site: Energy err = ', Eng-Eng_gs, ' Eg = ', Eng, ' norma = ', psi.norma)
    return Eng

def run_tdvp_2site_group(psi, H, dt=1., sweeps=10, opts=None, Eng_gs=0):
    """
    Run a faw sweeps of dmrg_1site_sweep. Returns energy
    """
    env = None
    
    for _ in range(sweeps):
        env = tdvp.tdvp_sweep_2site_group(psi, H, env=env, dt=dt, dtype='float64', hermitian=True, fermionic=False, opts_svd=opts)
        Eng = env.measure()
        print('2site_group: Energy err = ', Eng-Eng_gs, ' Eg = ', Eng, ' norma = ', psi.norma)
    return Eng


def test_full_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    D_total = 16
    dt = .125*1j  # sign dt = +1j for imaginary time evolution 
    sweeps = 20
    opts_svd = {'tol': 1e-6, 'D_total': D_total}
    Eng_gs = -4.758770483143633

    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)    
    """
    psi = ops_full.mps_ones(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first', normalize=True)
    Eng = run_tdvp_1site(psi, H, dt = dt, sweeps = sweeps, opts=opts_svd, Eng_gs=Eng_gs)

    psi = ops_full.mps_ones(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first', normalize=True)
    Eng = run_tdvp_2site(psi, H, dt = dt, sweeps = sweeps, opts=opts_svd, Eng_gs=Eng_gs)
    """

    psi = ops_full.mps_ones(N=N, Dmax=D_total, d=2)
    psi.canonize_sweep(to='first', normalize=True)
    Eng = run_tdvp_2site_group(psi, H, dt = dt, sweeps = sweeps, opts=opts_svd, Eng_gs=Eng_gs)


def test_Z2_tdvp():
    """
    Initialize random mps of full tensors and runs a few sweeps of dmrg1 with Hamiltonian of XX model.
    """
    N = 8
    D_total = 16
    dt = .125*1j  # sign dt = +1j for imaginary time evolution 
    sweeps = 20
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    Eng_parity0 = -4.758770483143633
    Eng_parity1 = -4.411474127809773
    Eng_gs = -4.758770483143633
    
    H = ops_Z2.mpo_XX_model(N=N, t=1, mu=0)
    
    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first', normalize=True)
    Eng = run_tdvp_1site(psi, H, dt = dt, sweeps = sweeps, opts=opts_svd, Eng_gs=Eng_parity0)
    
    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=1)
    psi.canonize_sweep(to='first', normalize=True)
    Eng = run_tdvp_1site(psi, H, dt = dt, sweeps = sweeps, opts=opts_svd, Eng_gs=Eng_parity1)
    
    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first', normalize=True)
    Eng = run_tdvp_2site(psi, H, dt = dt, sweeps = sweeps, opts=opts_svd, Eng_gs=Eng_gs)
    
    psi = ops_Z2.mps_random(N=N, Dblock=16, total_parity=0)
    psi.canonize_sweep(to='first', normalize=True)
    Eng = run_tdvp_2site_group(psi, H, dt = dt, sweeps = sweeps, opts=opts_svd, Eng_gs=Eng_gs)


def test_dmrg_total_full():
    """
    Check tdvp_OBC with measuring additional expectation values
    """
    N = 8
    D_total = 32
    opts_svd = {'tol': 1e-6, 'D_total': D_total}

    H = ops_full.mpo_XX_model(N=N, t=1, mu=0)    
    M = None
    measure_O = None

    Eng_gs = -4.758770483143633
    dt = .125*1j
    tmax = dt*30.

    version = '1site'
    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    print()
    env, E, dE, out = mps.tdvp.tdvp_OBC(psi=psi, tmax=tmax, dt=dt, H=H, M=M, measure_O=measure_O, version=version, opts_svd=opts_svd)
    print('1site: Energy - Eref= ', E-Eng_gs)

    version = '2site'
    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    print()
    env, E, dE, out = mps.tdvp.tdvp_OBC(psi=psi, tmax=tmax, dt=dt, H=H, M=M, measure_O=measure_O, version=version, opts_svd=opts_svd)
    print('2site: Energy - Eref= ', E-Eng_gs)
    
    version = '2site_group'
    psi = ops_full.mps_random(N=N, Dmax=32, d=2)
    psi.canonize_sweep(to='first')
    print()
    env, E, dE, out = mps.tdvp.tdvp_OBC(psi=psi, tmax=tmax, dt=dt, H=H, M=M, measure_O=measure_O, version=version, opts_svd=opts_svd)
    print('2site_group: Energy - Eref= ', E-Eng_gs)



if __name__ == "__main__":
    # pass
    #test_full_tdvp()
    #test_Z2_tdvp()
    test_dmrg_total_full()
