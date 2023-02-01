import numpy as np
import pytest
import logging
import argparse
import yast
import yast.tn.peps as peps
import yast.tn.mps as mps
import time
from yast.tn.peps.operators.gates import gates_hopping, gate_local_fermi_sea, gate_local_Hubbard
from yast.tn.peps.als import _als_update
from yast.tn.peps import initialize_peps_purification
from yast.tn.peps.ctm import sample, nn_bond, CtmEnv2Mps, nn_avg, ctmrg_, init_rand, one_site_avg, Local_CTM_Env

from yast.tn.mps import Env2, Env3


try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg


def test_NTU_spinless():

    lattice = 'rectangle'
    boundary = 'finite'
    purification = 'True'
    xx = 3
    yy = 4
    D = 4
    chi = 10
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    beta_end = 0.01
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'

    dims = (xx, yy)
    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)

    opt = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()
    ancilla='True'
    GA_nn, GB_nn = gates_hopping(t, dbeta, fid, fc, fcdag, purification=purification)  # nn gate for 2D fermi sea
    G_loc = gate_local_fermi_sea(mu, dbeta, fid, fc, fcdag, purification=purification) # local gate for spinless fermi sea
    Gate = {'loc': G_loc, 'nn':{'GA': GA_nn, 'GB': GB_nn}}
    if purification == 'True':
        gamma = initialize_peps_purification(fid, net) # initialized at infinite temperature
    
    time_steps = round(beta_end / dbeta)
    for nums in range(time_steps):
        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        gamma, info = _als_update(gamma, Gate, D, step, tr_mode, env_type='NTU') # fix_bd = 0 refers to unfixed symmetry sectors

    # convergence criteria for CTM based on total energy
    chi = 40 # environmental bond dimension
    cutoff = 1e-10
    max_sweeps=50 
    tol = 1e-7   # difference of some observable must be lower than tolernace

    env = init_rand(gamma, tc = ((0,) * fid.config.sym.NSYM,), Dc=(1,))  # initialization with random tensors 

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    cf_energy_old = 0

    for step in ctmrg_(gamma, env, chi, cutoff, max_sweeps, iterator_step=4, AAb_mode=0, flag=None):
        
        assert step.sweeps % 4 == 0 # stop every 4th step as iteration_step=4
        obs_hor, obs_ver =  nn_avg(gamma, step.env, ops)

        cdagc = 0.5*(abs(obs_hor.get('cdagc')) + abs(obs_ver.get('cdagc')))
        ccdag = 0.5*(abs(obs_hor.get('ccdag')) + abs(obs_ver.get('ccdag')))

        cf_energy = - (cdagc + ccdag) * (2 * xx * yy - xx - yy)

        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy


    ###  we try to find out the right boundary vector of the left-most column or 0th row
    ########## 3x3 lattice ########
    ###############################
    ##### (0,0) (1,0) (2,0) #######
    ##### (0,1) (1,1) (2,1) #######
    ##### (0,2) (1,2) (2,2) #######
    ###############################

    psi = gamma.boundary_mps()
    opts = {'D_total': chi}

    for r_index in range(net.Ny-1,-1,-1):
        print('r_index: ',r_index)
        Bctm = CtmEnv2Mps(net, step.env, index=r_index, index_type='r')  # right boundary of r_index th column through CTM environment tensors

       # assert all(Bctm[i].get_shape() == psi[i].get_shape() for i in range(net.Nx))
        print(abs(mps.vdot(psi, Bctm)) / (psi.norm() * Bctm.norm()))
        assert pytest.approx(abs(mps.vdot(psi, Bctm)) / (psi.norm() * Bctm.norm()), rel=1e-8) == 1.0

        psi0 = psi.copy()
        O = gamma.mpo(index=r_index, index_type='column')
        psi = mps.zipper(O, psi0, opts)  # right boundary of (r_index-1) th column through zipper
        mps.variational_(psi, O, psi0, method='1site', max_sweeps=2)

    nn, hh = fcdag @ fc, fc @ fcdag
    projectors = [nn, hh]
    out = sample(gamma, step.env, projectors)
    print(out)

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinless()

