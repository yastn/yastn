import numpy as np
import pytest
import logging
import argparse
import yast
import yast.tn.peps as peps
import yast.tn.mps as mps
import time
from yast.tn.peps.operators.gates import gates_hopping, gate_local_fermi_sea, gate_local_Hubbard
from yast.tn.peps.evolution import _als_update
from yast.tn.peps import initialize_peps_purification
from yast.tn.peps.ctm import sample, nn_bond, CtmEnv2Mps, nn_avg, ctmrg_, init_rand, one_site_avg, Local_CTM_Env
from yast.tn.mps import Env2, Env3


try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg


def test_NTU_spinfull():

    lattice = 'rectangle'
    boundary = 'finite'
    purification = 'True'
    xx = 5
    yy = 5
    D = 12
    chi = 20
    UU = 12
    mu_up, mu_dn = 0, 0 # chemical potential
    t_up, t_dn = 1., 1. # hopping amplitude
    beta_end = 0.03
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'

    dims = (yy, xx)
    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)

    opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    ancilla='True'
    GA_nn_up, GB_nn_up = gates_hopping(t_up, dbeta, fid, fc_up, fcdag_up, purification=purification)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, dbeta, fid, fc_dn, fcdag_dn, purification=purification)
    G_loc = gate_local_Hubbard(mu_up, mu_dn, UU, dbeta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, purification=purification)
    Gate = {'loc':G_loc, 'nn':{'GA_up':GA_nn_up, 'GB_up':GB_nn_up, 'GA_dn':GA_nn_dn, 'GB_dn':GB_nn_dn}}

    if purification == 'True':
        Gamma = initialize_peps_purification(fid, net) # initialized at infinite temperature
    
    time_steps = round(beta_end / dbeta)
    for nums in range(time_steps):
        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        Gamma, info = _als_update(Gamma, Gate, D, step, tr_mode, env_type='NTU') 

    # convergence criteria for CTM based on total energy
    chi = 40 # environmental bond dimension
    cutoff = 1e-10
    max_sweeps=50 
    tol = 1e-7   # difference of some observable must be lower than tolernace

    env = init_rand(Gamma, tc = ((0,) * fid.config.sym.NSYM,), Dc=(1,))  # initialization with random tensors 

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

    cf_energy_old = 0

    for step in ctmrg_(Gamma, env, chi, cutoff, max_sweeps, iterator_step=4, AAb_mode=0, flag=None):
        
        assert step.sweeps % 4 == 0 # stop every 4th step as iteration_step=4
        obs_hor, obs_ver =  nn_avg(Gamma, step.env, ops)

        cdagc_up = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))
        ccdag_up = 0.5*(abs(obs_hor.get('ccdag_up')) + abs(obs_ver.get('ccdag_up')))
        cdagc_dn = 0.5*(abs(obs_hor.get('cdagc_dn')) + abs(obs_ver.get('cdagc_dn')))
        ccdag_dn = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))

        cf_energy =  - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) * (2 * xx * yy - xx - yy)

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

    
    psi = Gamma.boundary_mps()
    opts = {'D_total': chi}

    for r_index in range(net.Ny-1,-1,-1):
        print(r_index)
        Bctm = CtmEnv2Mps(net, env, index=r_index, index_type='r')   # right boundary of r_index th column through CTM environment tensors
        #assert all(Bctm[i].get_shape() == psi[i].get_shape() for i in range(net.Nx))
        print(abs(mps.vdot(psi, Bctm)) / (psi.norm() * Bctm.norm()))
        assert pytest.approx(abs(mps.vdot(psi, Bctm)) / (psi.norm() * Bctm.norm()), rel=1e-10) == 1.0

        psi0 = psi.copy()
        O = Gamma.mpo(index=r_index, index_type='column')
        psi = mps.zipper(O, psi0, opts)  # right boundary of (r_index-1) th column through zipper
        mps.variational_(psi, O, psi0, method='1site', max_sweeps=2)


    n_up = fcdag_up @ fc_up 
    n_dn = fcdag_dn @ fc_dn 
    h_up = fc_up @ fcdag_up 
    h_dn = fc_dn @ fcdag_dn 

    nn_up, nn_dn, nn_do, nn_hole = n_up @ h_dn, n_dn @ h_up, n_up @ n_dn, h_up @ h_dn
    projectors = [nn_up, nn_dn, nn_do, nn_hole]
    out = sample(Gamma, env, projectors)
    print(out)

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinfull()

