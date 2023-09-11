import numpy as np
import pytest
import logging
import argparse
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps
import time
from yastn.tn.fpeps.operators.gates import gates_hopping, gate_local_Hubbard
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_peps_purification
from yastn.tn.fpeps.ctm import sample, CtmEnv2Mps, nn_exp_dict, ctmrg
from yastn.tn.mps import Env2, Env3
from yastn.tn.fpeps import _auxiliary


try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg


def not_working_test_sampling_spinfull():

    lattice = 'square'
    boundary = 'obc'
    purification = 'True'
    xx = 3
    yy = 3
    tot_sites = (xx * yy)
    Ds = 12
    chi = 20
    U = 12
    mu_up, mu_dn = 0, 0 # chemical potential
    t_up, t_dn = 1., 1. # hopping amplitude
    beta_end = 0.03
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'
    coeff = 0.25 # for purification; 0.5 for ground state calculation and 1j*0.5 for real-time evolution
    trotter_step = coeff * dbeta  

    dims = (yy, xx)
    net = fpeps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)

    opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    GA_nn_up, GB_nn_up = gates_hopping(t_up, trotter_step, fid, fc_up, fcdag_up)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, trotter_step, fid, fc_dn, fcdag_dn)
    g_loc = gate_local_Hubbard(mu_up, mu_dn, U, trotter_step, fid, fc_up, fc_dn, fcdag_up, fcdag_dn)
    g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn)]

    if purification == 'True':
        peps = initialize_peps_purification(fid, net) # initialized at infinite temperature
    
    gates = gates_homogeneous(peps, g_nn, g_loc)

    time_steps = round(beta_end / dbeta)
    opts_svd_ntu = {'D_total': Ds, 'tol_block': 1e-15}

    for nums in range(time_steps):
        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        peps, _ =  evolution_step_(peps, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 

    # convergence criteria for CTM based on total energy
    chi = 40 # environmental bond dimension
    tol = 1e-10
    max_sweeps=50 
    tol_exp = 1e-7   # difference of some observable must be lower than tolernace
    
    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

    cf_energy_old = 0
    opts_svd_ctm = {'D_total': chi, 'tol': tol}

    for step in ctmrg(peps, max_sweeps, iterator_step=4, AAb_mode=0, opts_svd=opts_svd_ctm):
        
        assert step.sweeps % 4 == 0 # stop every 4th step as iteration_step=4
        obs_hor, obs_ver =  nn_exp_dict(peps, step.env, ops)

        cdagc_up = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + 
                  sum(abs(val) for val in obs_ver.get('cdagc_up').values()))

        ccdag_up = (sum(abs(val) for val in obs_hor.get('ccdag_up').values()) + 
                  sum(abs(val) for val in obs_ver.get('ccdag_up').values()))

        cdagc_dn = (sum(abs(val) for val in obs_hor.get('cdagc_dn').values()) + 
                  sum(abs(val) for val in obs_ver.get('cdagc_dn').values()))

        ccdag_dn = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + 
                  sum(abs(val) for val in obs_ver.get('cdagc_up').values()))

        cf_energy =  - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) / tot_sites

        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

    ###  we try to find out the right boundary vector of the left-most column or 0th row
    ########## 3x3 lattice ########
    ###############################
    ##### (0,0) (1,0) (2,0) #######
    ##### (0,1) (1,1) (2,1) #######
    ##### (0,2) (1,2) (2,2) #######
    ###############################

    phi = peps.boundary_mps()
    opts = {'D_total': chi}

    for r_index in range(net.Ny-1,-1,-1):
        Bctm = CtmEnv2Mps(net, step.env, index=r_index, index_type='r')   # right boundary of r_index th column through CTM environment tensors
        #assert all(Bctm[i].get_shape() == peps[i].get_shape() for i in range(net.Nx))
        print(abs(mps.vdot(phi, Bctm)) / (phi.norm() * Bctm.norm()))
        assert pytest.approx(abs(mps.vdot(phi, Bctm)) / (phi.norm() * Bctm.norm()), rel=1e-10) == 1.0

        phi0 = phi.copy()
        O = peps.mpo(index=r_index, index_type='column')
        phi = mps.zipper(O, phi0, opts)  # right boundary of (r_index-1) th column through zipper
        mps.compression_(phi, (O, phi0), method='1site', max_sweeps=2)

    n_up = fcdag_up @ fc_up 
    n_dn = fcdag_dn @ fc_dn 
    h_up = fc_up @ fcdag_up 
    h_dn = fc_dn @ fcdag_dn 

    nn_up, nn_dn, nn_do, nn_hole = n_up @ h_dn, n_dn @ h_up, n_up @ n_dn, h_up @ h_dn
    projectors = [nn_up, nn_dn, nn_do, nn_hole]
    out = sample(peps, step.env, projectors)
    print(out)

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    not_working_test_sampling_spinfull()

