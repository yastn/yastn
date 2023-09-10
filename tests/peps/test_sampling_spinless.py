import numpy as np
import pytest
import logging
import argparse
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps
import time
from yastn.tn.fpeps.operators.gates import gates_hopping, gate_local_fermi_sea
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_peps_purification
from yastn.tn.fpeps.ctm import sample, CtmEnv2Mps, nn_exp_dict, ctmrg
from yastn.tn.fpeps import _auxiliary

try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg


def not_working_test_sampling_spinless():

    lattice = 'square'
    boundary = 'obc'
    purification = 'True'
    xx = 3
    yy = 3
    tot_sites = (xx * yy)
    Ds = 5
    chi = 10
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    beta_end = 0.01
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'
    coeff = 0.25 # for purification; 0.5 for ground state calculation and 1j*0.5 for real-time evolution
    trotter_step = coeff * dbeta  

    dims = (xx, yy)
    net = fpeps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)

    opt = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    GA_nn, GB_nn = gates_hopping(t, trotter_step, fid, fc, fcdag)  # nn gate for 2D fermi sea
    g_loc = gate_local_fermi_sea(mu, trotter_step, fid, fc, fcdag) # local gate for spinless fermi sea
    g_nn = [(GA_nn, GB_nn)]

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

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    cf_energy_old = 0
    opts_svd_ctm = {'D_total': chi, 'tol': tol}

    for step in ctmrg(peps, max_sweeps, iterator_step=1, AAb_mode=0, opts_svd=opts_svd_ctm):
        
        assert step.sweeps % 1 == 0 # stop every 4th step as iteration_step=4
        obs_hor, obs_ver =  nn_exp_dict(peps, step.env, ops)

        cdagc = (sum(abs(val) for val in obs_hor.get('cdagc').values()) + sum(abs(val) for val in obs_ver.get('cdagc').values()))
        ccdag = (sum(abs(val) for val in obs_hor.get('ccdag').values()) + sum(abs(val) for val in obs_ver.get('ccdag').values()))

        cf_energy = - (cdagc + ccdag) / tot_sites

        print("Energy : ", cf_energy)
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
        Bctm = CtmEnv2Mps(net, step.env, index=r_index, index_type='r')  # right boundary of r_index th column through CTM environment tensors
        assert pytest.approx(abs(mps.vdot(phi, Bctm)) / (phi.norm() * Bctm.norm()), rel=1e-8) == 1.0
        phi0 = phi.copy()
        O = peps.mpo(index=r_index, index_type='column')
        phi = mps.zipper(O, phi0, opts)  # right boundary of (r_index-1) th column through zipper
        mps.compression_(phi, (O, phi0), method='1site', max_sweeps=2)

    nn, hh = fcdag @ fc, fc @ fcdag
    projectors = [nn, hh]
    out = sample(peps, step.env, projectors)
    print(out)

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    not_working_test_sampling_spinless()

