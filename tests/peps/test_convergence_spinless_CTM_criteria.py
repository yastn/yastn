""" Test the expectation values of spin 1/2 fermions with analytical values of fermi sea """
import numpy as np
import pytest
import logging
import argparse
import yast
import yast.tn.peps as peps
import time
from yast.tn.peps.operators.gates import gates_hopping, gate_local_fermi_sea, gate_local_Hubbard
from yast.tn.peps.NTU import ntu_update, initialize_peps_purification
from yast.tn.peps.CTM import nn_avg, ctmrg_, init_rand, one_site_avg, Local_CTM_Env, nn_bond

try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg

sym='U1'
opt = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

n = fcdag @ fc

def test_NTU_spinfull_finite():

    lattice = 'rectangle'
    boundary = 'finite'
    purification = 'True'   # real-time evolution not imaginary
    xx = 3
    yy = 3
    D = 6
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    fix_bd = 0
    step = 'two-step'
    tr_mode = 'optimal'
    
    dims = (xx, yy)

    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_MU_%1.5f_T_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, mu, t, sym)
    state = np.load("fully_filled_initialized_spinless_tensors_%s.npy" % (file_name), allow_pickle=True).item()
    
    beta = 3
    sv_beta = round(beta * yast.BETA_MULTIPLIER)
    tpeps = peps.Peps(lattice, dims, boundary)
    for sind in tpeps.sites():
        tpeps[sind] = yast.load_from_dict(config=fid.config, d=state.get((sind, sv_beta))) 
    
    # convergence criteria for CTM based on total energy
    chi = 30 # environmental bond dimension
    cutoff = 1e-10
    max_sweeps=8 
    tol = 1e-7   # difference of some observable must be lower than tolernace

    env = init_rand(tpeps, tc = ((0,) * fid.config.sym.NSYM,), Dc=(1,))  # initialization with random tensors 

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    tl_sum_old1, tr_sum_old1, bl_sum_old1, br_sum_old1, tl_sum_old2, tr_sum_old2, bl_sum_old2, br_sum_old2 = 0, 0, 0, 0, 0, 0, 0, 0
    exp_rel_diff = 0

    bd_h = peps.Bond(site_0 = (2, 0), site_1=(2, 1), dirn='h')
    bd_v = peps.Bond(site_0 = (0, 1), site_1=(1, 1), dirn='v')

    for step in ctmrg_(tpeps, env, chi, cutoff, max_sweeps, iterator_step=1, AAb_mode=0, flag=None):
        
        assert step.sweeps % 1 == 0 # stop every 4th step as iteration_step=4
            
        print("##############################################################################################")
        print("###################################    SWEEP  ", step.sweeps)
        print("#############################################################################################")

        ms1, ms2 = (1,1), (2,2)
        tlc1, trc1, blc1, brc1, tlc2, trc2, blc2, brc2  = step.env[ms1].tl, step.env[ms1].tr, step.env[ms1].bl, step.env[ms1].br, step.env[ms2].tl, step.env[ms2].tr, step.env[ms2].bl, step.env[ms2].br
        U_tl1, S_tl1, V_tl1 = yast.svd(tlc1)
        U_tl2, S_tl2, V_tl2 = yast.svd(tlc2)
        U_tr1, S_tr1, V_tr1 = yast.svd(trc1)
        U_tr2, S_tr2, V_tr2 = yast.svd(trc2)
        U_bl1, S_bl1, V_bl1 = yast.svd(blc1)
        U_bl2, S_bl2, V_bl2 = yast.svd(blc2)
        U_br1, S_br1, V_br1 = yast.svd(brc1)
        U_br2, S_br2, V_br2 = yast.svd(brc2)
                
        tl_sum1, tl_sum2 = np.sum(np.diag(S_tl1.to_numpy())), np.sum(np.diag(S_tl2.to_numpy()))
        tr_sum1, tr_sum2 = np.sum(np.diag(S_tr1.to_numpy())), np.sum(np.diag(S_tr2.to_numpy()))
        bl_sum1, bl_sum2 = np.sum(np.diag(S_bl1.to_numpy())), np.sum(np.diag(S_bl2.to_numpy()))
        br_sum1, br_sum2 = np.sum(np.diag(S_br1.to_numpy())), np.sum(np.diag(S_br2.to_numpy()))

        print("##################    site   ", ms1)

        print("differnce top-left corner: ", (tl_sum1 - tl_sum_old1)/tl_sum_old1)
        print("differnce top-right coner: ", (tr_sum1 - tr_sum_old1)/tr_sum_old1)
        print("difference bottom-left coner: ", (bl_sum1 - bl_sum_old1)/bl_sum_old1)
        print("difference bottom-right coner: ", (br_sum1 - br_sum_old1)/br_sum_old1)

        print("##################    site   ", ms2)

        print("differnce top-left corner: ", (tl_sum2 - tl_sum_old2)/tl_sum_old2)
        print("differnce top-right coner: ", (tr_sum2 - tr_sum_old2)/tr_sum_old2)
        print("difference bottom-left coner: ", (bl_sum2 - bl_sum_old2)/bl_sum_old2)
        print("difference bottom-right coner: ", (br_sum2 - br_sum_old2)/br_sum_old2)

        tl_sum_old1, tl_sum_old2 = tl_sum1, tl_sum2
        tr_sum_old1, tr_sum_old2  = tr_sum1, tr_sum2
        bl_sum_old1, bl_sum_old2 = bl_sum1, bl_sum2
        br_sum_old1, br_sum_old2 = br_sum1, br_sum2

        ###   calculate fermionic bond correlators at each sweep as a covergence criteria  ###

        exact_correlation_bond = 0.1767723804184942

        nn_CTM_bond = 0.5*(abs(nn_bond(tpeps, env, ops['cdagc'], bd_v)) + abs(nn_bond(tpeps, env, ops['ccdag'], bd_v)))
        exp_rel_diff = (nn_CTM_bond - exact_correlation_bond)/exact_correlation_bond

        print("relative difference expectation value: ", exp_rel_diff)
        


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinfull_finite()


