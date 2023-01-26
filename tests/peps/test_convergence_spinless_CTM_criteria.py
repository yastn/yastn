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
    boundary = 'infinite'
    purification = 'True'   # real-time evolution not imaginary
    xx = 3
    yy = 3
    D = 8
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    fix_bd = 0
    step = 'two-step'
    tr_mode = 'optimal'
    
    dims = (xx, yy)

    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_MU_%1.5f_T_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, mu, t, sym)
    state = np.load("spinless_tensors_%s.npy" % (file_name), allow_pickle=True).item()
    
    beta = 3
    exact_energy_bond = -0.19489891 # at beta = 3

    sv_beta = round(beta * yast.BETA_MULTIPLIER)
    tpeps = peps.Peps(lattice, dims, boundary)
    for sind in tpeps.sites():
        tpeps[sind] = yast.load_from_dict(config=fid.config, d=state.get((sind, sv_beta))) 
    
    # convergence criteria for CTM based on total energy
    chi = 32 # environmental bond dimension
    cutoff = 1e-10
    max_sweeps=40
    tol = 1e-7   # difference of some observable must be lower than tolernace

    env = init_rand(tpeps, tc = ((0,) * fid.config.sym.NSYM,), Dc=(1,))  # initialization with random tensors 

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    singular_vecs  = {}
    singular_vecs_old = {}

    diff_norm_sing = {}
    average_diff_norm_sing = {}

    for step in ctmrg_(tpeps, env, chi, cutoff, max_sweeps, iterator_step=1, AAb_mode=0, flag=None):
        
        assert step.sweeps % 1 == 0 # stop every 4th step as iteration_step=4
            
        print("##############################################################################################")
        print("###################################    SWEEP  ", step.sweeps)
        print("#############################################################################################")


        ##### a criteria by calculating norms of singular vectors of corner transfer matrices of each site

        for ms in tpeps.sites():

            print("###################### SITE:  ", ms)

            _, S_tl, _ = yast.svd(step.env[ms].tl)
            singular_vecs[ms, 'tl'] = S_tl
            _, S_tr, _ = yast.svd(step.env[ms].tr)
            singular_vecs[ms, 'tr'] = S_tr
            _, S_bl, _ = yast.svd(step.env[ms].bl)
            singular_vecs[ms, 'bl'] = S_bl
            _, S_br, _ = yast.svd(step.env[ms].br)
            singular_vecs[ms, 'br'] = S_br
            
            if step.sweeps > 1:      

                diff_norm_sing[ms, 'tl'] =  ((singular_vecs[ms, 'tl'] - singular_vecs_old[ms, 'tl']).norm())
                diff_norm_sing[ms, 'tr'] = ((singular_vecs[ms, 'tr'] - singular_vecs_old[ms, 'tr']).norm())
                diff_norm_sing[ms, 'bl'] =  ((singular_vecs[ms, 'bl'] - singular_vecs_old[ms, 'bl']).norm())
                diff_norm_sing[ms, 'br'] =  ((singular_vecs[ms, 'br'] - singular_vecs_old[ms, 'br']).norm())
                average_diff_norm_sing['ms'] = 0.25 * (diff_norm_sing[ms, 'tl']+ diff_norm_sing[ms, 'tr']+ diff_norm_sing[ms, 'bl']+diff_norm_sing[ms, 'br'])
                print("average difference norm singular ", average_diff_norm_sing['ms'])
            
            singular_vecs_old[ms, 'tl'] = singular_vecs[ms, 'tl']
            singular_vecs_old[ms, 'tr'] = singular_vecs[ms, 'tr']
            singular_vecs_old[ms, 'bl'] = singular_vecs[ms, 'bl']
            singular_vecs_old[ms, 'br'] = singular_vecs[ms, 'br']

        ###   calculate energy at each sweep as a covergence criteria  ###

        obs_hor, obs_ver =  nn_avg(tpeps, step.env, ops)
        cdagc = 0.5*(abs(obs_hor.get('cdagc')) + abs(obs_ver.get('cdagc')))
        ccdag = 0.5*(abs(obs_hor.get('ccdag')) + abs(obs_ver.get('ccdag')))
        ctm_energy_bond = - 0.5 * (cdagc + ccdag)  # average energy per bond

        rel_diff_an_energy = (ctm_energy_bond-exact_energy_bond)/exact_energy_bond # relative difference with analytical result
        rel_diff_ctm_energy = (ctm_energy_bond-ctm_energy_bond_old)/ctm_energy_bond_old # relative difference with analytical result

        ctm_energy_bond_old = ctm_energy_bond

        with open("ctm_iteration_convergence_scheme_%s.txt" % file_name, "a+") as f:
            f.write('{:.3f} {:.2e} {:.2e} {:+.6f}\n'.format(beta, ntu_error_up, ntu_error_dn, energy))

        print("rel diff in energy: ", rel_diff_an_energy)


        


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinfull_finite()


