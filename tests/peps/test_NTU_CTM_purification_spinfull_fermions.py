""" Test the expectation values of spin 1/2 fermions with analytical values of fermi sea """
import numpy as np
import pytest
import logging
import argparse
import yast
import yast.tn.peps as peps
import time
from yast.tn.peps.operators.gates import gates_hopping, gate_local_Hubbard
from yast.tn.peps.evolution import evolution_step_, gates_homogeneous
from yast.tn.peps import initialize_peps_purification
from yast.tn.peps.ctm import nn_avg, ctmrg, init_rand, one_site_avg, nn_bond

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')

n_up = fcdag_up @ fc_up
n_dn = fcdag_dn @ fc_dn
n_int = n_up @ n_dn

def test_NTU_spinfull_finite():

    lattice = 'rectangle'
    boundary = 'finite'
    purification = 'True'
    xx = 3
    yy = 2
    D = 12
    mu_up, mu_dn = 0, 0 # chemical potential
    t_up, t_dn = 1, 1 # hopping amplitude
    U = 0
    beta_end = 0.1
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'

    dims = (xx, yy)
    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)
    
    GA_nn_up, GB_nn_up = gates_hopping(t_up, dbeta, fid, fc_up, fcdag_up, purification=purification)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, dbeta, fid, fc_dn, fcdag_dn, purification=purification)
    g_loc = gate_local_Hubbard(mu_up, mu_dn, U, dbeta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, purification=purification)
    g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn)]

    if purification == 'True':
        psi = initialize_peps_purification(fid, net) # initialized at infinite temperature
    
    gates = gates_homogeneous(psi, g_nn, g_loc)

    time_steps = round(beta_end / dbeta)
    opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}

    for nums in range(time_steps):

        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        psi, _ =  evolution_step_(psi, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 
    
    # convergence criteria for CTM based on total energy
    chi = 40 # environmental bond dimension
    tol = 1e-10 # truncation of singular values of CTM projectors
    max_sweeps=50 
    tol_exp = 1e-7   # difference of some observable must be lower than tolernace

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

    cf_energy_old = 0

    opts_svd_ctm = {'D_total': chi, 'tol': tol}


    for step in ctmrg(psi, max_sweeps, iterator_step=1, AAb_mode=0, opts_svd=opts_svd_ctm):
        
        assert step.sweeps % 1 == 0 # stop every 2nd step as iteration_step=2

        doc, _, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites

        obs_hor, obs_ver =  nn_avg(psi, step.env, ops)

        cdagc_up = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))
        ccdag_up = 0.5*(abs(obs_hor.get('ccdag_up')) + abs(obs_ver.get('ccdag_up')))
        cdagc_dn = 0.5*(abs(obs_hor.get('cdagc_dn')) + abs(obs_ver.get('cdagc_dn')))
        ccdag_dn = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))

        cf_energy = (xx * yy) * U * doc - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) * (2 * xx * yy - xx - yy)

        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy


    bd_h = peps.Bond(site_0 = (2, 0), site_1=(2, 1), dirn='h')
    bd_v = peps.Bond(site_0 = (0, 1), site_1=(1, 1), dirn='v')

    nn_CTM_bond_1_up = 0.5*(abs(nn_bond(psi, step.env, ops['cdagc_up'], bd_h)) + abs(nn_bond(psi, step.env, ops['ccdag_up'], bd_h)))
    nn_CTM_bond_2_up = 0.5*(abs(nn_bond(psi, step.env, ops['cdagc_up'], bd_v)) + abs(nn_bond(psi, step.env, ops['ccdag_up'], bd_v)))
    nn_CTM_bond_1_dn = 0.5*(abs(nn_bond(psi, step.env, ops['cdagc_dn'], bd_h)) + abs(nn_bond(psi, step.env, ops['ccdag_dn'], bd_h)))
    nn_CTM_bond_2_dn = 0.5*(abs(nn_bond(psi, step.env, ops['cdagc_dn'], bd_v)) + abs(nn_bond(psi, step.env, ops['ccdag_dn'], bd_v)))

    nn_bond_1_exact = 0.024917101651703362 # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (1,1) and (1,2)
    nn_bond_2_exact = 0.024896433958165112  # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (0,0) and (1,0)

    print(nn_CTM_bond_1_up)
    print(nn_CTM_bond_1_dn)
    print(nn_CTM_bond_2_up)
    print(nn_CTM_bond_2_dn)

    assert pytest.approx(nn_CTM_bond_1_up, abs=1e-5) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_1_dn, abs=1e-5) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2_up, abs=1e-5) == nn_bond_2_exact
    assert pytest.approx(nn_CTM_bond_2_dn, abs=1e-5) == nn_bond_2_exact


def test_NTU_spinfull_infinite():

    lattice = 'checkerboard'
    boundary = 'infinite'
    purification = 'True'
    D = 12
    chi = 40
    mu_up, mu_dn = 0, 0 # chemical potential
    t_up, t_dn = 1, 1 # hopping amplitude
    beta_end = 0.1
    U=0
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'
    net = peps.Peps(lattice=lattice, boundary=boundary)

    opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')

    GA_nn_up, GB_nn_up = gates_hopping(t_up, dbeta, fid, fc_up, fcdag_up, purification=purification)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, dbeta, fid, fc_dn, fcdag_dn, purification=purification)
    g_loc = gate_local_Hubbard(mu_up, mu_dn, U, dbeta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, purification=purification)
    g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn)]

    if purification == 'True':
        psi = initialize_peps_purification(fid, net) # initialized at infinite temperature
    
    gates = gates_homogeneous(psi, g_nn, g_loc)
    time_steps = round(beta_end / dbeta)
    opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}

    for nums in range(time_steps):

        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        psi, _ =  evolution_step_(psi, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 
    
    # convergence criteria for CTM based on total energy
    chi = 40 # environmental bond dimension
    tol = 1e-10 # truncation of singular values of CTM projectors
    max_sweeps=50 
    tol_exp = 1e-7   # difference of some observable must be lower than tolernace

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

    cf_energy_old = 0
    opts_svd_ctm = {'D_total': chi, 'tol': tol}

    for step in ctmrg(psi, max_sweeps, iterator_step=2, AAb_mode=0, opts_svd=opts_svd_ctm):
        
        assert step.sweeps % 2 == 0 # stop every 2nd step as iteration_step=2

        doc, _, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites

        obs_hor, obs_ver =  nn_avg(psi, step.env, ops)

        cdagc_up = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))
        ccdag_up = 0.5*(abs(obs_hor.get('ccdag_up')) + abs(obs_ver.get('ccdag_up')))
        cdagc_dn = 0.5*(abs(obs_hor.get('cdagc_dn')) + abs(obs_ver.get('cdagc_dn')))
        ccdag_dn = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))

        cf_energy = - (cdagc_up + ccdag_up +cdagc_dn + ccdag_dn) * 0.25

        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

    ob_hor, ob_ver = nn_avg(psi, step.env, ops)

    nn_CTM = 0.25 * (abs(ob_hor.get('cdagc_up')) + abs(ob_ver.get('ccdag_up'))+ abs(ob_ver.get('cdagc_dn'))+ abs(ob_ver.get('ccdag_dn')))
    print(nn_CTM)

    nn_exact = 0.02481459 # analytical nn fermionic correlator at beta = 0.1 for 2D infinite lattice with checkerboard ansatz

    assert pytest.approx(nn_CTM, abs=1e-3) == nn_exact

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinfull_finite()
    test_NTU_spinfull_infinite()
 

