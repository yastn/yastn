import numpy as np
import pytest
import logging
import argparse
import yast
import yast.tn.peps as peps
import time
from yast.tn.peps.operators.gates import gates_hopping, gate_local_fermi_sea, gate_local_Hubbard
from yast.tn.peps.NTU import ntu_update, initialize_peps_purification
from yast.tn.peps.CTM import GetEnv, nn_bond


def test_NTU_spinless():

    try:
        from .configs import config_U1_R_fermionic as cfg
        # cfg is used by pytest to inject different backends and divices
    except ImportError:
        from configs import config_U1_R_fermionic as cfg

    lattice = 'rectangle'
    boundary = 'finite'
    purification = 'True'
    xx = 3
    yy = 2
    D = 6
    chi = 20
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    beta_end = 0.1
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'

    dims = (yy, xx)
    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()
    ancilla='True'
    GA_nn, GB_nn = gates_hopping(t, dbeta, fid, fc, fcdag, ancilla=ancilla, purification=purification)  # nn gate for 2D fermi sea
    G_loc = gate_local_fermi_sea(mu, dbeta, fid, fc, fcdag, ancilla=ancilla, purification=purification) # local gate for spinless fermi sea
    Gate = {'loc': G_loc, 'nn':{'GA': GA_nn, 'GB': GB_nn}}

    if purification == 'True':
        Gamma = initialize_peps_purification(fid, net) # initialized at infinite temperature

    time_steps = round(beta_end / dbeta)

    for nums in range(time_steps):

        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        
        Gamma, info =  ntu_update(Gamma, net, fid, Gate, D, step, tr_mode, fix_bd=0) # fix_bd = 0 refers to unfixed symmetry sectors
    
    nbit = 10
    opts = {'chi': round(chi), 'cutoff': 1e-10, 'nbitmax': round(nbit), 'prec' : 1e-8, 'tcinit' : ((0,) * fid.config.sym.NSYM,), 'Dcinit' : (1,)}
    env = GetEnv(A=Gamma, net=net, **opts, AAb_mode=0)

    bd_h = peps.Bond(site_0 = (1, 1), site_1=(1, 2), dirn='h')
    bd_v = peps.Bond(site_0 = (0, 0), site_1=(1, 0), dirn='v')

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    nn_CTM_bond_1 = 0.5*(abs(nn_bond(Gamma, net, env, ops['cdagc'], bd_h)) + abs(nn_bond(Gamma, net, env, ops['ccdag'], bd_h)))
    nn_CTM_bond_2 = 0.5*(abs(nn_bond(Gamma, net, env, ops['cdagc'], bd_v)) + abs(nn_bond(Gamma, net, env, ops['ccdag'], bd_v)))

    print(nn_CTM_bond_1, nn_CTM_bond_2)

    nn_bond_1_exact = 0.02489643395816514 # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (1,1) and (1,2)
    nn_bond_2_exact = 0.0249171016517031  # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (0,0) and (1,0)
    assert pytest.approx(nn_CTM_bond_1, abs=1e-8) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2, abs=1e-8) == nn_bond_2_exact


def test_NTU_spinfull():

    try:
        from .configs import config_U1xU1_R_fermionic as cfg
        # cfg is used by pytest to inject different backends and divices
    except ImportError:
        from configs import config_U1xU1_R_fermionic as cfg

    lattice = 'rectangle'
    boundary = 'finite'
    purification = 'True'
    xx = 3
    yy = 2
    D = 12
    chi = 40
    mu_up, mu_dn = 0, 0 # chemical potential
    t_up, t_dn = 1, 1 # hopping amplitude
    U = 0
    beta_end = 0.1
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'

    dims = (yy, xx)
    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    ancilla='True'
    GA_nn_up, GB_nn_up = gates_hopping(t_up, dbeta, fid, fc_up, fcdag_up, ancilla=ancilla, purification=purification)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, dbeta, fid, fc_dn, fcdag_dn, ancilla=ancilla, purification=purification)
    G_loc = gate_local_Hubbard(mu_up, mu_dn, U, dbeta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, ancilla=ancilla, purification=purification)
    Gate = {'loc':G_loc, 'nn':{'GA_up':GA_nn_up, 'GB_up':GB_nn_up, 'GA_dn':GA_nn_dn, 'GB_dn':GB_nn_dn}}

    if purification == 'True':
        Gamma = initialize_peps_purification(fid, net) # initialized at infinite temperature

    time_steps = round(beta_end / dbeta)

    for nums in range(time_steps):

        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        
        Gamma, info =  ntu_update(Gamma, net, fid, Gate, D, step, tr_mode, fix_bd=0) # fix_bd = 0 refers to unfixed symmetry sectors
    
    nbit = 10
    opts = {'chi': round(chi), 'cutoff': 1e-10, 'nbitmax': round(nbit), 'prec' : 1e-8, 'tcinit' : ((0,) * fid.config.sym.NSYM,), 'Dcinit' : (1,)}
    env = GetEnv(A=Gamma, net=net, **opts, AAb_mode=0)

    bd_h = peps.Bond(site_0 = (1, 1), site_1=(1, 2), dirn='h')
    bd_v = peps.Bond(site_0 = (0, 0), site_1=(1, 0), dirn='v')

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

    nn_CTM_bond_1_up = 0.5*(abs(nn_bond(Gamma, net, env, ops['cdagc_up'], bd_h)) + abs(nn_bond(Gamma, net, env, ops['ccdag_up'], bd_h)))
    nn_CTM_bond_2_up = 0.5*(abs(nn_bond(Gamma, net, env, ops['cdagc_up'], bd_v)) + abs(nn_bond(Gamma, net, env, ops['ccdag_up'], bd_v)))
    nn_CTM_bond_1_dn = 0.5*(abs(nn_bond(Gamma, net, env, ops['cdagc_dn'], bd_h)) + abs(nn_bond(Gamma, net, env, ops['ccdag_dn'], bd_h)))
    nn_CTM_bond_2_dn = 0.5*(abs(nn_bond(Gamma, net, env, ops['cdagc_dn'], bd_v)) + abs(nn_bond(Gamma, net, env, ops['ccdag_dn'], bd_v)))

    print(nn_CTM_bond_1_up, nn_CTM_bond_2_up, nn_CTM_bond_1_dn, nn_CTM_bond_2_dn)

    nn_bond_1_exact = 0.02489643395816514 # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (1,1) and (1,2)
    nn_bond_2_exact = 0.0249171016517031  # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (0,0) and (1,0)
    assert pytest.approx(nn_CTM_bond_1_up, abs=2e-4) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_1_dn, abs=2e-4) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2_up, abs=2e-4) == nn_bond_2_exact
    assert pytest.approx(nn_CTM_bond_2_dn, abs=2e-4) == nn_bond_2_exact


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinless()
    test_NTU_spinfull()

