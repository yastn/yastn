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

    
    O = Gamma.mpo(index=1)
    psi = Gamma.boundary_mps()

    for x in O.A:
        print(x)

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinless()

