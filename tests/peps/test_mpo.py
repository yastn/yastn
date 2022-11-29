import numpy as np
import pytest
import logging
import argparse
import yast
import yast.tn.peps as peps
import yast.tn.mps as mps
import time
from yast.tn.peps.operators.gates import gates_hopping, gate_local_fermi_sea, gate_local_Hubbard
from yast.tn.peps.NTU import ntu_update, initialize_peps_purification
from yast.tn.peps.CTM import GetEnv, nn_bond, CtmEnv2Mps
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
    yy = 3
    D = 12
    chi = 5
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    beta_end = 0.01
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
        Gamma, info = ntu_update(Gamma, net, fid, Gate, D, step, tr_mode, fix_bd=0) # fix_bd = 0 refers to unfixed symmetry sectors

    nbit = 10
    opts = {'chi': round(chi), 'cutoff': 1e-10, 'nbitmax': round(nbit), 'prec' : 1e-8, 'tcinit' : ((0,) * fid.config.sym.NSYM,), 'Dcinit' : (1,)}
    env = GetEnv(A=Gamma, net=net, **opts, AAb_mode=0)


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
        Bctm = CtmEnv2Mps(net, env, index=r_index, index_type='r')  # bottom boundary of 0th row through CTM environment tensors
        print(*(Bctm[i].get_shape() for i in range(3)))
        print(*(psi[i].get_shape() for i in range(3)))
        print(mps.vdot(psi, Bctm) / (psi.norm() * Bctm.norm()))

        psi0 = psi.copy()
        O = Gamma.mpo(index=r_index, index_type='column')
        psi = mps.zipper(O, psi0, opts)  # bottom boundary of 0th row through zipper
        mps.variational_(psi, O, psi0, method='1site', max_sweeps=2)



if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinless()

