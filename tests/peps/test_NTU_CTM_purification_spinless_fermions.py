""" Test the expectation values of spinless fermions with analytical values of fermi sea for finite and infinite lattices """
import numpy as np
import pytest
import logging
import argparse
import yast
import yast.tn.peps as peps
import time
from yast.tn.peps.operators.gates import gates_hopping, gate_local_fermi_sea
from yast.tn.peps.evolution import evolution_step_, gates_homogeneous
from yast.tn.peps import initialize_peps_purification
from yast.tn.peps.ctm import nn_avg, ctmrg_, init_rand, nn_bond
try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg


def test_NTU_spinless_finite():

    lattice = 'rectangle'
    boundary = 'finite'
    purification = 'True'
    xx = 3
    yy = 2
    D = 6
    chi = 20
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    beta_end = 0.2
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'

    dims = (xx, yy)
    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)
   
    opt = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    GA_nn, GB_nn = gates_hopping(t, dbeta, fid, fc, fcdag, purification=purification)  # nn gate for 2D fermi sea
    g_loc = gate_local_fermi_sea(mu, dbeta, fid, fc, fcdag, purification=purification) # local gate for spinless fermi sea
    g_nn = [(GA_nn, GB_nn)]

    if purification == 'True':
        psi = initialize_peps_purification(fid, net) # initialized at infinite temperature

    gates = gates_homogeneous(psi, g_nn, g_loc)
    time_steps = round(beta_end / dbeta)

    for nums in range(time_steps):

        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        psi, _ =  evolution_step_(psi, gates, D, step, tr_mode, env_type='NTU') 
    
    # convergence criteria for CTM based on total energy
    chi = 40 # environmental bond dimension
    cutoff = 1e-10
    max_sweeps=50 
    tol = 1e-7   # difference of some observable must be lower than tolernace

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    cf_energy_old = 0

    for step in ctmrg_(psi, chi, cutoff, max_sweeps, iterator_step=2, AAb_mode=0, fix_signs=False):
        
        assert step.sweeps % 2 == 0 # stop every 4th step as iteration_step=2
        obs_hor, obs_ver =  nn_avg(psi, step.env, ops)

        cdagc = 0.5*(abs(obs_hor.get('cdagc')) + abs(obs_ver.get('cdagc')))
        ccdag = 0.5*(abs(obs_hor.get('ccdag')) + abs(obs_ver.get('ccdag')))

        cf_energy = - (cdagc + ccdag) * (2 * xx * yy - xx - yy)

        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

    bd_h = peps.Bond(site_0 = (2, 0), site_1=(2, 1), dirn='h')
    bd_v = peps.Bond(site_0 = (0, 1), site_1=(1, 1), dirn='v')


    nn_CTM_bond_1 = 0.5*(abs(nn_bond(psi, step.env, ops['cdagc'], bd_h)) + abs(nn_bond(psi, step.env, ops['ccdag'], bd_h)))
    nn_CTM_bond_2 = 0.5*(abs(nn_bond(psi, step.env, ops['cdagc'], bd_v)) + abs(nn_bond(psi, step.env, ops['ccdag'], bd_v)))

    print(nn_CTM_bond_1, nn_CTM_bond_2)

    nn_bond_1_exact = 0.04934701696955436 # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (1,1) and (1,2)
    nn_bond_2_exact = 0.049185554490429065  # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (0,0) and (1,0)
    assert pytest.approx(nn_CTM_bond_1, abs=1e-6) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2, abs=1e-6) == nn_bond_2_exact

def test_NTU_spinless_infinite():

    lattice = 'rectangle'
    boundary = 'infinite'
    purification = 'True'
    xx = 3
    yy = 2
    D = 8
    chi = 40
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    beta_end = 0.2
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'
    net = peps.Peps(lattice=lattice, dims=(xx, yy), boundary=boundary)
    opt = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    GA_nn, GB_nn = gates_hopping(t, dbeta, fid, fc, fcdag, purification=purification)  # nn gate for 2D fermi sea
    g_loc = gate_local_fermi_sea(mu, dbeta, fid, fc, fcdag, purification=purification) # local gate for spinless fermi sea
    g_nn = [(GA_nn, GB_nn)]

    if purification == 'True':
        psi = initialize_peps_purification(fid, net) # initialized at infinite temperature

    gates = gates_homogeneous(psi, g_nn, g_loc)
    time_steps = round(beta_end / dbeta)

    for nums in range(time_steps):

        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        psi, _ =  evolution_step_(psi, gates, D, step, tr_mode, env_type='NTU') # fix_bd = 0 refers to unfixed symmetry sectors    

    # convergence criteria for CTM based on total energy
    chi = 40 # environmental bond dimension
    cutoff = 1e-10
    max_sweeps=50 
    tol = 1e-7   # difference of some observable must be lower than tolernace

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    cf_energy_old = 0

    for step in ctmrg_(psi, chi, cutoff, max_sweeps, iterator_step=1, AAb_mode=0):
        
        assert step.sweeps % 1 == 0 # stop every 2nd step as iteration_step=2
        obs_hor, obs_ver =  nn_avg(psi, step.env, ops)

        cdagc = 0.5*(abs(obs_hor.get('cdagc')) + abs(obs_ver.get('cdagc')))
        ccdag = 0.5*(abs(obs_hor.get('ccdag')) + abs(obs_ver.get('ccdag')))

        cf_energy = - (cdagc + ccdag) * (2 * xx * yy - xx - yy)

        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

    ob_hor, ob_ver = nn_avg(psi, step.env, ops)

    nn_CTM = 0.5 * (abs(ob_hor.get('cdagc')) + abs(ob_ver.get('ccdag')))
    print(nn_CTM)

    nn_exact = 0.04856353 # analytical nn fermionic correlator at beta = 0.2 for 2D infinite lattice with checkerboard ansatz

    assert pytest.approx(nn_CTM, abs=1e-5) == nn_exact

if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    test_NTU_spinless_finite()
    #test_NTU_spinless_infinite()


