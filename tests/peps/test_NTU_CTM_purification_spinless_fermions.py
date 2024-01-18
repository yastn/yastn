""" Test the expectation values of spinless fermions with analytical values of fermi sea for finite and infinite lattices """
import numpy as np
import pytest
import logging
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.gates import gates_hopping, gate_local_fermi_sea
from yastn.tn.fpeps import product_peps
from yastn.tn.fpeps.ctm import nn_exp_dict, ctmrg, EV2ptcorr
try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg


def test_NTU_spinless_finite():

    boundary = 'obc'
    purification = 'True'
    xx = 3
    yy = 2
    tot_sites = (xx * yy)
    D = 6
    chi = 30 # environmental bond dimension
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    beta_end = 0.2
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'

    coeff = 0.25 # for purification; 0.5 for ground state calculation and 1j*0.5 for real-time evolution
    trotter_step = coeff * dbeta

    dims = (xx, yy)
    geometry = fpeps.SquareLattice(dims, boundary)  # shape = (rows, columns)
    #net = fpeps.Peps(geometry)

    opt = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    G_hop = gates_hopping(t, trotter_step, fid, fc, fcdag)  # nn gate for 2D fermi sea
    g_loc = gate_local_fermi_sea(mu, trotter_step, fid, fc, fcdag) # local gate for spinless fermi sea
    g_nn = [(G_hop)]

    if purification == 'True':
        peps = product_peps(geometry, fid) # initialized at infinite temperature

    gates = fpeps.gates_homogeneous(peps, g_nn, g_loc)
    time_steps = round(beta_end / dbeta)

    env = fpeps.EnvNTU(peps)
    opts = {"D_total": D, 'tol_block': 1e-15, "gradual_truncation": False, "initialization": "EAT"}  # truncation options  # should be also bond dimension

    for nums in range(time_steps):
        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        info = fpeps.evolution_step_(env, gates, opts)

    # convergence criteria for CTM based on total energy
    chi = 40 # environmental bond dimension
    tol = 1e-10 # truncation of singular values of ctm projectors
    max_sweeps=50
    tol_exp = 1e-7   # difference of some expectation value must be lower than tolerance

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    cf_energy_old = 0

    opts_svd_ctm = {'D_total': chi, 'tol': tol}

    for step in ctmrg(peps, max_sweeps, iterator_step=2, AAb_mode=0, fix_signs=False, opts_svd=opts_svd_ctm):

        assert step.sweeps % 2 == 0 # stop every 2nd step as iteration_step=2
        obs_hor, obs_ver =  nn_exp_dict(peps, step.env, ops)

        cdagc = (sum(abs(val) for val in obs_hor.get('cdagc').values()) + sum(abs(val) for val in obs_ver.get('cdagc').values()))
        ccdag = (sum(abs(val) for val in obs_hor.get('ccdag').values()) + sum(abs(val) for val in obs_ver.get('ccdag').values()))

        cf_energy = - (cdagc + ccdag) / tot_sites

        print("energy: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

    bd_h = fpeps.Bond(site0=(2, 0), site1=(2, 1))
    bd_v = fpeps.Bond(site0=(0, 1), site1=(1, 1))

    nn_CTM_bond_1 = 0.5*(abs(EV2ptcorr(peps, step.env, ops['cdagc'], bd_h.site0, bd_h.site1)) + abs(EV2ptcorr(peps, step.env, ops['ccdag'], bd_h.site0, bd_h.site1)))
    nn_CTM_bond_2 = 0.5*(abs(EV2ptcorr(peps, step.env, ops['cdagc'], bd_v.site0, bd_v.site1)) + abs(EV2ptcorr(peps, step.env, ops['ccdag'], bd_v.site0, bd_v.site1)))

    nn_bond_1_exact = 0.04934701696955436 # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (1,1) and (1,2)
    nn_bond_2_exact = 0.049185554490429065  # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (0,0) and (1,0)
    assert pytest.approx(nn_CTM_bond_1, abs=1e-6) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2, abs=1e-6) == nn_bond_2_exact

def test_NTU_spinless_infinite():
    purification = 'True'
    D = 8
    chi = 40
    mu = 0 # chemical potential
    t = 1 # hopping amplitude
    beta_end = 0.2
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'
    coeff = 0.25 # for purification; 0.5 for ground state calculation and 1j*0.5 for real-time evolution
    trotter_step = coeff * dbeta
    geometry = fpeps.CheckerboardLattice()
    opt = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    GA_nn, GB_nn = gates_hopping(t, trotter_step, fid, fc, fcdag)  # nn gate for 2D fermi sea
    g_loc = gate_local_fermi_sea(mu, trotter_step, fid, fc, fcdag) # local gate for spinless fermi sea
    g_nn = [(GA_nn, GB_nn)]

    if purification == 'True':
        peps = product_peps(geometry, fid) # initialized at infinite temperature

    gates = fpeps.gates_homogeneous(peps, g_nn, g_loc)
    time_steps = round(beta_end / dbeta)

    env = fpeps.EnvNTU(peps)
    opts = {"D_total": D, 'tol_block': 1e-15, "gradual_truncation": False, "initialization": "EAT"}  # truncation options  # should be also bond dimension

    for nums in range(time_steps):
        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        info = fpeps.evolution_step_(env, gates, opts)

    # convergence criteria for CTM based on total energy
    chi = 40 # environmental bond dimension
    tol = 1e-10 # CTM projectors
    max_sweeps=50
    tol_exp = 1e-7   # difference of some observable must be lower than tolernace

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    cf_energy_old = 0

    opts_svd_ctm = {'D_total': chi, 'tol': tol}

    for step in ctmrg(peps, max_sweeps, iterator_step=1, AAb_mode=0, opts_svd=opts_svd_ctm):

        assert step.sweeps % 1 == 0 # stop every 2nd step as iteration_step=2
        obs_hor, obs_ver =  nn_exp_dict(peps, step.env, ops)

        cdagc = (sum(abs(val) for val in obs_hor.get('cdagc').values()) + sum(abs(val) for val in obs_ver.get('cdagc').values()))
        ccdag = (sum(abs(val) for val in obs_hor.get('ccdag').values()) + sum(abs(val) for val in obs_ver.get('ccdag').values()))

        cf_energy = - (cdagc + ccdag) * 0.125

        print("energy: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

    obs_hor, obs_ver = nn_exp_dict(peps, step.env, ops)

    nn_CTM = 0.125 * (cdagc + ccdag)

    nn_exact = 0.04856353 # analytical nn fermionic correlator at beta = 0.2 for 2D infinite lattice with checkerboard ansatz

    assert pytest.approx(nn_CTM, abs=1e-5) == nn_exact

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinless_finite()
    test_NTU_spinless_infinite()


