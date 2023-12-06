""" Test the expectation values of spin 1/2 fermions with analytical values of fermi sea """
import numpy as np
import pytest
import logging
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.operators.gates import gates_hopping, gate_local_Hubbard
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import product_peps
from yastn.tn.fpeps.ctm import nn_exp_dict, ctmrg, one_site_dict, EV2ptcorr

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')

n_up = fcdag_up @ fc_up
n_dn = fcdag_dn @ fc_dn
n_int = n_up @ n_dn

def test_NTU_spinful_finite():

    lattice = 'square'
    boundary = 'obc'
    purification = 'True'
    xx = 3
    yy = 2
    tot_sites = (xx * yy)
    D = 12
    mu_up, mu_dn = 0, 0 # chemical potential
    t_up, t_dn = 1, 1 # hopping amplitude
    U = 0
    beta_end = 0.1
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'
    coeff = 0.25 # for purification; 0.5 for ground state calculation and 1j*0.5 for real-time evolution
    trotter_step = coeff * dbeta  

    dims = (xx, yy)
    geometry = fpeps.SquareLattice(lattice, dims, boundary)  # shape = (rows, columns)
    
    GA_nn_up, GB_nn_up = gates_hopping(t_up, trotter_step, fid, fc_up, fcdag_up)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, trotter_step, fid, fc_dn, fcdag_dn)
    g_loc = gate_local_Hubbard(mu_up, mu_dn, U, trotter_step, fid, n_up, n_dn)
    g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn)]

    if purification == 'True':
        peps = product_peps(geometry, fid) # initialized at infinite temperature
    
    gates = gates_homogeneous(peps, g_nn, g_loc)

    time_steps = round(beta_end / dbeta)
    opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}

    for nums in range(time_steps):

        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        peps, _ =  evolution_step_(peps, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 
    
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

    for step in ctmrg(peps, max_sweeps, iterator_step=1, AAb_mode=0, opts_svd=opts_svd_ctm):
        
        assert step.sweeps % 1 == 0 # stop every 2nd step as iteration_step=2

        d_oc = one_site_dict(peps, step.env, n_int) # first entry of the function gives average of one-site observables of the sites

        obs_hor, obs_ver =  nn_exp_dict(peps, step.env, ops)

        cdagc_up = 0.5 * (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + 
                  sum(abs(val) for val in obs_ver.get('cdagc_up').values()))

        ccdag_up = 0.5 * (sum(abs(val) for val in obs_hor.get('ccdag_up').values()) + 
                  sum(abs(val) for val in obs_ver.get('ccdag_up').values()))

        cdagc_dn = 0.5 * (sum(abs(val) for val in obs_hor.get('cdagc_dn').values()) + 
                  sum(abs(val) for val in obs_ver.get('cdagc_dn').values()))

        ccdag_dn = 0.5 * (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + 
                  sum(abs(val) for val in obs_ver.get('cdagc_up').values()))


        cf_energy = U * sum(d_oc.values()) - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) / tot_sites

        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

    bd_h = fpeps.Bond(site_0 = (2, 0), site_1=(2, 1), dirn='h')
    bd_v = fpeps.Bond(site_0 = (0, 1), site_1=(1, 1), dirn='v')

    nn_CTM_bond_1_up = 0.5*(abs(EV2ptcorr(peps, step.env, ops['cdagc_up'], bd_h.site_0, bd_h.site_1)) + abs(EV2ptcorr(peps, step.env, ops['ccdag_up'], bd_h.site_0, bd_h.site_1)))
    nn_CTM_bond_2_up = 0.5*(abs(EV2ptcorr(peps, step.env, ops['cdagc_up'], bd_v.site_0, bd_v.site_1)) + abs(EV2ptcorr(peps, step.env, ops['ccdag_up'], bd_v.site_0, bd_v.site_1)))
    nn_CTM_bond_1_dn = 0.5*(abs(EV2ptcorr(peps, step.env, ops['cdagc_dn'], bd_h.site_0, bd_h.site_1)) + abs(EV2ptcorr(peps, step.env, ops['ccdag_dn'], bd_h.site_0, bd_h.site_1)))
    nn_CTM_bond_2_dn = 0.5*(abs(EV2ptcorr(peps, step.env, ops['cdagc_dn'], bd_v.site_0, bd_v.site_1)) + abs(EV2ptcorr(peps, step.env, ops['ccdag_dn'], bd_v.site_0, bd_v.site_1)))

    nn_bond_1_exact = 0.024917101651703362 # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (1,1) and (1,2)
    nn_bond_2_exact = 0.024896433958165112  # analytical nn fermionic correlator at beta = 0.1 for 2D finite lattice (2,3) bond bond between (0,0) and (1,0)

    assert pytest.approx(nn_CTM_bond_1_up, abs=1e-5) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_1_dn, abs=1e-5) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2_up, abs=1e-5) == nn_bond_2_exact
    assert pytest.approx(nn_CTM_bond_2_dn, abs=1e-5) == nn_bond_2_exact


def test_NTU_spinful_infinite():

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
    coeff = 0.25 # for purification; 0.5 for ground state calculation and 1j*0.5 for real-time evolution
    trotter_step = coeff * dbeta  
    geometry = fpeps.SquareLattice(lattice=lattice, boundary=boundary)

    GA_nn_up, GB_nn_up = gates_hopping(t_up, trotter_step, fid, fc_up, fcdag_up)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, trotter_step, fid, fc_dn, fcdag_dn)
    g_loc = gate_local_Hubbard(mu_up, mu_dn, U, trotter_step, fid, n_up, n_dn)
    g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn)]

    if purification == 'True':
        peps = product_peps(geometry, fid) # initialized at infinite temperature
    
    gates = gates_homogeneous(peps, g_nn, g_loc)
    time_steps = round(beta_end / dbeta)
    opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}

    for nums in range(time_steps):

        beta = (nums + 1) * dbeta
        logging.info("beta = %0.3f" % beta)
        peps, _ =  evolution_step_(peps, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 
    
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

    for step in ctmrg(peps, max_sweeps, iterator_step=2, AAb_mode=0, opts_svd=opts_svd_ctm):
        
        assert step.sweeps % 2 == 0 # stop every 2nd step as iteration_step=2

        obs_hor, obs_ver =  nn_exp_dict(peps, step.env, ops)

        cdagc_up = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + 
                  sum(abs(val) for val in obs_ver.get('cdagc_up').values()))

        ccdag_up = (sum(abs(val) for val in obs_hor.get('ccdag_up').values()) + 
                  sum(abs(val) for val in obs_ver.get('ccdag_up').values()))

        cdagc_dn = (sum(abs(val) for val in obs_hor.get('cdagc_dn').values()) + 
                  sum(abs(val) for val in obs_ver.get('cdagc_dn').values()))

        ccdag_dn = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + 
                  sum(abs(val) for val in obs_ver.get('cdagc_up').values()))


        cf_energy = - (cdagc_up + ccdag_up +cdagc_dn + ccdag_dn) * 0.125

        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

    nn_CTM = 0.125 * 0.5 * (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) 

    nn_exact = 0.02481459 # analytical nn fermionic correlator at beta = 0.1 for 2D infinite lattice with checkerboard ansatz

    assert pytest.approx(nn_CTM, abs=1e-3) == nn_exact

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinful_finite()
    test_NTU_spinful_infinite()
 

