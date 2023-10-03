""" Test the expectation values of spin 1/2 fermions with analytical values of fermi sea """
import numpy as np
import pytest
import logging
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.operators.gates import gates_hopping, gates_Heisenberg_spinful
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_peps_purification, initialize_diagonal_basis
from yastn.tn.fpeps.ctm import nn_exp_dict, ctmrg

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

opt = yastn.operators.SpinfulFermions_GP(sym='U1xU1xZ2_GP', backend=cfg.backend, default_device=cfg.default_device)
fid, fc_up, fc_dn, fcdag_up, fcdag_dn, n_up, n_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d'), opt.n(spin='u'), opt.n(spin='d')
Sz = 0.5*(n_up-n_dn)
Sp = fcdag_up @ fc_dn
Sm = fcdag_dn @ fc_up
n = n_up + n_dn


def test_NTU_tJ_finite():

    lattice = 'square'
    boundary = 'obc'
    xx = 4
    yy = 4
    tot_sites = (xx * yy)
    D = 6
    t_up, t_dn = 1, 1 # hopping amplitude
 
    beta_end = 0.02
    dbeta = 0.01
    step = 'two-step'
    tr_mode = 'optimal'
    coeff = 0.25 # for purification; 0.5 for ground state calculation and 1j*0.5 for real-time evolution
    trotter_step = coeff * dbeta  
    J_z = 0.4
    J_ex = 0.4
    dims = (xx, yy)
    net = fpeps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)
    GA_nn_up, GB_nn_up = gates_hopping(t_up, trotter_step, fid, fc_up, fcdag_up)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, trotter_step, fid, fc_dn, fcdag_dn)
    GA_HS, GB_HS = gates_Heisenberg_spinful(J_z, J_ex, trotter_step, Sz, Sp, Sm, n)

    g_loc = fid
    g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn), (GA_HS, GB_HS)]

    #peps = initialize_peps_purification(fid, net) # initialized at infinite temperature
  
    ho = fid - n_up - n_dn  # eventually zero blocks should be removed

    nn_up, nn_dn, nn_hole = n_up, n_dn, ho
    projectors = [nn_up, nn_dn, nn_hole]
    out_init_1 =  {(0, 0): 0, (0, 1): 1, (0, 2): 0, (0, 3): 1,(1, 0): 0, (1, 1): 1, (1, 2): 0, 
                   (1, 3): 1,(2, 0): 0, (2, 1): 1, (2, 2): 0, (2, 3): 1,(3, 0): 0, (3, 1): 1, 
                   (3, 2): 0, (3, 3): 1}     # Neel state
    out_init_2 =  {(0, 0): 2, (0, 1): 1, (0, 2): 0, (0, 3): 1,(1, 0): 0, (1, 1): 1, (1, 2): 2,
                    (1, 3): 2,(2, 0): 0, (2, 1): 1, (2, 2): 0, (2, 3): 1,(3, 0): 1, (3, 1): 1, 
                    (3, 2): 0, (3, 3): 0}   # doped

    # peps initialization
    peps = initialize_diagonal_basis(projectors, net, out_init_2) 
    
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
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn},
           'Spm': {'l': fcdag_up, 'r': fc_dn},
           'Smp': {'l': fcdag_dn, 'r': fc_up},
           'SzSz': {'l': 0.5*(n_up-n_dn), 'r': 0.5*(n_up-n_dn)},
           'nn': {'l': n, 'r': n}}

    cf_energy_old = 0
    opts_svd_ctm = {'D_total': chi, 'tol': tol}
    for step in ctmrg(peps, max_sweeps, iterator_step=2, AAb_mode=0, opts_svd=opts_svd_ctm):
        
        assert step.sweeps % 2 == 0 # stop every 2nd step as iteration_step=2
        obs_hor, obs_ver =  nn_exp_dict(peps, step.env, ops)
        cdagc_up = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + sum(abs(val) for val in obs_ver.get('cdagc_up').values()))
        ccdag_up = (sum(abs(val) for val in obs_hor.get('ccdag_up').values()) + sum(abs(val) for val in obs_ver.get('ccdag_up').values()))
        cdagc_dn = (sum(abs(val) for val in obs_hor.get('cdagc_dn').values()) + sum(abs(val) for val in obs_ver.get('cdagc_dn').values()))
        ccdag_dn = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + sum(abs(val) for val in obs_ver.get('cdagc_up').values()))

        Spm = (sum(abs(val) for val in obs_hor.get('SpSm').values()) + sum(abs(val) for val in obs_ver.get('SpSm').values()))
        Smp = (sum(abs(val) for val in obs_hor.get('SmSp').values()) + sum(abs(val) for val in obs_ver.get('SmSp').values()))
        SzSz = (sum(abs(val) for val in obs_hor.get('SzSz').values()) + sum(abs(val) for val in obs_ver.get('SzSz').values()))
        nana = (sum(abs(val) for val in obs_hor.get('nn').values()) + sum(abs(val) for val in obs_ver.get('nn').values()))

        cf_energy = (- (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) +  (0.5 * J_ex * Spm + 0.5 * J_ex * Smp + J_z * SzSz - 0.25 * J_ex * nana)) / tot_sites

        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

 
if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_tJ_finite()
 
