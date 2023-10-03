import numpy as np
import logging
import argparse
import yastn
import yastn.tn.fpeps as peps
import time
from yastn.tn.fpeps.operators.gates import gates_hopping, gate_local_Hubbard, gates_Heisenberg_spinful
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_peps_purification, initialize_Neel_spinful
from yastn.tn.fpeps.ctm import nn_exp_dict, ctmrg, one_site_dict

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

def benchmark_NTU_hubbard(lattice, boundary, purification, xx, yy, D, sym, J_z, J_ex, t_up, t_dn, chi, step, tr_mode, fix_bd):

    dims = (xx, yy)
    tot_sites = xx * yy
    
    net = peps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yastn.operators.SpinfulFermions_GP(sym='U1xU1xZ2_GP', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn, n_up, n_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d'), opt.n(spin='u'), opt.n(spin='d')
    n = n_up + n_dn
    Sz = 0.5*(n_up-n_dn)
    Sp = fcdag_up @ fc_dn
    Sm = fcdag_dn @ fc_up

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn},
           'SzSz': {'l': Sz, 'r': Sz},
           'SpSm': {'l': Sp, 'r': Sm},
           'SmSp': {'l': Sm, 'r': Sp},
           'nn': {'l': n, 'r': n},}
    
    opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}
    mdata = {}
    dbeta = 0.1

    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_initial_dbeta_%1.3f_%s_%s_Ds_%s_J_z_%1.4f_J_ex_%1.4f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, dbeta, tr_mode, step, D, J_z, J_ex, t_up, t_dn, sym)
    beta = 0
    tol = 1e-10 # truncation of singular values of CTM projectors
    max_sweeps=100  # ctm param
    tol_exp = 1e-6 
    energy_old = 0

    psi = initialize_Neel_spinful(fc_up, fc_dn, fcdag_up, fcdag_dn, net) # initialized in Néel configuration

    for _ in range(20000):
        beta = beta + dbeta
        sv_beta = int(beta * yastn.BETA_MULTIPLIER)
        logging.info("beta = %0.3f" % beta)

        coeff = 0.5 # 0.25 for purification; 0.5 for ground state calculation and 1j*0.5 for real-time evolution
        trotter_step = coeff * dbeta  

        GA_nn_up, GB_nn_up = gates_hopping(t_up, trotter_step, fid, fc_up, fcdag_up)
        GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, trotter_step, fid, fc_dn, fcdag_dn)
        SA, SB = gates_Heisenberg_spinful(J_z, J_ex, trotter_step, Sz, Sp, Sm, n)

        g_loc = fid
        g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn), (SA, SB)]

        gates = gates_homogeneous(psi, g_nn, g_loc)
         
        psi, info =  evolution_step_(psi, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 
        print(info)

        for ms in net.sites():
            logging.info("shape of peps tensor = " + str(ms) + ": " +str(psi[ms].get_shape()))
            xs = psi[ms].unfuse_legs((0, 1, 2))
            for l in range(6):
                print(xs.get_leg_structure(axis=l))
            
        opts_svd_ctm = {'D_total': chi, 'tol': tol}
        ctm_energy_old = 0

        for step in ctmrg(psi, max_sweeps, iterator_step=2, AAb_mode=0, opts_svd=opts_svd_ctm):

            assert step.sweeps % 2 == 0 # stop every 3rd step as iteration_step=3
            obs_hor, obs_ver =  nn_exp_dict(psi, step.env, ops)

            cdagc_up = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + sum(abs(val) for val in obs_ver.get('cdagc_up').values()))
            ccdag_up = (sum(abs(val) for val in obs_hor.get('ccdag_up').values()) + sum(abs(val) for val in obs_ver.get('ccdag_up').values()))
            cdagc_dn = (sum(abs(val) for val in obs_hor.get('cdagc_dn').values()) + sum(abs(val) for val in obs_ver.get('cdagc_dn').values()))
            ccdag_dn = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + sum(abs(val) for val in obs_ver.get('cdagc_up').values()))

            SzSz_v = J_z * (sum(abs(val) for val in obs_hor.get('SzSz').values()) + sum(abs(val) for val in obs_ver.get('SzSz').values()))
            SpSm_v = 0.5 * J_ex * (sum(abs(val) for val in obs_hor.get('SpSm').values()) + sum(abs(val) for val in obs_ver.get('SpSm').values()))
            SmSp_v = 0.5 * J_ex * (sum(abs(val) for val in obs_hor.get('SmSp').values()) + sum(abs(val) for val in obs_ver.get('SmSp').values()))
            nn_v = - 0.25 * J_ex * (sum(abs(val) for val in obs_hor.get('nn').values()) + sum(abs(val) for val in obs_ver.get('nn').values()))

            ctm_energy = (- (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) + SzSz_v + SpSm_v + SmSp_v + nn_v)/ tot_sites 
           
            print("expectation value: ", ctm_energy)
            if abs(ctm_energy - ctm_energy_old) < tol_exp:
                break # here break if the relative differnece is below tolerance
            ctm_energy_old = ctm_energy

        print('energy: ', ctm_energy)
        if ctm_energy > energy_old:
            beta = beta_old
            psi = psi_old
            dbeta = dbeta/2.0
            continue

        x = {(ms, sv_beta): psi[ms].save_to_dict() for ms in psi.sites()}
        mdata.update(x)
        np.save("neel_initialized_ground_state_tJ_spinful_tensors_%s.npy" % (file_name), mdata)
        if step=='svd-update':
            continue
        ntu_error_up = np.mean(np.sqrt(info['ntu_error'][::2]))
        ntu_error_dn = np.mean(np.sqrt(info['ntu_error'][1::2]))
        logging.info('ntu error up: %.4e' % ntu_error_up)
        logging.info('ntu error dn: %.4e' % ntu_error_dn)

        svd_error_up = np.mean(np.sqrt(info['svd_error'][::2]))
        svd_error_dn = np.mean(np.sqrt(info['svd_error'][1::2]))
        logging.info('svd error up: %.4e' % svd_error_up)
        logging.info('svd error dn: %.4e' % svd_error_dn)

        with open("NTU_error_ground_state_%s.txt" % file_name, "a+") as f:
            f.write('{:.3f} {:.4e} {:.4e} {:+.6f}\n'.format(beta, ntu_error_up, ntu_error_dn, ctm_energy))
        with open("SVD_error_ground_state_%s.txt" % file_name, "a+") as f:
            f.write('{:.3f} {:.4e} {:.4e} {:+.6f}\n'.format(beta, svd_error_up, svd_error_dn, ctm_energy))

        energy_old = ctm_energy
        psi_old = psi
        beta_old = beta

    
if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='square')     # lattice shape
    parser.add_argument("-x", type=int, default=4)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=4)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='obc') # boundary
    parser.add_argument("-p", type=str, default='False') # bool
    parser.add_argument("-D", type=int, default=6) 
    parser.add_argument("-S", default='U1xU1xZ2_GP')   # symmetry -- Z2xZ2 or U1xU1
    parser.add_argument("-TUP", type=float, default=1.)        # hopping_up
    parser.add_argument("-TDOWN", type=float, default=1.)      # hopping_down
    parser.add_argument("-JZ", type=float, default=0.4)       
    parser.add_argument("-JEX", type=float, default=0.4)        
    parser.add_argument("-X", type=int, default=30)   # dbeta
    parser.add_argument("-STEP", default='one-step')           # truncation can be done in 'one-step' or 'two-step'. note that truncations done 
                                                               # with svd update or when we fix the symmetry sectors are always 'one-step'
    parser.add_argument("-MODE", default='optimal')             # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    parser.add_argument("-FIXED", type=int, default=0)         # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    args = parser.parse_args()

    tt = time.time()
    benchmark_NTU_hubbard(lattice = args.L, boundary = args.B,  xx=args.x, yy=args.y, D=args.D, sym=args.S, 
                                t_up=args.TUP, t_dn=args.TDOWN, purification=args.p, J_z = args.JZ, J_ex=args.JEX,
                                chi=args.X, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))


