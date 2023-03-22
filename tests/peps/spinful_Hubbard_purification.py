import numpy as np
import logging
import argparse
import yast
import yast.tn.peps as peps
import time
from yast.tn.peps.operators.import_distribution import import_distribution
from yast.tn.peps.operators.gates import gates_hopping, gate_local_Hubbard
from yast.tn.peps.evolution import evolution_step_, gates_homogeneous
from yast.tn.peps import initialize_peps_purification, initialize_Neel_spinfull

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

def benchmark_NTU_hubbard(lattice, boundary, purification, xx, yy, D, interval, sym, mu_up, mu_dn, U, t_up, t_dn, beta_end, dbeta, step, tr_mode, fix_bd):

    dims = (xx, yy)
    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')

    if purification == 'True':
        psi = initialize_peps_purification(fid, net) # initialized at infinite temperature
    elif purification == 'False':
        psi = initialize_Neel_spinfull(fid, fc_up, fc_dn, fcdag_up, fcdag_dn, net) # initialized at infinite temperature

    GA_nn_up, GB_nn_up = gates_hopping(t_up, dbeta, fid, fc_up, fcdag_up, purification=purification)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, dbeta, fid, fc_dn, fcdag_dn, purification=purification)
    g_loc = gate_local_Hubbard(mu_up, mu_dn, U, dbeta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, purification=purification)
    g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn)]

    gates = gates_homogeneous(psi, g_nn, g_loc)
    time_steps = round(beta_end / dbeta)
    obs_int = (interval / beta_end) * time_steps

    if isinstance(D, str):
        list_dist_Hubbard = import_distribution('Hubbard')   # import relevant distributions for the Hubbard model
        if step!='one-step':
            step='one-step'
            print('Automatically shifting to one-step for a choice of truncation to fixed sector sizes')
        D = list_dist_Hubbard[D]
        Ds = sum(D.values())
        opts_svd_ntu = {'D_total':Ds, 'D_block': D, 'tol_block': 1e-15}
    elif isinstance(D, int):
        opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}
        Ds = D 

    
    mdata = {}
    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, Ds, U, mu_up, mu_dn, t_up, t_dn, sym)

    for nums in range(time_steps):
        beta = (nums + 1) * dbeta
        sv_beta = int(beta * yast.BETA_MULTIPLIER)
        logging.info("beta = %0.3f" % beta)
        psi, info =  evolution_step_(psi, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 

        for ms in net.sites():
            logging.info("shape of peps tensor = " + str(ms) + ": " +str(psi[ms].get_shape()))
            xs = psi[ms].unfuse_legs((0, 1))
            for l in range(4):
                print(xs.get_leg_structure(axis=l))

        if (nums + 1) % int(obs_int) == 0:
            x = {(ms, sv_beta): psi[ms].save_to_dict() for ms in psi.sites()}
            mdata.update(x)
            np.save("purification_Hubbard_spinfull_tensors_%s.npy" % (file_name), mdata)
        if step=='svd-update':
            continue
        ntu_error_up = np.mean(np.sqrt(info['ntu_error'][::2]))
        ntu_error_dn = np.mean(np.sqrt(info['ntu_error'][1::2]))
        logging.info('ntu error up: %.2e' % ntu_error_up)
        logging.info('ntu error dn: %.2e' % ntu_error_dn)

        svd_error_up = np.mean(np.sqrt(info['svd_error'][::2]))
        svd_error_dn = np.mean(np.sqrt(info['svd_error'][1::2]))
        logging.info('svd error up: %.2e' % svd_error_up)
        logging.info('svd error dn: %.2e' % svd_error_dn)

        with open("NTU_error_%s.txt" % file_name, "a+") as f:
            f.write('{:.3f} {:.3e} {:.3e}\n'.format(beta, ntu_error_up, ntu_error_dn))
        with open("SVD_error_%s.txt" % file_name, "a+") as f:
            f.write('{:.3f} {:.3e} {:.3e}\n'.format(beta, svd_error_up, svd_error_dn))

    
if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='checkerboard')     # lattice shape
    parser.add_argument("-x", type=int, default=2)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=2)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='infinite') # boundary
    parser.add_argument("-p", type=str, default='True') # bool
    parser.add_argument("-D", type=str, default='9_1')            # bond dimension or distribution of virtual legs of peps tensors,  input can be either an 
                                                               # integer representing total bond dimension or a string labeling a sector-wise 
                                                               # distribution of bond dimensions in yaps.operators.import_distribution
    parser.add_argument("-I", type=float, default=0.1)         # interval of beta at which we want to measure observables
    parser.add_argument("-S", default='U1xU1xZ2')             # symmetry -- Z2xZ2 or U1xU1
    parser.add_argument("-M_UP", type=float, default=0.0)      # chemical potential up
    parser.add_argument("-M_DOWN", type=float, default=0.0)    # chemical potential down
    parser.add_argument("-U", type=float, default=0.)          # hubbard interaction
    parser.add_argument("-TUP", type=float, default=1.)        # hopping_up
    parser.add_argument("-TDOWN", type=float, default=1.)      # hopping_down
    parser.add_argument("-E", type=float, default=0.2)         # beta end
    parser.add_argument("-DBETA", type=float, default=0.005)   # dbeta
    parser.add_argument("-STEP", default='two-step')           # truncation can be done in 'one-step' or 'two-step'. note that truncations done 
                                                               # with svd update or when we fix the symmetry sectors are always 'one-step'
    parser.add_argument("-MODE", default='optimal')             # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    parser.add_argument("-FIXED", type=int, default=0)         # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    args = parser.parse_args()

    tt = time.time()
    benchmark_NTU_hubbard(lattice = args.L, boundary = args.B,  xx=args.x, yy=args.y, D=args.D, interval = args.I, sym=args.S, 
                                mu_up=args.M_UP, mu_dn=args.M_DOWN, U=args.U, t_up=args.TUP, t_dn=args.TDOWN, purification=args.p,
                                beta_end=args.E, dbeta=args.DBETA, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))


# to run, type in terminal : taskset -c 0-13 nohup python -u NTU_spinfull_Hubbard.py -L 'rectangle' -B 'finite' -p 'False' -x 5 -y 5 -D 4 -I 0.1 -S 'U1xU1_ind' -M_UP 0 -M_DOWN 0 -U 12 -TUP 1 -TDOWN 1 -E 20 -STEP 'two-step' -DBETA 0.005 -MODE 'optimal' -FIXED 0 > neel_initialized_spinfull_finite_5_5_MU_0_U_12_D_4.out &
# for implementing U1xU1_ind symmetry, bond dimension 16, chemical potential up and down 0, hopping rate of 1, hubbard interaction 0
# and if we want to evaluate expectation values at an interval 0.1 of beta