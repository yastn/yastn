import numpy as np
import logging
import argparse
import yast
import yast.tn.peps as peps
import time
from yast.tn.peps.operators.gates import gates_hopping, gate_local_fermi_sea
from yast.tn.peps.NTU import ntu_update, initialize_peps_purification, initialize_spinless_filled
try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg

def benchmark_NTU_hubbard(lattice, boundary, purification, xx, yy, D, interval, sym, mu, t, beta_end, dbeta, step, tr_mode, fix_bd):

    dims = (xx, yy)
    net = peps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()
    ancilla = 'True'
    GA_nn, GB_nn = gates_hopping(t, dbeta, fid, fc, fcdag, ancilla=ancilla, purification=purification)  # nn gate for 2D fermi sea
    G_loc = gate_local_fermi_sea(mu, dbeta, fid, fc, fcdag, ancilla=ancilla, purification=purification) # local gate for spinless fermi sea
    Gate = {'loc': G_loc, 'nn':{'GA': GA_nn, 'GB': GB_nn}}

    if purification == 'True':
        Gamma = initialize_peps_purification(fid, net) # initialized at infinite temperature
        print('yes')
    elif purification == 'False':
        Gamma = initialize_spinless_filled(fid, fc, fcdag, net) # initialized at infinite temperature
        print('no')

    time_steps = round(beta_end / dbeta)
    obs_int = (interval / beta_end) * time_steps
    
    mdata = {}
    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_MU_%1.5f_T_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, mu, t, sym)

    for nums in range(time_steps):
        beta = (nums + 1) * dbeta
        sv_beta = int(beta * yast.BETA_MULTIPLIER)
        logging.info("beta = %0.3f" % beta)
         
        Gamma, info =  ntu_update(Gamma, Gate, D, step, tr_mode, fix_bd=0) # fix_bd = 0 refers to unfixed symmetry sectors
        print(info)

        for ms in net.sites():
            logging.info("shape of peps tensor = " + str(ms) + ": " +str(Gamma[ms].get_shape()))
            xs = Gamma[ms].unfuse_legs((0, 1))
            for l in range(4):
                print(xs.get_leg_structure(axis=l))

        if (nums + 1) % int(obs_int) == 0:
            x = {(ms, sv_beta): Gamma._data[ms].save_to_dict() for ms in Gamma._data.keys()}
            mdata.update(x)
            np.save("fully_filled_initialized_spinless_tensors_%s.npy" % (file_name), mdata)
        if step=='svd-update':
            continue

        ntu_error = np.mean(np.sqrt(info['ntu_error']))
        logging.info('ntu error up: %.2e' % ntu_error)

        svd_error = np.mean(np.sqrt(info['svd_error']))
        logging.info('ntu error up: %.2e' % svd_error)

        with open("NTU_error_%s.txt" % file_name, "a+") as f:
            f.write('{:.3f} {:.2e}\n'.format(beta, ntu_error))
        with open("SVD_error_%s.txt" % file_name, "a+") as f:
            f.write('{:.3f} {:.2e}\n'.format(beta, svd_error))
        

if __name__== '__main__':
    logging.basicConfig(level='INFO')
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle')     # lattice shape
    parser.add_argument("-x", type=int, default=3)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=3)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='False') # bool
    parser.add_argument("-D", type=int, default=4)             
    parser.add_argument("-I", type=float, default=0.1)         # interval of beta at which we want to measure observables
    parser.add_argument("-S", default='U1')             # symmetry -- Z2xZ2 or U1xU1
    parser.add_argument("-M", type=float, default=0.0)      # chemical potential up
    parser.add_argument("-T", type=float, default=1.)        # hopping_up
    parser.add_argument("-E", type=float, default=10)         # beta end
    parser.add_argument("-DBETA", type=float, default=0.005)   # dbeta
    parser.add_argument("-STEP", default='two-step')           # truncation can be done in 'one-step' or 'two-step'. note that truncations done 
    parser.add_argument("-MODE", default='optimal')             # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    parser.add_argument("-FIXED", type=int, default=0)         # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    args = parser.parse_args()

    tt = time.time()
    benchmark_NTU_hubbard(lattice = args.L, boundary = args.B,  xx=args.x, yy=args.y, D=args.D, interval = args.I, sym=args.S, 
                                mu=args.M, t=args.T, purification=args.p, beta_end=args.E, dbeta=args.DBETA, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))


# to run, type in terminal : taskset -c 0-13 nohup python -u NTU_spinless_fermi_sea.py -L 'rectangle' -B 'finite' -p 'False' -x 3 -y 3 -D 4 -I 0.1 -S 'U1' -M 0 -T 1 -E 20 -STEP 'two-step' -DBETA 0.005 -MODE 'optimal' -FIXED 0 > fully_filled_initialized_spinless_finite_3_3_MU_0_D_4.out &
# for implementing U1xU1_ind symmetry, bond dimension 16, chemical potential up and down 0, hopping rate of 1, hubbard interaction 0
# and if we want to evaluate expectation values at an interval 0.1 of beta