# this script analyzes the saved peps tensor, calculates expectation values using CTM and error analysis
import numpy as np
import logging
import argparse
import yast
import time
import yast
import yast.tn.peps as peps
from yast.tn.peps.NTU import ntu_update, initialize_peps_purification, initialize_Neel_spinfull
from yast.tn.peps.operators.gates import gates_hopping, gate_local_fermi_sea
import pytest
from yast.tn.peps.CTM import nn_avg, nn_bond, one_site_avg, measure_one_site_spin, fPEPS_2layers, Local_CTM_Env
try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg


def evolution_hole_spinless(lattice, boundary, beta_gs, purification, xx, yy, D_gs, D_evol, sym, chi, interval, beta_end, dbeta, mu, t, step, tr_mode, fix_bd):
    
    print('BETA GROUND STATE: ',beta_gs)
    purification='False'
    dims=(xx, yy)
    peps_gs = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)c
    opt = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    n = fcdag @ fc 

    sv_beta = round(beta_gs * yast.BETA_MULTIPLIER)  # beta_gs is the temperature at which the ground state has converged
    dict_list_env = []
    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_MU_%1.5f_T_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D_gs, mu, t, sym)

    for ms in peps_gs.sites():
        dict_list_env.extend([('cortl', ms), ('cortr', ms), ('corbl', ms), ('corbr', ms), ('strt', ms), ('strb', ms), ('strl', ms), ('strr', ms)])
        
    state1 = np.load("ctm_environment_beta_%1.1f_chi_%1.1f_%s.npy" % (beta_gs, chi, file_name), allow_pickle=True).item()
    env = {ind: yast.load_from_dict(config=fid.config, d=state1[(*ind, sv_beta)]) for ind in dict_list_env}

    env1 = {}
    for ms in peps_gs.sites():
        env1[ms] = Local_CTM_Env()
        env1[ms].tl = env['cortl',ms] 
        env1[ms].tr = env['cortr',ms] 
        env1[ms].bl = env['corbl',ms] 
        env1[ms].br = env['corbr',ms] 
        env1[ms].t = env['strt',ms] 
        env1[ms].l = env['strl',ms] 
        env1[ms].r = env['strr',ms] 
        env1[ms].b = env['strb',ms] 

    state = np.load("fully_filled_initialized_spinless_tensors_%s.npy" % (file_name), allow_pickle=True).item()
    peps_gs._data = {sind: yast.load_from_dict(config=fid.config, d=state.get((sind, sv_beta))) for sind in peps_gs.sites()}
    peps_gs._data = {ms: peps_gs[ms].unfuse_legs(axes=(0, 1)) for ms in peps_gs.sites()}      

    def exp_1s(A, ms, env1, op): 
        val_op = measure_one_site_spin(A, ms, env1, op=op)
        val_norm = measure_one_site_spin(A, ms, env1, op=None)
        return (val_op/val_norm)

    target_site = (round((peps_gs.Nx-1)*0.5), round((peps_gs.Ny-1)*0.5))   # middle of an odd x odd lattice

    n_bf = exp_1s(peps_gs[target_site], target_site, env1, n)

    print('before n: ', n_bf)


    #################################################
    ##### operator to be applied to create hole #####
    #################################################

    print(peps_gs._data[target_site].get_shape())
    print(peps_gs._data[target_site].get_signature())

    hh = fc@fcdag
    f = hh
    Am = f.fuse_legs(axes=[(0, 1)])  # particle
    for s in (-1, 1, 1, -1):
        Am = Am.add_leg(axis=0, s=s)
    peps_gs[target_site] = Am

    print(peps_gs._data[target_site].get_shape())
    print(peps_gs._data[target_site].get_signature())
 
    n_af = exp_1s(peps_gs[target_site], target_site, env1, fid*1e-25+n) 
    print('after n: ', n_af)

    peps_gs._data = {ms: peps_gs._data[ms].fuse_legs(axes=((0, 1), (2, 3), 4)) for ms in peps_gs.sites()}

    ############################################################################################################
    ##################       time evolution after injecting hole at the center          ########################
    ############################################################################################################

    Gamma = peps_gs

    purification='Time'
    ancilla = 'True'
    GA_nn, GB_nn = gates_hopping(t, dbeta, fid, fc, fcdag, ancilla=ancilla, purification=purification)  # nn gate for 2D fermi sea
    G_loc = gate_local_fermi_sea(mu, dbeta, fid, fc, fcdag, ancilla=ancilla, purification=purification) # local gate for spinless fermi sea
    Gate = {'loc': G_loc, 'nn':{'GA': GA_nn, 'GB': GB_nn}}
    
    purification='time'
    time_steps = round(beta_end / dbeta)
    obs_int = (interval / beta_end) * time_steps

    mdata = {}
    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_MU_%1.5f_T_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D_evol, mu, t, sym)

    for nums in range(time_steps):
        beta = (nums + 1) * dbeta
        sv_beta = int((beta-dbeta) * yast.BETA_MULTIPLIER)
        logging.info("beta = %0.3f" % beta)
        if (nums + 1) % int(obs_int) == 0:
            x = {(ms, sv_beta): Gamma._data[ms].save_to_dict() for ms in Gamma._data.keys()}
            mdata.update(x)
            np.save("hole_initialized_real_time_evolution_fermi_sea_spinless_tensors_%s.npy" % (file_name), mdata)

        Gamma, info =  ntu_update(Gamma, Gate, D_evol, step, tr_mode, fix_bd) # fix_bd = 0 refers to unfixed symmetry sectors
        print(info)

        for ms in Gamma.sites():
            logging.info("shape of peps tensor = " + str(ms) + ": " +str(Gamma._data[ms].get_shape()))
            xs = Gamma._data[ms].unfuse_legs((0, 1))
            for l in range(4):
                print(xs.get_leg_structure(axis=l))

        
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
    parser.add_argument("-L", default='rectangle') # lattice shape
    parser.add_argument("-x", type=int, default=3)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=3)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='False') # bool
    parser.add_argument("-D_gs", type=int, default=4)
    parser.add_argument("-D_evol", type=int, default=6) 
    parser.add_argument("-BETA_GS", type=float, default=0.1) # beta of initial tensor corresponding to ground state
    parser.add_argument("-S", default='U1') # symmetry -- 'Z2_spinless' or 'U1_spinless'
    parser.add_argument("-X", type=float, default=20) # chi_multiple
    parser.add_argument("-I", type=float, default=0.01) # interval
    parser.add_argument("-D_BETA", type=float, default=0.001) # location
    parser.add_argument("-BETA_END", type=float, default=3) # location
    parser.add_argument("-M", type=float, default=0.)   # location                                                                                                 
    parser.add_argument("-T", type=float, default=1.)    # location
    parser.add_argument("-STEP", default='two-step')        # location
    parser.add_argument("-MODE", default='optimal')        # location
    parser.add_argument("-FIXED", type=int, default=0)   
    args = parser.parse_args()

    tt = time.time()
    evolution_hole_spinless(lattice = args.L, boundary = args.B, beta_gs=args.BETA_GS,  xx=args.x, yy=args.y,D_gs=args.D_gs, chi=args.X, D_evol=args.D_evol, interval = args.I, sym=args.S, 
                                mu=args.M, t=args.T, purification=args.p, beta_end=args.BETA_END, dbeta=args.D_BETA, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))

# to run, type in terminal : taskset -c 0-6 nohup python -u NTU_spinfull_Hubbard_hole_motion.py -L 'rectangle' -x 5 -y 5 -BETA_GS 20.0 -B 'finite' -p 'False' -D_gs 4 -X 20 -D_evol 12 -S 'U1xU1_ind' -I 0.001 -D_BETA 0.001 -BETA_END 3 -U 12 -MU_UP 0.0 -MU_DOWN 0.0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > hole_motion_spinfull_5_5_beta_gs_20_D_4_U_12_T_1.out &
# bond dimension 12 and if we want to store PEPS tensors at an interval 0.1 of beta for 0 chemical potential and a hopping rate of 1