# testing the value of nn correlators for spinfull fermi sea at beta=0.5
# for a unit cell of size (2x2). The results have been benchmarked with analytical values
import numpy as np
import logging
import argparse
import yast
import time
import yast
import yast.tn.peps as peps
import pytest
from yast.tn.peps.CTM import nn_avg, one_site_avg, Local_CTM_Env


try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

def Hubbard_GS(lattice, boundary, purification, xx, yy, D, sym, chi, interval, beta_end, beta_start, mu, t, step, tr_mode, fix_bd):

    dims=(xx, yy)
    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    n = fcdag @ fc

    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_MU_%1.5f_T_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, mu, t, sym)
    state = np.load("fully_filled_initialized_spinless_tensors_%s.npy" % (file_name), allow_pickle=True).item()
 
    ops = {'cdagc': {'l': fid, 'r': fid},
           'ccdag': {'l': fid, 'r': fid}}

    num_ints = round((beta_end-beta_start)/interval)+1
    beta_range = np.linspace(beta_start, beta_end, num_ints)
    print(beta_range)

    imb = np.zeros((num_ints, 9))
    s=0
    for beta in beta_range:

        print('BETA: ', beta)
        sv_beta = round(beta * yast.BETA_MULTIPLIER)
        tpeps = peps.Peps(net.lattice, net.dims, net.boundary)

        tpeps._data = {sind: yast.load_from_dict(config=fid.config, d=state.get((sind, sv_beta))) for sind in tpeps.sites()}
        dict_list_env = []
        for ms in net.sites():
            d1 = [('cortl', ms), ('cortr', ms), ('corbl', ms), ('corbr', ms), ('strt', ms), ('strb', ms), ('strl', ms), ('strr', ms)]
            dict_list_env.extend(d1)

        state1 = np.load("ctm_environment_beta_%1.1f_chi_%1.1f_%s.npy" % (beta, chi, file_name), allow_pickle=True).item()
        env = {ind: yast.load_from_dict(config=fid.config, d=state1[(*ind, sv_beta)]) for ind in dict_list_env}

        env1 = {}
        for ms in net.sites():
            env1[ms] = Local_CTM_Env()
            env1[ms].tl = env['cortl',ms] 
            env1[ms].tr = env['cortr',ms] 
            env1[ms].bl = env['corbl',ms] 
            env1[ms].br = env['corbr',ms] 
            env1[ms].t = env['strt',ms] 
            env1[ms].l = env['strl',ms] 
            env1[ms].r = env['strr',ms] 
            env1[ms].b = env['strb',ms] 

        print('density of electron')
        dens, mat = one_site_avg(tpeps, env1, n)

        obs_hor, obs_ver = nn_avg(tpeps, env1, ops)

        cdagc = 0.5*(abs(obs_hor.get('cdagc')) + abs(obs_ver.get('cdagc')))
        ccdag = 0.5*(abs(obs_hor.get('ccdag')) + abs(obs_ver.get('ccdag')))
        
        imb[s,0] = beta
        imb[s,3] = dens
        imb[s,4] = cdagc
        imb[s,5] = ccdag
        imb[s,8] = - (cdagc + ccdag)*(2*xx*yy-xx-yy)
        s = s+1

    np.savetxt("spinless_chi_%1.1f_%s.txt" % (chi, file_name), imb, fmt='%.6f')

if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle') # lattice shape
    parser.add_argument("-x", type=int, default=3)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=3)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='False') # bool
    parser.add_argument("-D", type=int, default=4) # bond dimension of peps tensors
    parser.add_argument("-S", default='U1') # symmetry -- 'Z2_spinless' or 'U1_spinless'
    parser.add_argument("-X", type=float, default=20) # chi_multiple
    parser.add_argument("-I", type=float, default=0.5) # interval
    parser.add_argument("-BETA_START", type=float, default=0.1) # location
    parser.add_argument("-BETA_END", type=float, default=0.2) # location
    parser.add_argument("-M", type=float, default=0.)   # location                                                                                                 
    parser.add_argument("-T", type=float, default=1.) # location                                                                                           
    parser.add_argument("-STEP", default='two-step')        # location
    parser.add_argument("-MODE", default='optimal')        # location
    parser.add_argument("-FIXED", type=int, default=0)   
    args = parser.parse_args()

    tt = time.time()
    Hubbard_GS(lattice=args.L, boundary=args.B, xx=args.x, yy=args.y, D=args.D, sym=args.S,  chi=args.X, purification=args.p,
     interval=args.I, beta_start=args.BETA_START, beta_end=args.BETA_END, mu = args.M, t = args.T, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))

# to run, type in terminal : taskset -c 14-27 nohup python -u observables_Hubbard_ground_state.py -L 'rectangle' -x 5 -y 5 -B 'finite' -p 'False' -D 4 -S 'U1xU1_ind' -X 20 -I 2 -BETA_START 10 -BETA_END 20 -U 12 -MU_UP 0 -MU_DOWN 0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > vals_spinfull_5_5_D_4_gs_U_12_MU_0_T_1.out &
# for implementing U1 symmetry, ctm precision 1e-7,
# bond dimension 10 and if we want to store PEPS tensors at an interval 0.1 of beta for 0 chemical potential, a hopping rate of 1 and 1step tru