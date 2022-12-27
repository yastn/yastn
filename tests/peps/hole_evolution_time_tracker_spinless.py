# testing the value of nn correlators for spinfull fermi sea at beta=0.5
# for a unit cell of size (2x2). The results have been benchmarked with analytical values

import numpy as np
import logging
import argparse
import time
import yast
import yast.tn.peps as peps
from yast.tn.peps.CTM import Local_CTM_Env
try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg
from yast.tn.peps.CTM import nn_avg, nn_bond, one_site_avg, Local_CTM_Env, EVcorr_diagonal, measure_one_site_spin

def expectation_values_hole(lattice, boundary, purification, xx, yy, D, sym, chi, interval, beta_start, beta_end, mu, t, step, tr_mode, fix_bd):

    dims=(xx, yy)
    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    n = fcdag @ fc      
    
    x_target = round((net.Nx-1)*0.5)
    y_target = round((net.Ny-1)*0.5)
    target_site = (x_target, y_target)

    def exp_1s(A, ms, env1, op): 
        val_op = measure_one_site_spin(A, ms, env1, op=op)
        val_norm = measure_one_site_spin(A, ms, env1, op=None)
        return (val_op/val_norm)


    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_MU_%1.5f_T_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, mu, t, sym)
    state = np.load("hole_initialized_real_time_evolution_fermi_sea_spinless_tensors_%s.npy" % (file_name), allow_pickle=True).item()
 
    num_ints = round((beta_end-beta_start)/interval)+1
   # beta_range = np.linspace(beta_start, beta_end, num_ints)

    beta_range = np.array([0.099, 0.199, 0.299, 0.399, 0.499, 0.599, 0.699, 0.799, 
    0.899, 0.999, 1.099, 1.199, 1.299, 1.399, 1.499])
    print(beta_range)

    imb = np.zeros((num_ints, 5))
    s=0
    mat = {}
    for beta in beta_range:

        print('BETA: ', beta)
        sv_beta = round(beta * yast.BETA_MULTIPLIER)
        tpeps = peps.Peps(net.lattice, net.dims, net.boundary)
        tpeps._data = {sind: yast.load_from_dict(config=fid.config, d=state.get((sind, sv_beta))) for sind in tpeps.sites()}
        tpeps._data = {ms: tpeps._data[ms].unfuse_legs(axes=(0, 1)) for ms in tpeps.sites()}      

        dict_list_env = []
        for ms in net.sites():
            dict_list_env.extend([('cortl', ms), ('cortr', ms), ('corbl', ms), ('corbr', ms), ('strt', ms), ('strb', ms), ('strl', ms), ('strr', ms)])

        state1 = np.load("ctm_environment_real_time_hole_beta_%1.3f_chi_%1.1f_%s.npy" % (beta, chi, file_name), allow_pickle=True).item()
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

        print('particle density')
        density_avg, density_central, mat_den = one_site_avg(tpeps, env1, n, flag=None)

        imb[s,0] = beta
        imb[s,1] = density_central

       # print(mat_hole)
        print(mat_den)
       # print(mat_dn)
        #print(mat_do)

        x = {('density', sv_beta): mat_den}
        mat.update(x)
        s = s+1

    np.savetxt("spinless_chi_%1.1f_%s.txt" % (chi, file_name), imb, fmt='%.6f')
    np.save("lattice_view_hole_initialized_real_time_evolution_fermi_sea_spinless_tensors_%s.npy" % (file_name), mat)


if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle') # lattice shape
    parser.add_argument("-x", type=int, default=3)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=3)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='time') # bool
    parser.add_argument("-D", type=int, default=6) # bond dimension of peps tensors
    parser.add_argument("-S", default='U1') # symmetry -- 'Z2_spinless' or 'U1_spinless'
    parser.add_argument("-X", type=float, default=30) # chi_multiple
    parser.add_argument("-I", type=float, default=0.1) # interval
    parser.add_argument("-BETA_START", type=float, default=0.1) # location
    parser.add_argument("-BETA_END", type=float, default=1.5) # location
    parser.add_argument("-M", type=float, default=0.)   # location                                                                                                 
    parser.add_argument("-T", type=float, default=1.) # location                                                                                           
    parser.add_argument("-STEP", default='two-step')        # location
    parser.add_argument("-MODE", default='optimal')        # location
    parser.add_argument("-FIXED", type=int, default=0)   
    args = parser.parse_args()

    tt = time.time()
    expectation_values_hole(lattice=args.L, boundary=args.B, xx=args.x, yy=args.y, D=args.D, sym=args.S,  chi=args.X, purification=args.p,
     interval=args.I, beta_start=args.BETA_START, beta_end=args.BETA_END, mu = args.M, t = args.T, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))

# to run, type in terminal : taskset -c 14-27 nohup python -u hole_evolution_time_tracker.py -L 'rectangle' -x 5 -y 5 -B 'finite' -p 'False' -D 8 -S 'U1xU1_ind' -X 30 -I 0.01 -BETA_START 0.01 -BETA_END 0.04 -U 12 -MU_UP 0 -MU_DOWN 0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > neel_initialized_vals_spinfull_5_5_D_12_gs_U_12_MU_0_T_1.out &
# for implementing U1 symmetry, ctm precision 1e-7,
# bond dimension 10 and if we want to store PEPS tensors at an interval 0.1 of beta for 0 chemical potential, a hopping rate of 1 and 1step truncation
