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
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg
from yast.tn.peps.CTM import nn_avg, nn_bond, one_site_avg, Local_CTM_Env, EVcorr_diagonal, measure_one_site_spin

def expectation_values_hole(lattice, boundary, purification, xx, yy, D, sym, chi, interval, beta_start, beta_end, UU, mu_up, mu_dn, t_up, t_dn, step, tr_mode, fix_bd):

    dims=(xx, yy)
    opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')

    n_up = fcdag_up @ fc_up 
    n_dn = fcdag_dn @ fc_dn
    n_int = n_up @ n_dn
    hole_density = (fid-n_up)@(fid-n_dn)
    
    x_target = round((xx-1)*0.5)
    y_target = round((yy-1)*0.5)
    target_site = (x_target, y_target)

    def exp_1s(A, ms, env1, op): 
        val_op = measure_one_site_spin(A, ms, env1, op=op)
        val_norm = measure_one_site_spin(A, ms, env1, op=None)
        return (val_op/val_norm)


    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, UU, mu_up, mu_dn, t_up, t_dn, sym)
    state = np.load("hole_initialized_real_time_evolution_Hubbard_model_spinfull_tensors_%s.npy" % (file_name), allow_pickle=True).item()
    num_ints = round((beta_end-beta_start)/interval)+1
    beta_range = np.linspace(beta_start, beta_end, num_ints)
    print(beta_range)

    imb = np.zeros((num_ints, 5))
    s=0
    mat = {}
    for beta in beta_range:

        print('BETA: ', beta)
        sv_beta = round(beta * yast.BETA_MULTIPLIER)
        tpeps = peps.Peps(lattice, dims, boundary)
        for sind in tpeps.sites():
            tpeps[sind] = yast.load_from_dict(config=fid.config, d=state.get((sind, sv_beta)))
            tpeps[sind] = tpeps[sind].unfuse_legs(axes=(0, 1))    

        dict_list_env = []
        for ms in tpeps.sites():
            dict_list_env.extend([('cortl', ms), ('cortr', ms), ('corbl', ms), ('corbr', ms), ('strt', ms), ('strb', ms), ('strl', ms), ('strr', ms)])

        state1 = np.load("ctm_environment_real_time_hole_beta_%1.3f_chi_%1.1f_%s.npy" % (beta, chi, file_name), allow_pickle=True).item()
        env = {ind: yast.load_from_dict(config=fid.config, d=state1[(*ind, sv_beta)]) for ind in dict_list_env}
        env1 = {}
        for ms in tpeps.sites():
            env1[ms] = Local_CTM_Env()
            env1[ms].tl = env['cortl',ms] 
            env1[ms].tr = env['cortr',ms] 
            env1[ms].bl = env['corbl',ms] 
            env1[ms].br = env['corbr',ms] 
            env1[ms].t = env['strt',ms] 
            env1[ms].l = env['strl',ms] 
            env1[ms].r = env['strr',ms] 
            env1[ms].b = env['strb',ms]

        print('hole_density')
        hole_density_avg, hole_density_central, mat_hole = one_site_avg(tpeps, net, env1, hole_density, central='yes')
     
        print('spin polarization up')
        up_avg, up_central, mat_up = one_site_avg(tpeps, net, env1, n_up, central='yes')

        print('spin polarization down')
        down_avg, down_central, mat_dn = one_site_avg(tpeps, net, env1, n_dn, central='yes')

        print('double occupancy')
        double_occupancy_avg, double_occupancy_central, mat_do = one_site_avg(tpeps, net, env1, n_int, central='yes')

       # print('diagonal correlations')
        
       # tpeps._data[target_site] = tpeps._data[target_site].unfuse_legs(axes=4).unfuse_legs(axes=5).fuse_legs(axes=(0, 1, 2, 3, 6, (4, 5))) # t l b r str [s a]
       # corrs_1 = EVcorr_diagonal(env1, ops, tpeps, x_l=x_target-1, y_l=y_target+1, x_r=x_target, y_r=y_target+2)   #(l: (2,1) --- r: (1,2))
       # corrs_2 = EVcorr_diagonal(env1, ops, tpeps, x_l=x_target-1, y_l=y_target, x_r=x_target, y_r=y_target+1)   #(l: (1,2) --- r: (2,3))
       # corrs_3 = EVcorr_diagonal(env1, ops, tpeps, x_l=x_target+1, y_l=y_target, x_r=x_target+1, y_r=y_target+1) #(l: (3,2) --- r: (2,3))
       # corrs_4 = EVcorr_diagonal(env1, ops, tpeps, x_l=x_target, y_l=y_target-1, x_r=x_target+1, y_r=y_target)   #(l: (2,1) --- r: (3,2))

        imb[s,0] = beta
        imb[s,1] = hole_density_central
        imb[s,2] = up_central
        imb[s,3] = down_central
        imb[s,4] = double_occupancy_central

       # print(mat_hole)
        print(mat_up)
       # print(mat_dn)
        #print(mat_do)

        x = {('hole', sv_beta): mat_hole, ('up', sv_beta):mat_up, ('dn', sv_beta):mat_dn, ('do', sv_beta):mat_do}
        mat.update(x)
        s = s+1

    np.savetxt("spinfull_chi_%1.1f_%s.txt" % (chi, file_name), imb, fmt='%.6f')
    np.save("lattice_view_hole_initialized_real_time_evolution_Hubbard_model_spinfull_tensors_%s.npy" % (file_name), mat)


if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle') # lattice shape
    parser.add_argument("-x", type=int, default=5)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=5)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='time') # bool
    parser.add_argument("-D", type=int, default=8) # bond dimension of peps tensors
    parser.add_argument("-S", default='U1xU1_ind') # symmetry -- 'Z2_spinless' or 'U1_spinless'
    parser.add_argument("-X", type=float, default=30) # chi_multiple
    parser.add_argument("-I", type=float, default=0.01) # interval
    parser.add_argument("-BETA_START", type=float, default=0.01) # location
    parser.add_argument("-BETA_END", type=float, default=0.04) # location
    parser.add_argument("-U", type=float, default=12.)       # location                                                                                             
    parser.add_argument("-MU_UP", type=float, default=0.)   # location                                                                                                 
    parser.add_argument("-MU_DOWN", type=float, default=0.) # location                                                                                           
    parser.add_argument("-T_UP", type=float, default=1.)    # location
    parser.add_argument("-T_DOWN", type=float, default=1.)  # location
    parser.add_argument("-STEP", default='two-step')        # location
    parser.add_argument("-MODE", default='optimal')         # location
    parser.add_argument("-FIXED", type=int, default=0)   
    args = parser.parse_args()

    tt = time.time()
    expectation_values_hole(lattice=args.L, boundary=args.B, xx=args.x, yy=args.y, D=args.D, sym=args.S,  chi=args.X, purification=args.p,
     interval=args.I, beta_start=args.BETA_START, beta_end=args.BETA_END, UU=args.U, mu_up = args.MU_UP, mu_dn = args.MU_DOWN, t_up = args.T_UP, t_dn = args.T_DOWN, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))

# to run, type in terminal : taskset -c 14-27 nohup python -u hole_evolution_time_tracker.py -L 'rectangle' -x 5 -y 5 -B 'finite' -p 'False' -D 8 -S 'U1xU1_ind' -X 30 -I 0.01 -BETA_START 0.01 -BETA_END 0.04 -U 12 -MU_UP 0 -MU_DOWN 0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > neel_initialized_vals_spinfull_5_5_D_12_gs_U_12_MU_0_T_1.out &
# for implementing U1 symmetry, ctm precision 1e-7,
# bond dimension 10 and if we want to store PEPS tensors at an interval 0.1 of beta for 0 chemical potential, a hopping rate of 1 and 1step truncation
