# testing the value of nn correlators for spinfull fermi sea at beta=0.5
# for a unit cell of size (2x2). The results have been benchmarked with analytical values
import numpy as np
import logging
import argparse
import yast
import multiprocess as mp
import dill
import os
import time
import yast
import yast.tn.peps as peps
import pytest
from yast.tn.peps.ctm import Local_CTM_Env, measure_one_site_spin, check_consistency_tensors

try:
    from ..configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

def Hubbard_GS(lattice, boundary, purification, xx, yy, D, sym, chi, UU, mu_up, mu_dn, t_up, t_dn, step, tr_mode, fix_bd):

    dims=(xx, yy)
    opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    n_up = fcdag_up @ fc_up
    n_dn = fcdag_dn @ fc_dn
    n_int = n_up @ n_dn
    h = (fid - n_up) @ (fid - n_dn)

    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, UU, mu_up, mu_dn, t_up, t_dn, sym)
    state = np.load("neel_initialized_Hubbard_spinfull_tensors_%s.npy" % (file_name), allow_pickle=True).item()

    beta = 3.0
    sv_beta = beta*yast.BETA_MULTIPLIER
    tpeps = peps.Peps(lattice, dims, boundary)
    for sind in tpeps.sites():
        tpeps[sind] = yast.load_from_dict(config=fid.config, d=state.get((sind, sv_beta))) 
    dict_list_env = []

    for ms in tpeps.sites():
        d1 = [('cortl', ms), ('cortr', ms), ('corbl', ms), ('corbr', ms), ('strt', ms), ('strb', ms), ('strl', ms), ('strr', ms)]
        dict_list_env.extend(d1)

    state1 = np.load("ctm_environment_beta_%1.5f_chi_%1.1f_%s.npy" % (beta, chi, file_name), allow_pickle=True).item()
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

    tpeps = check_consistency_tensors(tpeps)

    def one_site_exp(Am, env):
       
        val_up = measure_one_site_spin(Am, env, op=n_up)
        val_dn = measure_one_site_spin(Am, env, op=n_dn)
        val_do = measure_one_site_spin(Am, env, op=n_int)
        val_hole = measure_one_site_spin(Am, env, op=h)
        val_norm = measure_one_site_spin(Am, env, op=None)

        val_up = val_up/val_norm   # expectation value of spin-up polarization
        val_dn = val_dn/val_norm   # expectation value of spin-up polarization
        val_do = val_do/val_norm   # expectation value of spin-up polarization
        val_hole = val_hole/val_norm   # expectation value of spin-up polarization

        """print('density of polarization up: ', val_up)
        print('density of polarization down: ', val_dn)
        print('density of double occupancy: ', val_do)
        print('density of hole: ', val_hole)"""
        return val_up, val_dn, val_hole, val_hole

    # Multiple processor execution
    num_processors = mp.cpu_count()
    print(f"Number of processors: {num_processors}")
    
    pool = mp.Pool(processes=num_processors)
    start_time= time.time()
    results = pool.map(lambda s: one_site_exp(tpeps[s], env1[s]), tpeps.sites())
    end_time = time.time()
    print(f"Multiple processor execution time: {end_time - start_time:.2f} seconds")

    # combine the results into separate lists for each expectation value
    vals_up, vals_dn, vals_do, vals_hole = zip(*results)

    # compute the average of each expectation value
    avg_up = sum(vals_up) / len(vals_up)
    avg_dn = sum(vals_dn) / len(vals_dn)
    avg_do = sum(vals_do) / len(vals_do)
    avg_hole = sum(vals_hole) / len(vals_hole)

    print(avg_up, avg_dn, avg_do, avg_hole)


    #np.savetxt("spinfull_chi_%1.1f_%s.txt" % (chi, file_name), mat_up, fmt='%.6f')

if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle') # lattice shape
    parser.add_argument("-x", type=int, default=3)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=3)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='False') # bool
    parser.add_argument("-D", type=int, default=6) # bond dimension of peps tensors
    parser.add_argument("-S", default='U1xU1xZ2') # symmetry -- 'Z2_spinless' or 'U1_spinless'
    parser.add_argument("-X", type=float, default=20) # chi_multiple
    parser.add_argument("-U", type=float, default=12.)       # location                                                                                             
    parser.add_argument("-MU_UP", type=float, default=0.)   # location                                                                                                 
    parser.add_argument("-MU_DOWN", type=float, default=0.) # location                                                                                           
    parser.add_argument("-T_UP", type=float, default=1.)    # location
    parser.add_argument("-T_DOWN", type=float, default=1.)  # location
    parser.add_argument("-STEP", default='two-step')        # location
    parser.add_argument("-MODE", default='optimal')         # location
    parser.add_argument("-FIXED", type=int, default=0)   
    args = parser.parse_args()
    Hubbard_GS(lattice=args.L, boundary=args.B, xx=args.x, yy=args.y, D=args.D, sym=args.S,  chi=args.X, purification=args.p,
     UU=args.U, mu_up = args.MU_UP, mu_dn = args.MU_DOWN, t_up = args.T_UP, t_dn = args.T_DOWN, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)

# to run, type in terminal : taskset -c 0-13 nohup python -u test_parallel_lattice.py -L 'rectangle' -x 7 -y 7 -B 'finite' -p 'False' -D 7 -S 'U1xU1xZ2' -X 35 -U 12 -MU_UP 0 -MU_DOWN 0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > vals_spinfull_7_7_D_7_gs_U_12_MU_0_T_1.out &
