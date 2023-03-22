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
from yast.tn.peps.ctm import ctmrg, nn_avg, one_site_avg

try:
    from ..configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

def Hubbard_GS(lattice, boundary, purification, xx, yy, D, sym, chi, U, mu_up, mu_dn, t_up, t_dn, step, tr_mode, fix_bd):

    dims=(xx, yy)
    opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    n_up = fcdag_up @ fc_up
    n_dn = fcdag_dn @ fc_dn
    n_int = n_up @ n_dn
    h = (fid - n_up) @ (fid - n_dn)

    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, U, mu_up, mu_dn, t_up, t_dn, sym)
    state = np.load("neel_initialized_Hubbard_spinfull_tensors_%s.npy" % (file_name), allow_pickle=True).item()
    
    beta = 5.0    
  #  for beta in beta_range:
    
    sv_beta = round(beta * yast.BETA_MULTIPLIER)
    
    psi = peps.Peps(lattice, dims, boundary)
    for sind in psi.sites():
        psi[sind] =  yast.load_from_dict(config=fid.config, d=state.get((sind, sv_beta))) 
    print('BETA: ', beta)
    nbit = 20
    
    cf_energy_old = 0
    tol_exp = 1e-6
    tol=1e-10 # cutoff for singular values from CTM projectors
    opts_svd = {'D_total': chi, 'tol': tol}

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

    max_sweeps= 5

    start_time= time.time()

    for step in ctmrg(psi, max_sweeps, iterator_step=1, AAb_mode=0, opts_svd=opts_svd):
        assert step.sweeps % 1 == 0 # stop every 2nd step as iteration_step=2

        doc, _, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites

        obs_hor, obs_ver =  nn_avg(psi, step.env, ops)

        cdagc_up = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))
        ccdag_up = 0.5*(abs(obs_hor.get('ccdag_up')) + abs(obs_ver.get('ccdag_up')))
        cdagc_dn = 0.5*(abs(obs_hor.get('cdagc_dn')) + abs(obs_ver.get('cdagc_dn')))
        ccdag_dn = 0.5*(abs(obs_hor.get('cdagc_up')) + abs(obs_ver.get('cdagc_up')))

        cf_energy = (xx * yy) * U * doc - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) * (2 * xx * yy - xx - yy)
        print("expectation value: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy
    
    end_time= time.time()
    print(f"CTM with parallelization execution time: {end_time - start_time:.2f} seconds")

    mdata={}                                                                                 
    for ms in psi.sites():
        xm = {('cortl', ms, sv_beta): step.env[ms].tl.save_to_dict(), ('cortr', ms, sv_beta): step.env[ms].tr.save_to_dict(),
        ('corbl', ms, sv_beta): step.env[ms].bl.save_to_dict(), ('corbr', ms, sv_beta): step.env[ms].br.save_to_dict(),
        ('strt', ms, sv_beta): step.env[ms].t.save_to_dict(), ('strb', ms, sv_beta): step.env[ms].b.save_to_dict(),
        ('strl', ms, sv_beta): step.env[ms].l.save_to_dict(), ('strr', ms, sv_beta): step.env[ms].r.save_to_dict()}
        mdata.update(xm)
  
    
    with open("ctm_environment_beta_%1.5f_chi_%1.1f_%s.npy" % (beta, chi, file_name), 'wb') as f:
            np.save(f, mdata, allow_pickle=True)


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
     U=args.U, mu_up = args.MU_UP, mu_dn = args.MU_DOWN, t_up = args.T_UP, t_dn = args.T_DOWN, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)

# to run, type in terminal : taskset -c 0-13 nohup python -u test_parallel_lattice.py -L 'rectangle' -x 7 -y 7 -B 'finite' -p 'False' -D 7 -S 'U1xU1xZ2' -X 35 -U 12 -MU_UP 0 -MU_DOWN 0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > vals_spinfull_7_7_D_7_gs_U_12_MU_0_T_1.out &
