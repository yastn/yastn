# this script analyzes the saved peps tensor, calculates expectation values using CTM and error analysis
import numpy as np
import logging
import argparse
import time
import yast
import yast.tn.peps as peps
from yast.tn.peps.CTM import GetEnv, Local_CTM_Env

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg


def CtmEnv_Hubbard(lattice, boundary, purification, xx, yy, D, sym, chi, interval, beta_end, beta_start, UU, mu_up, mu_dn, t_up, t_dn, step, tr_mode, fix_bd):
    
    dims = (xx, yy)

    net = peps.Peps(lattice, dims, boundary)  # shape = (rows, columns)
    print(net.sites())
    opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')

    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_%s_%s_Ds_%s_U_%1.2f_MU_UP_%1.5f_MU_DN_%1.5f_T_UP_%1.2f_T_DN_%1.2f_%s" % (lattice, dims[1], dims[0], boundary, purification, fix_bd, tr_mode, step, D, UU, mu_up, mu_dn, t_up, t_dn, sym)
    state = np.load("neel_initialized_Hubbard_spinfull_tensors_%s.npy" % (file_name), allow_pickle=True).item()

    num_ints = round((beta_end-beta_start)/interval)+1
    beta_range = np.linspace(beta_start, beta_end, num_ints)
    print(beta_range)
    
    for beta in beta_range:
        
        sv_beta = round(beta * yast.BETA_MULTIPLIER)
        tpeps = peps.Peps(net.lattice, net.dims, net.boundary)

        tpeps._data = {sind: yast.load_from_dict(config=fid.config, d=state.get((sind, sv_beta))) for sind in net.sites()}
        print('BETA: ', beta)
        nbit = 10
        opts = {'chi': round(chi), 'cutoff': 1e-10, 'nbitmax': round(nbit), 'prec' : 1e-7, 'tcinit' : ((0,) * fid.config.sym.NSYM,), 'Dcinit' : (1,)}
        env = GetEnv(A=tpeps, net=net, **opts, AAb_mode=0)
        mdata={}                                                                                 
        for ms in net.sites():
            xm = {('cortl', ms, sv_beta): env[ms].tl.save_to_dict(), ('cortr', ms, sv_beta): env[ms].tr.save_to_dict(),
            ('corbl', ms, sv_beta): env[ms].bl.save_to_dict(), ('corbr', ms, sv_beta): env[ms].br.save_to_dict(),
            ('strt', ms, sv_beta): env[ms].t.save_to_dict(), ('strb', ms, sv_beta): env[ms].b.save_to_dict(),
            ('strl', ms, sv_beta): env[ms].l.save_to_dict(), ('strr', ms, sv_beta): env[ms].r.save_to_dict()}
            mdata.update(xm)
  
        with open("ctm_environment_beta_%1.1f_chi_%1.1f_%s.npy" % (beta, chi, file_name), 'wb') as f:
            np.save(f, mdata, allow_pickle=True)

if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle') # lattice shape
    parser.add_argument("-x", type=int, default=5)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=5)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='False') # bool
    parser.add_argument("-D", type=int, default=4) # bond dimension of peps tensors
    parser.add_argument("-S", default='U1xU1_ind') # symmetry -- 'Z2_spinless' or 'U1_spinless'
    parser.add_argument("-X", type=float, default=20) # chi_multiple
    parser.add_argument("-I", type=float, default=0.5) # interval
    parser.add_argument("-BETA_START", type=float, default=0.5) # location
    parser.add_argument("-BETA_END", type=float, default=2) # location
    parser.add_argument("-U", type=float, default=0.)       # location                                                                                             
    parser.add_argument("-MU_UP", type=float, default=0.)   # location                                                                                                 
    parser.add_argument("-MU_DOWN", type=float, default=0.) # location                                                                                           
    parser.add_argument("-T_UP", type=float, default=1.)    # location
    parser.add_argument("-T_DOWN", type=float, default=1.)  # location
    parser.add_argument("-STEP", default='two-step')        # location
    parser.add_argument("-MODE", default='optimal')        # location
    parser.add_argument("-FIXED", type=int, default=0)   
    args = parser.parse_args()

    tt = time.time()
    CtmEnv_Hubbard(lattice=args.L, boundary=args.B, xx=args.x, yy=args.y, D=args.D, sym=args.S,  chi=args.X, purification=args.p,
     interval=args.I, beta_start=args.BETA_START, beta_end=args.BETA_END, UU=args.U, mu_up = args.MU_UP, mu_dn = args.MU_DOWN, 
     t_up = args.T_UP, t_dn = args.T_DOWN, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))

# to run, type in terminal : taskset -c 14-27 nohup python -u generate_CtmEnv_Hubbard.py -L 'rectangle' -x 5 -y 5 -B 'finite' -p 'False' -D 4 -S 'U1xU1_ind' -X 20 -I 2 -BETA_START 10.0 -BETA_END 20.0 -U 12 -MU_UP 0 -MU_DOWN 0 -T_UP 1 -T_DOWN 1 -STEP 'two-step' -MODE 'optimal' -FIXED 0 > env_spinfull_5_5_gs_U_12_D_4_MU_0_T_1.out &
# for  ctm precision 1e-7,
# bond dimension 12 and if we want to store PEPS tensors at an interval 0.1 of beta for 0 chemical potential and a hopping rate of 1