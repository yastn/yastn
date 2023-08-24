import numpy as np
import logging
import argparse
import yastn
import yastn.tn.fpeps as peps
import time
from yastn.tn.fpeps.operators.gates import gates_hopping, gate_local_Hubbard
from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
from yastn.tn.fpeps import initialize_post_sampling_spinful_sz_basis, initialize_n_0p875_pattern_1, initialize_Neel_spinful
from yastn.tn.fpeps.ctm import sample, nn_avg, ctmrg, one_site_avg

try:
    from .configs import config_Z2_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and devices
except ImportError:
    from configs import config_Z2_R_fermionic as cfg

def NTU_hubbard_HF_METTS(lattice, boundary, purification, xx, yy, D, sym, mu_up, mu_dn, U, t_up, t_dn, beta_target, dbeta, chi, num_iter, step, tr_mode, fix_bd):

    dims = (xx, yy)
    net = peps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)
    opt = yastn.operators.SpinfulFermions(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    n_up = fcdag_up @ fc_up
    n_dn = fcdag_dn @ fc_dn
    n_int = n_up @ n_dn
    tot_sites = xx * yy

    GA_nn_up, GB_nn_up = gates_hopping(t_up, dbeta, fid, fc_up, fcdag_up, purification=purification)
    GA_nn_dn, GB_nn_dn = gates_hopping(t_dn, dbeta, fid, fc_dn, fcdag_dn, purification=purification)
    g_loc = gate_local_Hubbard(mu_up, mu_dn, U, dbeta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, purification=purification)
    g_nn = [(GA_nn_up, GB_nn_up), (GA_nn_dn, GB_nn_dn)]
    mdata = {}
    ns = {}
    num_steps = int(np.round(beta_target/dbeta))
    opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}
    
    file_name = "shape_%s_Nx_%1.0f_Ny_%1.0f_boundary_%s_purification_%s_fixed_bd_%1.1f_target_beta_%1.3f_%s_%s_Ds_%s_U_%1.2f_%s" % (lattice, dims[0], dims[1], boundary, purification, fix_bd, beta_target, tr_mode, step, D, U, sym)

    psi = initialize_n_0p875_pattern_1(fc_up, fc_dn, fcdag_up, fcdag_dn, net) # this initialization enforces a filling of n=0.875
   
    gates = gates_homogeneous(psi, g_nn, g_loc)
    mdata_ctm={}                                                                                 


    ###################################################
    ########    imaginary time evolution part     #####
    ###################################################

    for itr in range(num_iter):
        for num in range(num_steps):
            beta = (num+1)*dbeta
            logging.info("beta = %0.3f" % beta)
            psi, info =  evolution_step_(psi, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu) 
            print(info)
            for ms in net.sites():
                logging.info("shape of peps tensor = " + str(ms) + ": " +str(psi[ms].get_shape()))
                xs = psi[ms].unfuse_legs((0, 1))
                for l in range(4):
                    print(xs.get_leg_structure(axis=l))

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
                f.write('{:.3f} {:.3e} {:.3e} \n'.format(beta, svd_error_up, svd_error_dn))

        # save the tensor at target beta
        x = {(ms, itr+1): psi[ms].save_to_dict() for ms in psi.sites()}
        mdata.update(x)
        np.save("sz_basis_METTS_Hubbard_hf_tensors_target_beta_%1.1f_%s.npy" % (beta, file_name), mdata)

        # calculate observables with ctm 

        tol = 1e-10 # truncation of singular values of CTM projectors
        max_sweeps=100  # ctm param
        tol_exp = 1e-6
        opts_svd_ctm = {'D_total': chi, 'tol': tol}

        tot_energy_old = 0

        ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

        for step in ctmrg(psi, max_sweeps, iterator_step=3, AAb_mode=0, opts_svd=opts_svd_ctm):

            assert step.sweeps % 3 == 0 # stop every 3rd step as iteration_step=3
            
            obs_hor_mean, obs_ver_mean, obs_hor_sum, obs_ver_sum =  nn_avg(psi, step.env, ops)
            cdagc_up = obs_hor_sum.get('cdagc_up') + obs_ver_sum.get('cdagc_up')
            ccdag_up = - obs_hor_sum.get('ccdag_up') - obs_ver_sum.get('ccdag_up')
            cdagc_dn = obs_hor_sum.get('cdagc_dn') + obs_ver_sum.get('cdagc_dn')
            ccdag_dn = - obs_hor_sum.get('ccdag_dn') - obs_ver_sum.get('ccdag_dn')
            
            mean_int, _ = one_site_avg(psi, step.env, n_int) # first entry of the function gives average of one-site observables of the sites
            tot_energy =  U * mean_int - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn)/tot_sites 

            print("Energy : ", tot_energy)
            if abs(tot_energy - tot_energy_old) < tol_exp:
                break # here break if the relative differnece is below tolerance
            tot_energy_old = tot_energy

        # save the CTMRG environmental tensors
        for ms in psi.sites():
            xm = {('cortl', ms, itr+1): step.env[ms].tl.save_to_dict(), ('cortr', ms, itr+1): step.env[ms].tr.save_to_dict(),
            ('corbl', ms, itr+1): step.env[ms].bl.save_to_dict(), ('corbr', ms, itr+1): step.env[ms].br.save_to_dict(),
            ('strt', ms, itr+1): step.env[ms].t.save_to_dict(), ('strb', ms, itr+1): step.env[ms].b.save_to_dict(),
            ('strl', ms, itr+1): step.env[ms].l.save_to_dict(), ('strr', ms, itr+1): step.env[ms].r.save_to_dict()}
            mdata_ctm.update(xm)
  
        np.save("sz_basis_ctm_envs_METTS_initialize_ctm_environment_%s.npy" % (file_name), mdata_ctm)

        # calculation of energy of the central part of the lattice for even x even sized lattices
        mean_density, _ = one_site_avg(psi, step.env, (n_up+n_dn))
        print('density ', mean_density)

        with open("sz_basis_observables_from_METTS_Hubbard_hf_target_beta_%1.1f_%s.txt" % (beta, file_name), "a+") as f:
                f.write('{:.1f} {:.5f} {:.5f} {:.5f}\n'.format(itr+1, tot_energy, mean_int, mean_density))

        n_up = fcdag_up @ fc_up 
        n_dn = fcdag_dn @ fc_dn 
        h_up = fc_up @ fcdag_up 
        h_dn = fc_dn @ fcdag_dn 

        nn_up, nn_dn, nn_do, nn_hole = n_up @ h_dn, n_dn @ h_up, n_up @ n_dn, h_up @ h_dn
        projectors = [nn_up, nn_dn, nn_do, nn_hole]
        out = sample(psi, step.env, projectors)
        print(out)

        ns.update({itr+1:out})

        np.save("samples_generated_from_%s.npy" % file_name, ns)

        psi = initialize_post_sampling_spinful_sz_basis(fc_up, fc_dn, fcdag_up, fcdag_dn, net, out)


if __name__== '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("-L", default='rectangle')     # lattice shape
    parser.add_argument("-x", type=int, default=4)   # lattice dimension in x-dirn
    parser.add_argument("-y", type=int, default=4)   # lattice dimension in y-dirn
    parser.add_argument("-B", type=str, default='finite') # boundary
    parser.add_argument("-p", type=str, default='True') # purifciation can be 'True' or 'False' or 'Time'
    parser.add_argument("-D", type=int, default=4)            # bond dimension or distribution of virtual legs of peps tensors,  input can be either an 
                                                               # integer representing total bond dimension or a string labeling a sector-wise 
                                                               # distribution of bond dimensions in yastn.fpeps.operators.import_distribution
    parser.add_argument("-S", default='Z2')             # symmetry -- Z2xZ2 or U1xU1
    parser.add_argument("-M_UP", type=float, default=0.0)      # chemical potential up
    parser.add_argument("-M_DOWN", type=float, default=0.0)    # chemical potential down
    parser.add_argument("-U", type=float, default=8.)          # hubbard interaction
    parser.add_argument("-TUP", type=float, default=1.)        # hopping_up
    parser.add_argument("-TDOWN", type=float, default=1.)      # hopping_down
    parser.add_argument("-BT", type=float, default=4)        # target inverse temperature beta
    parser.add_argument("-DBETA", type=float, default=0.2)      # dbeta
    parser.add_argument("-X", type=int, default=20)        # chi --- environmental bond dimension for CTM
    parser.add_argument("-ITER", type=int, default=1)        # chi --- environmental bond dimension for CTM
    parser.add_argument("-STEP", default='one-step')           # truncation can be done in 'one-step' or 'two-step'. note that truncations done 
                                                               # with svd update or when we fix the symmetry sectors are always 'one-step'
    parser.add_argument("-MODE", default='optimal')             # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    parser.add_argument("-FIXED", type=int, default=0)         # truncation mode can be svd (without NTU), normal (NTU without EAT), optimal (NTU with EAT)
    args = parser.parse_args()

    tt = time.time()
    NTU_hubbard_HF_METTS(lattice = args.L, boundary = args.B,  xx=args.x, yy=args.y, D=args.D, sym=args.S, dbeta=args.DBETA,
                                mu_up=args.M_UP, mu_dn=args.M_DOWN, beta_target=args.BT, U=args.U, t_up=args.TUP, t_dn=args.TDOWN, num_iter=args.ITER, purification=args.p,
                                chi=args.X, step=args.STEP, tr_mode=args.MODE, fix_bd=args.FIXED)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))


# to run, type in terminal : taskset -c 7-14 nohup python -u NTU_Hubbard_hf_METTS_sz_basis.py -L 'rectangle' -B 'finite' -p 'True' -x 4 -y 4 -D 4 -X 20 -S 'Z2' -M_UP 0.0 -M_DOWN 0.0 -TUP 1.0 -TDOWN 1.0 -DBETA 0.05 -BT 1.0 -ITER 5000 -STEP 'one-step' -MODE 'optimal' -FIXED 0 > sz_basis_METTS_Hubbard_4_4_MU_0_T_1_D_4_beta_target_1.out &
