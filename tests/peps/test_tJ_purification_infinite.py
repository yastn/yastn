import numpy as np
# import pytest
import yastn
import yastn.tn.fpeps as fpeps
import logging
import argparse
import time
# import os
from yastn.tn.fpeps._evolution import accumulated_truncation_error

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg

FIX_dimensions = {21:{(-1,-1,0):1, (-1,0,1):2, (-1, 1, 0):2, (0, -1, 1):2, (0, 0, 0):7, (0, 1, 1):2, (1, -1, 0):2, (1, 0, 1):2, (1, 1, 0):1},
                  19:{(-1,0,1):2, (-1, 1, 0):2, (0, -1, 1):2, (0, 0, 0):7, (0, 1, 1):2, (1, -1, 0):2, (1, 0, 1):2},
                  18:{(-1,0,1):2, (-1, 1, 0):2, (0, -1, 1):2, (0, 0, 0):6, (0, 1, 1):2, (1, -1, 0):2, (1, 0, 1):2},
                  17:{(-1,0,1):2, (-1, 1, 0):2, (0, -1, 1):2, (0, 0, 0):5, (0, 1, 1):2, (1, -1, 0):2, (1, 0, 1):2},
                  16:{(-1,0,1):2, (-1, 1, 0):2, (0, -1, 1):2, (0, 0, 0):4, (0, 1, 1):2, (1, -1, 0):2, (1, 0, 1):2},
                  13:{(-1,0,1):2, (-1, 1, 0):1, (0, -1, 1):2, (0, 0, 0):3, (0, 1, 1):2, (1, -1, 0):1, (1, 0, 1):2},
                  9: {(-1,0,1):1, (-1, 1, 0):1, (0, -1, 1):1, (0, 0, 0):3, (0, 1, 1):1, (1, -1, 0):1, (1, 0, 1):1},
                  8: {(-1,0,1):1, (-1, 1, 0):1, (0, -1, 1):1, (0, 0, 0):2, (0, 1, 1):1, (1, -1, 0):1, (1, 0, 1):1},
                  7: {(-1,0,1):1, (-1, 1, 0):1, (0, -1, 1):1, (0, 0, 0):1, (0, 1, 1):1, (1, -1, 0):1, (1, 0, 1):1},}

def NTU_tJ_Purification(load_file, chemical_potential, D, chi, sym, J, Jz, t, beta0, beta_target, dbeta, step, ntu_environment, verbose, step_per_save):

    coef = 0.25

    # dims = (xx, yy)
    # tot_sites = int(xx*yy)
    # net = fpeps.SquareLattice((2, 2), "infinite") # shape = (rows, columns)
    net = fpeps.CheckerboardLattice()
    opt = yastn.operators.SpinfulFermions_tJ(sym = sym, backend=cfg.backend, default_device=cfg.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn, n_up, n_dn, Sz, Sp, Sm = opt.I(), \
                                                                    opt.c(spin='u'),  opt.c(spin='d'), \
                                                                    opt.cp(spin='u'), opt.cp(spin='d'), \
                                                                    opt.n(spin='u'),  opt.n(spin='d'), \
                                                                    opt.Sz(), opt.Sp(), opt.Sm()

    psi = fpeps.product_peps(net, fid)

    if load_file == "None":
        pass
    else:
        load_data = np.load(load_file, allow_pickle=True).item()
        for site in psi.sites():
            try:
                psi[site] = yastn.load_from_dict(config=cfg, d=load_data[(site, 1)])
            except:
                psi[site] = yastn.load_from_dict(config=cfg, d=load_data[site])
        # print("File Loaded!")

    n = n_up + n_dn

    # projectors = [h, n_dn, n_up]

    # filling = {(0, 0): 1, (0, 1): 2,
    #             (1, 0): 0, (1, 1): 0,
    #             }

    # psi = initialize_diagonal_basis(projectors, net, filling)
    sweep = 0
    tJ_PEPS_tensors_directory = "../tJ3_PEPS_tensors/"

    g_hopping_up = fpeps.gates.gate_nn_hopping(t, dbeta * coef, fid, fc_up, fcdag_up)
    g_hopping_dn = fpeps.gates.gate_nn_hopping(t, dbeta * coef, fid, fc_dn, fcdag_dn)
    g_heisenberg = fpeps.gates.gates_Heisenberg_spinful(dbeta * coef, Jz, J, J, Sz, Sp, Sm, n, fid)
    g_loc        = fpeps.gates.gate_local_occupation(chemical_potential, dbeta * coef, fid, n)
    gates        = fpeps.gates.distribute(net, gates_nn=[g_hopping_up, g_hopping_dn, g_heisenberg], gates_local=g_loc)

    opts_svd_ctm = {'D_total': chi, 'tol': 1e-8, 'truncate_multiplets': True}
    # opts_svd_ctm = {'D_total': chi, 'tol': 1e-7}

    # mdata_ctm={}
    # data_mag = {}
    # data_density = {}
    # ns = {}
    num_steps = int(np.round((beta_target - beta0)/dbeta))
    if step == "one-step":
        opts_svd = {"D_total": D, 'tol_block': 1e-15}
    elif step == "FIX":
        opts_svd = {"D_total":D, "D_block": FIX_dimensions[D], 'tol_block': 1e-15}
    elif step == "two-step":
        opts_svd = [{"D_total": 2 * D, 'tol_block': 1e-17},
                    {"D_total": D, 'tol_block': 1e-15}]
    max_sweeps = 100

    if t != 0.0:
        file_name = "TEST_TJ_Purification_Inifinite_D" + str(D) + "_Sym" + sym + "_NTUEnvironment" + ntu_environment + \
                    "_dbeta" + str(dbeta) + "_beta0_" + str(float(beta0)) +"_beta" + str(float(beta_target)) + \
                    "_J" + str(J) + "_Jz" + str(Jz) + "_mu" + str(chemical_potential) + "_" + step
    else:
        file_name = "TEST_TJ_Purification_Inifinite_D" + str(D) + "_Sym" + sym + "_NTUEnvironment" + ntu_environment + \
                    "_dbeta" + str(dbeta) + "_beta0_" + str(float(beta0)) +"_beta" + str(float(beta_target)) + \
                    "_J" + str(J) + "inf_Jz" + str(Jz) + "inf_mu" + str(chemical_potential) + "_" + step

    if step == "one-step":
        directory = tJ_PEPS_tensors_directory + sym + "_" + ntu_environment + "/" + "D=" + str(D) + "/mu=" + str(chemical_potential) + "/"
        directory_verbose = tJ_PEPS_tensors_directory + "Verbose/" + sym + "_" + ntu_environment + "/" + "D=" + str(D) + "/mu=" + str(chemical_potential) + "/"
    elif step == "FIX":
        directory = tJ_PEPS_tensors_directory + sym + "_" + ntu_environment + "/" + "D=" + str(D) + "/mu=" + str(chemical_potential) + "/FIX/"
        directory_verbose = tJ_PEPS_tensors_directory + "Verbose/" + sym + "_" + ntu_environment + "/" + "D=" + str(D) + "/mu=" + str(chemical_potential) + "/FIX/"
    elif step == "two-step":
        directory = tJ_PEPS_tensors_directory + sym + "_" + ntu_environment + "/" + "D=" + str(D) + "/mu=" + str(chemical_potential) + "/two-step/"
        directory_verbose = tJ_PEPS_tensors_directory + "Verbose/" + sym + "_" + ntu_environment + "/" + "D=" + str(D) + "/mu=" + str(chemical_potential) + "/two-step/"

    if ntu_environment == "Full":
        env_evolution = fpeps.EnvNTU(psi, which='NN')
        if load_file != "None":
            try:
                file_name_split = load_file.split("/")
                ctm_dir = ''
                for ii in range(len(file_name_split) - 1):
                    if ii == 1:
                        ctm_dir = ctm_dir + file_name_split[ii] + "/CTM/"
                    else:
                        ctm_dir = ctm_dir + file_name_split[ii] + "/"
                ctm_dir = ctm_dir + "chi=" + str(chi) + "/2siteSYM/"
                to_be_loaded_ctm_file_name = file_name_split[-1][:-4] + "_CTM_2siteSYM.npy"
                # print(ctm_dir)
                # print(to_be_loaded_ctm_file_name)
                ctm_data = np.load(ctm_dir + to_be_loaded_ctm_file_name, allow_pickle=True).item()
                env_evolution_full_temp = fpeps.load_from_dict(config=cfg, d=ctm_data)
                env_evolution_full = fpeps.EnvCTM(psi, init="eye")
                env_evolution_full.initialize_ctm_with_old_ctm(psi, env_evolution_full_temp)
                # print("CTM file loaded!")
            except:
                # print("Cannot find CTMRG file!")
                env_evolution_full = fpeps.EnvCTM(psi, init="eye")
                ctm_dir = tJ_PEPS_tensors_directory + "CTM/" + sym + "_" + ntu_environment + "/" + "D=" + str(D) + "/mu=" + str(chemical_potential) + "/chi=" + str(chi) + "/2siteSYM/"
                # print(ctm_dir)
        else:
            ctm_dir = tJ_PEPS_tensors_directory + "CTM/" + sym + "_" + ntu_environment + "/" + "D=" + str(D) + "/mu=" + str(chemical_potential) + "/chi=" + str(chi) + "/2siteSYM/"
            # print(ctm_dir)
    else:
        env_evolution = fpeps.EnvNTU(psi, which=ntu_environment)

    # for itr in range(num_iter):
    if ntu_environment == 'Full':
        num_steps = num_steps + 1
    for num in range(num_steps):
        beta = (num+1)*dbeta + beta0
        logging.info("beta = %0.3f" % beta)
        # if ntu_environment == 'Full' and beta >= 0.014:
        if ntu_environment == 'Full':
            try:
                try: # Try calculated from loaded CTM environment
                    for ctm_step in env_evolution_full.ctmrg_(max_sweeps=max_sweeps, iterator_step=1, opts_svd=opts_svd_ctm, method='2site', corner_tol=1e-6):
                        pass
                    infos =  fpeps.evolution_step_(env_evolution_full, gates, opts_svd=opts_svd, max_iter=100)
                    sweep = ctm_step.sweeps
                    # print("Fast full update from previous step")
                except: # Initialize CTMRG using identity
                    env_evolution_full = fpeps.EnvCTM(psi, init='eye')
                    for _ in env_evolution_full.ctmrg_(max_sweeps=max_sweeps, iterator_step=1, opts_svd=opts_svd_ctm, method='2site', corner_tol=1e-6):
                        pass
                    infos =  fpeps.evolution_step_(env_evolution_full, gates, opts_svd=opts_svd, max_iter=100)
                    sweep = ctm_step.sweeps
                    # print("Fast full update from identity")

                # if num % step_per_save == 0:
                #     try:
                #         np.save(ctm_dir + "PEPS_tensor_%s_CTM_2siteSYM.npy" % (file_name + "_Step_" + str(num)), env_evolution_full.save_to_dict())
                #     except:
                #         os.makedirs(os.path.dirname(ctm_dir))
                #         np.save(ctm_dir + "PEPS_tensor_%s_CTM_2siteSYM.npy" % (file_name + "_Step_" + str(num)), env_evolution_full.save_to_dict())
            except: # If Full update cannot be done, then do NTU
                # print("Fast full update failed")
                infos =  fpeps.evolution_step_(env_evolution, gates, opts_svd=opts_svd, max_iter=100)
                # print("NTU update")

        else:
            infos =  fpeps.evolution_step_(env_evolution, gates, opts_svd=opts_svd, max_iter=100)
            # print("NTU update")

        # for ms in net.sites():
        #     logging.info("shape of peps tensor = " + str(ms) + ": " +str(psi[ms].get_shape()))
        #     xs = psi[ms].unfuse_legs((0, 1))
        #     for l in range(4):
        #         print(xs.get_leg_structure(axis=l))

        if step=='svd-update':
            continue

        ntu_error_max = accumulated_truncation_error([infos])
        ntu_error_mean = accumulated_truncation_error([infos], statistics="mean")
        logging.info('ntu error max: %.2e' % ntu_error_max)
        logging.info('ntu error mean: %.2e' % ntu_error_mean)

        # try:
        #     with open(directory + "NTU_error_%s.txt" % file_name, "a+") as f:
        #         f.write('{:.3f} {:.3e} {:.3e} {:.0f}\n'.format(beta, ntu_error_max, ntu_error_mean, sweep))
        # except:
        #     os.makedirs(os.path.dirname(directory))
        #     with open(directory + "NTU_error_%s.txt" % file_name, "a+") as f:
        #         f.write('{:.3f} {:.3e} {:.3e} {:.0f}\n'.format(beta, ntu_error_max, ntu_error_mean, sweep))

        # if verbose == "True":
        #     try:
        #         with open(directory_verbose + "Verbose_%s.txt" % file_name, "a+") as f:
        #             gate_count = 0
        #             f.write('{:.3f} {:.3e} {:.3e}\n'.format(beta, ntu_error_max, ntu_error_mean))
        #             for info in infos:
        #                 f.write("# Gate: " + str(gate_count) + "\n")
        #                 gate_count = gate_count + 1
        #                 f.write("Truncation error: " + str(info.truncation_error) + "\n")
        #                 f.write("Truncation error list: " + str(info.truncation_errors) + "\n")
        #                 f.write("Number of iterations: " + str(info.iterations) + "\n")
        #                 f.write("Non-hermitian part: " + str(info.nonhermitian_part) + "\n")
        #                 f.write("Min_eigenvalue / max_eigenvalue: " + str(info.min_eigenvalue) + "\n")
        #                 f.write("Fixed eigenvalues: " + str(info.wrong_eigenvalues) + "\n")
        #                 f.write("Pinv cutoff: " + str(info.pinv_cutoffs) + "\n")
        #                 f.write("EAT error: " + str(info.EAT_error) + "\n")
        #             site = psi.sites()[0]
        #             for leg in range(0, 4):
        #                 psi_unfused = psi[site].unfuse_legs(axes=(0, 1))
        #                 for it in range(len(psi_unfused.get_legs()[leg].t)):
        #                     f.write(str(psi_unfused.get_legs()[leg].t[it][0:2]) + ":" + str(psi_unfused.get_legs()[leg].D[it]) + ", ")
        #                 f.write("\n")
        #             f.write("\n")
        #     except:
        #         os.makedirs(os.path.dirname(directory_verbose))
        #         with open(directory_verbose + "Verbose_%s.txt" % file_name, "a+") as f:
        #             gate_count = 0
        #             f.write('{:.3f} {:.3e} {:.3e}\n'.format(beta, ntu_error_max, ntu_error_mean))
        #             for info in infos:
        #                 f.write("# Gate: " + str(gate_count) + "\n")
        #                 gate_count = gate_count + 1
        #                 f.write("Truncation error: " + str(info.truncation_error) + "\n")
        #                 f.write("Truncation error list: " + str(info.truncation_errors) + "\n")
        #                 f.write("Number of iterations: " + str(info.iterations) + "\n")
        #                 f.write("Non-hermitian part: " + str(info.nonhermitian_part) + "\n")
        #                 f.write("Min_eigenvalue / max_eigenvalue: " + str(info.min_eigenvalue) + "\n")
        #                 f.write("Fixed eigenvalues: " + str(info.wrong_eigenvalues) + "\n")
        #                 f.write("Pinv cutoff: " + str(info.pinv_cutoffs) + "\n")
        #                 f.write("EAT error: " + str(info.EAT_error) + "\n")
        #             site = psi.sites()[0]
        #             site = psi.sites()[0]
        #             for leg in range(0, 4):
        #                 psi_unfused = psi[site].unfuse_legs(axes=(0, 1))
        #                 for it in range(len(psi_unfused.get_legs()[leg].t)):
        #                     f.write(str(psi_unfused.get_legs()[leg].t[it][0:2]) + ":" + str(psi_unfused.get_legs()[leg].D[it]) + ", ")
        #                 f.write("\n")
        #             f.write("\n")
        # elif verbose == "NTUerror":
        #     try:
        #         with open(directory_verbose + "Verbose_%s.txt" % file_name, "a+") as f:
        #             f.write('{:.3f} {:.3e} {:.3e}\n'.format(beta, ntu_error_max, ntu_error_mean))
        #             for info in infos:
        #                 f.write(str(info.truncation_error) + " ")
        #             f.write("\n")
        #     except:
        #         os.makedirs(os.path.dirname(directory_verbose))
        #         with open(directory_verbose + "Verbose_%s.txt" % file_name, "a+") as f:
        #             f.write('{:.3f} {:.3e} {:.3e}\n'.format(beta, ntu_error_max, ntu_error_mean))
        #             for info in infos:
        #                 f.write(str(info.truncation_error) + " ")
        #             f.write("\n")


        # if ((num  + 1) % step_per_save == 0):

        #     # save the tensor at target beta
        #     mdata = {}
        #     # x = {(ms, itr+1): psi[ms].save_to_dict() for ms in psi.sites()}
        #     x = {(ms, 1): psi[ms].save_to_dict() for ms in psi.sites()}
        #     # print(x)
        #     mdata.update(x)
        #     np.save(directory + "PEPS_tensor_%s.npy" % (file_name + "_Step_" + str(num + 1)), mdata)



if __name__== '__main__':
    logging.basicConfig(level='INFO')

    file_name = "None"
    # file_name = "PEPS_tensor_TEST_TJ_Purification_D9_chi40_10x10_dbeta0.01_beta0_2.0_beta3.0_J0.5_Jz0.5_Step_50.npy"


    parser = argparse.ArgumentParser()
    parser.add_argument("-LoadFile", default=file_name)
    parser.add_argument("-D", type=int, default=21)            # bond dimension or distribution of virtual legs of peps tensors,  input can be either an
                                                               # integer representing total bond dimension or a string labeling a sector-wise
                                                               # distribution of bond dimensions in yastn.fpeps.operators.import_distribution
    parser.add_argument("-S", default='U1xU1xZ2')             # symmetry -- Z2 or U1xU1 or U1xU1xZ2
    parser.add_argument("-MU", type=float, default=3.0)      # chemical potential
    parser.add_argument("-J", type=float, default=0.5)          # Heisenberg SpSm + h.c. interaction
    parser.add_argument("-Jz", type=float, default=0.5)          # Heisenberg SzSz interaction
    parser.add_argument("-T", type=float, default=1.0)      # hopping
    parser.add_argument("-BT0", type=float, default = 0.0) # Beginning of the target inverse temperature beta
    parser.add_argument("-BT", type=float, default=1.0)        # target inverse temperature beta
    parser.add_argument("-DBETA", type=float, default=0.005)      # dbeta
    parser.add_argument("-STEP", default='one-step')           # truncation can be done in 'one-step' or 'two-step'. note that truncations done
                                                               # with svd update or when we fix the symmetry sectors are always 'one-step'
    parser.add_argument("-NTUEnvironment", default='NN') # 'NN', 'NN+' 'NN++', 'NNN'
    parser.add_argument("-Verbose", default='False') # Verbose mode
    parser.add_argument("-StepPerSave", type=int, default=50) # Verbose mode
    parser.add_argument("-X", type=int, default=48) # Verbose mode
    args = parser.parse_args()

    tt = time.time()

    NTU_tJ_Purification(load_file = args.LoadFile, chemical_potential=args.MU, D=args.D, chi=args.X, sym=args.S, dbeta=args.DBETA,
                                beta0 = args.BT0, beta_target=args.BT, J=args.J, Jz=args.Jz, t=args.T,
                                step=args.STEP, ntu_environment=args.NTUEnvironment, verbose = args.Verbose, step_per_save=args.StepPerSave)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))


#python tJ_purification_infinite.py -LoadFile "PEPS_tensor_TEST_TJ_Purification_Inifinite_D16_SymU1xU1xZ2_NTUEnvironmentNN+_dbeta0.005_beta0_0.0_beta10.0_J0.5_Jz0.5_mu1.8_one-step_Step_1700.npy" -D 16 -BT0 8.5 -BT 9.0 -MU 1.8 -NTUEnvironment NN+ -STEP "one-step" -Verbose "True" -StepPerSave 20
