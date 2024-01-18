# """ Test the expectation values of spinless fermions with analytical values of fermi sea for finite and infinite lattices """
# import numpy as np
# import pytest
# import logging
# import yastn
# import yastn.tn.fpeps as fpeps
# from yastn.tn.fpeps.gates._gates import gates_hopping, gate_local_fermi_sea
# from yastn.tn.fpeps.evolution import evolution_step_, gates_homogeneous
# from yastn.tn.fpeps import initialize_peps_purification, MpsEnv
# from yastn.tn.fpeps.ctm import nn_exp_dict, ctmrg, EV2ptcorr
# import yastn.tn.mps as mps
# try:
#     from .configs import config_U1_R_fermionic as cfg
#     # cfg is used by pytest to inject different backends and divices
# except ImportError:
#     from configs import config_U1_R_fermionic as cfg


# def test_envs_spinless_finite():
#     lattice = 'square'
#     boundary = 'obc'
#     purification = 'True'
#     xx = 3
#     yy = 2
#     tot_sites = (xx * yy)
#     D = 6
#     chi = 30 # environmental bond dimension
#     mu = 0 # chemical potential
#     t = 1 # hopping amplitude
#     beta_end = 0.2
#     dbeta = 0.01
#     step = 'single-step'
#     tr_mode = 'optimal'

#     coeff = 0.25 # for purification; 0.5 for ground state calculation and 1j*0.5 for real-time evolution
#     trotter_step = coeff * dbeta

#     dims = (xx, yy)
#     net = fpeps.Lattice(lattice, dims, boundary)  # shape = (rows, columns)

#     opt = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
#     fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

#     GA_nn, GB_nn = gates_hopping(t, trotter_step, fid, fc, fcdag)  # nn gate for 2D fermi sea
#     g_loc = gate_local_fermi_sea(mu, trotter_step, fid, fc, fcdag) # local gate for spinless fermi sea
#     g_nn = [(GA_nn, GB_nn)]

#     if purification == 'True':
#         peps = initialize_peps_purification(fid, net) # initialized at infinite temperature

#     gates = gates_homogeneous(peps, g_nn, g_loc)
#     time_steps = round(beta_end / dbeta)

#     opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}
#     for nums in range(time_steps):

#         beta = (nums + 1) * dbeta
#         logging.info("beta = %0.3f" % beta)
#         peps, _ =  evolution_step_(peps, gates, step, tr_mode, env_type='NTU', opts_svd=opts_svd_ntu)

#     # convergence criteria for CTM based on total energy
#     chi = 40 # environmental bond dimension
#     tol = 1e-10 # truncation of singular values of ctm projectors
#     max_sweeps=50
#     tol_exp = 1e-12   # difference of some expectation value must be lower than tolerance

#     ops = {'cdagc': {'l': fcdag, 'r': fc},
#            'ccdag': {'l': fc, 'r': fcdag}}

#     cf_energy_old = 0

#     opts_svd_ctm = {'D_total': chi, 'tol': tol}

#     env = fpeps.ctm.init_ones(peps)
#     for step in ctmrg(peps, max_sweeps, iterator_step=2, AAb_mode=0, fix_signs=False, env=env, opts_svd=opts_svd_ctm):

#         assert step.sweeps % 2 == 0 # stop every 2nd step as iteration_step=2
#         obs_hor, obs_ver =  nn_exp_dict(peps, step.env, ops)

#         cdagc = (sum(abs(val) for val in obs_hor.get('cdagc').values()) + sum(abs(val) for val in obs_ver.get('cdagc').values()))
#         ccdag = (sum(abs(val) for val in obs_hor.get('ccdag').values()) + sum(abs(val) for val in obs_ver.get('ccdag').values()))

#         cf_energy = - (cdagc + ccdag) / tot_sites

#         print("energy: ", cf_energy)
#         if abs(cf_energy - cf_energy_old) < tol_exp:
#             break # here break if the relative differnece is below tolerance
#         cf_energy_old = cf_energy

#     mpsenv = MpsEnv(peps, opts_svd=opts_svd_ctm)
#     for ny in range(peps.Ny):
#         vR0 = step.env.env2mps(index=ny, index_type='r')
#         vR1 = mpsenv.env2mps(index=ny, index_type='r')
#         vL0 = step.env.env2mps(index=ny, index_type='l')
#         vL1 = mpsenv.env2mps(index=ny, index_type='l')

#         print(mps.vdot(vR0, vR1) / (vR0.norm() * vR1.norm()))  # problem with phase in peps?
#         print(mps.vdot(vL0, vL1) / (vL0.norm() * vL1.norm()))
#         assert abs(abs(mps.vdot(vR0, vR1)) / (vR0.norm() * vR1.norm()) - 1) < 1e-7
#         assert abs(abs(mps.vdot(vL0, vL1)) / (vL0.norm() * vL1.norm()) - 1) < 1e-7


# if __name__ == '__main__':
#     logging.basicConfig(level='INFO')
#     test_envs_spinless_finite()
