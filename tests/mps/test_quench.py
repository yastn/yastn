# """ mps.tdvp """
# import logging
# import pytest
# import yastn.tn.mps as mps
# try:
#     from .configs import config_dense as cfg
#    # cfg is used by pytest to inject different backends and divices
# except ImportError:
#     from configs import config_dense as cfg

# tol=1e-8


# def test_Ising_quench():
#     """
#     Initialize initial state in paramagnetic phase of transverse Ising model, and quench it to zero transverse field.
#     """
#     N = 7
#     D_total = 4
#     dt = -.25
#     sweeps = 10
#     opts_svd = {'tol': 1e-6, 'D_total': D_total}

#     gi, gf = 1, 0
#     Ji, Jf = 0, 1

#     def drive(s, gi, gf, Ji, Jf):
#         # s in [0, 1]
#         g = gi * (1 - s) + gf * s
#         J = Ji * (1 - s) + Jf * s
#         return J, g

#     total_time, dt = 4, 0.1

#     time = 0
#     while time < total_time + 1e-8:

    


#     Eng_gs = {0: -2.232050807568877, 1: -1.982050807568877}
#     H = ops_Z2.mpo_nn_hopping(N=N, t=1, mu=0.25)

#     for parity in (0, 1):
#         for version in ('1site', '2site'):
#             psi = ops_Z2.mps_random(N=N, Dblock=D_total/2, total_parity=parity)
#             psi.canonize_(to='first')
#             run_tdvp_imag(psi, H, dt=dt, Eng_gs=Eng_gs[parity], sweeps=sweeps, version=version, opts_svd=opts_svd)



# if __name__ == "__main__":
#     logging.basicConfig(level='INFO')
#     test_Ising_quench()
