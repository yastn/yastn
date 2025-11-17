import numpy as np
# import pytest
import yastn
import yastn.tn.fpeps as fpeps
import logging
import argparse
import time
import os
from yastn.tn.fpeps._evolution import accumulated_truncation_error
from yastn.tn.fpeps._peps import Peps
from yastn.tn.fpeps._geometry import Site, Bond

def calculate_energy(env, fc_up, fcdag_up, fc_dn, fcdag_dn, Sm, Sp, Sz, n_up, n_dn):

    J = 1 / 2
    # print(env.psi.bonds())
    tot_sites = 2
    cdagc_up = env.measure_nn(fcdag_up, fc_up)  # calculate for all unique bonds
    cdagc_dn = env.measure_nn(fcdag_dn, fc_dn)  # -> {bond: value}
    ccdag_dn = env.measure_nn(fc_dn, fcdag_dn)
    ccdag_up = env.measure_nn(fc_up, fcdag_up)
    SmSp = env.measure_nn(Sm, Sp)
    SpSm = env.measure_nn(Sp, Sm)
    SzSz = env.measure_nn(Sz, Sz)
    nn = env.measure_nn(n_up + n_dn, n_up + n_dn)
    expt_n_up = env.measure_1site(n_up)
    expt_n_dn = env.measure_1site(n_dn)


    energy = -1 * (sum(cdagc_up.values()) - sum(ccdag_up.values()) + sum(cdagc_dn.values()) - sum(ccdag_dn.values())) + \
                J / 2 * (sum(SmSp.values()) + sum(SpSm.values())) + J * sum(SzSz.values()) - J / 4 * sum(nn.values()) - \
                2.0 * (sum(expt_n_up.values()) + sum(expt_n_dn.values()))
    energy = energy / tot_sites
    return energy.real

def test(initialization="EAT", opts_svd_evolution={"D_total": 6, 'tol_block': 1e-15}):

    NTUEnv="NN"
    beta_end = 0.1
    dbeta = 0.1
    coef = 0.25
    beta_start = 0
    sym="U1xU1xZ2"

    opt = yastn.operators.SpinfulFermions_tJ(sym = sym, backend="np", default_device="cpu", default_dype="complex128")
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn, n_up, n_dn, h, Sz, Sp, Sm = opt.I(), \
                                                                    opt.c(spin='u'),  opt.c(spin='d'), \
                                                                    opt.cp(spin='u'), opt.cp(spin='d'), \
                                                                    opt.n(spin='u'),  opt.n(spin='d'), \
                                                                    opt.h(), \
                                                                    opt.Sz(), opt.Sp(), opt.Sm()
    net = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(net, fid)

    g_tJ = fpeps.gates.gate_nn_tJ(1 / 2, 1, 1, 2.0 / 4, 2.0 / 4, 2.0 / 4, 2.0 / 4, dbeta * coef, fid, fc_up, fcdag_up, fc_dn, fcdag_dn)

    gates = []

    for bond in net.bonds():
        gates.append(g_tJ._replace(sites=bond))

    for bond in net.bonds()[::-1]:
        gates.append(g_tJ._replace(sites=bond))

    beta_start = 0
    num_of_step = round((-beta_start + beta_end) / dbeta)

    # env_evolution = fpeps.EnvNTU(psi, which=NTUEnv)
    # psi_config = psi.config
    opts_svd_evolution = opts_svd_evolution
    env_evolution = fpeps.EnvNTU(psi, which=NTUEnv)

    for step in range(num_of_step):
        beta = beta_start + (step + 1) * dbeta
        logging.info("beta = %0.3f" % beta)

        infos =  fpeps.evolution_step_(env_evolution, gates, opts_svd=opts_svd_evolution, max_iter=100, initialization=initialization)

        ntu_error_max = accumulated_truncation_error([infos], statistics="max")
        ntu_error_mean = accumulated_truncation_error([infos], statistics="mean")

        logging.info('ntu error max: %.2e' % ntu_error_max)
        logging.info('ntu error mean: %.2e' % ntu_error_mean)


    opts_svd_ctm = {'D_total': 24, 'tol': 1e-8, 'truncate_multiplets': True}
    corner_tol = 1e-5
    max_sweeps = 50

    env = fpeps.EnvCTM(psi, init="eye")

    for ctmrg_step in env.ctmrg_(max_sweeps=max_sweeps, iterator=True, opts_svd=opts_svd_ctm, method='2site', corner_tol=corner_tol):
        pass


    return calculate_energy(env, fc_up, fcdag_up, fc_dn, fcdag_dn, Sm, Sp, Sz, n_up, n_dn)


truncation_methods_dict = {"EAT":{"D_total": 6, 'tol_block': 1e-15},
                           "SVD":{"D_total": 6, 'tol_block': 1e-15},
                        #    "ZMT10":{"D_total": 6, 'tol_block': 1e-15},
                           "ZMT1svd":{"D_total": 6, 'tol_block': 1e-15, 'preD':12},
                           "ZMT1eat":{"D_total": 6, 'tol_block': 1e-15, 'preD':12},
                        #    "ZMT30":{"D_total": 6, 'tol_block': 1e-15},
                           "ZMT3svd":{"D_total": 6, 'tol_block': 1e-15, 'preD':12},
                           "ZMT3eat":{"D_total": 6, 'tol_block': 1e-15, 'preD':12},
                        #    "ZMT3zmt10":{"D_total": 6, 'tol_block': 1e-15, 'preD':12},
                           "ZMT3zmt1eat":{"D_total": 6, 'tol_block': 1e-15, 'preD':12},
                           "ZMT3zmt1svd":{"D_total": 6, 'tol_block': 1e-15, 'preD':12},
                           }

if __name__== '__main__':
    logging.basicConfig(level='INFO')
    for key in truncation_methods_dict.keys():
        assert(test(key, truncation_methods_dict[key]) + 1.6403 <1e-3)
        print(key, "pass")
