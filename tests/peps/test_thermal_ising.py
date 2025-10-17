import numpy as np
# import pytest
import yastn
import yastn.tn.fpeps as fpeps
import logging
import argparse
import time
import os
# from yastn.tn.fpeps._evolution import truncate_
from yastn.tn.fpeps._peps import Peps
# from predisentangler import *
from itertools import pairwise
# from generate_initial_state import *

def Ising(D, chi, dbeta, ntu_environment):

    sym = "dense"
    net = fpeps.CheckerboardLattice()
    psi = Peps(net)

    opts_svd = {"D_total": D, 'tol': 1e-10}

    opt = yastn.operators.Spin12(sym=sym, backend="np", default_device="cpu")
    Id, Z, X = opt.I(), opt.z(), opt.x()
    psi = fpeps.product_peps(net, Id)

    dbeta = 0.01
    env_evolution = fpeps.EnvNTU(psi, which=ntu_environment)

    diff, num_of_iter = 0, 0

    for step in range(180):

        sum_err = 0
        ntu_iterations = 0

        for bond in net.bonds():

            gate = fpeps.gates.gate_nn_Ising(-1, dbeta / 4, Id, Z, bond=bond)
            infos = fpeps.evolution_step_(env_evolution, [gate], opts_svd=opts_svd, max_iter=1, initialization="EAT")
            # psi.apply_gate_(gate)
            # infos = []
            # for s0, s1 in pairwise(gate.sites):
            #     info = truncate_(env_evolution, opts_svd, (s0, s1), max_iter=100, initialization="EAT")
            #     infos.append(info)

            for info in infos:
                ntu_iterations += info.iterations['eat_opt']
            sum_err = sum_err + min(list(infos[0].truncation_errors.values()))

        for site in net.sites():
            gate = [fpeps.gates.gate_local_field(2.9,  dbeta / 4, Id, X, site=site), fpeps.gates.gate_local_field(5e-4, dbeta / 2, Id, Z, site=site), fpeps.gates.gate_local_field(2.9,  dbeta / 4, Id, X, site=site)]
            infos = fpeps.evolution_step_(env_evolution, gate, opts_svd=opts_svd, max_iter=1, initialization="EAT")
            # for gate in [fpeps.gates.gate_local_field(2.9,  dbeta / 4, Id, X, site=site), fpeps.gates.gate_local_field(5e-4, dbeta / 2, Id, Z, site=site), fpeps.gates.gate_local_field(2.9,  dbeta / 4, Id, X, site=site)]:
            #     psi.apply_gate_(gate)

        for bond in net.bonds()[::-1]:

            gate = fpeps.gates.gate_nn_Ising(-1, dbeta / 4, Id, Z, bond=bond)
            infos = fpeps.evolution_step_(env_evolution, [gate], opts_svd=opts_svd, max_iter=1, initialization="EAT")
            # psi.apply_gate_(gate)
            # infos = []
            # for s0, s1 in pairwise(gate.sites):
            #     info = truncate_(env_evolution, opts_svd, (s0, s1), max_iter=100, initialization="EAT")
            #     infos.append(info)

            for info in infos:
                ntu_iterations += info.iterations['eat_opt']

            sum_err = sum_err + min(list(infos[0].truncation_errors.values()))

        print('%.3f %.3e %.0f' % ((step + 1) * dbeta, sum_err / 4, ntu_iterations / 8))

    env = fpeps.EnvCTM(psi, init="eye")
    opts_svd_ctm = {'D_total': chi, 'tol': 1e-8, 'truncate_multiplets': True}
    for _ in env.ctmrg_(max_sweeps=50, iterator_step=1, opts_svd=opts_svd_ctm, method='2site', corner_tol=1e-5):
        pass
    print(env.measure_1site(Z))


if __name__== '__main__':
    logging.basicConfig(level='INFO')
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", type=int, default=6)
    parser.add_argument("-DBETA", type=float, default=0.01)
    parser.add_argument("-NTUEnvironment", default='NN+')
    parser.add_argument("-X", type=int, default=25)
    args = parser.parse_args()
    tt = time.time()

    Ising(D=args.D, chi=args.X, dbeta=args.DBETA, ntu_environment=args.NTUEnvironment)
    logging.info('Elapsed time: %0.2f s.', (time.time() - tt))
