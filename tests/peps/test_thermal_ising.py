import numpy as np
# import pytest
import yastn
import yastn.tn.fpeps as fpeps
import logging
import argparse
import time
from yastn.tn.fpeps._peps import Peps

def Ising(D, chi, dbeta, ntu_environment, method):

    sym = "dense"
    net = fpeps.CheckerboardLattice()
    psi = Peps(net)

    opts_svd = {"D_total": D, 'tol': 1e-10}

    opt = yastn.operators.Spin12(sym=sym, backend="np", default_device="cpu")
    Id, Z, X = opt.I(), opt.z(), opt.x()
    psi = fpeps.product_peps(net, Id)

    hx = 2.9
    hz = 5e-4
    beta = 1.8
    dbeta = 0.01
    steps = int(beta / dbeta)
    dbeta = beta / steps

    if 'BP' in ntu_environment:
        env_evolution = fpeps.EnvBP(psi, which=ntu_environment)
        env_evolution.iterate_(max_sweeps=100, diff_tol=1e-8)
    else:
        env_evolution = fpeps.EnvNTU(psi, which=ntu_environment)

    gateZZ = fpeps.gates.gate_nn_Ising(-1, dbeta / 4, Id, Z)
    gateX = fpeps.gates.gate_local_field(hx,  dbeta / 4, Id, X)
    gateZ = fpeps.gates.gate_local_field(hz, dbeta / 4, Id, Z)
    gates = fpeps.gates.distribute(net, gates_nn=[gateZZ], gates_local=[gateX, gateZ], symmetrize=False)
    gates = gates[::-1] + gates

    H = - fpeps.fkron(Z, Z) - hz * (fpeps.fkron(Z, Id) + fpeps.fkron(Id, Z)) / 4 - hx * (fpeps.fkron(X, Id) + fpeps.fkron(Id, X)) / 4
    gate = fpeps.gates.gate_nn_exp(dbeta / 4, Id, H)
    gates = fpeps.gates.distribute(net, gates_nn=[gate])

    infoss = []
    for step in range(steps):
        infos = fpeps.evolution_step_(env_evolution, gates, opts_svd=opts_svd, max_iter=1, initialization="EAT", method=method)
        if 'BP' in ntu_environment:
            env_evolution.iterate_(max_sweeps=100, diff_tol=1e-8)
        infoss.append(infos)
        Delta = fpeps.accumulated_truncation_error(infoss)
        #print(f"{step+1}  {Delta=:0.4e}")

    env = fpeps.EnvCTM(psi, init="eye")
    opts_svd_ctm = {'D_total': chi, 'tol': 1e-8, 'truncate_multiplets': True}
    for info in env.ctmrg_(max_sweeps=50, iterator_step=1, opts_svd=opts_svd_ctm, method='2site', corner_tol=1e-5):
        pass
        #print(info)
    print(f"{D=}, {chi=}, {dbeta}, {ntu_environment=} {method=}")
    print("<X> = ", env.measure_1site(X))
    print("<Z> = ", env.measure_1site(Z))
    # print("<ZZ> = ", env.measure_nn(Z, Z))
    print("Delta =", Delta)



if __name__== '__main__':
    # logging.basicConfig(level='INFO')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-D", type=int, default=5)
    # parser.add_argument("-DBETA", type=float, default=0.01)
    # parser.add_argument("-NTUEnvironment", default='NN+')
    # parser.add_argument("-X", type=int, default=20)
    # parser.add_argument("-method", type=str, default='mpo')

    # args = parser.parse_args()
    # tt = time.time()

    # Ising(D=args.D, chi=args.X, dbeta=args.DBETA, ntu_environment=args.NTUEnvironment, method=args.method)

    # logging.info('Elapsed time: %0.2f s.', (time.time() - tt))

    for D in [12]:
        for ee in ['NN', 'NN+', 'BP', 'NN+BP']:
            for method in ['nn', 'mpo']:
                Ising(D, chi=25, dbeta=0.01, ntu_environment=ee, method=method)
