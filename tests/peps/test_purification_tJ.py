# Copyright 2025 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import logging


def purification_tJ(config_kwargs, mu):
    D = 8
    chi = 16
    sym = "U1xU1xZ2"
    coef = 0.25
    J = 0.5
    t = 1
    beta_target = 0.05
    dbeta = 0.025
    ntu_environment = "NN+"

    tot_sites = int(2 * 3)
    net = fpeps.SquareLattice((2, 3), "obc")
    opt = yastn.operators.SpinfulFermions_tJ(sym = sym, **config_kwargs)
    I = opt.I()
    c_up, c_dn = opt.c(spin='u'), opt.c(spin='d')
    cdag_up, cdag_dn = opt.cp(spin='u'), opt.cp(spin='d')
    n_up, n_dn = opt.n(spin='u'), opt.n(spin='d')
    n = n_up + n_dn
    Sz, Sp, Sm = opt.Sz(), opt.Sp(), opt.Sm()
    #
    psi = fpeps.product_peps(net, I)
    #
    num_steps = round(beta_target / dbeta)
    dbeta = beta_target / num_steps
    #
    # g_hopping_up = fpeps.gates.gate_nn_hopping(t, dbeta * coef, I, c_up, cdag_up)
    # g_hopping_dn = fpeps.gates.gate_nn_hopping(t, dbeta * coef, I, c_dn, cdag_dn)
    # g_heisenberg = fpeps.gates.gate_nn_tJ(J, 0, 0, 0, 0, 0, 0, dbeta * coef, I, c_up, cdag_up, c_dn, cdag_dn)
    # g_loc = fpeps.gates.gate_local_occupation(mu, dbeta * coef, I, n)
    # gates = fpeps.gates.distribute(net, gates_nn=[g_hopping_up, g_hopping_dn, g_heisenberg], gates_local=g_loc)

    # g_tj = fpeps.gates.gate_nn_tJ(J, t, t, 0, 0, 0, 0, dbeta * coef, I, c_up, cdag_up, c_dn, cdag_dn)
    # gates = fpeps.gates.distribute(net, gates_nn=g_tj, gates_local=g_loc)

    gates = [fpeps.gates.gate_nn_tJ(J, t, t, mu/4, mu/4, mu/4, mu/4, dbeta * coef, I, c_up, cdag_up, c_dn, cdag_dn, bond) for bond in net.bonds()]
    # correct boundary terms with local chemical potential
    gates += [fpeps.gates.gate_local_occupation(mu/2, dbeta * coef, I, n, site=(0, 0)),
              fpeps.gates.gate_local_occupation(mu/4, dbeta * coef, I, n, site=(0, 1)),
              fpeps.gates.gate_local_occupation(mu/2, dbeta * coef, I, n, site=(0, 2)),
              fpeps.gates.gate_local_occupation(mu/2, dbeta * coef, I, n, site=(1, 0)),
              fpeps.gates.gate_local_occupation(mu/4, dbeta * coef, I, n, site=(1, 1)),
              fpeps.gates.gate_local_occupation(mu/2, dbeta * coef, I, n, site=(1, 2))]
    # symmetrize
    gates = gates + gates[::-1]

    env_evolution = fpeps.EnvNTU(psi, which=ntu_environment)

    opts_svd = {"D_total": D, 'tol_block': 1e-15}

    beta = 0
    for _ in range(num_steps):
        beta += dbeta
        logging.info("beta = %0.3f" % beta)
        fpeps.evolution_step_(env_evolution, gates, opts_svd=opts_svd)

    # calculate observables with ctm
    max_sweeps = 10  # ctm param
    tol_exp = 1e-8
    opts_svd_ctm = {'D_total': chi}

    env = fpeps.EnvCTM(psi, init="eye")
    env.update_(opts_svd=opts_svd_ctm, method="2site")

    print("Time evolution done")
    info = env.ctmrg_(max_sweeps=max_sweeps, method="2site", opts_svd=opts_svd_ctm, corner_tol=tol_exp)

    # calculate expectation values
    cdagc_up = env.measure_nn(cdag_up, c_up)  # calculate for all unique bonds
    cdagc_dn = env.measure_nn(cdag_dn, c_dn)  # -> {bond: value}
    ccdag_dn = env.measure_nn(c_dn, cdag_dn)
    ccdag_up = env.measure_nn(c_up, cdag_up)
    SmSp = env.measure_nn(Sm, Sp)
    SpSm = env.measure_nn(Sp, Sm)
    SzSz = env.measure_nn(Sz, Sz)
    nn = env.measure_nn(n, n)

    n_total = env.measure_1site(n)
    nu = env.measure_1site(n_up)
    nd = env.measure_1site(n_dn)

    energy = -t * (sum(cdagc_up.values()) - sum(ccdag_up.values()) + sum(cdagc_dn.values()) - sum(ccdag_dn.values()))
    energy += J / 2 * (sum(SmSp.values()) + sum(SpSm.values())) + J * sum(SzSz.values()) - J / 4 * sum(nn.values())
    energy -= mu * sum(n_total.values())
    energy = energy / tot_sites

    density = sum([*nu.values(), *nd.values()]) / tot_sites
    print(info)
    print("CTMRG done")

    return energy, density, cdagc_up, cdagc_dn, SpSm, SzSz, nn, nu, nd


@pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
@pytest.mark.parametrize('mu', [0.0, 2.0, 4.0])
def test_purification_tJ(config_kwargs, mu):
    #
    # key are mu's
    energy_ED   = {4.0: -2.944895, 2.0: -1.476636, 0.0: -0.092255}
    density_ED  = {4.0:  0.711440, 2.0:  0.690445, 0.0:  0.668632}

    cdagc_up_ED = {4.0: {((0, 0), (1, 0)): 0.0051242685, ((0, 0), (0, 1)): 0.0051188873, ((0, 0), (1, 1)): -1.601062e-05, ((0, 0), (0, 2)): -7.977839e-06, ((0, 0), (1, 2)): -1.28507318e-06, ((0, 1), (1, 1)): 0.00511307671},
                   2.0: {((0, 0), (1, 0)): 0.0053343311, ((0, 0), (0, 1)): 0.0053292537, ((0, 0), (1, 1)): -8.320902e-06, ((0, 0), (0, 2)): -4.131369e-06, ((0, 0), (1, 2)): -1.41302066e-06, ((0, 1), (1, 1)): 0.00532370373},
                   0.0: {((0, 0), (1, 0)): 0.0055292673, ((0, 0), (0, 1)): 0.0055245495, ((0, 0), (1, 1)): 3.6422994e-07, ((0, 0), (0, 2)): 2.1213986e-07, ((0, 0), (1, 2)): -1.52494194e-06, ((0, 1), (1, 1)): 0.00551932130}}

    cdagc_dn_ED = {4.0: {((0, 0), (1, 0)): 0.0051242685, ((0, 0), (0, 1)): 0.0051188873, ((0, 0), (1, 1)): -1.601062e-05, ((0, 0), (0, 2)): -7.977839e-06, ((0, 0), (1, 2)): -1.28507318e-06, ((0, 1), (1, 1)): 0.00511307671},
                   2.0: {((0, 0), (1, 0)): 0.0053343311, ((0, 0), (0, 1)): 0.0053292537, ((0, 0), (1, 1)): -8.320902e-06, ((0, 0), (0, 2)): -4.131369e-06, ((0, 0), (1, 2)): -1.41302066e-06, ((0, 1), (1, 1)): 0.00532370373},
                   0.0: {((0, 0), (1, 0)): 0.0055292673, ((0, 0), (0, 1)): 0.0055245495, ((0, 0), (1, 1)): 3.6422994e-07, ((0, 0), (0, 2)): 2.1213986e-07, ((0, 0), (1, 2)): -1.52494194e-06, ((0, 1), (1, 1)): 0.00551932130}}

    SpSm_ED =     {4.0: {((0, 0), (1, 0)): -0.001590451, ((0, 0), (0, 1)): -0.001592043, ((0, 0), (1, 1)): 1.3364900e-05, ((0, 0), (0, 2)): 6.6836772e-06, ((0, 0), (1, 2)): -8.69206438e-08, ((0, 1), (1, 1)): -0.0015936644},
                   2.0: {((0, 0), (1, 0)): -0.001497955, ((0, 0), (0, 1)): -0.001499530, ((0, 0), (1, 1)): 1.2141434e-05, ((0, 0), (0, 2)): 6.0731073e-06, ((0, 0), (1, 2)): -7.64104221e-08, ((0, 1), (1, 1)): -0.0015011307},
                   0.0: {((0, 0), (1, 0)): -0.001404800, ((0, 0), (0, 1)): -0.001406345, ((0, 0), (1, 1)): 1.0951276e-05, ((0, 0), (0, 2)): 5.4792413e-06, ((0, 0), (1, 2)): -6.65275291e-08, ((0, 1), (1, 1)): -0.0014079145}}

    SzSz_ED =     {4.0: {((0, 0), (1, 0)): -0.000795225, ((0, 0), (0, 1)): -0.000796021, ((0, 0), (1, 1)): 6.6824504e-06, ((0, 0), (0, 2)): 3.3418386e-06, ((0, 0), (1, 2)): -4.34603219e-08, ((0, 1), (1, 1)): -0.0007968322},
                   2.0: {((0, 0), (1, 0)): -0.000748977, ((0, 0), (0, 1)): -0.000749765, ((0, 0), (1, 1)): 6.0707170e-06, ((0, 0), (0, 2)): 3.0365536e-06, ((0, 0), (1, 2)): -3.82052110e-08, ((0, 1), (1, 1)): -0.0007505653},
                   0.0: {((0, 0), (1, 0)): -0.000702400, ((0, 0), (0, 1)): -0.000703172, ((0, 0), (1, 1)): 5.4756382e-06, ((0, 0), (0, 2)): 2.7396206e-06, ((0, 0), (1, 2)): -3.32637645e-08, ((0, 1), (1, 1)): -0.0007039572}}

    nn_ED =       {4.0: {((0, 0), (1, 0)): 0.5059222217, ((0, 0), (0, 1)): 0.5064989447, ((0, 0), (1, 1)): 0.50633914839, ((0, 0), (0, 2)): 0.50576181600, ((0, 0), (1, 2)): 0.5057615358391, ((0, 1), (1, 1)): 0.50707634359},
                   2.0: {((0, 0), (1, 0)): 0.4765072444, ((0, 0), (0, 1)): 0.4770781863, ((0, 0), (1, 1)): 0.47690499585, ((0, 0), (0, 2)): 0.47633342024, ((0, 0), (1, 2)): 0.4763331111101, ((0, 1), (1, 1)): 0.47764983396},
                   0.0: {((0, 0), (1, 0)): 0.4468820973, ((0, 0), (0, 1)): 0.4474429974, ((0, 0), (1, 1)): 0.44725689102, ((0, 0), (0, 2)): 0.44669533940, ((0, 0), (1, 2)): 0.4466950017614, ((0, 1), (1, 1)): 0.44800462648}}

    n_ED  =       {4.0: {(0, 0): 0.3555845661, (1, 0): 0.3555845661, (0, 2): 0.3555845661, (1, 2): 0.3555845661, (0, 1): 0.3559902764, (1, 1): 0.3559902764},
                   2.0: {(0, 0): 0.3450844497, (1, 0): 0.3450844497, (0, 2): 0.3450844497, (1, 2): 0.3450844497, (0, 1): 0.3454983123, (1, 1): 0.3454983123},
                   0.0: {(0, 0): 0.3341762261, (1, 0): 0.3341762261, (0, 2): 0.3341762261, (1, 2): 0.3341762261, (0, 1): 0.3345960764, (1, 1): 0.3345960764}}
    #
    print(f"Calculations for {mu=}")
    #
    energy, density, cdagc_up, cdagc_dn, SpSm, SzSz, nn, n_up, n_dn = purification_tJ(config_kwargs, mu)
    assert abs(energy - energy_ED[mu]) < 5e-4

    for bond in [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((0, 0), (1, 0))]:
        assert abs(cdagc_up[bond] - cdagc_up_ED[mu][bond]) / abs(cdagc_up_ED[mu][bond]) < 2e-3
        assert abs(cdagc_dn[bond] - cdagc_dn_ED[mu][bond]) / abs(cdagc_dn_ED[mu][bond]) < 2e-3
        assert abs(SpSm[bond] - SpSm_ED[mu][bond]) / abs(SpSm_ED[mu][bond]) < 2e-3
        assert abs(SzSz[bond] - SzSz_ED[mu][bond]) / abs(SzSz_ED[mu][bond]) < 2e-3
        assert abs(nn[bond] - nn_ED[mu][bond]) / abs(nn_ED[mu][bond]) < 2e-3

    assert abs(density - density_ED[mu]) < 1e-4
    for site in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]:
        assert abs(n_up[site] - n_ED[mu][site]) / n_ED[mu][site] < 1e-4
        assert abs(n_dn[site] - n_ED[mu][site]) / n_ED[mu][site] < 1e-4


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", '--long_tests'])
