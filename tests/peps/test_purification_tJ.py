import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import logging

try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg


def purification_tJ(chemical_potential):

    D = 8
    chi = 40
    sym = "U1xU1xZ2"
    coef = 0.25
    J = 0.5
    Jz = 0.5
    t = 1
    beta_target = 0.05
    dbeta = 0.005
    ntu_environment = "NN"

    tot_sites = int(2 * 3)
    net = fpeps.SquareLattice((2, 3), "obc")
    opt = yastn.operators.SpinfulFermions_tJ(sym = sym, backend=cfg.backend, default_device=cfg.default_device)
    fid = opt.I()
    fc_up, fc_dn = opt.c(spin='u'), opt.c(spin='d')
    fcdag_up, fcdag_dn = opt.cp(spin='u'), opt.cp(spin='d')
    n_up, n_dn = opt.n(spin='u'), opt.n(spin='d')
    n = n_up + n_dn
    Sz, Sp, Sm = opt.Sz(), opt.Sp(), opt.Sm()
    #
    psi = fpeps.product_peps(net, fid)
    #
    num_steps = round(beta_target / dbeta)
    dbeta = beta_target / num_steps
    #
    g_hopping_up = fpeps.gates.gate_nn_hopping(t, dbeta * coef, fid, fc_up, fcdag_up)
    g_hopping_dn = fpeps.gates.gate_nn_hopping(t, dbeta * coef, fid, fc_dn, fcdag_dn)
    g_heisenberg = fpeps.gates.gates_Heisenberg_spinful(dbeta * coef, Jz, J, J, Sz, Sp, Sm, n, fid)
    g_loc = fpeps.gates.gate_local_occupation(chemical_potential, dbeta * coef, fid, n)
    gates = fpeps.gates.distribute(net, gates_nn=[g_hopping_up, g_hopping_dn, g_heisenberg], gates_local=g_loc)

    env_evolution = fpeps.EnvNTU(psi, which=ntu_environment)

    opts_svd = {"D_total": D, 'tol_block': 1e-15}

    beta = 0
    for _ in range(num_steps):
        beta += dbeta
        logging.info("beta = %0.3f" % beta)
        fpeps.evolution_step_(env_evolution, gates, opts_svd=opts_svd)

    # calculate observables with ctm
    tol = 1e-8 # truncation of singular values of CTM projectors
    max_sweeps = 100  # ctm param
    tol_exp = 1e-6
    # opts_svd_ctm = {'D_total': chi, 'tol': tol, 'policy': 'lowrank', 'D_block': chi // 8}
    opts_svd_ctm = {'D_total': chi, 'tol': tol}

    env = fpeps.EnvCTM(psi, init="eye")
    env.update_(opts_svd=opts_svd_ctm, method="2site")

    print("Time evolution done")
    for out in env.ctmrg_(max_sweeps=max_sweeps, iterator_step=1, method="2site", opts_svd=opts_svd_ctm, corner_tol=tol_exp):
        pass

    # calculate expectation values
    cdagc_up = env.measure_nn(fcdag_up, fc_up)  # calculate for all unique bonds
    cdagc_dn = env.measure_nn(fcdag_dn, fc_dn)  # -> {bond: value}
    ccdag_dn = env.measure_nn(fc_dn, fcdag_dn)
    ccdag_up = env.measure_nn(fc_up, fcdag_up)
    SmSp = env.measure_nn(Sm, Sp)
    SpSm = env.measure_nn(Sp, Sm)
    SzSz = env.measure_nn(Sz, Sz)
    nn = env.measure_nn(n, n)

    n_total = env.measure_1site(n)
    nu = env.measure_1site(n_up)
    nd = env.measure_1site(n_dn)

    energy = -t * (sum(cdagc_up.values()) - sum(ccdag_up.values()) + sum(cdagc_dn.values()) - sum(ccdag_dn.values())) + \
                J / 2 * (sum(SmSp.values()) + sum(SpSm.values())) + Jz * sum(SzSz.values()) - J / 4 * sum(nn.values()) - \
                sum(n_total.values()) * chemical_potential
    energy = energy / tot_sites

    mean_density = sum(env.measure_1site(n).values()) / tot_sites
    print(out)
    print("CTMRG done")

    return (energy, mean_density, list(cdagc_up.values()), list(cdagc_dn.values()), list(SpSm.values()), list(SzSz.values()), list(nn.values()), list(nu.values()), list(nd.values()))


def test_purification_tJ():
    mu_list = [4.0, 2.0, 0.0]

    energy_ED  = {4.0: -2.944895, 2.0: -1.476636, 0.0: -0.092255}
    density_ED = {4.0:  0.711440, 2.0:  0.690445, 0.0:  0.668632}

    cdagc_up_ED = {4.0: {((0, 0), (1, 0)): 0.0051242685725287895, ((0, 0), (0, 1)): 0.005118887340003729, ((0, 0), (1, 1)): -1.60106267146614e-05, ((0, 0), (0, 2)): -7.977839795621708e-06, ((0, 0), (1, 2)): -1.2850731871040807e-06, ((0, 1), (1, 1)): 0.005113076713279957},
                   2.0: {((0, 0), (1, 0)): 0.005334331154560933, ((0, 0), (0, 1)): 0.005329253764495156, ((0, 0), (1, 1)): -8.320902120925753e-06, ((0, 0), (0, 2)): -4.131369887258778e-06, ((0, 0), (1, 2)): -1.413020664357699e-06, ((0, 1), (1, 1)): 0.00532370373574041},
                   0.0: {((0, 0), (1, 0)): 0.005529267312233714, ((0, 0), (0, 1)): 0.0055245495513449525, ((0, 0), (1, 1)): 3.642299456184801e-07, ((0, 0), (0, 2)): 2.121398605395238e-07, ((0, 0), (1, 2)): -1.5249419417730037e-06, ((0, 1), (1, 1)): 0.005519321307547488}}

    cdagc_dn_ED = {4.0: {((0, 0), (1, 0)): 0.0051242685725287895, ((0, 0), (0, 1)): 0.00511888734000373, ((0, 0), (1, 1)): -1.6010626714661406e-05, ((0, 0), (0, 2)): -7.977839795621712e-06, ((0, 0), (1, 2)): -1.2850731871040807e-06, ((0, 1), (1, 1)): 0.005113076713279957},
                   2.0: {((0, 0), (1, 0)): 0.005334331154560933, ((0, 0), (0, 1)): 0.005329253764495156, ((0, 0), (1, 1)): -8.320902120925756e-06, ((0, 0), (0, 2)): -4.131369887258773e-06, ((0, 0), (1, 2)): -1.4130206643576993e-06, ((0, 1), (1, 1)): 0.0053237037357404095},
                   0.0: {((0, 0), (1, 0)): 0.005529267312233715, ((0, 0), (0, 1)): 0.0055245495513449525, ((0, 0), (1, 1)): 3.642299456184818e-07, ((0, 0), (0, 2)): 2.121398605395187e-07, ((0, 0), (1, 2)): -1.5249419417730035e-06, ((0, 1), (1, 1)): 0.005519321307547488}}

    SpSm_ED =     {4.0: {((0, 0), (1, 0)): -0.0015904516040125315, ((0, 0), (0, 1)): -0.0015920439157563497, ((0, 0), (1, 1)): 1.3364900857681796e-05, ((0, 0), (0, 2)): 6.683677250204739e-06, ((0, 0), (1, 2)): -8.692064387654903e-08, ((0, 1), (1, 1)): -0.001593664466363279},
                   2.0: {((0, 0), (1, 0)): -0.0014979553434482993, ((0, 0), (0, 1)): -0.0014995300453349571, ((0, 0), (1, 1)): 1.2141434065674859e-05, ((0, 0), (0, 2)): 6.073107359341195e-06, ((0, 0), (1, 2)): -7.641042217884863e-08, ((0, 1), (1, 1)): -0.001501130726100453},
                   0.0: {((0, 0), (1, 0)): -0.0014048008292682078, ((0, 0), (0, 1)): -0.0014063456949771065, ((0, 0), (1, 1)): 1.0951276548530065e-05, ((0, 0), (0, 2)): 5.479241334056487e-06, ((0, 0), (1, 2)): -6.652752915056496e-08, ((0, 1), (1, 1)): -0.0014079145452911346}}

    SzSz_ED =     {4.0: {((0, 0), (1, 0)): -0.0007952258020062653, ((0, 0), (0, 1)): -0.0007960219578781714, ((0, 0), (1, 1)): 6.6824504288433544e-06, ((0, 0), (0, 2)): 3.341838625104223e-06, ((0, 0), (1, 2)): -4.346032194283496e-08, ((0, 1), (1, 1)): -0.0007968322331816355},
                   2.0: {((0, 0), (1, 0)): -0.0007489776717241464, ((0, 0), (0, 1)): -0.0007497650226674793, ((0, 0), (1, 1)): 6.070717032838325e-06, ((0, 0), (0, 2)): 3.0365536796704063e-06, ((0, 0), (1, 2)): -3.8205211089551144e-08, ((0, 1), (1, 1)): -0.000750565363050228},
                   0.0: {((0, 0), (1, 0)): -0.0007024004146340985, ((0, 0), (0, 1)): -0.0007031728474885535, ((0, 0), (1, 1)): 5.4756382742637486e-06, ((0, 0), (0, 2)): 2.739620667026213e-06, ((0, 0), (1, 2)): -3.326376457564234e-08, ((0, 1), (1, 1)): -0.0007039572726455672}}

    nn_ED =       {4.0: {((0, 0), (1, 0)): 0.5059222217646266, ((0, 0), (0, 1)): 0.506498944714555, ((0, 0), (1, 1)): 0.5063391483978517, ((0, 0), (0, 2)): 0.5057618160085865, ((0, 0), (1, 2)): 0.5057615358391269, ((0, 1), (1, 1)): 0.507076343595331},
                   2.0: {((0, 0), (1, 0)): 0.47650724445220005, ((0, 0), (0, 1)): 0.4770781863056769, ((0, 0), (1, 1)): 0.47690499585627116, ((0, 0), (0, 2)): 0.4763334202415265, ((0, 0), (1, 2)): 0.47633311111015614, ((0, 1), (1, 1)): 0.4776498339627556},
                   0.0: {((0, 0), (1, 0)): 0.44688209737269735, ((0, 0), (0, 1)): 0.4474429974809053, ((0, 0), (1, 1)): 0.4472568910206833, ((0, 0), (0, 2)): 0.4466953394019484, ((0, 0), (1, 2)): 0.44669500176140686, ((0, 1), (1, 1)): 0.44800462648996614}}

    n_ED  =       {4.0: {0: 0.35558456616282075, 2: 0.3559902764928251},
                   2.0: {0: 0.3450844497724358, 2: 0.34549831230933825},
                   0.0: {0: 0.3341762261156182, 2: 0.3345960764790318}}

    for mu in mu_list:

        dict_bond = {((0, 0), (0, 1)):0,
                     ((0, 1), (0, 2)):1,
                     ((1, 0), (1, 1)):2,
                     ((1, 1), (1, 2)):3,
                     ((0, 0), (1, 0)):4, ((0, 1), (1, 1)):5, ((0, 2), (1, 2)):6}
        print(f"Calculations for {mu=}")
        energy, density, cdagc_up, cdagc_dn, SpSm, SzSz, nn, n_up, n_dn = purification_tJ(mu)
        assert abs(energy - energy_ED[mu]) < 1e-3
        assert abs(density - density_ED[mu]) < 1e-3

        for bond in [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((0, 0), (1, 0))]:
            assert  abs(cdagc_up[dict_bond[bond]] - cdagc_up_ED[mu][bond]) / abs(cdagc_up_ED[mu][bond]) < 2e-3
            assert  abs(cdagc_dn[dict_bond[bond]] - cdagc_dn_ED[mu][bond]) / abs(cdagc_dn_ED[mu][bond]) < 2e-3
            assert  abs(SpSm[dict_bond[bond]] - SpSm_ED[mu][bond]) / abs(SpSm_ED[mu][bond]) < 2e-3
            assert  abs(SzSz[dict_bond[bond]] - SzSz_ED[mu][bond]) / abs(SzSz_ED[mu][bond]) < 2e-3
            assert  abs(nn[dict_bond[bond]] - nn_ED[mu][bond]) / abs(nn_ED[mu][bond]) < 2e-3

        assert  abs(n_up[0] - n_ED[mu][0]) / abs(n_ED[mu][0]) < 1e-3
        assert  abs(n_up[2] - n_ED[mu][2]) / abs(n_ED[mu][2]) < 1e-3

        assert  abs(n_dn[0] - n_ED[mu][0]) / abs(n_ED[mu][0]) < 1e-3
        assert  abs(n_dn[2] - n_ED[mu][2]) / abs(n_ED[mu][2]) < 1e-3


if __name__== '__main__':
    test_purification_tJ()
