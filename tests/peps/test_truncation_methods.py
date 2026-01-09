import pytest
import yastn
import yastn.tn.fpeps as fpeps
import logging

@pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
@pytest.mark.parametrize("initialization, opts_svd_evolution",
                        [("EAT", {"D_total": 6, 'tol_block': 1e-15}),
                         ("SVD", {"D_total": 6, 'tol_block': 1e-15}),
                         ("ZMT1svd", {"D_total": 6, 'tol_block': 1e-15, 'preD':12}),
                         ("ZMT1eat", {"D_total": 6, 'tol_block': 1e-15, 'preD':12}),
                         ("ZMT3svd", {"D_total": 6, 'tol_block': 1e-15, 'preD':12}),
                         ("ZMT3eat", {"D_total": 6, 'tol_block': 1e-15, 'preD':12}),
                         ("ZMT3zmt1eat", {"D_total": 6, 'tol_block': 1e-15, 'preD':12}),
                         ("ZMT3zmt1svd", {"D_total": 6, 'tol_block': 1e-15, 'preD':12})])
def test_truncation_methods(config_kwargs, initialization, opts_svd_evolution):

    NTUEnv="NN"
    beta_end = 0.1
    dbeta = 0.1
    coef = 0.25
    beta_start = 0
    sym="U1xU1xZ2"

    opt = yastn.operators.SpinfulFermions_tJ(sym = sym, **config_kwargs)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn, n_up, n_dn, Sz, Sp, Sm = opt.I(), \
                                                                    opt.c(spin='u'),  opt.c(spin='d'), \
                                                                    opt.cp(spin='u'), opt.cp(spin='d'), \
                                                                    opt.n(spin='u'),  opt.n(spin='d'), \
                                                                    opt.Sz(), opt.Sp(), opt.Sm()
    net = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(net, fid)

    J, t, mu = 0.5, 1.0, 2.0
    g_tJ = fpeps.gates.gate_nn_tJ(J, t, t, mu / 4, mu / 4, mu / 4, mu / 4, dbeta * coef, fid, fc_up, fcdag_up, fc_dn, fcdag_dn)

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

        ntu_error_max = fpeps.accumulated_truncation_error([infos], statistics="max")
        ntu_error_mean = fpeps.accumulated_truncation_error([infos], statistics="mean")

        logging.info('ntu error max: %.2e' % ntu_error_max)
        logging.info('ntu error mean: %.2e' % ntu_error_mean)


    opts_svd_ctm = {'D_total': 24, 'tol': 1e-8, 'truncate_multiplets': True}
    corner_tol = 1e-5
    max_sweeps = 50

    env = fpeps.EnvCTM(psi, init="eye")

    info = env.ctmrg_(max_sweeps=max_sweeps, opts_svd=opts_svd_ctm, method='2x2', corner_tol=corner_tol)
    assert info.converged

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


    energy = -t * (sum(cdagc_up.values()) - sum(ccdag_up.values()) + sum(cdagc_dn.values()) - sum(ccdag_dn.values())) + \
                J / 2 * (sum(SmSp.values()) + sum(SpSm.values())) + J * sum(SzSz.values()) - J / 4 * sum(nn.values()) - \
                mu * (sum(expt_n_up.values()) + sum(expt_n_dn.values()))
    energy = energy / tot_sites

    assert abs(energy + 1.6403) < 1e-3


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    pytest.main([__file__, "-vs", "--durations=0", '--long_tests'])
