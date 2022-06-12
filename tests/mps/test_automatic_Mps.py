""" adding set of the Mps-s, automatic Mps generator """
import yamps
try:
    from . import ops_dense
    from . import ops_U1
except ImportError:
    import ops_dense
    import ops_U1


def test_gen_XX_dmrg_dense():
    N = 7
    t, mu = 1., .2
    H = ops_dense.mpo_gen_XX(N, t, mu)
    Eng_gs = -3.427339492125848

    Dmax = 32
    cutoff_sweep = 10
    cutoff_dE = 1e-13
    opts_svd = {'tol': 1e-6, 'D_total': Dmax}

    version = '2site'
    psi = ops_dense.mps_random(N=N, Dmax=Dmax, d=2).canonize_sweep(to='first')
    env = yamps.dmrg(psi, H, version=version, max_sweeps=cutoff_sweep, atol=cutoff_dE, opts_svd=opts_svd)
    assert abs(env.measure()-Eng_gs)<1e-12


def test_gen_XX_dmrg_U1():
    N = 7
    t, mu = 1., .2
    H = ops_U1.mpo_gen_XX(N, t, mu)
    Eng_sectors = {2: -2.861972627395668,
                   3: -3.427339492125848,
                   4: -3.227339492125848}

    Dmax = 32
    cutoff_sweep = 10
    cutoff_dE = 1e-13
    opts_svd = {'tol': 1e-6, 'D_total': Dmax}

    version = '2site'
    for total_occ, E_target in Eng_sectors.items():
        psi = ops_U1.mps_random(N=N, Dblocks=[Dmax, 2, Dmax], total_charge=total_occ).canonize_sweep(to='first')
        env = yamps.dmrg(psi, H, version=version, max_sweeps=cutoff_sweep, atol=cutoff_dE, opts_svd=opts_svd)
        assert abs(env.measure()-E_target) < 1e-12



if __name__ == "__main__":
    test_gen_XX_dmrg_dense()
    test_gen_XX_dmrg_U1()
