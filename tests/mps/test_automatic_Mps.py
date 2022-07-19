""" adding set of the Mps-s, automatic Mps generator """
import yamps
import generate_random
import generate_automatic
try:
    from .configs import config_dense, config_dense_fermionic
    from .configs import config_U1, config_U1_fermionic
    from .configs import config_Z2, config_Z2_fermionic
except ImportError:
    from configs import config_dense, config_dense_fermionic
    from configs import config_U1, config_U1_fermionic
    from configs import config_Z2, config_Z2_fermionic


tol = 1e-12


def test_gen_XX_dmrg_dense():
    N = 7
    t, mu = 1., .2
    H = generate_automatic.mpo_XX_model(config_dense_fermionic, N, t, mu)
    Eng_gs = -3.427339492125848

    Dmax = 32
    cutoff_sweep = 10
    cutoff_dE = 1e-13
    opts_svd = {'tol': 1e-6, 'D_total': Dmax}

    version = '2site'
    psi = generate_random.mps_random(config_dense_fermionic, N=N, Dmax=Dmax, d=2).canonize_sweep(to='first')
    env = yamps.dmrg(psi, H, version=version, max_sweeps=cutoff_sweep, atol=cutoff_dE, opts_svd=opts_svd)
    assert abs(env.measure() - Eng_gs) < tol


def test_gen_XX_dmrg_U1():
    N = 7
    t, mu = 1., .2
    H = generate_automatic.mpo_XX_model(config_U1_fermionic, N, t, mu)
    Eng_sectors = {2: -2.861972627395668,
                   3: -3.427339492125848,
                   4: -3.227339492125848}

    Dmax = 32
    cutoff_sweep = 10
    cutoff_dE = 1e-13
    opts_svd = {'tol': 1e-6, 'D_total': Dmax}

    version = '2site'
    for total_occ, E_target in Eng_sectors.items():
        psi = generate_random.mps_random(config_U1_fermionic, N=N, Dblocks=[Dmax, 2, Dmax], total_charge=total_occ).canonize_sweep(to='first')
        env = yamps.dmrg(psi, H, version=version, max_sweeps=cutoff_sweep, atol=cutoff_dE, opts_svd=opts_svd)
        assert abs(env.measure() - E_target) <  tol



if __name__ == "__main__":
    test_gen_XX_dmrg_dense()
    test_gen_XX_dmrg_U1()
