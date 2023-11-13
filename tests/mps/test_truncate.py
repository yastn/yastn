""" truncation of mps """
import pytest
import yastn
import yastn.tn.mps as mps
try:
    from .configs import config_dense as cfg
except ImportError:
    from configs import config_dense as cfg
# pytest modifies cfg to inject different backends and devices during tests


def test_truncate(config=cfg, tol=1e-12):
    """ Test mps.truncate_ on random input states. """
    opts_config = {} if config is None else \
                {'backend': config.backend, 'default_device': config.default_device}
    #
    # initialize random MPO
    #
    N = 14
    Di = 35  # initial D
    Df, svd_tol = 15, 6e-2  # truncation parameters
    for sym, n, seed in [("dense", (), 0), ("Z2", (0,), 1), ("Z2", (1,), 2)]:
        ops = yastn.operators.Spin12(sym=sym, **opts_config)
        I = mps.product_mpo(ops.I(), N)
        ops.random_seed(seed=seed)
        #
        #  truncate MPO keeping the norm;
        #
        psi = 2 * mps.random_mpo(I, D_total=Di)  # extra factor
        assert psi.get_bond_dimensions() == (1, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
        psipsi = mps.vdot(psi, psi)
        #
        # canonize psi to the last site; retains the norm in psi.factor for the record
        #
        psi.canonize_(to='last', normalize=False)
        # assert psi.get_bond_dimensions() == (1, 4, 16, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
        assert abs(mps.vdot(psi, psi) / psipsi - 1) < tol
        assert psi.is_canonical(to='last', tol=tol)
        #
        # truncate phi while canonizing to the first site;
        # keep the norm of the original state psi in phi.factor
        #
        phi = psi.shallow_copy()
        discarded = phi.truncate_(to='first', opts_svd={'D_total': Df}, normalize=False)
        assert phi.get_bond_dimensions() == (1, 4, Df, Df, Df, Df, Df, Df, Df, Df, Df, Df, Df, 4, 1)
        #
        phiphi = mps.vdot(phi, phi)
        phipsi = mps.vdot(phi, psi)
        #
        assert abs(phiphi / phipsi - 1) < tol
        assert abs(phipsi / psipsi + discarded ** 2 - 1) < tol
        #
        # norm of phi is phi.factor; individual tensors are properly canonized
        assert phi.is_canonical(to='first', tol=tol)

        #
        # truncate MPS forgetting the norm;
        #
        psi = 0.5j * mps.random_mps(I, D_total=Di, n=n)  #extra factor
        assert psi.get_bond_dimensions() == (1, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
        psipsi = mps.vdot(psi, psi)
        #
        # canonize psi to the first site; keep track of the norm
        #
        psi.canonize_(to='first', normalize=False)
        assert abs(mps.vdot(psi, psi) / psipsi - 1) < tol
        assert psi.is_canonical(to='first', tol=tol)
        #
        #  truncate phi while canonizing to the last site; truncate based on tolerance
        #
        phi = psi.shallow_copy()
        discarded = phi.truncate_(to='last', opts_svd={'tol': svd_tol})  # normalize=True
        #
        svals = phi.get_Schmidt_values()
        for sval in svals:
            assert min(sval._data) / max(sval.data) > svd_tol * 0.8  # 0.8 as local truncations affect each other
        #
        phiphi = mps.vdot(phi, phi)
        phipsi = mps.vdot(phi, psi)
        assert abs(mps.vdot(phi, phi) - 1) < tol
        assert abs(phipsi ** 2 / psipsi + discarded ** 2 - 1) < tol



def test_zipper(config=cfg, tol=1e-12):
    """ Test mps.zipper on random input states. """
    opts_config = {} if config is None else \
                {'backend': config.backend, 'default_device': config.default_device}
    #
    N = 13
    for sym, n, seed in [("dense", (), 0), ("Z2", (0,), 1), ("Z2", (1,), 2)]:
        #
        #  zipper for MPO @ MPO
        #
        Dai, Dbi = 13, 8  # initial D
        ops = yastn.operators.Spin12(sym=sym, **opts_config)
        I = mps.product_mpo(ops.I(), N)
        ops.random_seed(seed=seed)
        a = mps.random_mpo(I, D_total=Dai)
        b = mps.random_mpo(I, D_total=Dbi)
        ab = a @ b
        vabab = mps.vdot(ab, ab)
        #
        Df = 31  # final D
        c, discarded = mps.zipper(a, b, opts_svd={'D_total': Df}, normalize=False, return_discarded=True)
        #
        vcc = mps.vdot(c, c)
        vcab = mps.vdot(c, ab)
        assert c.is_canonical(to='first', tol=tol)
        assert abs(vcab / vcc - 1) < tol
        assert abs(vcab / vabab + discarded**2 - 1)  <  0.1  # discarded is only an estimate
        assert c.get_bond_dimensions() == (1, Df, Df, Df, Df, Df, Df, Df, Df, Df, Df, 16, 4, 1)

        #
        #  zipper for MPO @ MPS; forgetting the norm
        #
        Dai, Dbi = 17, 11  # initial D
        a = mps.random_mpo(I, D_total=Dai).canonize_(to='last', normalize=False)
        b = mps.random_mps(I, D_total=Dbi, n=n).canonize_(to='first', normalize=False)
        # some random canonization to move norm to .factor
        ab = a @ b
        vabab = mps.vdot(ab, ab)
        #
        c, discarded = mps.zipper(a, b, opts_svd={'tol': 1e-1}, return_discarded=True)  # normalize=True,
        #
        assert c.is_canonical(to='first', tol=tol)
        assert abs(mps.vdot(c, c) - 1) < tol
        #
        vcab = mps.vdot(c, ab)
        assert abs(vcab ** 2 / vabab + discarded**2 - 1)  <  0.1  # discarded is only an estimate


def test_compression(config=cfg, tol=1e-12):
    """ Test mps.zipper on random input states. """
    opts_config = {} if config is None else \
                {'backend': config.backend, 'default_device': config.default_device}
    #
    N = 13
    for sym, n, seed in [("dense", (), 0), ("Z2", (0,), 1), ("Z2", (1,), 2)]:
        #
        #  compression for MPO @ MPO; initialize with zipper
        #
        Dai, Dbi = 13, 8  # initial D
        ops = yastn.operators.Spin12(sym=sym, **opts_config)
        I = mps.product_mpo(ops.I(), N)
        ops.random_seed(seed=seed)
        a = 2.0 * mps.random_mpo(I, D_total=Dai)  # add extra factors
        b = 0.5 * mps.random_mpo(I, D_total=Dbi)
        ab = a @ b
        vabab = mps.vdot(ab, ab)
        #
        Df = 29  # final D
        c = mps.zipper(a, b, opts_svd={'D_total': Df})
        #
        #  optimize overlap
        #
        out = mps.compression_(c, (a, b), method='1site', overlap_tol=1e-4, Schmidt_tol=1e-4,
                         max_sweeps = 100, normalize=False)
        #
        vcc = mps.vdot(c, c)
        vcab = mps.vdot(c, ab)
        assert c.is_canonical(to='first', tol=tol)
        assert abs(vcab / vcc - 1) < tol
        assert c.get_bond_dimensions() == (1, 4, 16, Df, Df, Df, Df, Df, Df, Df, Df, 16, 4, 1)
        # converged within desired tolerance
        assert out.doverlap < 1e-4 and out.max_dSchmidt < 1e-4 and out.sweeps < 100
        # absolut convergence is still far from perfect for this Df.
        assert 0.4 < vcab / vabab < 1

        #
        # compression for MPO @ MPS; initialize with zipper
        #
        Dai, Dbi = 17, 11  # initial D
        a = 3.0 * mps.random_mpo(I, D_total=Dai).canonize_(to='last', normalize=False)
        b = mps.random_mps(I, D_total=Dbi, n=n).canonize_(to='first', normalize=False)
        # some random canonization to move norm to .factor
        ab = a @ b
        vabab = mps.vdot(ab, ab)
        #
        c = mps.zipper(a, b, opts_svd={'D_total': 13})  # normalize=True
        #
        #  optimize overlap, testing iterator
        #
        overlap_old, doverlap_old = 0, 1
        for out in mps.compression_(c, (a, b),
                                    method='2site', opts_svd={'D_total': 13},
                                    max_sweeps = 2, iterator_step=1):  # normalize=True
            assert doverlap_old > out.doverlap and overlap_old < out.overlap
            overlap_old, doverlap_old = out.overlap, out.doverlap
            assert out.max_discarded_weight > 0
        #
        assert c.is_canonical(to='first', tol=tol)
        assert abs(mps.vdot(c, c) - 1) < tol

        #
        # compression for MPS; initialize with truncate_
        #
        Dai = 57
        a = mps.random_mps(I, D_total=Dai, n=n).canonize_(to='last', normalize=False)
        c = a.shallow_copy()
        c.truncate_(to='first', opts_svd={'D_total': 13}, normalize=False)
        #
        #  optimize overlap
        #
        out = mps.compression_(c, (a,), method='2site', opts_svd={'D_total': 13},
                               Schmidt_tol=1e-3, max_sweeps = 20, normalize=False)
        #
        assert out.max_dSchmidt < 1e-3  # is converged based on Schmidt values
        vaa, vca, vcc = mps.vdot(a, a), mps.vdot(c, a), mps.vdot(c, c)
        assert c.is_canonical(to='first', tol=tol)
        assert abs(vca / vcc - 1) < tol
        assert 0.6 < vca / vaa  < 1


if __name__ == "__main__":
    test_truncate()
    test_zipper()
    test_compression()
