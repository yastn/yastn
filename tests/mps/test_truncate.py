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
    Df, svd_tol = 15, 1e-2  # truncation parameters
    for sym, n, seed in [("dense", (), 0), ("Z2", (0,), 1), ("Z2", (1,), 2)]:
        ops = yastn.operators.Spin12(sym=sym, **opts_config)
        I = mps.product_mpo(ops.I(), N)
        ops.random_seed(seed=seed)
        #
        #  truncate MPO keeping the norm;
        #
        psi = mps.random_mpo(I, D_total=Di)
        assert psi.get_bond_dimensions() == (1, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
        psipsi = mps.vdot(psi, psi)
        #
        # canonize psi to the last site; retains the norm in psi.factor for the record
        #
        psi.canonize_(to='last', normalize=False)
        assert psi.get_bond_dimensions() == (1, 4, 16, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
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
        assert abs(phiphi / psipsi - 1) < tol
        #
        phipsi = mps.vdot(phi, psi)
        assert abs(phipsi ** 2 / psipsi ** 2 + discarded ** 2 - 1) < tol
        #
        # norm of phi is phi.factor; individual tensors are properly canonized
        assert phi.is_canonical(to='first', tol=tol)

        #
        # truncate MPS forgetting the norm;
        #
        psi = mps.random_mps(I, D_total=Di, n=n)
        assert psi.get_bond_dimensions() == (1, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
        #
        # canonize psi to the first site; forgets the norm
        #
        psi.canonize_(to='first')
        assert abs(mps.vdot(psi, psi) - 1) < tol
        assert psi.is_canonical(to='first', tol=tol)
        #
        #  truncate phi while canonizing to the last site; truncate based on tolerance
        #
        phi = psi.shallow_copy()
        discarded = phi.truncate_(to='last', opts_svd={'tol': svd_tol})
        #
        assert abs(mps.vdot(phi, phi) - 1) < tol
        svals = phi.get_Schmidt_values()
        for sval in svals:
            assert min(sval._data) / max(sval.data) > svd_tol * 0.8  # 0.8 as local truncations affect each other
        #
        phipsi = mps.vdot(phi, psi)
        assert abs(phipsi ** 2 + discarded ** 2 - 1) < tol



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
        psi, discarded = mps.zipper(a, b, opts_svd={'D_total': Df, 'tol': 1e-13}, return_discarded=True)
        #
        assert psi.is_canonical(to='first', tol=tol)
        vpsipsi = mps.vdot(psi, psi)
        # assert abs(vpsipsi / vabab - 1) < tol  TODO: why this does not work?
        vpsiab = mps.vdot(psi, ab)
        print(vpsiab, vabab, discarded)
        # assert abs(vpsiab ** 2 / vabab + discarded ** 2 - 1) < tol
        # assert psi.get_bond_dimensions() == (1, 2, 4, 8, 16, Df, Df, Df, Df, 16, 8, 4, 2, 1)

        phi = ab.shallow_copy()
        phi.canonize_(to='last', normalize=False).truncate_(to='first', opts_svd={'D_total': Df}, normalize=False)
        print(vpsiab ** 2 / vabab , discarded**2)

        #
        #  zipper for MPO @ MPS; forgetting the norm
        #
        Dai, Dbi = 17, 11  # initial D
        a = mps.random_mpo(I, D_total=Dai)
        b = mps.random_mps(I, D_total=Dbi, n=n)
        ab = a @ b
        vabab = mps.vdot(ab, ab)
        #
        psi, discarded = mps.zipper(a, b, opts_svd={'tol': 1e-1}, return_discarded=True)
        #
        assert psi.is_canonical(to='first', tol=tol)
        vpsipsi = mps.vdot(psi, psi)
        assert abs(vpsipsi - 1) < tol
        vpsiab = mps.vdot(psi, ab / ab.norm())
        print(vpsiab ** 2, discarded ** 2)
        # assert abs(vpsiab ** 2 + discarded ** 2 - 1) < tol
        # assert psi.get_bond_dimensions() == (1, 2, 4, 8, 16, Df, Df, Df, Df, 16, 8, 4, 2, 1)

        phi = ab.shallow_copy()
        phi.canonize_(to='last', normalize=False).truncate_(to='first', opts_svd={'D_total': Df}, normalize=False)
        # print(vpsipsi / vabab, mps.vdot(phi, phi) / vabab)



if __name__ == "__main__":
    # test_truncate()
    test_zipper()



