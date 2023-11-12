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
    """ Initialize random mps and checks mps.truncate_ """
    opts_config = {} if config is None else \
                {'backend': config.backend, 'default_device': config.default_device}
    #
    # initialize random MPO
    #
    N = 13
    Di = 35  # initial D
    ops = yastn.operators.Spin12(sym='dense', **opts_config)
    I = mps.product_mpo(ops.I(), N)
    ops.random_seed(seed=0)
    psi = mps.random_mpo(I, D_total=Di)
    assert psi.get_bond_dimensions() == (1, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
    psipsi = mps.vdot(psi, psi)
    #
    # canonize psi to the last site; retains the norm in psi.factor for the record
    #
    psi.canonize_(to='last', normalize=False)
    assert psi.get_bond_dimensions() == (1, 4, 16, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
    assert abs(mps.vdot(psi, psi) / psipsi - 1) < tol
    assert psi.is_canonical(to='last', tol=tol)
    #
    # truncate phi while canonizing to the first site;
    # keep the norm of the original state psi in phi.factor
    #
    #
    Df = 17  # final D
    phi = psi.shallow_copy()
    discarded = phi.truncate_(to='first', opts_svd={'D_total': Df}, normalize=False)
    assert phi.get_bond_dimensions() == (1, 4, 16, Df, Df, Df, Df, Df, Df, Df, Df, 16, 4, 1)
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
    # truncate MPS forgetting the norm of phi
    #
    psi = mps.random_mps(I, D_total=Di)
    assert psi.get_bond_dimensions() == (1, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
    #
    # canonize psi to the first site; forgets the norm
    #
    psi.canonize_(to='first')
    assert psi.get_bond_dimensions() == (1, Di, Di, Di, Di, Di, Di, Di, 32, 16, 8, 4, 2, 1)
    assert abs(mps.vdot(psi, psi) - 1) < tol
    assert psi.is_canonical(to='first', tol=tol)
    #
    #  truncate phi while canonizing to the last site; truncate based on tolerance
    svd_tol = 1e-1
    phi = psi.shallow_copy()
    discarded = phi.truncate_(to='last', opts_svd={'tol': svd_tol})
    #
    assert abs(mps.vdot(phi, phi) - 1) < tol
    svals = phi.get_Schmidt_values()
    for sval in svals:
        assert min(sval._data) / max(sval.data) > svd_tol * 0.8  # 0.8 as local truncations affect rach other
    #
    phipsi = mps.vdot(phi, psi)
    assert abs(phipsi ** 2 + discarded ** 2 - 1) < tol



if __name__ == "__main__":
    test_truncate()



