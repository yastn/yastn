""" examples for addition of the Mps-s """
import pytest
import yastn.tn.mps as mps
import yastn
try:
    from .configs import config_dense as cfg
    # pytest modifies cfg to inject different backends and devices during tests
except ImportError:
    from configs import config_dense as cfg

tol = 1e-12

def test_mps_properties():
    """ Test reading MPS/MPO properties """

    N = 11
    ops = yastn.operators.Spin12(sym="dense", backend=cfg.backend, default_device=cfg.default_device)
    I = mps.product_mpo(ops.I(), N)
    psi = mps.random_mps(I, D_total=16)
    psi.canonize_(to='last').canonize_(to='first')

    assert psi.N == N
    assert len(psi) == N

    bds_mps = (1, 2, 4, 8, 16, 16, 16, 16, 8, 4, 2, 1)
    assert len(bds_mps) == N + 1
    assert psi.get_bond_dimensions() == bds_mps
    assert psi.get_bond_charges_dimensions() == [{(): bd} for bd in bds_mps]  # empty charge for dense

    legs = psi.get_virtual_legs()
    assert all(leg.D == (bd,) for leg, bd in zip(legs, bds_mps))
    assert all(leg.s == -1 for leg in legs)
    assert len(legs) == N + 1

    legs = psi.get_physical_legs()
    assert all(leg == ops.space() for leg in legs) and len(legs) == N

    H = mps.random_mpo(I, D_total=23)
    H.canonize_(to='last').canonize_(to='first')
    bds_mpo = (1, 4, 16, 23, 23, 23, 23, 23, 23, 16, 4, 1)
    assert (len(bds_mpo) == N + 1) and (H.get_bond_dimensions() == bds_mpo)
    assert H.get_bond_charges_dimensions() == [{(): bd} for bd in bds_mpo]  # empty charge for dense

    legs = H.get_virtual_legs()
    assert all(leg.D == (bd,) for leg, bd in zip(legs, bds_mpo))
    assert all(leg.s == -1 for leg in legs) and (len(legs) == N + 1)

    legs = H.get_physical_legs()
    assert all(leg == (ops.space(), ops.space().conj()) for leg in legs)


if __name__ == "__main__":
    test_mps_properties()
