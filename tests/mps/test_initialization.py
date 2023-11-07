""" product and random mps initialization """
import pytest
import numpy as np
import yastn
import yastn.tn.mps as mps
try:
    from .configs import config_dense as cfg
    # pytest modifies cfg to inject different backends and devices during tests
except ImportError:
    from configs import config_dense as cfg

tol = 1e-12



def random_mps_spinless_fermions(N=10, D_total=16, sym='Z2', n=1, config=None):
    """
    Generate random MPS of N sites, with bond dimension D_total,
    tensors with symmetry sym and total charge n.
    """

    # config is used here by pytest to inject backend and device for testing
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)

    generate = mps.Generator(N, ops)
    generate.random_seed(seed=0)
    psi = generate.random_mps(D_total=D_total, n=n)
    #
    # the same can be done without employing Generator class
    #
    ops.random_seed(seed=0)
    I = mps.product_mpo(ops.I(), N)
    psi2 = mps.random_mps(I, D_total=D_total, n=n)
    assert (psi - psi2).norm() < tol

    return psi


def random_mpo_spinless_fermions(N=10, D_total=16, sym='Z2', config=None):
    """
    Generate random MPO of N sites, with bond dimension D_total and tensors with symmetry sym.
    """
    if config is None:
        ops = yastn.operators.SpinlessFermions(sym=sym)
    else:  # config is used here by pytest to inject backend and device for testing
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend, default_device=config.default_device)

    generate = mps.Generator(N, ops)
    H = generate.random_mpo(D_total=D_total)
    return H


def test_generate_random_mps():
    N = 10
    D_total = 16
    bds = (1,) + (D_total,) * (N - 1) + (1,)

    for sym, nn in (('Z2', (0,)), ('Z2', (1,)), ('U1', (N // 2,))):
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        generate = mps.Generator(N, ops)
        I = generate.I()
        assert pytest.approx(mps.measure_overlap(I, I).item(), rel=tol) == 2 ** N
        O = I @ I + (-1 * I)
        assert pytest.approx(O.norm().item(), abs=tol) == 0

        n0 = (0,) * len(nn)
        psi = random_mps_spinless_fermions(N, D_total, sym, nn)
        leg = psi[psi.first].get_legs(axes=0)
        assert leg.t == (nn,) and leg.s == -1
        leg = psi[psi.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = psi.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])

        H = random_mpo_spinless_fermions(N, D_total, sym)
        leg = H[H.first].get_legs(axes=0)
        assert leg.t == (n0,) and leg.s == -1
        leg = H[H.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = H.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])


def test_product_mps():
    """ test mps.product_mps"""
    for sym, nl, nr in [('U1', (1,), (0,)), ('Z2', (1,), (0,)), ('dense', (), ())]:
        ops = yastn.operators.Spin12(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        vp1 = ops.vec_z(val=+1)
        vm1 = ops.vec_z(val=-1)

        psi = mps.product_mps(vectors=[vp1, vm1], N=7)  #  mps of [vp1, vm1, vp1, vm1, vp1, vm1, vp1]

        assert pytest.approx(mps.vdot(psi, psi).item(), rel=tol) == 1.0
        assert psi.virtual_leg('first').t == (nl,)
        assert psi.virtual_leg('last').t == (nr,)
        assert mps.measure_1site(psi, ops.z(), psi) == {0: +1.0, 1: -1.0, 2: +1.0, 3: -1.0, 4: +1.0, 5: -1.0, 6: +1.0}

    for sym, ntot in [('U1', (4,)), ('Z2', (0,))]:
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        v0 = ops.vec_n(val=0)
        v1 = ops.vec_n(val=1)

        psi = mps.product_mps(vectors=[v1, v0, v1, v0, v0, v1, v1])

        assert pytest.approx(mps.vdot(psi, psi).item(), rel=tol) == 1.0
        assert psi.virtual_leg('first').t == (ntot,)
        assert mps.measure_1site(psi, ops.n(), psi) == {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 1.0}


def test_product_mpo():

    for sym, nl, nr in [('U1', (0,), (0,)), ('Z2', (0,), (0,)), ('dense', (), ())]:
        ops = yastn.operators.Spin12(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        N = 8
        I = mps.product_mpo(ops.I(), N=8)

        assert pytest.approx(mps.vdot(I, I).item(), rel=tol) == 2 ** N
        assert (I @ I - I).norm() < tol
        assert I.virtual_leg('first').t == (nl,)
        assert I.virtual_leg('last').t == (nr,)


if __name__ == "__main__":
    test_generate_random_mps()
    test_product_mps()
    test_product_mpo()
