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
    #
    # load local operators
    #
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject backend and device for testing
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)
    #
    ops.random_seed(seed=0)  # fix seed
    I = mps.product_mpo(ops.I(), N)  # identity MPS
    #
    # random MPS of matching physical dimension
    #
    psi = mps.random_mps(I, D_total=D_total, n=n)
    #
    # the same can be done employing Generator class
    #
    generate = mps.Generator(N, ops)
    generate.random_seed(seed=0)
    phi = generate.random_mps(D_total=D_total, n=n)
    assert (psi - phi).norm() < 1e-12 * psi.norm()
    #
    return psi


def random_mpo_spinless_fermions(N=10, D_total=16, sym='Z2', config=None):
    """
    Generate random MPO of N sites, with bond dimension D_total and tensors with symmetry sym.
    """
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject backend and device for testing
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)
    #
    I = mps.product_mpo(ops.I(), N)  # identity MPS
    ops.random_seed(seed=0)  # fix seed
    H = mps.random_mpo(I, D_total=D_total)  # random MPO
    #
    # same with Generator class
    #
    generate = mps.Generator(N, ops)
    generate.random_seed(seed=0)
    G = generate.random_mpo(D_total=D_total)
    assert (H - G).norm() < 1e-12 * H.norm()
    #
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
        psi = random_mps_spinless_fermions(N, D_total, sym, nn, config=cfg)
        leg = psi[psi.first].get_legs(axes=0)
        assert leg.t == (nn,) and leg.s == -1
        leg = psi[psi.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = psi.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])

        H = random_mpo_spinless_fermions(N, D_total, sym, config=cfg)
        leg = H[H.first].get_legs(axes=0)
        assert leg.t == (n0,) and leg.s == -1
        leg = H[H.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = H.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])


def test_mixed_dimensions_mpo_and_transpose():
    N = 5
    b1 = yastn.ones(config=cfg, s=(1, -1), D=(2, 3))
    b2 = yastn.ones(config=cfg, s=(1, -1), D=(3, 4))
    # reference mpo
    ref = mps.product_mpo([b1, b2], N)
    #
    # random mpo with physical dimensions matching ref
    #
    H = mps.random_mpo(ref, D_total=5)
    #
    H_legs = H.get_physical_legs()
    assert H_legs == ref.get_physical_legs()
    phys_dims = [(legs[0].D, legs[1].D) for legs in H_legs]
    assert phys_dims == [((2,), (3,)), ((3,), (4,)), ((2,), (3,)), ((3,), (4,)), ((2,), (3,))]
    #
    psi_ket = mps.random_mps(ref, D_total=4)
    ket_legs = psi_ket.get_physical_legs()
    assert [leg.D for leg in ket_legs] == [(2,), (3,), (2,), (3,), (2,)]
    #
    # conjugate transpose to change reference to bra
    #
    psi_bra = mps.random_mps(ref.conj().T, D_total=3)  #
    bra_legs = psi_bra.get_physical_legs()
    assert [leg.D for leg in bra_legs] == [(3,), (4,), (3,), (4,), (3,)]
    #
    with pytest.raises(yastn.YastnError):
        mps.vdot(psi_bra, H, psi_ket)
        # Bond dimensions do not match.
    #
    # expectation value in positive operator H @ H.conj().T
    #
    assert mps.vdot(psi_bra, H.conj().T @ H, psi_bra) > 0
    assert mps.vdot(psi_ket, H @ H.conj().T, psi_ket) > 0




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
    """ test mps.product_mpo"""
    for sym, nl, nr in [('U1', (0,), (0,)), ('Z2', (0,), (0,)), ('dense', (), ())]:
        ops = yastn.operators.Spin12(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        N = 8
        I = mps.product_mpo(ops.I(), N=8)

        assert pytest.approx(mps.vdot(I, I).item(), rel=tol) == 2 ** N
        assert (I @ I - I).norm() < tol
        assert I.virtual_leg('first').t == (nl,)
        assert I.virtual_leg('last').t == (nr,)


def test_getitem_setitem():
    """ raise exceptions in mps getitem setitem"""
    psi3 = mps.Mps(N=8)
    psi4 = mps.Mpo(N=8)
    T3 = yastn.ones(config=cfg, s=(-1, 1, 1), D=(2, 3, 4))
    T4 = yastn.ones(config=cfg, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))

    psi3[1] = T3
    psi4[1] = T4
    assert psi3[1] == T3

    with pytest.raises(yastn.YastnError):
        psi3[5] = T4
        # MpsMpo: Tensor rank should be 3.
    with pytest.raises(yastn.YastnError):
        psi4[5] = T3
        # MpsMpo: Tensor rank should be 4.
    with pytest.raises(yastn.YastnError):
        psi4[5.] = T4
        # MpsMpo: n should be an integer in 0, 1, ..., N-1.
    with pytest.raises(yastn.YastnError):
        psi3[9] = T3
        # MpsMpo: n should be an integer in 0, 1, ..., N-1.
    with pytest.raises(yastn.YastnError):
        psi3[(4, 5)]
        #  MpsMpo does not have site with index (4, 5)


if __name__ == "__main__":
    test_generate_random_mps()
    test_product_mps()
    test_product_mpo()
    test_getitem_setitem()
    test_mixed_dimensions_mpo_and_transpose()
