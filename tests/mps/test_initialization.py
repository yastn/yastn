""" product and random mps initialization """
import pytest
import numpy as np
import yastn
import yastn.tn.mps as mps
try:
    from .configs import config_dense as cfg
except ImportError:
    from configs import config_dense as cfg
# pytest modifies cfg to inject different backends and devices during tests


def test_MpsMpo_getitem_setitem():
    """ raise exceptions in MpsMpo, __getitem__, __setitem__"""
    #
    #  empry Mps and Mpo
    N = 8
    psi3 = mps.Mps(N=N)
    psi4 = mps.Mpo(N=N)
    #
    for psi in (psi3, psi4):
        assert psi.pC is None and len(psi.A) == len(psi) == N
        assert all(psi[n] is None for n in psi.sweep())

    #
    #  directly assigning tensors to sites
    T3 = yastn.ones(config=cfg, s=(-1, 1, 1), D=(2, 3, 4))
    T4 = yastn.ones(config=cfg, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    psi3[1] = T3
    psi4[1] = T4
    assert psi3[1] == T3
    #
    #  raising in getitem setitem
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
    #
    #  raising in MpsMpo
    with pytest.raises(yastn.YastnError):
        mps.Mps(N=0)
        # Number of Mps sites N should be a positive integer.
    with pytest.raises(yastn.YastnError):
        mps.Mps(N=3.)
        # Number of Mps sites N should be a positive integer.
    with pytest.raises(yastn.YastnError):
        mps.MpsMpo(N=5, nr_phys=3)
        # Number of physical legs, nr_phys, should be 1 or 2.
    #
    #  raising in sweep
    with pytest.raises(yastn.YastnError):
        psi = mps.Mps(5)
        psi.sweep(to='center')
        # "to" in sweep should be in "first" or "last"


def test_product_mps(config=cfg, tol=1e-12):
    """ test mps.product_mps """

    opts_config = {} if config is None else \
            {'backend': config.backend,
            'default_device': config.default_device}

    for sym, nl, nr in [('U1', (1,), (0,)), ('Z2', (1,), (0,)), ('dense', (), ())]:
        ops = yastn.operators.Spin12(sym=sym, **opts_config)
        vp1 = ops.vec_z(val=+1)
        vm1 = ops.vec_z(val=-1)

        psi = mps.product_mps(vectors=[vp1, vm1], N=7)  #  mps of [vp1, vm1, vp1, vm1, vp1, vm1, vp1]

        assert pytest.approx(mps.vdot(psi, psi).item(), rel=tol) == 1.0
        assert psi.virtual_leg('first').t == (nl,)
        assert psi.virtual_leg('last').t == (nr,)
        assert psi.pC is None and len(psi.A) == len(psi) == 7
        assert all(psi[site].s == (-1, 1, 1) for site in psi.sweep())

        assert mps.measure_1site(psi, ops.z(), psi) == {0: +1.0, 1: -1.0, 2: +1.0, 3: -1.0, 4: +1.0, 5: -1.0, 6: +1.0}

    for sym, ntot in [('U1', (4,)), ('Z2', (0,))]:
        ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)
        v0 = ops.vec_n(val=0)
        v1 = ops.vec_n(val=1)

        psi = mps.product_mps(vectors=[v1, v0, v1, v0, v0, v1, v1])

        assert pytest.approx(mps.vdot(psi, psi).item(), rel=tol) == 1.0
        assert psi.virtual_leg('first').t == (ntot,)
        assert psi.virtual_leg('last').t == ((0,),)
        assert psi.pC is None and len(psi.A) == len(psi) == 7
        assert all(psi[site].s == (-1, 1, 1) for site in psi.sweep())

        assert mps.measure_1site(psi, ops.n(), psi) == {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 1.0}

        with pytest.raises(yastn.YastnError):
            psi = mps.product_mps(v1.conj(), N=4)
            # Vector should have ndim = 1 with the signature s = (1,).
        with pytest.raises(yastn.YastnError):
            psi = mps.product_mps(v1.add_leg(), N=4)
            # Vector should have ndim = 1 with the signature s = (1,).

    psi = mps.random_dense_mps(N=5, D=4, d=3, **opts_config)
    assert len(psi) == 5
    assert psi.get_bond_dimensions() == (1, 4, 4, 4, 4, 1)
    assert (leg.D == (3,) for leg in psi.get_physical_legs())
    assert opts_config["default_device"] in psi[0].device
    assert opts_config["backend"] == psi.config.backend


def test_product_mpo(config=cfg, tol=1e-12):
    """ test mps.product_mpo """
    #
    opts_config = {} if config is None else \
        {'backend': config.backend,
        'default_device': config.default_device}
    #
    # initializing identity MPO
    for sym, nl, nr in [('U1', (0,), (0,)), ('Z2', (0,), (0,)), ('dense', (), ())]:
        ops = yastn.operators.Spin12(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        N = 8
        I = mps.product_mpo(ops.I(), N=8)

        assert pytest.approx(mps.vdot(I, I).item(), rel=tol) == 2 ** N
        assert (I @ I - I).norm() < tol
        assert I.virtual_leg('first').t == (nl,)
        assert I.virtual_leg('last').t == (nr,)
        assert all(I[site].s == (-1, 1, 1, -1) for site in I.sweep())

        with pytest.raises(yastn.YastnError):
            mps.product_mpo(ops.I().T, N=8)
            # Operator should have ndim = 2 with the signature s = (1, -1).

    H = mps.random_dense_mpo(N=5, D=4, d=3, **opts_config)
    assert len(H) == 5
    assert H.get_bond_dimensions() == (1, 4, 4, 4, 4, 1)
    assert (leg[0].D == (3,) and leg[1].D == (3,) for leg in H.get_physical_legs())
    assert opts_config["default_device"] in H[0].device
    assert opts_config["backend"] == H.config.backend


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
    # pytest uses config to inject various backends and devices for testing
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
    # pytest uses config to inject various backends and devices for testing
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
    """
    test states from random_mpo_spinless_fermions, random_mps_spinless_fermions
    """
    N = 10
    D_total = 16
    bds = (1,) + (D_total,) * (N - 1) + (1,)
    #
    for sym, nn in (('Z2', (0,)), ('Z2', (1,)), ('U1', (N // 2,))):
        n0 = (0,) * len(nn)
        psi = random_mps_spinless_fermions(N, D_total, sym, nn, config=cfg)
        leg = psi[psi.first].get_legs(axes=0)
        assert leg.t == (nn,) and leg.s == -1
        leg = psi[psi.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = psi.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])  # > D_total / 2 as randomness might not allow saturation.

        H = random_mpo_spinless_fermions(N, D_total, sym, config=cfg)
        leg = H[H.first].get_legs(axes=0)
        assert leg.t == (n0,) and leg.s == -1
        leg = H[H.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = H.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])


def test_mixed_dims_mpo_and_transpose():
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
    # expectation value in positive operator H @ H.conj().T
    #
    assert mps.vdot(psi_bra, H.conj().T @ H, psi_bra) > 0
    assert mps.vdot(psi_ket, H @ H.conj().T, psi_ket) > 0
    #
    # final tests of transpose
    #
    assert psi_ket.T == psi_ket
    phys_dims = [(legs[0].D, legs[1].D) for legs in H.T.get_physical_legs()]
    assert phys_dims == [((3,), (2,)), ((4,), (3,)), ((3,), (2,)), ((4,), (3,)), ((3,), (2,))]



def test_MpsMpo_properties(config=cfg):
    """ Test reading MPS/MPO properties """
    opts_config = {} if config is None else \
                {'backend': config.backend, 'default_device': config.default_device}

    N = 11
    ops = yastn.operators.Spin12(sym="dense", **opts_config)
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


def test_MpsMpo_copy(config=cfg):
    """ Initialize random mps of full tensors and checks copying. """
    opts_config = {} if config is None else \
                {'backend': config.backend, 'default_device': config.default_device}

    N = 16
    ops = yastn.operators.Spin1(sym='Z3', **opts_config)
    generate = mps.Generator(N=N, operators=ops)

    psi = generate.random_mps(n=(1,), D_total=16)
    phi = psi.copy()
    assert len(psi.A) == len(phi.A) == len(psi) == N
    for n in psi.sweep():
        assert np.allclose(psi[n].to_numpy(), phi[n].to_numpy()) and psi[n] is not phi[n]
    assert psi.A is not phi.A and psi is not phi

    psi = generate.random_mpo(D_total=16)
    psi.orthogonalize_site_(4)
    phi = psi.clone()
    assert len(psi.A) == len(phi.A) == len(psi) + 1 == N + 1
    for n in psi.A:
        assert np.allclose(psi[n].to_numpy(), phi[n].to_numpy()) and psi[n] is not phi[n]
    assert psi.A is not phi.A and psi is not phi

    phi = psi.shallow_copy()
    assert psi.A is not phi.A and psi is not phi
    assert all(psi[n] is phi[n] for n in psi.A)



if __name__ == "__main__":
    test_MpsMpo_getitem_setitem()
    test_product_mps()
    test_product_mpo()
    test_generate_random_mps()
    test_mixed_dims_mpo_and_transpose()
    test_MpsMpo_properties()
    test_MpsMpo_copy()
