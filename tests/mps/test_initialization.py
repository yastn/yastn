# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" product and random Mps initialization """
import re
import numpy as np
import pytest
import yastn
import yastn.tn.mps as mps


def test_MpsMpoOBC_getitem_setitem(config_kwargs):
    """ raise exceptions in MpsMpoOBC, __getitem__, __setitem__"""
    #
    #  empty Mps and Mpo
    N = 8
    psi3 = mps.Mps(N=N)
    psi4 = mps.Mpo(N=N)
    #
    for psi in (psi3, psi4):
        assert psi.pC is None and len(psi.A) == len(psi) == N
        assert all(psi[n] is None for n in psi.sweep())
    #
    #  directly assigning tensors to sites
    config = yastn.make_config(sym='none', **config_kwargs)
    T3 = yastn.ones(config , s=(-1, 1, 1), D=(2, 3, 4))
    T4 = yastn.ones(config, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    psi3[1] = T3
    psi4[1] = T4
    assert psi3[1] == T3
    #
    #  raising in getitem setitem
    with pytest.raises(yastn.YastnError,
                       match="MpsMpoOBC: Tensor rank should be 3."):
        psi3[5] = T4
    with pytest.raises(yastn.YastnError,
                       match="MpsMpoOBC: Tensor rank should be 4."):
        psi4[5] = T3
    with pytest.raises(yastn.YastnError,
                       match="MpsMpoOBC: n should be an integer in 0, 1, ..., N-1."):
        psi4[5.] = T4
    with pytest.raises(yastn.YastnError,
                       match="MpsMpoOBC: n should be an integer in 0, 1, ..., N-1."):
        psi3[9] = T3
    with pytest.raises(yastn.YastnError,
                       match=re.escape("MpsMpoOBC does not have site with index (4, 5).")):
        psi3[(4, 5)]
    #
    #  raising in MpsMpoOBC
    with pytest.raises(yastn.YastnError,
                       match="Number of Mps sites N should be a positive integer."):
        mps.Mps(N=0)
    with pytest.raises(yastn.YastnError,
                       match="Number of Mps sites N should be a positive integer."):
        mps.Mps(N=3.)
    with pytest.raises(yastn.YastnError,
                       match="Number of physical legs, nr_phys, should be 1 or 2."):
        mps.MpsMpoOBC(N=5, nr_phys=3)
    #
    #  raising in sweep
    with pytest.raises(yastn.YastnError,
                       match='"to" in sweep should be in "first" or "last".'):
        psi = mps.Mps(5)
        psi.sweep(to='center')


def test_product_mps(config_kwargs, tol=1e-12):
    """ test mps.product_mps """

    for sym, nl, nr in [('U1', (1,), (0,)), ('Z2', (1,), (0,)), ('dense', (), ())]:
        ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
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
        ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
        v0 = ops.vec_n(val=0)
        v1 = ops.vec_n(val=1)

        psi = mps.product_mps(vectors=[v1, v0, v1, v0, v0, v1, v1])

        assert pytest.approx(mps.vdot(psi, psi).item(), rel=tol) == 1.0
        assert psi.virtual_leg('first').t == (ntot,)
        assert psi.virtual_leg('last').t == ((0,),)
        assert psi.pC is None and len(psi.A) == len(psi) == 7
        assert all(psi[site].s == (-1, 1, 1) for site in psi.sweep())

        assert mps.measure_1site(psi, ops.n(), psi) == {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 1.0}

        with pytest.raises(yastn.YastnError,
                           match="Vector should have ndim = 1."):
            psi = mps.product_mps(v1.add_leg(), N=4)

    psi = mps.random_dense_mps(N=5, D=4, d=3, **config_kwargs)
    assert len(psi) == 5
    assert psi.get_bond_dimensions() == (1, 4, 4, 4, 4, 1)
    assert (leg.D == (3,) for leg in psi.get_physical_legs())
    assert config_kwargs["default_device"] in psi[0].device
    assert config_kwargs["backend"] == psi.config.backend.BACKEND_ID


def test_product_mpo(config_kwargs, tol=1e-12):
    """ test mps.product_mpo """
    #
    # initializing identity MPO
    for sym, nl, nr in [('U1', (0,), (0,)), ('Z2', (0,), (0,)), ('dense', (), ())]:
        ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
        N = 8
        I = mps.product_mpo(ops.I(), N=8)

        assert pytest.approx(mps.vdot(I, I).item(), rel=tol) == 2 ** N
        assert (I @ I - I).norm() < tol
        assert I.virtual_leg('first').t == (nl,)
        assert I.virtual_leg('last').t == (nr,)
        assert all(I[site].s == (-1, 1, 1, -1) for site in I.sweep())

        with pytest.raises(yastn.YastnError,
                           match="Operator should have ndim = 2."):
            mps.product_mpo(ops.vec_z(val=1), N=8)


    H = mps.random_dense_mpo(N=5, D=4, d=3, **config_kwargs)
    assert len(H) == 5
    assert H.get_bond_dimensions() == (1, 4, 4, 4, 4, 1)
    assert (leg[0].D == (3,) and leg[1].D == (3,) for leg in H.get_physical_legs())
    assert config_kwargs["default_device"] in H[0].device
    assert config_kwargs["backend"] == H.config.backend.BACKEND_ID


def random_mps_spinless_fermions(config_kwargs, sym='Z2', N=10, D_total=16, n=1):
    """
    Generate random MPS of N sites, with bond dimension D_total,
    tensors with symmetry sym and total charge n.
    """
    #
    # load local operators
    #
    ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
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


def random_mpo_spinless_fermions(config_kwargs, N=10, D_total=16, sym='Z2'):
    """
    Generate random MPO of N sites, with bond dimension D_total and tensors with symmetry sym.
    """
    ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
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


def test_generate_random_mps(config_kwargs):
    """
    test states from random_mpo_spinless_fermions, random_mps_spinless_fermions
    """
    N = 10
    D_total = 16
    bds = (1,) + (D_total,) * (N - 1) + (1,)
    #
    for sym, nn in (('Z2', (0,)), ('Z2', (1,)), ('U1', (N // 2,))):
        psi = random_mps_spinless_fermions(config_kwargs, sym, N, D_total, nn)
        n0 = psi.config.sym.zero()
        leg = psi[psi.first].get_legs(axes=0)
        assert leg.t == (nn,) and leg.s == -1
        leg = psi[psi.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = psi.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])  # > D_total / 2 as randomness might not allow saturation.

        H = random_mpo_spinless_fermions(config_kwargs, N, D_total, sym)
        leg = H[H.first].get_legs(axes=0)
        assert leg.t == (n0,) and leg.s == -1
        leg = H[H.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = H.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])
    #
    # test 2 charge sectors; and passing extra parameters to gaussian_leg
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    N = 5
    I = mps.product_mpo(ops.I(), N)  # identity MPS
    n_profile = [(2, 2), (2, 1), (1, 1), (1, 1), (0.5, 0.5), (0, 0)]
    psi = mps.random_mps(I, n=n_profile, D_total=20, sigma=2, method='rand')
    phi = mps.random_mps(I, n=n_profile, D_total=20, sigma=2)
    assert psi.get_bond_charges_dimensions() != phi.get_bond_charges_dimensions()
    #
    with pytest.raises(yastn.YastnError,
                       match="Random mps is a zero state. Check parameters, or try running again in this is due to randomness of the initialization."):
        # Impossible number of particles for given N.
        random_mps_spinless_fermions(config_kwargs, 'U1', N=5, D_total=4, n=20)

    with pytest.raises(yastn.YastnError,
                       match=re.escape("Wrong number of elements in 'n'. It should be a charge on the first virtual leg, or list of charges on all len(I) + 1 virtual legs.")):
        n_profile = [5, 4, 3]
        random_mps_spinless_fermions(config_kwargs, 'U1', N=5, D_total=4, n=n_profile)
        # Profile should speficy the first, or all virtual charges.

    with pytest.raises(yastn.YastnError,
                       match="The charge on the last virtual leg should be zero."):
        n_profile = [0, 0.5, 1, 1, 1.5, 2]
        random_mps_spinless_fermions(config_kwargs, 'U1', N=5, D_total=4, n=n_profile)
        # correct profile is e.g., [2, 1.5, 1, 1, 1, 0]

    n_profile = [2, 1.5, 1, 1, 1, 0]
    random_mps_spinless_fermions(config_kwargs, 'U1', N=5, D_total=4, n=n_profile)

    with pytest.raises(yastn.YastnError,
                       match="Charge on the first virtual leg is not consistent with tensor symmetry."):
        n_profile = [2, 1.5, 1, 1, 1, 0]
        random_mps_spinless_fermions(config_kwargs, 'Z2', N=5, D_total=4, n=n_profile)
        # for Z2 symmetry, the charge on first virtual leg should be 0 or 1.

    with pytest.raises(yastn.YastnError,
                       match="Random mpo is a zero state. Check parameters, or try running again in this is due to randomness of the initialization."):
        ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
        ops.random_seed(seed=0)  # fix seed
        I = mps.product_mpo(ops.I(), 100)  # identity MPS
        mps.random_mpo(I, D_total=1, sigma=4, method='rand')


def test_mixed_dims_mpo_and_transpose(config_kwargs):
    N = 5
    config = yastn.make_config(sym='none', **config_kwargs)
    b1 = yastn.ones(config, s=(1, -1), D=(2, 3))
    b2 = yastn.ones(config, s=(1, -1), D=(3, 4))
    # reference mpo
    ref = mps.product_mpo([b1, b2], N)
    #
    # random mpo with physical dimensions matching ref
    #
    H = mps.random_mpo(ref, D_total=5, dtype='complex128')
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
    psi_bra = mps.random_mps(ref.conj().T, D_total=3, dtype='complex128')  #
    bra_legs = psi_bra.get_physical_legs()
    assert [leg.D for leg in bra_legs] == [(3,), (4,), (3,), (4,), (3,)]
    #
    # expectation value in positive operator H @ H.conj().T
    #
    tmp = mps.vdot(psi_bra, H.conj().T @ H, psi_bra)
    assert tmp.real > 0
    assert abs(tmp.imag) < 1e-7
    tmp = mps.vdot(psi_ket, H @ H.conj().T, psi_ket)
    assert tmp.real > 0
    assert abs(tmp.imag) < 1e-7
    assert (H.conj().T - H.H).norm() < 1e-12 * H.norm()
    assert (psi_bra.conj() - psi_bra.H).norm() < 1e-12 * psi_bra.norm()
    assert (psi_bra.T - psi_bra).norm() < 1e-12 * psi_bra.norm()
    #
    # final tests of transpose
    #
    assert psi_ket.T == psi_ket
    phys_dims = [(legs[0].D, legs[1].D) for legs in H.T.get_physical_legs()]
    assert phys_dims == [((3,), (2,)), ((4,), (3,)), ((3,), (2,)), ((4,), (3,)), ((3,), (2,))]


def test_MpsMpoOBC_properties(config_kwargs):
    """ Test reading MPS/MPO properties """
    N = 11
    ops = yastn.operators.Spin12(sym="dense", **config_kwargs)
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


def test_MpsMpoOBC_copy(config_kwargs):
    """ Initialize random mps of full tensors and checks copying. """
    N = 16
    ops = yastn.operators.Spin1(sym='Z3', **config_kwargs)
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


@pytest.mark.parametrize('N', [6, 1])
def test_MpsMpoOBC_to_tensor(config_kwargs, N):
    """ Test turning MPS/MPO into a single tensor and back. """
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)

    generate = mps.Generator(N=N, operators=ops)
    psi = generate.random_mps(D_total=8)
    tpsi = psi.to_tensor()
    assert tpsi.get_shape() == (2,) * N
    mpsi = psi.to_matrix()
    assert mpsi.get_shape() == (2 ** N,)

    op = generate.random_mpo(D_total=8)
    top = op.to_tensor()
    assert top.get_shape() == (2,) * (N * 2)
    mop = op.to_matrix()
    assert mop.get_shape() == (2 ** N, 2 ** N)

    for to in ['first', 'last']:
        psi2 = mps.mps_from_tensor(tpsi, canonize=to)
        assert (psi - psi2).norm() < 1e-10
        assert psi2.is_canonical(to=to)

    for to in ['first', 'last']:
        op2 = mps.mpo_from_tensor(top, canonize=to)
        assert (op - op2).norm() < 1e-10
        assert op2.is_canonical(to=to)

    op2 = mps.mpo_from_tensor(top)  # canonize = 'balance'
    assert (op - op2).norm() < 1e-10
    if N > 1:
        assert all(not op2.is_canonical(to=to) for to in ['first', 'last'])



def test_MpoPBC_to_tensor(config_kwargs):
    """ Test turning MPS/MPO into a single tensor and back. """
    config = yastn.make_config(syn='dense', **config_kwargs)
    op = mps.MpoPBC(N=3)
    op[0] = yastn.rand(config, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    op[1] = yastn.rand(config, s=(-1, 1, 1, -1), D=(4, 6, 3, 7))
    op[2] = yastn.rand(config, s=(-1, 1, 1, -1), D=(3, 8, 2, 9))
    top = op.to_tensor()
    assert top.shape == (3, 5, 6, 7, 8, 9)
    mop = op.to_matrix()
    assert mop.shape == (3 * 6 * 8, 5 * 7 * 9)

    op = mps.MpoPBC(N=1)
    op[0] = yastn.rand(config, s=(-1, 1, 1, -1), D=(2, 3, 2, 5))
    top = op.to_tensor()
    assert top.shape == (3, 5)
    mop = op.to_matrix()
    assert mop.shape == (3, 5)



if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
