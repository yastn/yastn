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
""" Test operation of peps.Lattice and peps.Peps that inherits Lattice"""
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.envs._env_auxlliary import *

tol = 1e-12  #pylint: disable=invalid-name


def create_double_peps_tensor(config_kwargs, dtype='float64'):
    """
    Create fermionic U1-ymmetric DoublePepsTensor of shape (16, 25, 36, 49)
    """
    config = yastn.make_config(sym='U1', fermionic=True, **config_kwargs)
    # single peps tensor
    leg0 = yastn.Leg(config, s=-1, t=(-1, 0, 1), D=(1, 2, 1))
    leg1 = yastn.Leg(config, s=1,  t=(-1, 0, 1), D=(2, 1, 2))
    leg2 = yastn.Leg(config, s=1,  t=(-1, 0, 1), D=(3, 1, 2))
    leg3 = yastn.Leg(config, s=-1, t=(-1, 0, 1), D=(2, 2, 3))
    leg4 = yastn.Leg(config, s=1,  t=(0, 1), D=(1, 1))
    A = yastn.rand(config, legs=[leg0, leg1, leg2, leg3, leg4], dtype=dtype)
    return fpeps.DoublePepsTensor(bra=A, ket=A)


def test_double_peps_tensor_copy(config_kwargs):
    T0 = create_double_peps_tensor(config_kwargs)

    for op in [None,
               yastn.eye(T0.config, legs=T0.bra.get_legs(axes=4))]:
        T0.op = op
        T0_conj = T0.conj()
        T0_copy = T0.copy()
        T0_clone = T0.clone()
        dicts = [T0.to_dict(level=level) for level in [0, 1, 2]]
        T0_dict = [fpeps.DoublePepsTensor.from_dict(d) for d in dicts]
        T0_split = [yastn.from_dict(yastn.combine_data_and_meta(*yastn.split_data_and_meta(d))) for d in dicts]

        for new, ind in zip([T0_conj, T0_copy, T0_clone, *T0_dict, *T0_split],
                            [False, True, True, False, False, True, False, False, True]):
            assert T0.ket.are_independent(new.ket, independent=ind)
            assert T0.bra.are_independent(new.bra, independent=ind)
            if op is not None:
                assert op.are_independent(new.op, independent=ind)
            assert T0.swaps is not new.swaps
            assert T0.swaps == new.swaps


def test_double_peps_tensor_basic(config_kwargs):
    """
    Test transpose, fuse_layers, get_legs, get_shape in DoublePepsTensor.
    """
    T0 = create_double_peps_tensor(config_kwargs)
    T0.print_properties()

    f0 = T0.fuse_layers()
    assert T0.get_shape() == (16, 25, 36, 49)
    assert T0.get_shape(axes=1) == 25
    assert f0.get_shape() == (16, 25, 36, 49)
    assert T0.shape == (16, 25, 36, 49)
    assert f0.shape == (16, 25, 36, 49)
    assert T0.ndim == 4


    allowed_transpose = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
                         (0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0))

    for axes1 in allowed_transpose:
        T1 = T0.transpose(axes=axes1)
        r1 = f0.transpose(axes=axes1)
        f1 = T1.fuse_layers()
        assert T1.get_shape() == f1.get_shape()
        assert T1.get_legs() == f1.get_legs()
        assert (f1 - r1).norm() < tol
        for axes2 in allowed_transpose:
            T2 = T1.transpose(axes=axes2)
            r2 = r1.transpose(axes=axes2)
            f2 = T2.fuse_layers()
            assert (f2 - r2).norm() < tol


def test_double_peps_tensor_tensordot(config_kwargs):
    """
    Test tensordot with DoublePepsTensor (fermionic tensors).
    """
    T0 = create_double_peps_tensor(config_kwargs, dtype='complex128')
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    assert T0.config.fermionic is True

    allowed_transpose = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
                         (0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0))

    for n_vec in [0, 1]:
        for op in [None, ops.c()]:
            T0.set_operator_(op)
            for charge, axes in [[(1,), []], [(1,), ['k1']], [(-1,), ['b4', 'k1', 'k2', 'k4']]]:
                T0.add_charge_swaps_(charge, axes)
                f0 = T0.fuse_layers()
                l0 = yastn.Leg(f0.config, s=1, t=(-1, 0, 1), D=(1, 2, 1))
                l3 = yastn.Leg(f0.config, s=-1, t=(-3, 0, 2), D=(1, 1, 2))
                for axes1 in allowed_transpose:
                    T1 = T0.transpose(axes=axes1)
                    r1 = f0.transpose(axes=axes1)
                    lfs = T1.get_legs()

                    t01 = yastn.rand(f0.config, legs=[l0, lfs[0].conj(), lfs[1].conj(), l3], n=n_vec)
                    t12 = yastn.rand(f0.config, legs=[lfs[1].conj(), lfs[2].conj(), l3], n=n_vec)
                    t32 = yastn.rand(f0.config, legs=[l0, lfs[3].conj(), lfs[2].conj()], n=n_vec)
                    t30 = yastn.rand(f0.config, legs=[l0, lfs[3].conj(), lfs[0].conj(), l3, l3], n=n_vec)

                    a01 = r1.tensordot(t01, axes=((0, 1), (1, 2)))
                    b01 = T1.tensordot(t01, axes=((0, 1), (1, 2)))
                    c01 = yastn.tensordot(T1, t01, axes=((1, 0), (2, 1)))
                    assert (a01 - b01).norm() < tol
                    assert (a01 - c01).norm() < tol

                    a12 = yastn.tensordot(r1, t12, axes=((1, 2), (0, 1)))
                    b12 = yastn.tensordot(T1, t12, axes=((1, 2), (0, 1)))
                    c21 = yastn.tensordot(t12, T1, axes=((0, 1), (1, 2)))
                    assert (a12 - b12).norm() < tol
                    assert (a12 - c21.transpose(axes=(1, 2, 0))).norm() < tol

                    a32 = r1.tensordot(t32, axes=((3, 2), (1, 2)))
                    b32 = T1.tensordot(t32, axes=((3, 2), (1, 2)))
                    c23 = yastn.tensordot(t32, T1, axes=((2, 1), (2, 3)))
                    assert (a32 - b32).norm() < tol
                    assert (a32 - c23.transpose(axes=(1, 2, 0))).norm() < tol

                    a30 = yastn.tensordot(t30, r1, axes=((2, 1), (0, 3)))
                    b30 = yastn.tensordot(t30, T1, axes=((2, 1), (0, 3)))
                    c03 = yastn.tensordot(T1, t30, axes=((3, 0), (1, 2)))
                    assert (a30 - b30).norm() < tol
                    assert (a30 - c03.transpose(axes=(2, 3, 4, 0, 1))).norm() < tol


def test_double_peps_tensor_raises(config_kwargs):
    """
    Test exceptions in DoublePepsTensor.
    """
    T0 = create_double_peps_tensor(config_kwargs, dtype='complex128')
    lfs = T0.get_legs()
    t01 = yastn.rand(T0.config, legs=[lfs[0], lfs[0].conj(), lfs[1].conj(), lfs[3]])

    with pytest.raises(yastn.YastnError,
                       match="DoublePEPSTensor only supports permutations that retain legs' ordering"):
        T0.transpose(axes=(1, 0, 2, 3))
    with pytest.raises(yastn.YastnError,
                       match="DoublePEPSTensor only supports permutations that retain legs' ordering"):
        fpeps.DoublePepsTensor(bra=T0.ket, ket=T0.bra, transpose=(1, 2, 0, 3))
    with pytest.raises(yastn.YastnError,
                       match="DoublePepTensor.tensordot only supports contraction of exactly 2 legs"):
        T0.tensordot(t01, axes=(1, 1))
    with pytest.raises(yastn.YastnError,
                       match="DoublePepTensor.tensordot repeated axis in axes"):
        T0.tensordot(t01, axes=((2, 2), (1, 1)))
    with pytest.raises(yastn.YastnError,
                       match="DoublePepTensor.tensordot axes outside of tensor ndim"):
        T0.tensordot(t01, axes=((3, 4), (1, 2)))
    with pytest.raises(yastn.YastnError,
                       match="DoublePepTensor.tensordot, 2 axes of self should be neighbouring"):
        T0.tensordot(t01, axes=((1, 3), (1, 2)))
    with pytest.raises(yastn.YastnError,
                       match="Elements of axes should be 'b0', 'b1', 'b2', 'b3', 'b4', 'k0', 'k1', 'k2', 'k3', 'k4'."):
        T0.add_charge_swaps_(charge=t01.n, axes='any')


def test_auxlliary_contractions(config_kwargs):
    T0 = create_double_peps_tensor(config_kwargs, dtype='complex128')
    assert T0.config.fermionic is True
    #
    legs = T0.get_legs()
    bd = [identity_boundary(T0.config, leg.conj()) for leg in legs]
    f0 = T0.fuse_layers()
    #
    ctl_0 = cor_tl(A_bra=T0.bra, A_ket=T0.ket)  # [b b'] [r r']
    ctl_1 = yastn.ncon([f0, bd[0], bd[1]], [(1, 2, -0, -1), (1,), (2,)])
    ctl_2 = yastn.ncon([f0.unfuse_legs(axes=(0, 1))], [(1, 1, 2, 2, -0, -1)])
    assert (ctl_0 - ctl_1).norm() < tol
    assert (ctl_0 - ctl_2).norm() < tol
    #
    ctr_0 = cor_tr(A_bra=T0.bra, A_ket=T0.ket)  # [l l'] [b b']
    ctr_1 = yastn.ncon([f0, bd[0], bd[3]], [(1, -0, -1, 2), (1,), (2,)])
    ctr_2 = yastn.ncon([f0.unfuse_legs(axes=(0, 3))], [(1, 1, -0, -1, 2, 2)])
    assert (ctr_0 - ctr_1).norm() < tol
    assert (ctr_0 - ctr_2).norm() < tol
    #
    cbl_0 = cor_bl(A_bra=T0.bra, A_ket=T0.ket)  # [r r'] [t t']
    cbl_1 = yastn.ncon([f0, bd[1], bd[2]], [(-1, 1, 2, -0), (1,), (2,)])
    cbl_2 = yastn.ncon([f0.unfuse_legs(axes=(1, 2))], [(-1, 1, 1, 2, 2, -0)])
    assert (cbl_0 - cbl_1).norm() < tol
    assert (cbl_0 - cbl_2).norm() < tol
    #
    cbr_0 = cor_br(A_bra=T0.bra, A_ket=T0.ket)  # [t t'] [l l']
    cbr_1 = yastn.ncon([f0, bd[2], bd[3]], [(-0, -1, 1, 2), (1,), (2,)])
    cbr_2 = yastn.ncon([f0.unfuse_legs(axes=(2, 3))], [(-0, -1, 1, 1, 2, 2)])
    assert (cbr_0 - cbr_1).norm() < tol
    assert (cbr_0 - cbr_2).norm() < tol
    #
    el_0 = edge_l(A_bra=T0.bra, A_ket=T0.ket)  # [b b'] [r r'] [t t']
    el_1 = yastn.ncon([f0, bd[1]], [(-2, 1, -0, -1), (1,),])
    el_2 = yastn.ncon([f0.unfuse_legs(axes=1)], [(-2, 1, 1, -0, -1)])
    assert (el_0 - el_1).norm() < tol
    assert (el_0 - el_2).norm() < tol
    #
    et_0 = edge_t(A_bra=T0.bra, A_ket=T0.ket)  # [l l'] [b b'] [r r']
    et_1 = yastn.ncon([f0, bd[0]], [(1, -0, -1, -2), (1,),])
    et_2 = yastn.ncon([f0.unfuse_legs(axes=0)], [(1, 1, -0, -1, -2)])
    assert (et_0 - et_1).norm() < tol
    assert (et_0 - et_2).norm() < tol
    #
    er_0 = edge_r(A_bra=T0.bra, A_ket=T0.ket)  # [t t'] [l l'] [b b']
    er_1 = yastn.ncon([f0, bd[3]], [(-0, -1, -2, 1), (1,),])
    er_2 = yastn.ncon([f0.unfuse_legs(axes=3)], [(-0, -1, -2, 1, 1)])
    assert (er_0 - er_1).norm() < tol
    assert (er_0 - er_2).norm() < tol
    #
    eb_0 = edge_b(A_bra=T0.bra, A_ket=T0.ket)  # [r r'] [t t'] [l l']
    eb_1 = yastn.ncon([f0, bd[2]], [(-1, -2, 1, -0), (1,),])
    eb_2 = yastn.ncon([f0.unfuse_legs(axes=2)], [(-1, -2, 1, 1, -0)])
    assert (eb_0 - eb_1).norm() < tol
    assert (eb_0 - eb_2).norm() < tol
    #
    ht_0 = hair_t(T0.ket)  # b' b
    ht_1 = yastn.ncon([f0.unfuse_legs(axes=(0, 1, 2, 3))], [(1, 1, 2, 2, -1, -0, 3, 3)])
    assert (ht_0 - ht_1).norm() < tol
    #
    hb_0 = hair_b(T0.ket)  # t' t
    hb_1 = yastn.ncon([f0.unfuse_legs(axes=(0, 1, 2, 3))], [(-1, -0, 1, 1, 2, 2, 3, 3)])
    assert (hb_0 - hb_1).norm() < tol
    #
    hl_0 = hair_l(T0.ket)  # r' r
    hl_1 = yastn.ncon([f0.unfuse_legs(axes=(0, 1, 2, 3))], [(1, 1, 2, 2, 3, 3, -1, -0)])
    assert (hl_0 - hl_1).norm() < tol
    #
    hr_0 = hair_r(T0.ket)  # l' l
    hr_1 = yastn.ncon([f0.unfuse_legs(axes=(0, 1, 2, 3))], [(1, 1, -1, -0, 2, 2, 3, 3)])
    assert (hr_0 - hr_1).norm() < tol


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
