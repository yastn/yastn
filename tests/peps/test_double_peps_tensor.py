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

tol = 1e-12  #pylint: disable=invalid-name


def create_double_peps_tensor(config_kwargs, dtype='float64'):
    """
    Create fermionic U1-ymmetric DoublePepsTensor of shape (25, 16, 9, 4)
    """
    config = yastn.make_config(sym='U1', fermionic=True, **config_kwargs)
    # single peps tensor
    leg0 = yastn.Leg(config, s=-1, t=(-1, 0, 1), D=(1, 3, 1))
    leg1 = yastn.Leg(config, s=1,  t=(-1, 0, 1), D=(1, 2, 1))
    leg2 = yastn.Leg(config, s=1,  t=(-1, 0), D=(2, 1))
    leg3 = yastn.Leg(config, s=-1, t=(0, 1), D=(1, 1))
    leg4 = yastn.Leg(config, s=1,  t=(0, 1), D=(1, 1))
    A = yastn.rand(config, legs=[leg0, leg1, leg2, leg3, leg4], dtype=dtype)
    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    return fpeps.DoublePepsTensor(bra=A, ket=A)


def test_double_peps_tensor_basic(config_kwargs):
    """
    Test transpose, fuse_layers, get_legs, get_shape in DoublePepsTensor.
    """
    T0 = create_double_peps_tensor(config_kwargs)
    T0.print_properties()

    f0 = T0.fuse_layers()
    assert T0.get_shape() == (25, 16, 9, 4)
    assert f0.get_shape() == (25, 16, 9, 4)
    assert T0.ndim == 4

    for tmp in  [T0.copy(), T0.clone()]:
        assert yastn.are_independent(tmp.ket, T0.ket)
        assert yastn.are_independent(tmp.bra, T0.bra)

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
    assert T0.config.fermionic is True
    f0 = T0.fuse_layers()

    allowed_transpose = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
                         (0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0))

    l0 = yastn.Leg(f0.config, s=1, t=(0, 1), D=(1, 2))
    l3 = yastn.Leg(f0.config, s=-1, t=(0, 2), D=(1, 1))

    for axes1 in allowed_transpose:
        T1 = T0.transpose(axes=axes1)
        r1 = f0.transpose(axes=axes1)
        lfs = T1.get_legs()

        t01 = yastn.rand(f0.config, legs=[l0, lfs[0].conj(), lfs[1].conj(), l3])
        t12 = yastn.rand(f0.config, legs=[lfs[1].conj(), lfs[2].conj(), l3])
        t32 = yastn.rand(f0.config, legs=[l0, lfs[3].conj(), lfs[2].conj()])
        t30 = yastn.rand(f0.config, legs=[l0, lfs[3].conj(), lfs[0].conj(), l3, l3])

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
                       match="DoublePepTensor.tensordot 2 axes of b should be neighbouring"):
        T0.tensordot(t01, axes=((1, 2), (1, 3)))
    with pytest.raises(yastn.YastnError,
                       match="DoublePepTensor.tensordot axis outside of tensor ndim"):
        T0.tensordot(t01, axes=((3, 4), (1, 2)))
    with pytest.raises(yastn.YastnError,
                       match="DoublePepTensor.tensordot, 2 axes of self should be neighbouring"):
        T0.tensordot(t01, axes=((1, 3), (1, 2)))


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
