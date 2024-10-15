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
try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg


tol = 1e-12

def test_double_peps_tensor():
    """ Generate a few lattices veryfing expected output of some functions. """
    ops = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)

    # single peps tensor
    leg0 = yastn.Leg(ops.config, s=-1, t=(-1, 0, 1), D=(1, 3, 1))
    leg1 = yastn.Leg(ops.config, s=1,  t=(-1, 0, 1), D=(1, 2, 1))
    leg2 = yastn.Leg(ops.config, s=1,  t=(-1, 0), D=(2, 1))
    leg3 = yastn.Leg(ops.config, s=-1, t=(0, 1), D=(1, 1))
    leg4 = ops.space()

    A = yastn.rand(ops.config, legs=[leg0, leg1, leg2, leg3, leg4])
    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    # we keep the legs [t l] [b r] fused for efficiency of higher symmetries

    # create double peps tensor
    T0123 = fpeps.DoublePepsTensor(bra=A, ket=A)
    T0123.print_properties()

    f0123 = T0123.fuse_layers()
    assert T0123.get_shape() == (25, 16, 9, 4)
    assert f0123.get_shape() == (25, 16, 9, 4)
    assert T0123.ndim == 4

    co0123 = T0123.copy()
    cl0123 = T0123.clone()
    for tmp in  [co0123, cl0123]:
        assert yastn.are_independent(tmp.ket, T0123.ket)
        assert yastn.are_independent(tmp.bra, T0123.bra)

    T1230 = T0123.transpose(axes=(1, 2, 3, 0))
    f1230 = T1230.fuse_layers()
    assert T1230.get_shape() == (16, 9, 4, 25)
    assert f1230.get_shape() == (16, 9, 4, 25)

    T2301 = T1230.transpose(axes=(1, 2, 3, 0))
    f2301 = T2301.fuse_layers()
    assert T2301.get_shape() == (9, 4, 25, 16)
    assert f2301.get_shape() == (9, 4, 25, 16)

    T3012 = T1230.transpose(axes=(2, 3, 0, 1))
    f3012 = T3012.fuse_layers()
    assert T3012.get_shape() == (4, 25, 16, 9)
    assert f3012.get_shape() == (4, 25, 16, 9)

    T0321 = T1230.transpose(axes=(3, 2, 1, 0))
    f0321 = T0321.fuse_layers()
    assert T0321.get_shape(axes=1) == 4
    assert f0321.get_shape(axes=1) == 4
    assert T0321.get_legs(axes=0).t == ((-2,), (-1,), (0,), (1,), (2,))
    assert f0321.get_legs(axes=0).t == ((-2,), (-1,), (0,), (1,), (2,))

    T3210 = T3012.transpose(axes=(0, 3, 2, 1))
    f3210 = T3210.fuse_layers()
    assert  T3210.get_shape() == (4, 9, 16, 25)
    assert  f3210.get_shape() == (4, 9, 16, 25)

    T2103 = T0321.transpose(axes=(2, 3, 0, 1))
    f2103 = T2103.fuse_layers()
    assert  T2103.get_shape() == (9, 16, 25, 4)
    assert  f2103.get_shape() == (9, 16, 25, 4)

    T1032 = T0123.transpose(axes=(1, 0, 3, 2))
    f1032 = T1032.fuse_layers()
    assert  T1032.get_shape() == (16, 25, 4, 9)
    assert  f1032.get_shape() == (16, 25, 4, 9)


    # prepare legs for boundary vectors to attach
    legf0, legf1, legf2, legf3 = T0123.get_legs()

    t01 = yastn.rand(ops.config, legs=[leg1, legf1.conj(), legf0.conj(), leg0])
    assert t01.get_shape() == (4, 16, 25, 5)
    tt01a = T0123._attach_01(t01)
    tt01b = T2301._attach_23(t01)
    tt01c = f0123._attach_01(t01)
    tt01d = f2301._attach_23(t01)
    tt01e = co0123._attach_01(t01)
    tt01f = cl0123._attach_01(t01)

    assert tt01a.get_shape() == (4, 9, 5, 4)
    assert all((tmp - tt01a).norm() < tol for tmp in [tt01b, tt01c, tt01d, tt01e, tt01f])

    t10 = yastn.rand(ops.config, legs=[leg1, legf0.conj(), legf1.conj(), leg0])
    tt10a = T1032._attach_01(t10)
    tt10b = T3210._attach_23(t10)
    tt10c = f1032._attach_01(t10)
    tt10d = f3210._attach_23(t10)
    assert tt10a.get_shape() == (4, 4, 5, 9)
    assert all((tmp - tt10a).norm() < tol for tmp in [tt10b, tt10c, tt10d])

    t23 = yastn.rand(ops.config, legs=[leg1, legf3.conj(), legf2.conj(), leg0])
    assert t23.get_shape() == (4, 4, 9, 5)
    tt23a = T0123._attach_23(t23)
    tt23b = T2301._attach_01(t23)
    tt23c = f0123._attach_23(t23)
    tt23d = f2301._attach_01(t23)
    assert tt23a.get_shape() == (4, 25, 5, 16)
    assert all((tmp - tt23a).norm() < tol for tmp in [tt23b, tt23c, tt23d])

    t32 = yastn.rand(ops.config, legs=[leg1, legf2.conj(), legf3.conj(), leg0])
    tt32a = T1032._attach_23(t32)
    tt32b = T3210._attach_01(t32)
    tt32c = f1032._attach_23(t32)
    tt32d = f3210._attach_01(t32)
    assert tt32a.get_shape() == (4, 16, 5, 25)
    assert all((tmp - tt32a).norm() < tol for tmp in [tt32b, tt32c, tt32d])

    t12 = yastn.rand(ops.config, legs=[leg1, legf2.conj(), legf1.conj(), leg0])
    assert t12.get_shape() == (4, 9, 16, 5)
    tt12a = T1230._attach_01(t12)
    tt12b = T3012._attach_23(t12)
    tt12c = T0123._attach_12(t12)
    tt12d = f1230._attach_01(t12)
    tt12e = f3012._attach_23(t12)
    tt12f = f0123._attach_12(t12)
    assert tt12a.get_shape() == (4, 4, 5, 25)
    assert all((tmp - tt12a).norm() < tol for tmp in [tt12b, tt12c, tt12d, tt12e, tt12f])

    t21 = yastn.rand(ops.config, legs=[leg1, legf1.conj(), legf2.conj(), leg0])
    tt21a = T2103._attach_01(t21)
    tt21b = T0321._attach_23(t21)
    tt21c = f2103._attach_01(t21)
    tt21d = f0321._attach_23(t21)
    assert tt21a.get_shape() == (4, 25, 5, 4)
    assert all((tmp - tt21a).norm() < tol for tmp in [tt21b, tt21c, tt21d])

    t30 = yastn.rand(ops.config, legs=[leg1, legf0.conj(), legf3.conj(), leg0])
    assert t30.get_shape() == (4, 25, 4, 5)
    tt30a = T3012._attach_01(t30)
    tt30b = T1230._attach_23(t30)
    tt30c = T0123._attach_30(t30)
    tt30d = f3012._attach_01(t30)
    tt30e = f1230._attach_23(t30)
    tt30f = f0123._attach_30(t30)
    assert tt30a.get_shape() == (4, 16, 5, 9)
    assert all((tmp - tt30a).norm() < tol for tmp in [tt30b, tt30c, tt30d, tt30e, tt30f])

    t03 = yastn.rand(ops.config, legs=[leg1, legf3.conj(), legf0.conj(), leg0])
    tt03a = T0321._attach_01(t03)
    tt03b = T2103._attach_23(t03)
    tt03c = f0321._attach_01(t03)
    tt03d = f2103._attach_23(t03)
    assert tt03a.get_shape() == (4, 9, 5, 16)
    assert all((tmp - tt03a).norm() < tol for tmp in [tt03b, tt03c, tt03d])

    with pytest.raises(yastn.YastnError):
        T0123.transpose(axes=(1, 0, 2, 3))
        # DoublePEPSTensor only supports permutations that retain legs' ordering.
    with pytest.raises(yastn.YastnError):
        fpeps.DoublePepsTensor(bra=A, ket=A, transpose=(1, 2, 0, 3))
        # DoublePEPSTensor only supports permutations that retain legs' ordering.
    with pytest.raises(yastn.YastnError):
        T1230._attach_12(t12)
        # Transpositions not supported by _attach_12
    with pytest.raises(yastn.YastnError):
        T1230._attach_30(t30)
        # Transpositions not supported by _attach_30


if __name__ == '__main__':
    test_double_peps_tensor()
