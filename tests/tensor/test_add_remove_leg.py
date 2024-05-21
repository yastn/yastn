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
""" yastn.add_leg() yastn.remove_leg() """
import pytest
import yastn
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def _test_add_remove_leg(a):
    """ run a sequence of adding and removing axis operations. """
    b = yastn.add_leg(a)  # new axis with s=1 added as the last one; makes copy
    assert b.is_consistent()
    assert not yastn.are_independent(a, b)
    assert all(x == 0 for x in b.struct.n)  # tensor charge is set here to 0
    b = b.add_leg(axis=1, s=-1)
    b.is_consistent()

    c = b.remove_leg()  # removes last axis by default
    assert c.is_consistent()
    assert c.struct.n == a.struct.n
    c = c.remove_leg(axis=1)
    assert c.is_consistent()
    assert yastn.norm(a - c) < tol


def test_add_leg_basic():
    """ add_leg for various symmetries """
    # dense
    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    _test_add_remove_leg(a)

    # U1
    a = yastn.Tensor(config=config_U1, s=(-1, 1), n=-1)
    a.set_block(ts=(1, 0), Ds=(1, 1), val=1)  # creation operator
    b = yastn.Tensor(config=config_U1, s=(-1, 1), n=1)
    b.set_block(ts=(0, 1), Ds=(1, 1), val=1)  # annihilation operator
    _test_add_remove_leg(a)
    _test_add_remove_leg(b)

    ab1 = yastn.tensordot(a, b, axes=((), ()))  # outer product
    a = a.add_leg(s=1)
    b = b.add_leg(s=-1)
    ab2 = yastn.tensordot(a, b, axes=(2, 2))
    assert yastn.norm(ab1 - ab2) < tol

    # Z2xU1
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=[(0, 0), (1, 0), (0, 2), (1, 2)], D=(1, 2, 2, 4)),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, -2), (0, 2)], D=(2, 3)),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)], D=(2, 4, 6, 3, 6, 9))]
    a = yastn.rand(config=config_Z2xU1, n=(1, 2), legs=legs)
    assert a.get_shape() == (9, 3, 25)
    _test_add_remove_leg(a)

    # new axis with tensor charge set by hand.
    a = a.add_leg(s=-1, axis=0, t=(0, 2))
    assert a.struct.n == (1, 0)
    a.is_consistent()
    a = a.add_leg(s=1, t=(1, 0))
    assert a.struct.n == (0, 0)
    a.is_consistent()

    # mix adding/removing axes with fusions
    assert a.get_shape() == (1, 9, 3, 25, 1)
    a = a.fuse_legs(axes=((1, 0), 2, (3, 4)), mode='hard')
    assert a.get_shape() == (9, 3, 25)
    a = a.fuse_legs(axes=((0, 1), 2), mode='meta')
    assert a.get_shape() == (27, 25)

    a = a.add_leg(axis=1)
    assert a.get_shape() == (27, 1, 25)
    a = a.add_leg(axis=3)
    assert a.get_shape() == (27, 1, 25, 1)
    a = a.unfuse_legs(axes=0)
    assert a.get_shape() == (9, 3, 1, 25, 1)
    a = a.unfuse_legs(axes=(0, 3))
    assert a.get_shape(native=True) == (9, 1, 3, 1, 25, 1, 1)
    a.is_consistent()


def test_add_leg_fused():
    """ add_leg combined with fusion """
    leg0m = yastn.Leg(config_U1, s=-1, t=(-1,), D=(1,))
    leg0z = yastn.Leg(config_U1, s=1, t=(0,), D=(1,))
    leg0p = yastn.Leg(config_U1, s=-1, t=(1,), D=(1,))
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))

    legs = [leg0z, leg0m, leg0p, leg1.conj(), leg1]
    a = yastn.ones(config=config_U1, legs=legs)
    #
    # test fusions
    #
    for mode, mfs in [('hard', ((1,), (1,), (1,), (1,))),
                      ('meta', ((3, 2, 1, 1, 1), (1,), (3, 2, 1, 1, 1), (1,)))]:
        b = a.fuse_legs(axes=((0, 1), 2, 3, 4), mode=mode)
        b = b.fuse_legs(axes=((0, 1), 2, 3), mode=mode)
        legf = b.get_legs(axes=0)

        c = b.add_leg(axis=2, leg=legf)

        assert legf == c.get_legs(axes=2)
        assert c.mfs == mfs

        c = c.unfuse_legs(axes=(0, 2))
        c = c.unfuse_legs(axes=(0, 3))
        legc1 = c.get_legs()
        legc2 = legs[:4] + [leg0z, leg0m, leg0p,] + legs[4:]
        assert all(l1 == l2 for l1, l2 in zip(legc1, legc2))


def test_operators_chain():
    """
    Consider a sequence of operators "cp cp c c"
    add virtual legs connecting them, starting from the end
    """

    cdag = yastn.Tensor(config=config_U1, s=(1, -1), n=1)
    cdag.set_block(ts=(1, 0), Ds=(1, 1), val=1)
    c = yastn.Tensor(config=config_U1, s=(1, -1), n=-1)
    c.set_block(ts=(0, 1), Ds=(1, 1), val=1)

    nn = (0,) * len(c.n)
    o4 = yastn.add_leg(c, axis=-1, t=nn, s=-1)
    o4 = yastn.add_leg(o4, axis=0, s=1)
    nn = o4.get_legs(axes=0).t[0]

    o3 = yastn.add_leg(c, axis=-1, t=nn, s=-1)
    o3 = yastn.add_leg(o3, axis=0, s=1)
    nn = o3.get_legs(axes=0).t[0]

    o2 = yastn.add_leg(cdag, axis=-1, t=nn, s=-1)
    o2 = yastn.add_leg(o2, axis=0, s=1)
    nn = o2.get_legs(axes=0).t[0]

    o1 = yastn.add_leg(cdag, axis=-1, t=nn, s=-1)
    o1 = yastn.add_leg(o1, axis=0, s=1)
    nn = o1.get_legs(axes=0).t[0]
    assert nn == (0,) * len(c.n)

    T1 = yastn.ncon([cdag, cdag, c, c], [(-1, -5), (-2, -6), (-3 ,-7), (-4, -8)])
    T2 = yastn.ncon([o1, o2, o3, o4], [(4, -1, -5, 1), (1, -2, -6, 2), (2, -3 ,-7, 3), (3, -4, -8, 4)])
    assert yastn.norm(T1 - T2) < tol

    # special case when there are no blocks in the tensor
    a = yastn.Tensor(config=config_U1, s=(1, -1, 1, -1), n=1)
    a = a.remove_leg(axis=1)
    assert a.struct.s == (1, 1, -1)
    a = a.remove_leg(axis=1)
    assert a.struct.s == (1, -1)
    assert a.struct.n == (1,)


def test_add_leg_exceptions():
    """ handling exceptions in yastn.add_leg()"""
    leg = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    with pytest.raises(yastn.YastnError):
        a = yastn.eye(config=config_U1, legs=[leg, leg.conj()])
        a.add_leg(s=1)  # Cannot add axis to a diagonal tensor.
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, n=1, legs=[leg, leg.conj()])
        a.add_leg(s=1, t=(1, 0))  # len(t) does not match the number of symmetry charges.
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, n=1, legs=[leg, leg.conj()])
        a.add_leg(s=2)  # Signature of the new axis should be 1 or -1.
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, legs=[leg, leg.conj()])
        new_leg = yastn.Leg(config_U1, s=1, t=(-1,), D=(2,))
        a.add_leg(axis=1, leg=new_leg)  # Only the leg of dimension one can be added to the tensor.
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, legs=[leg, leg.conj()])
        new_leg = yastn.Leg(config_U1, s=1, t=(), D=())
        a.add_leg(axis=1, leg=new_leg)  # Only the leg of dimension one can be added to the tensor.
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, legs=[leg, leg.conj()])
        new_leg = yastn.Leg(config_U1, s=1, t=(-1, 0), D=(1, 1))
        a.add_leg(axis=1, leg=new_leg)  # Only the leg of dimension one can be added to the tensor.


def test_remove_leg_exceptions():
    """ handling exceptions in yastn.remove_leg()"""
    leg = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    with pytest.raises(yastn.YastnError):
        a = yastn.eye(config=config_U1, legs=[leg, leg.conj()])
        a.remove_leg(axis=1)  # Cannot remove axis to a diagonal tensor.
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, n=1, legs=[leg, leg.conj()])
        scalar = yastn.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))
        _ = scalar.remove_leg(axis=0)  # Cannot remove axis of a scalar tensor.
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, legs=[leg, leg.conj(), leg])
        a = a.fuse_legs(axes=((0, 1), 2), mode='meta')
        _ = a.remove_leg(axis=0)  # Axis to be removed cannot be fused.
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, legs=[leg, leg.conj(), leg, leg])
        a = a.fuse_legs(axes=((0, 1), 2, 3), mode='meta')
        a = a.fuse_legs(axes=(0, (1, 2)), mode='hard')
        _ = a.remove_leg(axis=0)  # Axis to be removed cannot be fused.
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, legs=[leg, leg.conj(), leg, leg])
        _ = a.remove_leg(axis=1)  # Axis to be removed must have single charge of dimension one.


if __name__ == '__main__':
    test_add_leg_basic()
    test_add_leg_fused()
    test_operators_chain()
    test_add_leg_exceptions()
    test_remove_leg_exceptions()
