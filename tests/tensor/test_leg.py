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
""" Test: yastn.Leg, get_legs() """
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_leg(config_kwargs):
    """ basic operations with yastn.Leg"""
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    assert str(leg) == 'Leg(sym=U1, s=1, t=((-1,), (0,), (1,)), D=(2, 3, 4), hist=o)'

    # flipping signature
    legc = leg.conj()
    assert leg.s == -legc.s
    assert not leg.is_fused()
    assert str(legc) == 'Leg(sym=U1, s=-1, t=((-1,), (0,), (1,)), D=(2, 3, 4), hist=o)'

    # order of provided charges (with corresponding bond dimensions) does not matter
    leg_unsorted = yastn.Leg(config_U1, s=1, t=(1, 0, -1), D=(4, 3, 2))
    assert leg_unsorted == leg
    assert hash(leg_unsorted) == hash(leg)

    legs = (yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(0, 4), D=(2, 4)),
            yastn.Leg(config_U1, s=1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(0,), D=(1,)))

    a = yastn.ones(config=config_U1, legs=legs)
    assert a.get_legs() == legs

    assert yastn.leg_union(legs[1], legs[2]) == yastn.Leg(config_U1, s=1, t=(-2, 0, 2, 4), D=(1, 2, 3, 4))

    assert a.get_legs(-1) == a.get_legs(3)


def test_random_leg(config_kwargs):
    #
    config_Z3 = yastn.make_config(sym='Z3', **config_kwargs)
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    #
    leg = yastn.random_leg(config_U1, n=0, s=1, D_total=1024)
    assert sum(leg.D) == 1024 and sum(t[0] * D for t, D in zip(leg.t, leg.D)) / 1024 < 0.2
    leg = yastn.random_leg(config_U1, s=1, D_total=1024)
    assert sum(leg.D) == 1024 and sum(t[0] * D for t, D in zip(leg.t, leg.D)) / 1024 < 0.2
    leg = yastn.random_leg(config_U1, n=1, s=1, D_total=1024)
    assert sum(leg.D) == 1024 and sum((t[0] - 1) * D for t, D in zip(leg.t, leg.D)) / 1024 < 0.2
    leg = yastn.random_leg(config_U1, s=1, n=0, D_total=1024, nonnegative=True)
    assert sum(leg.D) == 1024 and all(t[0] >= 0 for t in leg.t)
    leg = yastn.random_leg(config_Z3, s=1, n=2, D_total=1024)
    assert sum(leg.D) == 1024 and leg.t == ((0,), (1,), (2,))
    with pytest.raises(yastn.YastnError):
        yastn.random_leg(config_U1, n=(0, 0), D_total=1024)
        # len(n) is not consistent with provided symmetry.

    leg0 = yastn.Leg(config_U1, s=-1, t=(0, 1), D=(1, 1))
    # limit charges on random leg, to be consistent with provided legs (e.g. for creation of a tensor with 3 legs)
    leg = yastn.random_leg(config_U1, s=1, n=0, D_total=1024, legs=[leg0, leg0])
    assert sum(leg.D) == 1024 and leg.t == ((0,), (1,), (2,))
    leg = yastn.random_leg(config_U1, s=1, n=0, D_total=1024, legs=[leg0.conj(), leg0])
    assert sum(leg.D) == 1024 and leg.t == ((-1,), (0,), (1,))

    leg_dense = yastn.random_leg(yastn.make_config(), s=1, D_total=10)
    assert leg_dense.D == (10,) and leg_dense.t == ((),)

    leg0 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))
    leg1 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 4))

    leg = yastn.leg_product(leg0, leg1)
    print(leg)
    assert leg.is_fused()


def test_leg_meta_fusion(config_kwargs):
    """ test get_leg with meta-fused tensor"""
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    a = yastn.ones(config=config_U1, legs=[leg, leg, leg, leg.conj(), leg.conj()])
    assert a.get_legs([1, 3, 2, 4]) == (leg, leg.conj(), leg, leg.conj())

    a = a.fuse_legs(axes=((0, 1), (2, 3), 4), mode='meta')
    a = a.fuse_legs(axes=((0, 1), 2), mode='meta')
    legm = a.get_legs(0)
    assert legm.fusion == a.mfs[0] and legm.legs == (leg, leg, leg, leg.conj())
    assert legm.history() == 'm(m(oo)m(oo))'
    assert legm.is_fused()

    legt = a.get_legs((0, 1))
    assert legt[0] == legm
    assert legt[1] == leg.conj()

    b = yastn.ones(config=config_U1, legs=a.get_legs())
    assert yastn.norm(a - b) < tol

    a = yastn.ones(config=config_U1, s=(1, 1, 1, 1),
                  t=[(0, 1), (-1, 1), (-1, 0), (0,)],
                  D=[(2, 3), (1, 3), (1, 2), (2,)])
    legs = a.get_legs()

    a = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
    umlegs = yastn.leg_union(*a.get_legs())
    assert umlegs.legs[0] == yastn.leg_union(legs[0], legs[2])
    assert umlegs.legs[0] == yastn.leg_union(legs[1], legs[3])


def test_leg_hard_fusion(config_kwargs):
    """ legs with hard fusion """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(0, 2), D=(5, 4)),
            yastn.Leg(config_U1, s=-1, t=(0, 2), D=(2, 3)),
            yastn.Leg(config_U1, s=1, t=(0,), D=(5,))]
    a = yastn.ones(config=config_U1, legs=legs)
    af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    laf = af.get_legs()
    assert all(l.is_fused() for l in laf)

    bf = yastn.ones(config=config_U1, legs=laf)
    b = bf.unfuse_legs(axes=(0, 1))
    assert yastn.norm(a - b) < tol

    cf =  af.fuse_legs(axes=[(0, 1)], mode='meta')
    assert cf.get_legs(axes=0).history() == 'm(p(oo)p(oo))'


def test_leg_exceptions(config_kwargs):
    """ raising exceptions """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    # in initialization of new tensors with Leg
    legU1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    a = yastn.ones(config=config_U1, legs=[legU1, legU1.conj()])
    with pytest.raises(yastn.YastnError):
        b = a.fuse_legs(axes=[(0, 1)], mode='meta')
        yastn.eye(config_U1, legs=[b.get_legs(0)])
        # Diagonal tensor cannot be initialized with fused legs.
    with pytest.raises(yastn.YastnError):
        b = a.fuse_legs(axes=[(0, 1)], mode='hard')
        yastn.eye(config_U1, legs=[b.get_legs(0)])
        # Diagonal tensor cannot be initialized with fused legs.

    config_Z3 = yastn.make_config(sym='Z3', **config_kwargs)
    legZ3 = yastn.Leg(config_Z3, s=1, t=(0, 1, 2), D=(2, 3, 4))
    with pytest.raises(yastn.YastnError):
        a = yastn.ones(config=config_U1, legs=[legU1, legZ3])
        # Different symmetry of initialized tensor and some of the legs.

    # in initialization of Leg
    with pytest.raises(yastn.YastnError):
        _ = yastn.Leg(config_U1, s=2, t=(), D=())
        # Signature of Leg should be 1 or -1
    with pytest.raises(yastn.YastnError):
        _ = yastn.Leg(config_U1, s=1, t=(1, 0), D=(1,))
        # Number of provided charges and bond dimensions do not match sym.NSYM
    with pytest.raises(yastn.YastnError):
        _ = yastn.Leg(config_U1, s=1, t=(1,), D=(0,))
        # D should be a tuple of positive ints
    with pytest.raises(yastn.YastnError):
        _ = yastn.Leg(config_U1, s=1, t=(1,), D=(1.5,))
        # D should be a tuple of positive ints
    with pytest.raises(yastn.YastnError):
        _ = yastn.Leg(config_U1, s=1, t=(1.5,), D=(2,))
        # Charges should be ints
    with pytest.raises(yastn.YastnError):
        _ = yastn.Leg(config_U1, s=1, t=(1, 1), D=(2, 2))
        # Repeated charge index.
    with pytest.raises(yastn.YastnError):
        _ = yastn.Leg(config_Z3, s=1, t=(4,), D=(2,))
        # Provided charges are outside of the natural range for specified symmetry.

    # in leg_union
    leg_Z3 = yastn.Leg(config_Z3, s=1, t=(0, 1, 2), D=(2, 2, 2))
    leg = yastn.Leg(config_U1, s=1, t=(-1, 1), D=(2, 2))

    a = yastn.rand(config_U1, legs=[leg, leg, leg, leg])
    af1 = a.fuse_legs(axes=(0, (1, 2, 3)), mode='meta')
    af2 = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
    with pytest.raises(yastn.YastnError):
        yastn.leg_union(af1.get_legs(1), a.get_legs(1))
        # All arguments of leg_union should have consistent fusions.
    with pytest.raises(yastn.YastnError):
        yastn.leg_union(af1.get_legs(1), af2.get_legs(1))
        # Meta-fusions do not match.
    with pytest.raises(yastn.YastnError):
        yastn.leg_union(leg, leg_Z3)
        #  Provided legs have different symmetries.
    with pytest.raises(yastn.YastnError):
        yastn.leg_union(leg, leg.conj())
        # Provided legs have different signatures.
    with pytest.raises(yastn.YastnError):
        af1 = a.fuse_legs(axes=(0, (1, 2, 3)), mode='hard')
        af2 = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        yastn.leg_union(af1.get_legs(1), af2.get_legs(1))
        # Inconsistent numbers of hard-fused legs or sub-fusions order.
    with pytest.raises(yastn.YastnError):
        leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1), D=(2, 3))
        yastn.leg_union(leg, leg2)
        # Legs have inconsistent dimensions.
    with  pytest.raises(yastn.YastnError):
        b = yastn.rand(config_U1, legs=[leg, leg.conj(), leg, leg])
        af = a.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        bf = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        yastn.leg_union(af.get_legs(0), bf.get_legs(0))
        # Inconsistent signatures of fused legs.
    with  pytest.raises(yastn.YastnError):
        leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1), D=(2, 3))
        b = yastn.rand(config_U1, legs=[leg, leg2, leg, leg])
        af = a.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        bf = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        yastn.leg_union(af.get_legs(0), bf.get_legs(0))
        # Bond dimensions of fused legs do not match.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
