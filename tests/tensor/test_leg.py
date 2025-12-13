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
import numpy as np
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

    assert yastn.legs_union(legs[1], legs[2]) == yastn.Leg(config_U1, s=1, t=(-2, 0, 2, 4), D=(1, 2, 3, 4))
    assert a.get_legs(-1) == a.get_legs(3)

    leg0 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))
    leg1 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 4))
    leg = yastn.leg_product(leg0, leg1)
    assert str(leg) == 'Leg(sym=U1, s=1, t=((0,), (1,), (2,)), D=(4, 14, 12), hist=p(oo))'
    assert leg.is_fused()
    leg0u, leg1u = yastn.undo_leg_product(leg)
    assert leg0u == leg0
    assert leg1u == leg1


def check_gaussian_distribution(leg, nc, sigma, D_total):
    assert sum(leg.D) == D_total
    ref = [np.exp((t[0] - nc) ** 2 / (-2 * sigma ** 2)) for t in leg.t]
    ref = np.array(ref) / sum(ref)
    dis = np.array(leg.D) / D_total
    assert np.linalg.norm(ref - dis) < 0.1


def test_gaussian_leg(config_kwargs):
    #
    config_Z3 = yastn.make_config(sym='Z3', **config_kwargs)
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    #
    config_U1.backend.random_seed(seed=0)
    for D_total in [1024, 2043]:
        for method in ['round', 'rand']:
            for sigma in [1, 2]:
                leg = yastn.gaussian_leg(config_U1, n=0, s=1, D_total=D_total, sigma=sigma, method=method)
                check_gaussian_distribution(leg, nc=0, sigma=sigma, D_total=D_total)
                #
                leg = yastn.gaussian_leg(config_U1, s=1, D_total=D_total, sigma=sigma, method=method)
                check_gaussian_distribution(leg, nc=0, sigma=sigma, D_total=D_total)
                #
                leg = yastn.gaussian_leg(config_U1, n=1, s=1, D_total=D_total, sigma=sigma, method=method)
                check_gaussian_distribution(leg, nc=1, sigma=sigma, D_total=D_total)
                #
                leg = yastn.gaussian_leg(config_U1, s=1, n=0, D_total=D_total, sigma=sigma, nonnegative=True, method=method)
                check_gaussian_distribution(leg, nc=0, sigma=sigma, D_total=D_total)
                assert all(t[0] >= 0 for t in leg.t)
                #
                leg = yastn.gaussian_leg(config_U1, s=1, n=2.5, D_total=D_total, sigma=sigma, nonnegative=True, method=method)
                check_gaussian_distribution(leg, nc=2.5, sigma=sigma, D_total=D_total)
                assert all(t[0] >= 0 for t in leg.t)
                #
                leg = yastn.gaussian_leg(config_Z3, s=1, n=2, D_total=D_total, sigma=sigma, method=method)
                check_gaussian_distribution(leg, nc=2, sigma=sigma, D_total=D_total)
                assert leg.t == ((0,), (1,), (2,))
                #
                with pytest.raises(yastn.YastnError):
                    yastn.gaussian_leg(config_U1, n=(0, 0), D_total=D_total)
                    # len(n) is not consistent with provided symmetry.
                #
                leg0 = yastn.Leg(config_U1, s=-1, t=(0, 1), D=(1, 1))
                # limit charges on random leg, to be consistent with provided legs (e.g. for creation of a tensor with 3 legs)
                leg = yastn.gaussian_leg(config_U1, s=1, n=0, D_total=D_total, legs=[leg0, leg0], sigma=sigma, method=method)
                check_gaussian_distribution(leg, nc=0, sigma=sigma, D_total=D_total)
                assert leg.t == ((0,), (1,), (2,))
                #
                leg = yastn.gaussian_leg(config_U1, s=1, n=1.5, D_total=D_total, legs=[leg0.conj(), leg0], sigma=sigma, method=method)
                check_gaussian_distribution(leg, nc=1.5, sigma=sigma, D_total=D_total)
                assert leg.t == ((-1,), (0,), (1,))
                #
                config_dense = yastn.make_config()
                leg_dense = yastn.gaussian_leg(config_dense, s=1, D_total=D_total, sigma=sigma, method=method)
                assert leg_dense.D == (D_total,) and leg_dense.t == ((),)


def test_leg_meta_fusion(config_kwargs):
    """ test get_leg with meta-fused tensor"""
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    a = yastn.ones(config=config_U1, legs=[leg, leg, leg, leg.conj(), leg.conj()])
    assert a.get_legs([1, 3, 2, 4]) == (leg, leg.conj(), leg, leg.conj())

    am = a.fuse_legs(axes=((0, 1), (2, 3), 4), mode='meta')
    amm = am.fuse_legs(axes=((0, 1), 2), mode='meta')
    legmm = amm.get_legs(0)

    assert legmm.mf == amm.mfs[0]
    assert legmm.legs == (leg, leg, leg, leg.conj())
    assert legmm.history() == 'm(m(oo)m(oo))'
    assert legmm.is_fused()
    assert 'LegMeta' in str(legmm)

    legt = amm.get_legs((0, 1))
    assert legt[0] == legmm
    assert legt[1] == leg.conj()

    bmm = yastn.ones(config=config_U1, legs=amm.get_legs())
    assert yastn.norm(amm - bmm) < tol

    a = yastn.ones(config=config_U1, s=(1, 1, 1, 1),
                  t=[(0, 1), (-1, 1), (-1, 0), (0,)],
                  D=[(2, 3), (1, 3), (1, 2), (2,)])
    legs = a.get_legs()

    a = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
    umlegs = yastn.legs_union(*a.get_legs())
    assert umlegs.legs[0] == yastn.legs_union(legs[0], legs[2])
    assert umlegs.legs[0] == yastn.legs_union(legs[1], legs[3])

    ll = a.get_legs()
    assert ll[0].are_consistent(ll[1], sgn=1)
    assert not ll[0].are_consistent(leg)


def test_leg_hard_fusion(config_kwargs):
    """ legs with hard fusion """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(0, 2), D=(5, 4)),
            yastn.Leg(config_U1, s=-1, t=(0, 2), D=(2, 3)),
            yastn.Leg(config_U1, s=1, t=(0,), D=(5,))]
    a = yastn.ones(config=config_U1, legs=legs)

    af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    legsf = af.get_legs()
    assert all(legf.is_fused() for legf in legsf)
    assert str(legsf[1]) == 'Leg(sym=U1, s=-1, t=((0,), (2,)), D=(10, 15), hist=p(oo))'
    assert(legsf[0].are_consistent(legsf[1], sgn=1))

    l2, l3 = yastn.undo_leg_product(legsf[1])
    assert l2 == legs[2]
    assert l3 == legs[3]

    bf = yastn.ones(config=config_U1, legs=legsf)
    b = bf.unfuse_legs(axes=(0, 1))
    assert yastn.norm(a - b) < tol

    cf =  af.fuse_legs(axes=[(0, 1)], mode='meta')
    legmf = cf.get_legs(axes=0)
    assert legmf.history() == 'm(p(oo)p(oo))'


def test_leg_exceptions(config_kwargs):
    """ raising exceptions """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    # in initialization of new tensors with Leg
    legU1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    a = yastn.ones(config=config_U1, legs=[legU1, legU1.conj()])

    with pytest.raises(yastn.YastnError,
                       match='Diagonal tensor cannot be initialized with fused legs.'):
        b = a.fuse_legs(axes=[(0, 1)], mode='meta')
        yastn.eye(config_U1, legs=[b.get_legs(0)])

    with pytest.raises(yastn.YastnError,
                       match='Diagonal tensor cannot be initialized with fused legs.'):
        b = a.fuse_legs(axes=[(0, 1)], mode='hard')
        yastn.eye(config_U1, legs=[b.get_legs(0)])

    config_Z3 = yastn.make_config(sym='Z3', **config_kwargs)
    legZ3 = yastn.Leg(config_Z3, s=1, t=(0, 1, 2), D=(2, 3, 4))
    with pytest.raises(yastn.YastnError,
                       match='Different symmetry of initialized tensor and some of the legs.'):
        a = yastn.ones(config=config_U1, legs=[legU1, legZ3])

    # in initialization of Leg
    with pytest.raises(yastn.YastnError,
                       match='Signature of Leg should be 1 or -1'):
        _ = yastn.Leg(config_U1, s=2, t=(), D=())
    with pytest.raises(yastn.YastnError,
                       match='Number of provided charges and bond dimensions do not match sym.NSYM'):
        _ = yastn.Leg(config_U1, s=1, t=(1, 0), D=(1,))
    with pytest.raises(yastn.YastnError,
                       match='D should be a tuple of positive ints.'):
        _ = yastn.Leg(config_U1, s=1, t=(1,), D=(0,))
    with pytest.raises(yastn.YastnError,
                       match='D should be a tuple of positive ints.'):
        _ = yastn.Leg(config_U1, s=1, t=(1,), D=(1.5,))
    with pytest.raises(yastn.YastnError,
                       match='Charges should be tuples of ints.'):
        _ = yastn.Leg(config_U1, s=1, t=(1.5,), D=(2,))
    with pytest.raises(yastn.YastnError,
                       match='Repeated charge index.'):
        _ = yastn.Leg(config_U1, s=1, t=(1, 1), D=(2, 2))
    with pytest.raises(yastn.YastnError,
                       match='Provided charges are outside of the natural range for specified symmetry.'):
        _ = yastn.Leg(config_Z3, s=1, t=(4,), D=(2,))

    # in legs_union
    leg_Z3 = yastn.Leg(config_Z3, s=1, t=(0, 1, 2), D=(2, 2, 2))
    leg = yastn.Leg(config_U1, s=1, t=(-1, 1), D=(2, 2))

    a = yastn.rand(config_U1, legs=[leg, leg, leg, leg])
    af1 = a.fuse_legs(axes=(0, (1, 2, 3)), mode='meta')
    af2 = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
    with pytest.raises(yastn.YastnError,
                       match='All arguments of legs_union should have consistent fusions.'):
        yastn.legs_union(af1.get_legs(1), a.get_legs(1))
    with pytest.raises(yastn.YastnError,
                       match='Meta-fusions do not match.'):
        yastn.legs_union(af1.get_legs(1), af2.get_legs(1))
    with pytest.raises(yastn.YastnError,
                       match='Provided legs have different symmetries.'):
        yastn.legs_union(leg, leg_Z3)
    with pytest.raises(yastn.YastnError,
                       match='Provided legs have different signatures.'):
        yastn.legs_union(leg, leg.conj())
    with pytest.raises(yastn.YastnError,
                       match='Inconsistent numbers of hard-fused legs or sub-fusions order.'):
        af1 = a.fuse_legs(axes=(0, (1, 2, 3)), mode='hard')
        af2 = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        yastn.legs_union(af1.get_legs(1), af2.get_legs(1))
    with pytest.raises(yastn.YastnError,
                       match='Legs have inconsistent dimensions.'):
        leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1), D=(2, 3))
        yastn.legs_union(leg, leg2)
    with  pytest.raises(yastn.YastnError,
                        match='Inconsistent signatures of fused legs.'):
        b = yastn.rand(config_U1, legs=[leg, leg.conj(), leg, leg])
        af = a.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        bf = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        yastn.legs_union(af.get_legs(0), bf.get_legs(0))
    with  pytest.raises(yastn.YastnError,
                        match='Bond dimensions of fused legs do not match.'):
        leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1), D=(2, 3))
        b = yastn.rand(config_U1, legs=[leg, leg2, leg, leg])
        af = a.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        bf = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        yastn.legs_union(af.get_legs(0), bf.get_legs(0))


if __name__ == '__main__':
   pytest.main([__file__, "-vs", "--durations=0"])
