""" Test: yast.Leg, get_legs() """
import pytest
import yast
try:
    from .configs import config_U1, config_Z3
except ImportError:
    from configs import config_U1, config_Z3

tol = 1e-12  #pylint: disable=invalid-name


def test_leg():
    """ basic operations with yast.Leg"""
    # U1
    leg = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    print(leg)

    # flipping signature
    legc = leg.conj()
    assert leg.s == -legc.s
    print(legc)

    # order of provided charges (with corresponding bond dimensions) does not matter
    leg_unsorted = yast.Leg(config_U1, s=1, t=(1, 0, -1), D=(4, 3, 2))
    assert leg_unsorted == leg
    assert hash(leg_unsorted) == hash(leg)

    legs = (yast.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yast.Leg(config_U1, s=1, t=(0, 4), D=(2, 4)),
            yast.Leg(config_U1, s=1, t=(-2, 0, 2), D=(1, 2, 3)),
            yast.Leg(config_U1, s=1, t=(0,), D=(1,)))

    a = yast.ones(config=config_U1, legs=legs)
    assert a.get_legs() == legs

    assert yast.leg_union(legs[1], legs[2]) == yast.Leg(config_U1, s=1, t=(-2, 0, 2, 4), D=(1, 2, 3, 4))

    assert a.get_legs(-1) == a.get_legs(3)


def test_leg_meta_fusion():
    """ test get_leg with meta-fused tensor"""
    # U1
    leg = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    a = yast.ones(config=config_U1, legs=[leg, leg, leg, leg.conj(), leg.conj()])
    assert a.get_legs([1, 3, 2, 4]) == (leg, leg.conj(), leg, leg.conj())

    a = a.fuse_legs(axes=((0, 1), (2, 3), 4), mode='meta')
    a = a.fuse_legs(axes=((0, 1), 2), mode='meta')
    legm = a.get_legs(0)
    assert legm.fusion == a.mfs[0] and legm.legs == (leg, leg, leg, leg.conj())
    assert legm.history() == 'm(m(oo)m(oo))'

    legt = a.get_legs((0, 1))
    assert legt[0] == legm
    assert legt[1] == leg.conj()

    b = yast.ones(config=config_U1, legs=a.get_legs())
    assert yast.norm(a - b) < tol

    a = yast.ones(config=config_U1, s=(1, 1, 1, 1),
                  t=[(0, 1), (-1, 1), (-1, 0), (0,)],
                  D=[(2, 3), (1, 3), (1, 2), (2,)])
    legs = a.get_legs()

    a = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
    umlegs = yast.leg_union(*a.get_legs())
    assert umlegs.legs[0] == yast.leg_union(legs[0], legs[2])
    assert umlegs.legs[0] == yast.leg_union(legs[1], legs[3])


def test_leg_hard_fusion():
    """ legs with hard fusion """
    legs = [yast.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yast.Leg(config_U1, s=1, t=(0, 2), D=(5, 4)),
            yast.Leg(config_U1, s=-1, t=(0, 2), D=(2, 3)),
            yast.Leg(config_U1, s=1, t=(0,), D=(5,))]
    a = yast.ones(config=config_U1, legs=legs)
    af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')

    bf = yast.ones(config=config_U1, legs=af.get_legs())
    b = bf.unfuse_legs(axes=(0, 1))
    assert yast.norm(a - b) < tol

    cf =  af.fuse_legs(axes=[(0, 1)], mode='meta')
    assert cf.get_legs(axis=0).history() == 'm(p(oo)p(oo))'


def test_leg_exceptions():
    """ raising exceptions """
    # in initialization of new tensors with Leg
    legU1 = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    a = yast.ones(config=config_U1, legs=[legU1, legU1.conj()])
    with pytest.raises(yast.YastError):
        b = a.fuse_legs(axes=[(0, 1)], mode='meta')
        yast.eye(config_U1, legs=[b.get_legs(0)])
        # Diagonal tensor cannot be initialized with fused legs.
    with pytest.raises(yast.YastError):
        b = a.fuse_legs(axes=[(0, 1)], mode='hard')
        yast.eye(config_U1, legs=[b.get_legs(0)])
        # Diagonal tensor cannot be initialized with fused legs.

    legZ3 = yast.Leg(config_Z3, s=1, t=(0, 1, 2), D=(2, 3, 4))
    with pytest.raises(yast.YastError):
        a = yast.ones(config=config_U1, legs=[legU1, legZ3])
        # Different symmetry of initialized tensor and some of the legs.

    # in initialization of Leg
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config_U1, s=2, t=(), D=())
        # Signature of Leg should be 1 or -1
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config_U1, s=1, t=(1, 0), D=(1,))
        # Number of provided charges and bond dimensions do not match sym.NSYM
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config_U1, s=1, t=(1,), D=(0,))
        # D should be a tuple of positive ints
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config_U1, s=1, t=(1,), D=(1.5,))
        # D should be a tuple of positive ints
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config_U1, s=1, t=(1.5,), D=(2,))
        # Charges should be ints
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config_U1, s=1, t=(1, 1), D=(2, 2))
        # Repeated charge index.
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config_Z3, s=1, t=(4,), D=(2,))
        # Provided charges are outside of the natural range for specified symmetry.

    # in leg_union
    leg_Z3 = yast.Leg(config_Z3, s=1, t=(0, 1, 2), D=(2, 2, 2))
    leg = yast.Leg(config_U1, s=1, t=(-1, 1), D=(2, 2))

    a = yast.rand(config_U1, legs=[leg, leg, leg, leg])
    af1 = a.fuse_legs(axes=(0, (1, 2, 3)), mode='meta')
    af2 = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
    with pytest.raises(yast.YastError):
        yast.leg_union(af1.get_legs(1), a.get_legs(1))
        # All arguments of leg_union should have consistent fusions.
    with pytest.raises(yast.YastError):
        yast.leg_union(af1.get_legs(1), af2.get_legs(1))
        # Meta-fusions do not match.
    with pytest.raises(yast.YastError):
        yast.leg_union(leg, leg_Z3)
        #  Provided legs have different symmetries.
    with pytest.raises(yast.YastError):
        yast.leg_union(leg, leg.conj())
        # Provided legs have different signatures.
    with pytest.raises(yast.YastError):
        af1 = a.fuse_legs(axes=(0, (1, 2, 3)), mode='hard')
        af2 = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        yast.leg_union(af1.get_legs(1), af2.get_legs(1))
        # Inconsistent numbers of hard-fused legs or sub-fusions order.
    with pytest.raises(yast.YastError):
        leg2 = yast.Leg(config_U1, s=1, t=(-1, 1), D=(2, 3))
        yast.leg_union(leg, leg2)
        # Legs have inconsistent dimensions.
    with  pytest.raises(yast.YastError):
        b = yast.rand(config_U1, legs=[leg, leg.conj(), leg, leg])
        af = a.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        bf = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        yast.leg_union(af.get_legs(0), bf.get_legs(0))
        # Inconsistent signatures of fused legs.
    with  pytest.raises(yast.YastError):
        leg2 = yast.Leg(config_U1, s=1, t=(-1, 1), D=(2, 3))
        b = yast.rand(config_U1, legs=[leg, leg2, leg, leg])
        af = a.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        bf = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        yast.leg_union(af.get_legs(0), bf.get_legs(0))
        # Bond dimensions of fused legs do not match.


if __name__ == '__main__':
    test_leg()
    test_leg_meta_fusion()
    test_leg_hard_fusion()
    test_leg_exceptions()
