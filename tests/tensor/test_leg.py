""" Test: yast.Leg, get_legs() """
import pytest
import yast
try:
    from .configs import config_U1, config_Z3
except ImportError:
    from configs import config_U1, config_Z3

tol = 1e-12  #pylint: disable=invalid-name


def test_leg():
    leg = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))

    print(repr(leg.sym))
    print(leg.sym)
    print(str(leg.sym))


    # leg0 = yast.Legtest(sym=config_U1.sym, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    # print(leg0)
    # print(hash(leg))
    # print(hash(leg0))
    # leg1 = yast.Legtest(sym=config_U1.sym, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    # print(hash(leg1))

    # leg2 = yast.Legtest(sym=config_U1.sym, s=1, t=(1, 0, -1), D=(4, 3, 2))
    # print(hash(leg2))

    # leg3 = yast.Legtest(sym=config_U1.sym, s=1, t=(1, 0, -1), D=(4, 3, 2), to_verify=False)
    # print(hash(leg3))

    # assert leg2 == leg1

    # # flipping signature
    # legc = leg.conj()
    # assert leg.s == -legc.s

    # # order of provided charges (with corresponding bond dimensions) does not matter
    # leg_unsorted = yast.Leg(config_U1, s=1, t=(1, 0, -1), D=(4, 3, 2))
    # assert leg_unsorted == leg

    print(leg)

    legs = [yast.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yast.Leg(config_U1, s=1, t=(0, 2), D=(1, 2)),
            yast.Leg(config_U1, s=1, t=(-2, 0, 2), D=(1, 2, 3)),
            yast.Leg(config_U1, s=1, t=(0,), D=(1,))]

    a = yast.ones(config=config_U1, legs=legs)
    assert all(a.get_legs(n) == legs[n] for n in range(a.ndim))


def test_leg_meta():
    """ test get_leg with meta-fused tensor"""
    leg = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    a = yast.ones(config=config_U1, legs=[leg, leg, leg, leg.conj(), leg.conj()])
    assert a.get_legs([1, 3, 2, 4]) == (leg, leg.conj(), leg, leg.conj())

    a = a.fuse_legs(axes=((0, 1), (2, 3), 4), mode='meta')
    a = a.fuse_legs(axes=((0, 1), 2), mode='meta')
    legm = a.get_legs(0)
    assert legm.mf == a.mfs[0] and legm.legs == (leg, leg, leg, leg.conj())
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


def test_leg_initialization_exceptions():
    legU1 = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))

    a = yast.ones(config=config_U1, legs=[legU1, legU1.conj()])
    with pytest.raises(yast.YastError):
        b = a.fuse_legs(axes=[(0, 1)], mode='meta')
        yast.eye(config_U1, legs=[b.get_legs(0)])

    legZ3 = yast.Leg(config_Z3, s=1, t=(0, 1, 2), D=(2, 3, 4))
    with pytest.raises(yast.YastError):
        a = yast.ones(config=config_U1, legs=[legU1, legZ3])
        # Different symmetry of initialized tensor and some of the legs.




def test_leg_exceptions():
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config=config_U1, s=2, t=(), D=())
        # Signature of Leg should be 1 or -1
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config=config_U1, s=1, t=(1, 0), D=(1,))
        # Number of provided charges and bond dimensions do not match sym.NSYM
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config=config_U1, s=1, t=(1,), D=(0,))
        # D should be a tuple of positive ints
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config=config_U1, s=1, t=(1,), D=(1.5,))
        # D should be a tuple of positive ints
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config=config_U1, s=1, t=(1.5,), D=(2,))
        # Charges should be ints
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config=config_U1, s=1, t=(1, 1), D=(2, 2))
        # Repeated charge index.
    with pytest.raises(yast.YastError):
        _ = yast.Leg(config=config_Z3, s=1, t=(4,), D=(2,))
        # Provided charges are outside of the natural range for specified symmetry.


if __name__ == '__main__':
    test_leg()
    test_leg_meta()
    test_leg_exceptions()
    test_leg_initialization_exceptions()