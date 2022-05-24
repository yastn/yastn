""" Test: fill_tensor (which is called in: rand, zeros, ones), to_numpy, match_legs """
import pytest
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_leg():
    leg = yast.Leg(config_U1, s=1, t=(1, 0, 1), D=(2, 3, 4))

    legc = leg.conj()
    assert leg.s == -legc.s

    a = yast.ones(config=config_U1, s=(-1, 1, 1, 1),
                  t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                  D=((1, 2, 3), (1, 2), (1, 2, 3), 1))

    legs = [a.get_leg(n) for n in range(a.ndim)]

    print(legs)

    b = yast.rand(config=config_U1, legs=legs)
    b.show_properties()


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


if __name__ == '__main__':
    test_leg()
    test_leg_exceptions()
