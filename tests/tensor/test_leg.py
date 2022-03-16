""" Test: fill_tensor (which is called in: rand, zeros, ones), to_numpy, match_legs """
import pytest
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_leg():
    leg = yast.Leg(s=1, t=(1, 0, 1), D=(2, 3, 4))
    legc = leg.conj()
    assert leg.s == -legc.s

    with pytest.raises(yast.YastError):
        _ = yast.Leg(s=2, t=(), D=())
        # Signature of Leg should be 1 or -1
    with pytest.raises(yast.YastError):
        _ = yast.Leg(s=1, t=(1, 0))
        # Charges t and their dimensions D do not match

    a = yast.ones(config=config_U1, s=(-1, 1, 1, 1),
                  t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                  D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    legs = [a.get_leg_structure2(n) for n in range(a.ndim)]
    print(legs)
    b = yast.rand2(config=config_U1, legs=legs)
    b.show_properties()


if __name__ == '__main__':
    test_leg()
