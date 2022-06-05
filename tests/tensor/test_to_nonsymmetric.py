import pytest
import yast
try:
    from .configs import config_dense, config_U1
except ImportError:
    from configs import config_dense, config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_to_nonsymmetric_0():
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))

    an = a.to_nonsymmetric()
    bn = b.to_nonsymmetric()
    assert pytest.approx(yast.vdot(an, bn).item(), rel=tol) == yast.vdot(a, b).item()
    assert pytest.approx(yast.vdot(a, bn).item(), rel=tol) == yast.vdot(a, b).item()
    # for dense to_nonsymetric should result in the same config
    assert an.are_independent(a)
    assert bn.are_independent(b)
    assert an.is_consistent()
    assert bn.is_consistent()


def test_to_nonsymmetric_1():
    legs = [yast.Leg(config_U1, s=-1, t=(-1, 1, 0), D=(1, 2, 3)),
            yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
            yast.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]

    a = yast.rand(config=config_U1, legs=legs)

    legs[0] = yast.Leg(config_U1, s=-1, t=(-2, 1, 2), D=(1, 2, 3))
    legs[2] = yast.Leg(config_U1, s=1, t=(1, 2, 3), D=(8, 9, 10))

    b = yast.rand(config=config_U1, legs=legs)

    an = a.to_nonsymmetric(legs=dict(enumerate(b.get_legs())))
    bn = b.to_nonsymmetric(legs=dict(enumerate(a.get_legs())))
    assert pytest.approx(yast.vdot(an, bn).item(), rel=tol) == yast.vdot(a, b).item()
    with pytest.raises(yast.YastError):
        a.vdot(bn)
        # Two tensors have different symmetry rules.
    assert an.are_independent(a)
    assert bn.are_independent(b)
    assert an.is_consistent()
    assert bn.is_consistent()

    with pytest.raises(yast.YastError):
        _ = a.to_nonsymmetric(legs={5: legs[0]})
        # Specified leg out of ndim


if __name__ == '__main__':
    test_to_nonsymmetric_0()
    test_to_nonsymmetric_1()
