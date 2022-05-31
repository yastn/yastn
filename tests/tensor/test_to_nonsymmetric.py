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
    assert pytest.approx(an.vdot(bn).item(), rel=tol) == a.vdot(b).item()
    assert pytest.approx(a.vdot(bn).item(), rel=tol) == a.vdot(b).item()
    # for dense to_nonsymetric should result in the same config
    assert an.are_independent(a)
    assert bn.are_independent(b)
    assert an.is_consistent()
    assert bn.is_consistent()


def test_to_nonsymmetric_1():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-2, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    an = a.to_nonsymmetric(legs={0: b.get_legs(axis=0)})
    bn = b.to_nonsymmetric(legs={0: a.get_legs(axis=0)})
    assert pytest.approx(an.vdot(bn).item(), rel=tol) == a.vdot(b).item()
    with pytest.raises(yast.YastError):
        a.vdot(bn)
    assert an.are_independent(a)
    assert bn.are_independent(b)
    assert an.is_consistent()
    assert bn.is_consistent()


if __name__ == '__main__':
    test_to_nonsymmetric_0()
    test_to_nonsymmetric_1()
