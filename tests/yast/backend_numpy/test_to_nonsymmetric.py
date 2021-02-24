import yamps.yast as yast
import config_dense_R
import config_U1_R
import pytest

tol = 1e-12


def test_to_nonsymmetric_0():
    a = yast.rand(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = yast.rand(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))

    an = a.to_nonsymmetric()
    bn = a.to_nonsymmetric()
    an.scalar(bn)
    with pytest.raises(yast.YastError):
        a.scalar(bn)

def test_to_nonsymmetric_1():
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=((-2, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    an = a.to_nonsymmetric()
    bn = b.to_nonsymmetric()
    bn.scalar(an)
    with pytest.raises(yast.YastError):
        a.scalar(bn)


if __name__ == '__main__':
    test_to_nonsymmetric_0()
    test_to_nonsymmetric_1()
