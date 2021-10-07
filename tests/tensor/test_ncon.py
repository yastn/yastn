""" Test yast.ncon """
import pytest
import yast
try:
    from .configs import config_dense, config_U1
except ImportError:
    from configs import config_dense, config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_ncon_0():
    a = yast.rand(s=(1, 1, 1), D=(20, 3, 1), config=config_dense, dtype='complex128')
    b = yast.rand(s=(1, 1, -1), D=(4, 2, 20), config=config_dense, dtype='complex128')
    c = yast.rand(s=(-1, 1, 1, -1), D=(20, 30, 10, 10), config=config_dense, dtype='complex128')
    d = yast.rand(s=(1, 1, 1, -1), D=(30, 10, 20, 10), config=config_dense, dtype='complex128')
    e = yast.rand(s=(1, 1), D=(3, 1), config=config_dense, dtype='complex128')
    f = yast.rand(s=(-1, -1), D=(2, 4), config=config_dense, dtype='complex128')
    g = yast.rand(s=(1,), D=(5,), config=config_dense, dtype='complex128')
    h = yast.rand(s=(-1,), D=(6,), config=config_dense, dtype='complex128')

    x = yast.ncon([a, b], [[1, -1, -3], [-0, -2, 1]])
    assert x.get_shape() == (4, 3, 2, 1)

    y = yast.ncon([a, b, c, d], [[4, -2, -0], [-3, -1, 5], [4, 3, 1, 1], [3, 2, 5, 2]], [0, 1, 0, 1])
    assert y.get_shape() == (1, 2, 3, 4)

    z1 = yast.ncon([e, f], [[-2, -0], [-1, -3]], [0, 1])
    z2 = yast.ncon([f, e], [[-1, -3], [-2, -0]], [1, 0])
    z3 = yast.ncon([e, f], [[-2, -0], [-1, -3]], [0, 0])
    z4 = yast.ncon([f, e], [[-1, -3], [-2, -0]], [1, 1])
    assert z1.get_shape() == (1, 2, 3, 4)
    assert z2.get_shape() == (1, 2, 3, 4)
    assert z3.get_shape() == (1, 2, 3, 4)
    assert z4.get_shape() == (1, 2, 3, 4)
    assert z1.norm_diff(z2) < tol  # == 0.0
    assert (z3 - z4.conj()).norm() < tol  # == 0.0

    y1 = yast.ncon([a, b, c, d, g, h], [[4, -2, -0], [-3, -1, 5], [4, 3, 1, 1], [3, 2, 5, 2], [-4], [-5]], [1, 0, 1, 0, 1, 0])
    assert y1.get_shape() == (1, 2, 3, 4, 5, 6)

    y2 = yast.ncon([a, g, a], [[1, 2, 3], [-0], [1, 2, 3]], [0, 0, 1])
    assert y2.get_shape() == (5,)

    y3 = yast.ncon([a, g, a, b, a], [[1, 2, 3], [-4], [1, 2, 3], [-3, -1, 4], [4, -2, -0]], [0, 0, 1, 0, 0])
    assert y3.get_shape() == (1, 2, 3, 4, 5)

    y4 = yast.ncon([a, a, b, b], [[1, 2, 3], [1, 2, 3], [6, 5, 4], [6, 5, 4]], [0, 1, 0, 1])
    y5 = yast.ncon([a, a, b, b], [[6, 5, 4], [6, 5, 4], [1, 3, 2], [1, 3, 2]], [0, 1, 0, 1])
    assert isinstance(y4.item(), complex)
    assert isinstance(y5.item(), complex)
    assert yast.norm_diff(y4, y5) / yast.norm(y4) < tol  # == 0.0
    assert pytest.approx((a.norm().item() ** 2) * (b.norm().item() ** 2), rel=tol) == y4.item()


def test_ncon_1():
    a = yast.rand(config=config_U1, s=[-1, 1, -1], n=0,
                  D=((20, 10), (3, 3), (1, 1)), t=((1, 0), (1, 0), (1, 0)))
    b = yast.rand(config=config_U1, s=[1, 1, 1], n=1,
                  D=((4, 4), (2, 2), (20, 10)), t=((1, 0), (1, 0), (1, 0)))
    c = yast.rand(config=config_U1, s=[1, 1, 1, -1], n=1,
                  D=((20, 10), (30, 20), (10, 5), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))
    d = yast.rand(config=config_U1, s=[1, 1, -1, -1], n=0,
                  D=((30, 20), (10, 5), (20, 10), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))

    e = yast.ncon([a, b], [[1, -1, -3], [-0, -2, 1]])
    assert e.get_shape() == (8, 6, 4, 2)
    assert e.is_consistent()
    h = yast.ncon([a, b, c, d], [[4, -2, -0], [-3, -1, 5], [4, 3, 1, 1], [3, 2, 5, 2]], [0, 1, 0, 1])
    assert h.get_shape() == (2, 4, 6, 8)
    assert h.is_consistent()
    g = yast.ncon([a, a, a, b], [[1, 2, 3], [1, 2, 3], [4, -2, -0], [-3, -1, 4]], [0, 1, 1, 1])
    assert g.get_shape() == (2, 4, 6, 8)
    assert g.is_consistent()


if __name__ == '__main__':
    test_ncon_0()
    test_ncon_1()
