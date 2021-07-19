import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_add_0():
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='float64')
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='float64')

    c1 = a + 2 * b
    c2 = a.apxb(b, 2)

    assert c1.norm_diff(c2) < tol  # == 0.0
    assert yast.norm(c1 - c2) < tol  # == 0.0

    d = yast.rand(config=config_dense, isdiag=True, D=5, dtype='float64')
    d1 = d.copy()

    e1 = 2 * d1 - (d + d)
    e2 = 2 * d - d1 - d1

    assert e1.norm() < tol  # == 0.0
    assert e2.norm() < tol  # == 0.0

    assert a.are_independent(c1)
    assert a.are_independent(c2)
    assert b.are_independent(c1)
    assert b.are_independent(c2)
    assert d.are_independent(d1)


def test_add_1():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), dtype='float64')

    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), dtype='float64')

    c1 = a + 2 * b
    c2 = a.apxb(b, 2)
    assert c1.norm_diff(c2) < tol  # == 0.0

    d = yast.eye(config=config_U1, t=1, D=5)
    d1 = yast.eye(config=config_U1, t=2, D=5)

    e1 = 2 * d + d1
    e2 = d - 2 * d1
    e3 = d1 - 2 * d
    assert pytest.approx(e1.norm(), rel=tol) == 5
    assert pytest.approx(e2.norm(), rel=tol) == 5
    assert pytest.approx(e3.norm(), rel=tol) == 5

    assert a.are_independent(c1)
    assert a.are_independent(c2)
    assert b.are_independent(c1)
    assert b.are_independent(c2)


def test_add_2():
    a = yast.rand(config=config_Z2_U1, s=(-1, 1, 1, 1),
        t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 2), (1, -2), (1, 0), (1, 2)], [(0, 0), (0, 2)]],
        D=((1, 2, 2, 4), (2, 3), (2, 6, 3, 6, 9), (4, 7)), dtype='float64')
    b = yast.rand(config=config_Z2_U1, s=(-1, 1, 1, 1),
        t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)], [(0, 0)]],
        D=((1, 2, 2, 4), (2, 3), (2, 4, 6, 3, 6, 9), 4), dtype='float64')

    c1 = a - b * 2
    c2 = a.apxb(b, -2)

    assert pytest.approx(c1.norm_diff(c2), abs=tol) == 0

    leg_structures = {2: b.get_leg_structure(2), 3: a.get_leg_structure(3)}
    na = a.to_numpy(leg_structures)
    nb = b.to_numpy(leg_structures)
    nc = c1.to_numpy()

    assert pytest.approx(np.linalg.norm(nc - na + 2 * nb), abs=tol) == 0

    assert a.are_independent(c1)
    assert a.are_independent(c2)
    assert b.are_independent(c1)
    assert b.are_independent(c2)
    assert c1.is_consistent()
    assert c2.is_consistent()


def test_add_mismatch():
    """ handling pathological examples """
    a = yast.Tensor(config=config_U1, s=(1, -1, 1, -1))
    a.set_block(ts=(1, 1, 0, 0), Ds=(2, 2, 1, 1), val='rand')
    b = yast.Tensor(config=config_U1, s=(1, -1, 1, -1))
    b.set_block(ts=(1, 1, 1, 1), Ds=(1, 1, 1, 1), val='rand')
    c = a + b
    # c.is_consistent()


if __name__ == '__main__':
    test_add_0()
    test_add_1()
    test_add_2()
    test_add_mismatch()
