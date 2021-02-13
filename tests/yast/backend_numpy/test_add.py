import yamps.yast as yast
import config_dense_R
import config_U1_R
import config_Z2_U1_R
from math import isclose
import numpy as np

tol = 1e-12

def test_add_0():
    a = yast.rand(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = yast.rand(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))

    c1 = a + 2 * b
    c2 = a.apxb(b, 2)

    assert c1.norm_diff(c2) < tol  # == 0.0

    d = yast.rand(config=config_dense_R, isdiag=True, D=5)
    d1 = d.copy()

    e1 = 2 * d1 - (d + d)
    e2 = 2 * d - d1 - d1

    assert e1.norm() < tol  # == 0.0
    assert e2.norm() < tol  # == 0.0

    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)
    assert d.is_independent(d1)


def test_add_1():
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                    t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    c1 = a + 2 * b
    c2 = a.apxb(b, 2)
    assert c1.norm_diff(c2) < tol  # == 0.0

    d = yast.eye(config=config_U1_R, t=1, D=5)
    d1 = yast.eye(config=config_U1_R, t=2, D=5)

    e1 = 2 * d + d1
    e2 = d - 2 * d1
    e3 = d1 - 2 * d
    assert isclose(e1.norm(), 5, rel_tol=tol)
    assert isclose(e2.norm(), 5, rel_tol=tol)
    assert isclose(e3.norm(), 5, rel_tol=tol)

    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)


def test_add_2():
    a = yast.rand(config=config_Z2_U1_R, s=(-1, 1, 1, 1),
                    t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 2), (1, -2), (1, 0), (1, 2)], [(0, 0), (0, 2)]],
                    D=((1, 2, 2, 4), (2, 3), (2, 6, 3, 6, 9), (4, 7)))
    b = yast.rand(config=config_Z2_U1_R, s=(-1, 1, 1, 1),
                    t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)], [(0, 0)]],
                    D=((1, 2, 2, 4), (2, 3), (2, 4, 6, 3, 6, 9), 4))

    c1 = a - b * 2
    c2 = a.apxb(b, -2)

    assert c1.norm_diff(c2) < tol  # == 0.0

    leg_structures = {2: b.get_leg_structure(2), 3: a.get_leg_structure(3)}
    na = a.to_numpy(leg_structures)
    nb = b.to_numpy(leg_structures)
    nc = c1.to_numpy()

    assert isclose(np.linalg.norm(nc - na + 2*nb), 0, rel_tol=tol, abs_tol=tol)

    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)
    assert c1.is_consistent()
    assert c2.is_consistent()


if __name__ == '__main__':
    test_add_0()
    test_add_1()
    test_add_2()
