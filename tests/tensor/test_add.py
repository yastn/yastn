import yamps.tensor as tensor
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest


def test_add0():
    a = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))

    c1 = a + 2 * b
    c2 = a.apxb(b, 2)
    assert pytest.approx(c1.norm_diff(c2)) == 0
    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)

    d = tensor.rand(settings=settings_full, isdiag=True, D=5)
    d1 = d.copy()
    e1 = 2 * d1 - (d + d)
    e2 = 2 * d - d1 - d1
    assert pytest.approx(e1.norm()) == 0
    assert pytest.approx(e2.norm()) == 0
    assert d.is_independent(d1)


def test_add1():
    a = tensor.rand(settings=settings_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = tensor.rand(settings=settings_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    c1 = a + 2 * b
    c2 = a.apxb(b, 2)
    assert pytest.approx(c1.norm_diff(c2)) == 0
    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)

    d = tensor.eye(settings=settings_U1, isdiag=True, t=1, D=5)
    d1 = tensor.eye(settings=settings_U1, isdiag=True, t=2, D=5)

    e1 = 2 * d + d1
    e2 = d - 2 * d1
    e3 = d1 - 2 * d
    assert pytest.approx(e1.norm()) == e2.norm() == e3.norm() == 5.0


def test_add2():
    a = tensor.rand(settings=settings_Z2_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 0, 1)),
                    D=((1, 2), (1, 2), (3, 4), (3, 4), (5, 6), (5, 6), (7, 8), (7, 8, 9)))

    b = tensor.rand(settings=settings_Z2_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1, 2)),
                    D=((1, 2), (1, 2), (3, 4), (3, 4), (5, 6), (5, 6), (7, 8), (7, 9, 10)))

    c1 = a - b * 2
    c2 = a.apxb(b, -2)

    assert pytest.approx(c1.norm_diff(c2)) == 0
    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)

    a1 = a.transpose((0, 1, 2, 3))
    assert a1.is_independent(a)


if __name__ == '__main__':
    test_add0()
    test_add1()
    test_add2()
