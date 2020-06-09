from yamps.tensor import Tensor
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest


def test_add0():
    a = Tensor(settings=settings_full, s=(-1, 1, 1, -1))
    a.reset_tensor(D=(2, 3, 4, 5), val='rand')

    b = Tensor(settings=settings_full, s=(-1, 1, 1, -1))
    b.reset_tensor(D=(2, 3, 4, 5), val='rand')

    c1 = a + 2 * b
    c2 = a.apxb(b, 2)
    print('Norm diff = ', c1.norm_diff(c2))
    assert(abs(c1.norm_diff(c2) < 1e-8))
    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)


def test_add1():
    a = Tensor(settings=settings_U1, s=(-1, 1, 1, -1))
    a.reset_tensor(t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)), D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), val='rand')

    b = Tensor(settings=settings_U1, s=(-1, 1, 1, -1))
    b.reset_tensor(t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)), D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), val='rand')

    c1 = a + 2 * b
    c2 = a.apxb(b, 2)
    print('Norm diff = ', c1.norm_diff(c2))
    assert(abs(c1.norm_diff(c2) < 1e-8))
    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)


def test_add2():
    a = Tensor(settings=settings_Z2_U1, s=(-1, 1, 1, -1))
    a.reset_tensor(t=((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 0, 1)),
                   D=((1, 2), (1, 2), (3, 4), (3, 4), (5, 6), (5, 6), (7, 8), (7, 8, 9)),
                   val='rand')

    b = Tensor(settings=settings_Z2_U1, s=(-1, 1, 1, -1))
    b.reset_tensor(t=((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1, 2)),
                   D=((1, 2), (1, 2), (3, 4), (3, 4), (5, 6), (5, 6), (7, 8), (7, 9, 10)),
                   val='rand')

    c1 = a - b * 2
    c2 = a.apxb(b, -2)

    print('Norm diff = ', c1.norm_diff(c2))
    assert(abs(c1.norm_diff(c2) < 1e-8))
    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)


if __name__ == '__main__':
    test_add0()
    test_add1()
    test_add2()
