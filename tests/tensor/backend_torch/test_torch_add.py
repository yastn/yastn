import yamps.tensor as tensor
import settings_full_torch as settings_full
import settings_U1_torch as settings_U1
import settings_Z2_U1_torch as settings_Z2_U1
import settings_U1_U1_torch as settings_U1_U1
from math import isclose

rel_tol=1.0e-14

def test_add0():
    a = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))

    c1 = a + 2 * b
    c2 = a.apxb(b, 2)
    assert isclose(c1.norm_diff(c2),0,rel_tol=rel_tol)
    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)

    d = tensor.rand(settings=settings_full, isdiag=True, D=5)
    d1 = d.copy()
    e1 = 2 * d1 - (d + d)
    e2 = 2 * d - d1 - d1
    assert isclose(e1.norm(),0,rel_tol=rel_tol)
    assert isclose(e2.norm(),0,rel_tol=rel_tol)
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
    assert isclose(c1.norm_diff(c2),0,rel_tol=rel_tol)
    assert a.is_independent(c1)
    assert a.is_independent(c2)
    assert b.is_independent(c1)
    assert b.is_independent(c2)

    d = tensor.eye(settings=settings_U1, isdiag=True, t=1, D=5)
    d1 = tensor.eye(settings=settings_U1, isdiag=True, t=2, D=5)

    e1 = 2 * d + d1
    e2 = d - 2 * d1
    e3 = d1 - 2 * d
    assert isclose(e1.norm(),5,rel_tol=rel_tol)
    assert isclose(e2.norm(),5,rel_tol=rel_tol)
    assert isclose(e3.norm(),5,rel_tol=rel_tol)


def test_add2():
    a = tensor.rand(settings=settings_Z2_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 0, 1)),
                    D=((1, 2), (1, 2), (3, 4), (3, 4), (5, 6), (5, 6), (7, 8), (7, 8, 9)))

    b = tensor.rand(settings=settings_Z2_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1, 2)),
                    D=((1, 2), (1, 2), (3, 4), (3, 4), (5, 6), (5, 6), (7, 8), (7, 9, 10)))

    c1 = a - b * 2
    c2 = a.apxb(b, -2)

    assert isclose(c1.norm_diff(c2),0,rel_tol=rel_tol)
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
