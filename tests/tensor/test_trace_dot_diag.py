import yamps.tensor as tensor
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest
import numpy as np


def test_trace_dot_diag0():
    a = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 5, 3, 5))
    b = tensor.rand(settings=settings_full, isdiag=True, D=5)
    b1 = b.diag()

    r1 = a.trace_dot_diag(b, axis1=1, axis2=3)
    r2 = a.dot(b1, axes=((1, 3), (1, 0)))
    r3 = a.dot(b, axes=((1, 3), (1, 0)))
    r4 = b.dot(a, axes=((1, 0), (3, 1)))

    assert pytest.approx(r1.norm_diff(r2)) == 0
    assert pytest.approx(r1.norm_diff(r3)) == 0
    assert pytest.approx(r1.norm_diff(r4)) == 0


def test_trace_dot_diag1():
    a = tensor.rand(settings=settings_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (7, 8, 9)))
    b = tensor.rand(settings=settings_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    b1 = b.diag()

    r1 = a.trace_dot_diag(b, axis1=3, axis2=2)
    r2 = a.dot(b1, axes=((2, 3), (1, 0)))
    r3 = a.dot(b, axes=((2, 3), (1, 0)))
    r4 = b.dot(a, axes=((1, 0), (3, 2)))

    assert pytest.approx(r1.norm_diff(r2)) == 0
    assert pytest.approx(r1.norm_diff(r3)) == 0
    assert pytest.approx(r1.norm_diff(r4)) == 0


def test_trace_dot_diag2():
    a = tensor.rand(settings=settings_Z2_U1, s=(-1, -1, 1, 1),
                    t=[(0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2)],
                    D=[(2, 3), (3, 2), (4, 5), (5, 4), (4, 5), (5, 4), (2, 3), (3, 2)])
    b = tensor.eye(settings=settings_Z2_U1,
                   t=[[(0, 0), (0, 2), (1, 0), (1, 2), (1, 3)]],
                   D=[[6, 4, 9, 6, 5]])
    c = tensor.eye(settings=settings_Z2_U1,
                   t=[[(0, 0), (0, 3)]],
                   D=[[6, 3]])
    b1 = b.diag()

    r1 = a.trace_dot_diag(b, axis1=0, axis2=3)
    r2 = a.dot(b1, axes=((0, 3), (0, 1)))
    r3 = a.trace(axes=(0, 3))
    r4 = a.dot(b, axes=((0, 3), (0, 1)))
    r5 = b.dot(a, axes=((1, 0), (3, 0)))

    assert pytest.approx(r1.norm_diff(r2)) == 0
    assert pytest.approx(r1.norm_diff(r3)) == 0
    assert pytest.approx(r1.norm_diff(r4)) == 0
    assert pytest.approx(r1.norm_diff(r5)) == 0

    r6 = b.trace_dot_diag(c)
    assert not r6.isdiag and r6.ndim == 0
    assert r6.to_number() == 6.


if __name__ == '__main__':
    # test_trace_dot_diag0()
    # test_trace_dot_diag1()
    test_trace_dot_diag2()
