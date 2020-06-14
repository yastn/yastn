import yamps.tensor as tensor
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest
import numpy as np


def test_dot_diag0():
    a = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    b = tensor.rand(settings=settings_full, isdiag=True, D=5)
    b1 = b.diag()

    r1 = a.dot_diag(b, axis=1)
    r2 = b.dot_diag(a, axis=1)
    r3 = a.dot(b1, axes=(1, 1)).transpose((0, 3, 1, 2))

    assert pytest.approx(r1.norm_diff(r2)) == 0
    assert pytest.approx(r1.norm_diff(r3)) == 0

    a = tensor.randC(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    b = tensor.randC(settings=settings_full, s=(-1, 1), D=(7, 5))
    b = b.diag()  # 5x5 diagonal
    b1 = b.diag()

    r1 = a.dot_diag(b, axis=1, conj=(1, 0))
    r2 = b.dot_diag(a, axis=1, conj=(0, 1))
    r3 = a.dot_diag(b, axis=1, conj=(0, 1)).conj()
    r4 = b.dot_diag(a, axis=1, conj=(1, 0)).conj()
    r5 = b1.dot(a, axes=(0, 1), conj=(0, 1)).transpose((1, 0, 2, 3))

    assert pytest.approx(r1.norm_diff(r2)) == 0
    assert pytest.approx(r1.norm_diff(r3)) == 0
    assert pytest.approx(r1.norm_diff(r4)) == 0
    assert pytest.approx(r1.norm_diff(r5)) == 0


def test_dot_diag1():
    a = tensor.rand(settings=settings_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    b = tensor.rand(settings=settings_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    b1 = b.diag()

    r1 = a.dot_diag(b, axis=2)
    r2 = a.dot(b1, axes=(2, 1)).transpose((0, 1, 3, 2))
    assert pytest.approx(r1.norm_diff(r2)) == 0


def test_dot_diag2():
    a = tensor.rand(settings=settings_Z2_U1, s=(-1, -1, 1, 1),
                    t=[(0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2)],
                    D=[(2, 3), (3, 2), (4, 5), (5, 4), (4, 5), (5, 4), (2, 3), (3, 2)])
    b = tensor.eye(settings=settings_Z2_U1,
                   t=[[(0, 0), (0, 2), (1, 0), (1, 2), (1, 3)]],
                   D=[[6, 4, 9, 6, 5]])
    r1 = a.dot_diag(b, axis=0)
    assert pytest.approx(r1.norm_diff(a)) == 0

    r2 = b.dot_diag(b, axis=0)
    assert pytest.approx(r2.norm_diff(b)) == 0


if __name__ == '__main__':
    test_dot_diag0()
    test_dot_diag1()
    test_dot_diag2()
