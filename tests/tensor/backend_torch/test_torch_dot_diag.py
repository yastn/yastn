import yamps.tensor as tensor
import settings_full_torch as settings_full
import settings_U1_torch as settings_U1
import settings_Z2_U1_torch as settings_Z2_U1
import settings_U1_U1_torch as settings_U1_U1
from math import isclose
import numpy as np

rel_tol=1.0e-14

def test_dot_diag0():
    a = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    b = tensor.rand(settings=settings_full, isdiag=True, D=5)
    b1 = b.diag()

    r1 = a.dot_diag(b, axis=1)
    r2 = a.dot(b1, axes=(1, 1)).transpose((0, 3, 1, 2))
    r3 = a.dot(b, axes=(1, 1)).transpose((0, 3, 1, 2))
    r4 = b.dot(a, axes=(1, 1)).transpose((1, 0, 2, 3))

    assert isclose(r1.norm_diff(r2),0,rel_tol=rel_tol)
    assert isclose(r1.norm_diff(r3),0,rel_tol=rel_tol)
    assert isclose(r1.norm_diff(r4),0,rel_tol=rel_tol)

    a = tensor.randR(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    b = tensor.randR(settings=settings_full, s=(-1, 1), D=(7, 5))
    b = b.diag()  # 5x5 diagonal
    b1 = b.diag()

    r1 = a.dot_diag(b, axis=1, conj=(1, 0))
    r2 = a.dot_diag(b, axis=1, conj=(0, 1)).conj()
    r3 = b1.dot(a, axes=(0, 1), conj=(0, 1)).transpose((1, 0, 2, 3))
    r4 = b.dot(a, axes=(0, 1), conj=(0, 1)).transpose((1, 0, 2, 3))
    r5 = a.dot(b, axes=(1, 0), conj=(1, 0)).transpose((0, 3, 1, 2))

    assert isclose(r1.norm_diff(r2),0,rel_tol=rel_tol)
    assert isclose(r1.norm_diff(r3),0,rel_tol=rel_tol)
    assert isclose(r1.norm_diff(r4),0,rel_tol=rel_tol)
    assert isclose(r1.norm_diff(r5),0,rel_tol=rel_tol)


def test_dot_diag1():
    a = tensor.rand(settings=settings_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    b = tensor.rand(settings=settings_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    b1 = b.diag()

    r1 = a.dot_diag(b, axis=2)
    r2 = a.dot(b1, axes=(2, 1)).transpose((0, 1, 3, 2))
    r3 = a.dot(b, axes=(2, 1)).transpose((0, 1, 3, 2))
    r4 = b.dot(a, axes=(1, 2)).transpose((1, 2, 0, 3))

    assert isclose(r1.norm_diff(r2),0,rel_tol=rel_tol)
    assert isclose(r1.norm_diff(r3),0,rel_tol=rel_tol)
    assert isclose(r1.norm_diff(r4),0,rel_tol=rel_tol)


def test_dot_diag2():
    a = tensor.rand(settings=settings_Z2_U1, s=(-1, -1, 1, 1),
                    t=[(0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2)],
                    D=[(2, 3), (3, 2), (4, 5), (5, 4), (1, 2), (2, 1), (1, 3), (3, 1)])
    b = tensor.eye(settings=settings_Z2_U1,
                   t=[[(0, 0), (0, 2), (1, 0), (1, 2), (1, 3)]],
                   D=[[6, 4, 9, 6, 5]])
    c = tensor.eye(settings=settings_Z2_U1,
                   t=[[(0, 2), (0, 3)]],
                   D=[[4, 3]])
    b1 = b.diag()

    r1 = a.dot_diag(b, axis=0)
    r2 = a.dot(b, axes=(0, 0))._transpose_local((3, 0, 1, 2))
    r3 = b.dot(a, axes=(0, 0))

    assert isclose(r1.norm_diff(a),0,rel_tol=rel_tol)
    assert isclose(r2.norm_diff(a),0,rel_tol=rel_tol)
    assert isclose(r3.norm_diff(a),0,rel_tol=rel_tol)
    assert not r1.isdiag and not r1.isdiag and not r1.isdiag

    r4 = b.dot_diag(c, axis=0)
    assert r4.isdiag

    r5 = r4.to_numpy()
    assert r5.shape == (4, 4)
    assert np.trace(r5) == 4.


if __name__ == '__main__':
    test_dot_diag0()
    test_dot_diag1()
    test_dot_diag2()
