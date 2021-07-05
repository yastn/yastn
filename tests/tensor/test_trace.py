import numpy as np
import pytest
import yast
from .configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12


def test_trace_0():
    a = yast.ones(config=config_dense, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    x = a.to_numpy().transpose(0, 1, 3, 2)
    x = np.trace(np.reshape(x, (x.shape[0] * x.shape[1], -1)))
    b = a.trace(axes=(0, 2))
    c = b.trace(axes=(0, 1))
    assert pytest.approx(c.to_number(), rel=tol) == x

    a = yast.eye(config=config_dense, D=5)
    x1 = a.trace(axes=((), ()))
    x2 = a.trace()
    assert a.norm_diff(x1) < tol  # == 0.0
    assert pytest.approx(x2.to_number(), rel=tol) == 5


def test_trace_1():
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(2, 3), (4, 5), (6, 7), (6, 7), (4, 5), (2, 3)])
    x = a.to_numpy().transpose(0, 1, 2, 5, 4, 3)
    x = np.trace(np.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], -1)))
    b = a.trace(axes=(0, 5))
    b = b.trace(axes=(0, 3))
    b = b.trace(axes=(0, 1))
    assert pytest.approx(b.to_number(), rel=tol) == x

    a = yast.eye(config=config_U1, t=(1, 2, 3), D=(3, 4, 5))
    x1 = a.trace(axes=((), ()))
    x2 = a.trace()
    assert a.norm_diff(x1) < tol  # == 0.0
    assert pytest.approx(x2.to_number(), rel=tol) == 12


def test_trace_2():
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2_U1, s=(-1, -1, 1, 1),
                  t=[t1, t1, t1, t1],
                  D=[(6, 4, 9, 6), (20, 16, 25, 20), (20, 16, 25, 20), (6, 4, 9, 6)])
    x = a.to_numpy().transpose(0, 1, 3, 2)
    x = np.reshape(x, (x.shape[0] * x.shape[1], -1))
    x = np.trace(x)
    b = a.trace(axes=(0, 3))
    b = b.trace(axes=(0, 1))
    c = a.trace(axes=((0, 1), (3, 2)))
    assert pytest.approx(b.to_number(), rel=tol) == x
    assert pytest.approx(c.to_number(), rel=tol) == x

    a = yast.ones(config=config_Z2_U1, isdiag=True,
                  t=[[(0, 0), (1, 1), (2, 2)]],
                  D=[[2, 2, 2]])
    x1 = a.trace(axes=((), ()))
    x2 = a.trace()
    assert a.norm_diff(x1) < tol  # == 0.0
    assert pytest.approx(x2.to_number(), rel=tol) == 6


if __name__ == '__main__':
    test_trace_0()
    test_trace_1()
    test_trace_2()
