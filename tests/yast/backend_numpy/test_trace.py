import yamps.yast as yast
import config_dense_R
import config_U1_R
import config_Z2_U1_R
from math import isclose
import numpy as np

tol = 1e-12

def test_trace_0():
    a = yast.ones(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    x = a.to_dense().transpose(0, 1, 3, 2)
    x = np.trace(np.reshape(x, (x.shape[0] * x.shape[1], -1)))
    b = a.trace(axes=(0, 2))
    c = b.trace(axes=(0, 1))
    assert isclose(c.to_number(), x, rel_tol=tol)

    a = yast.eye(config=config_dense_R, D=5)
    x1 = a.trace(axes=((), ()))
    x2 = a.trace()
    assert a.norm_diff(x1) < tol  # == 0.0
    assert isclose(x2.to_number(), 5, rel_tol=tol)


def test_trace_1():
    a = yast.ones(config=config_U1_R, s=(-1, -1, -1, 1, 1, 1),
                    t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                    D=[(2, 3), (4, 5), (6, 7), (6, 7), (4, 5), (2, 3)])
    x = a.to_dense().transpose(0, 1, 2, 5, 4, 3)
    x = np.trace(np.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], -1)))
    b = a.trace(axes=(0, 5))
    b = b.trace(axes=(0, 3))
    b = b.trace(axes=(0, 1))
    assert isclose(b.to_number(), x, rel_tol=tol)

    a = yast.eye(config=config_U1_R, t=(1, 2, 3), D=(3, 4, 5))
    x1 = a.trace(axes=((), ()))
    x2 = a.trace()
    assert a.norm_diff(x1) < tol  # == 0.0
    assert isclose(x2.to_number(), 12, rel_tol=tol)


def test_trace_2():
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2_U1_R, s=(-1, -1, 1, 1),
                    t=[t1, t1, t1, t1],
                    D=[(6, 4, 9, 6), (20, 16, 25, 20), (20, 16, 25, 20), (6, 4, 9, 6)])
    x = a.to_dense().transpose(0, 1, 3, 2)
    x = np.reshape(x, (x.shape[0] * x.shape[1], -1))
    x = np.trace(x)
    b = a.trace(axes=(0, 3))
    b = b.trace(axes=(0, 1))
    c = a.trace(axes=((0, 1), (3, 2)))
    assert isclose(b.to_number(), x, rel_tol=tol)
    assert isclose(c.to_number(), x, rel_tol=tol)

    a = yast.ones(config=config_Z2_U1_R, isdiag=True,
                    t=[[(0, 0), (1, 1), (2, 2)]],
                    D=[[2, 2, 2]])
    x1 = a.trace(axes=((), ()))
    x2 = a.trace()
    assert a.norm_diff(x1) < tol  # == 0.0
    assert isclose(x2.to_number(), 6, rel_tol=tol)


if __name__ == '__main__':
    test_trace_0()
    test_trace_1()
    test_trace_2()
