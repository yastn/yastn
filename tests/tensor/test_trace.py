import yamps.tensor as tensor
import settings_full_R
import settings_U1_R
import settings_Z2_U1
import settings_U1_U1
import pytest
import numpy as np


def test_trace0():
    a = tensor.ones(settings=settings_full_R, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    x = a.to_numpy().transpose(0, 1, 3, 2)
    x = np.trace(np.reshape(x, (x.shape[0] * x.shape[1], -1)))
    b = a.trace(axes=(0, 2))
    c = b.trace(axes=(0, 1))
    assert pytest.approx(c.to_number()) == x

    a = tensor.eye(settings=settings_full_R, D=5)
    x1 = a.trace(axes=((), ()))
    x2 = a.trace()
    assert pytest.approx(a.norm_diff(x1)) == 0
    assert pytest.approx(x2.to_number()) == 5


def test_trace1():
    a = tensor.ones(settings=settings_U1_R, s=(-1, -1, -1, 1, 1, 1),
                    t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                    D=[(2, 3), (4, 5), (6, 7), (6, 7), (4, 5), (2, 3)])
    x = a.to_numpy().transpose(0, 1, 2, 5, 4, 3)
    x = np.trace(np.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], -1)))
    b = a.trace(axes=(0, 5))
    b = b.trace(axes=(0, 3))
    b = b.trace(axes=(0, 1))
    assert pytest.approx(b.to_number()) == x

    a = tensor.eye(settings=settings_U1_R, t=(1, 2, 3), D=(3, 4, 5))
    x1 = a.trace(axes=((), ()))
    x2 = a.trace()
    assert pytest.approx(a.norm_diff(x1)) == 0
    assert pytest.approx(x2.to_number()) == 12


def test_trace2():
    a = tensor.ones(settings=settings_Z2_U1, s=(-1, -1, 1, 1),
                    t=[(0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2)],
                    D=[(2, 3), (3, 2), (4, 5), (5, 4), (4, 5), (5, 4), (2, 3), (3, 2)])
    x = a.to_numpy().transpose(0, 1, 3, 2)
    x = np.reshape(x, (x.shape[0] * x.shape[1], -1))
    x = np.trace(x)
    b = a.trace(axes=(0, 3))
    b = b.trace(axes=(0, 1))
    c = a.trace(axes=((0, 1), (3, 2)))
    assert pytest.approx(b.to_number()) == x
    assert pytest.approx(c.to_number()) == x

    a = tensor.ones(settings=settings_Z2_U1, isdiag=True,
                    t=[[(0, 0), (1, 1), (2, 2)]],
                    D=[[2, 2, 2]])
    x1 = a.trace(axes=((), ()))
    x2 = a.trace()
    assert pytest.approx(a.norm_diff(x1)) == 0
    assert pytest.approx(x2.to_number()) == 6


if __name__ == '__main__':
    test_trace0()
    test_trace1()
    test_trace2()
