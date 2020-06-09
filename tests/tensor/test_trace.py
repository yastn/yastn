from yamps.tensor import Tensor
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest
import numpy as np


def test_trace0():
    a = Tensor(settings=settings_full, s=(-1, 1, 1, -1))
    a.reset_tensor(D=(2, 5, 2, 5), val='ones')
    x = a.to_numpy().transpose(0, 1, 3, 2)
    x = np.trace(np.reshape(x, (x.shape[0] * x.shape[1], -1)))
    b = a.trace(axes=(0, 2))
    c = b.trace(axes=(0, 1))
    pytest.approx(c.to_number()) == x


def test_trace1():
    a = Tensor(settings=settings_U1, s=(-1, -1, -1, 1, 1, 1))
    a.reset_tensor(t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                   D=[(2, 3), (4, 5), (6, 7), (6, 7), (4, 5), (2, 3)],
                   val='ones')
    x = a.to_numpy().transpose(0, 1, 2, 5, 4, 3)
    x = np.trace(np.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], -1)))
    b = a.trace(axes=(0, 5))
    b = b.trace(axes=(0, 3))
    b = b.trace(axes=(0, 1))
    pytest.approx(b.to_number()) == x


def test_trace2():
    a = Tensor(settings=settings_Z2_U1, s=(-1, -1, 1, 1))
    a.reset_tensor(t=[(0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2)],
                   D=[(2, 3), (3, 2), (4, 5), (5, 4), (4, 5), (5, 4), (2, 3),  (3, 2)],
                   val='ones')
    x = a.to_numpy().transpose(0, 1, 3, 2)
    x = np.reshape(x, (x.shape[0] * x.shape[1], -1))
    x = np.trace(x)
    b = a.trace(axes=(0, 3))
    b = b.trace(axes=(0, 1))
    c = a.trace(axes=((0, 1), (3, 2)))
    assert pytest.approx(b.to_number()) == x
    assert pytest.approx(c.to_number()) == x


if __name__ == '__main__':
    test_trace0()
    test_trace1()
    test_trace2()
