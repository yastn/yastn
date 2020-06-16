import yamps.tensor as tensor
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest
import numpy as np


def test_transpose0():
    a = tensor.ones(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    a._transpose_local(axes=(1, 3, 2, 0))
    b = a.to_numpy()
    assert b.shape == (3, 5, 4, 2)


def test_transpose1():
    a = tensor.ones(settings=settings_U1, s=(-1, -1, -1, 1, 1, 1),
                    t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                    D=[(2, 3), (4, 5), (6, 7), (6, 7), (4, 5), (2, 3)])
    a._transpose_local(axes=(1, 2, 3, 0, 5, 4))
    b = a.to_numpy()
    assert b.shape == (9, 13, 13, 5, 5, 9)


def test_transpose2():
    a = tensor.ones(settings=settings_Z2_U1, s=(-1, -1, 1, 1),
                    t=[(0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2)],
                    D=[(2, 3), (3, 2), (4, 5), (5, 4), (5, 6), (6, 5), (1, 2), (2, 1)])
    a._transpose_local(axes=(1, 2, 3, 0))
    b = a.to_numpy()
    assert b.shape == (81, 121, 9, 25)


if __name__ == '__main__':
    test_transpose0()
    test_transpose1()
    test_transpose2()
