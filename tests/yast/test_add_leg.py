from context import yast
from context import config_dense, config_U1, config_Z2_U1
import pytest
import numpy as np

tol = 1e-12


def test_aux_0():
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    c = yast.add_leg(a)
    c.is_consistent()
    assert a.are_independent(c)
    a.add_leg(inplace=True)
    assert yast.norm_diff(a, c) < tol
    assert a.are_independent(c)


def test_aux_1():
    a = yast.Tensor(config=config_U1, s=(-1, 1), n=-1)
    a.set_block(ts=(1, 0), Ds=(1, 1), val=1)

    b = yast.Tensor(config=config_U1, s=(-1, 1), n=1)
    b.set_block(ts=(0, 1), Ds=(1, 1), val=1)

    ab1 = yast.tensordot(a, b, axes=((), ()))

    yast.add_leg(a, s=1, inplace='True')
    yast.add_leg(b, s=-1, inplace='True')
    a.is_consistent()
    b.is_consistent()

    ab2 = yast.tensordot(a, b, axes=(2, 2))
    assert yast.norm_diff(ab1, ab2) < tol


def test_aux_2():
    a = yast.rand(config=config_Z2_U1, s=(-1, 1, 1), n=(1, 2),
                  t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 2), (1, -2), (1, 0), (1, 2)]],
                  D=((1, 2, 2, 4), (2, 3), (2, 6, 3, 6, 9)), dtype='float64')
    b = yast.rand(config=config_Z2_U1, s=(-1, 1, 1), n=(1, 0),
                  t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)]],
                  D=((1, 2, 2, 4), (2, 3), (2, 4, 6, 3, 6, 9)), dtype='float64')

    a.add_leg(s=1, t=(1, 0), inplace=True)
    a.add_leg(s=-1, axis=0, t=(0, 2), inplace=True)
    assert a.n == (0, 0)
    a.is_consistent()

    assert b.get_shape() == (9, 5, 30)
    b.fuse_legs(axes=(0, (1, 2)), inplace=True)
    assert b.get_shape() == (9, 75)

    b.add_leg(axis=1, inplace=True)
    b.add_leg(axis=3, inplace=True)
    assert b.get_shape() == (9, 1, 75, 1)
    assert b.get_shape(native=True) == (9, 1, 5, 30, 1)

    b.is_consistent()


if __name__ == '__main__':
    test_aux_0()
    test_aux_1()
    test_aux_2()
