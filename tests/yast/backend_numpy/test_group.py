import yamps.tensor.yast as yast
import config_dense_R
import config_U1_R
import config_Z2_U1_R
import pytest
import numpy as np


def group_ungroup01(a):
    c = a.group_legs(axes=(0, 1))
    d = c.ungroup_leg(axis=0)
    assert pytest.approx(a.norm_diff(d)) == 0
    assert a.is_independent(c)
    assert a.is_independent(d)
    assert c.is_independent(d)
    assert c.is_consistent()
    assert d.is_consistent()


def group_ungroup31(a):
    c = a.group_legs(axes=(3, 1))
    d = c.ungroup_leg(axis=2).transpose(axes=(0, 3, 1, 2))
    assert pytest.approx(a.norm_diff(d)) == 0
    assert a.is_independent(c)
    assert a.is_independent(d)
    assert c.is_independent(d)
    assert c.is_consistent()
    assert d.is_consistent()


def test_group0():
    a = yast.rand(config=config_dense_R, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    group_ungroup01(a)
    group_ungroup31(a)


def test_group1():
    a = yast.rand(config=config_U1_R, s=(-1, 1, -1, 1), n=1,
                    t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                    D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    group_ungroup01(a)
    group_ungroup31(a)

    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, 1),
                    t=((0, 2), (-2, 2), (-2, 0, 2), 0),
                    D=((1, 2), (1, 2), (1, 2, 3), 1))
    a.set_block(ts=(0, 0, 0, 0), Ds=(1, 3, 2, 1), val=np.arange(6))
    group_ungroup01(a)
    group_ungroup31(a)


def test_group2():
    a = yast.rand(config=config_Z2_U1_R, s=(-1, -1, 1, 1),
                    t=[((0, 0), (1, 0), (0, 2), (1, 2)), ((0, 0), (1, 0), (0, 2), (1, 2)), ((0, 0), (1, 0), (0, 2), (1, 2)), ((0, 0), (1, 0), (0, 2), (1, 2))],
                    D=[(6, 9, 4, 6), (20, 16, 25, 20), (12, 16, 9, 12), (6, 4, 9, 6)])
    group_ungroup01(a)
    group_ungroup31(a)

    a = yast.rand(config=config_Z2_U1_R, s=(-1, 1, 1, 1),
                    t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)], [(0, 0)]],
                    D=((1, 2, 2, 4), (2, 3), (2, 4, 6, 3, 6, 9), 4))
    a.set_block(ts=(0, 0, 0, 0, 0, 0, 0, 0), Ds=(1, 5, 4, 4), val=np.arange(80))
    group_ungroup01(a)
    group_ungroup31(a)


def group_ungroup_sparse(a):
    b = a.group_legs(axes=(0, 1))
    c = b.group_legs(axes=(1, 2), new_s=-1)
    d = c.ungroup_leg(axis=1)
    e = d.ungroup_leg(axis=0)

    assert pytest.approx(a.norm_diff(e)) == 0
    assert a.is_independent(c)
    assert a.is_independent(d)
    assert c.is_independent(d)
    assert c.is_consistent()
    assert d.is_consistent()


def test_group1_sparse():
    a = yast.Tensor(config=config_U1_R, s=(1, -1, 1, -1))
    a.set_block(ts=(0, 0, 1, 1), Ds=(1, 1, 2, 2), val='rand')
    a.set_block(ts=(1, 1, 0, 0), Ds=(2, 2, 1, 1), val='rand')
    #a.show_properties()
    group_ungroup_sparse(a)


if __name__ == '__main__':
    test_group0()
    test_group1()
    test_group2()
    test_group1_sparse()
