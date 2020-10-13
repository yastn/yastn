import yamps.tensor as tensor
import settings_full_R
import settings_U1_R
import settings_Z2_U1
import settings_U1_U1
import pytest
import numpy as np


def group_ungroup01(a):
    c, leg_order = a.group_legs(axes=(0, 1))
    d = c.ungroup_leg(axis=0, leg_order=leg_order)
    assert pytest.approx(a.norm_diff(d)) == 0
    assert a.is_independent(c)
    assert a.is_independent(d)
    assert c.is_independent(d)
    assert c.is_symmetric()
    assert d.is_symmetric()


def group_ungroup31(a):
    c, leg_order = a.group_legs(axes=(3, 1))
    d = c.ungroup_leg(axis=2, leg_order=leg_order)._transpose_local(axes=(0, 3, 1, 2))
    assert pytest.approx(a.norm_diff(d)) == 0
    assert a.is_independent(c)
    assert a.is_independent(d)
    assert c.is_independent(d)
    assert c.is_symmetric()
    assert d.is_symmetric()


def test_group0():
    a = tensor.rand(settings=settings_full_R, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    group_ungroup01(a)
    group_ungroup31(a)


def test_group1():
    a = tensor.rand(settings=settings_U1_R, s=(-1, 1, -1, 1), n=1,
                    t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                    D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    group_ungroup01(a)
    group_ungroup31(a)

    a = tensor.rand(settings=settings_U1_R, s=(-1, 1, 1, 1),
                    t=((0, 2), (-2, 2), (-2, 0, 2), 0),
                    D=((1, 2), (1, 2), (1, 2, 3), 1))
    a.set_block(ts=(0, 0, 0, 0), Ds=(1, 3, 2, 1), val=np.arange(6))
    group_ungroup01(a)
    group_ungroup31(a)


def test_group2():
    a = tensor.rand(settings=settings_Z2_U1, s=(-1, -1, 1, 1),
                    t=[(0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2)],
                    D=[(2, 3), (3, 2), (4, 5), (5, 4), (4, 3), (3, 4), (2, 3), (3, 2)])
    group_ungroup01(a)
    group_ungroup31(a)

    a = tensor.rand(settings=settings_Z2_U1, s=(-1, 1, 1, 1),
                    t=((0, 1), (0, 2), 0, (-2, 2), (0, 1), (-2, 0, 2), 0, 0),
                    D=((1, 2), (1, 2), 1, (1, 2), (2, 3), (1, 2, 3), 2, 2))
    a.set_block(ts=(0, 0, 0, 0, 0, 0, 0, 0), Ds=(1, 5, 4, 4), val=np.arange(80))
    group_ungroup01(a)
    group_ungroup31(a)


def group_ungroup_sparse(a):
    b, lo1 = a.group_legs(axes=(0, 1))
    c, lo2 = b.group_legs(axes=(1, 2), new_s=-1)
    print(lo1)
    b.show_properties()
    print(lo2)
    c.show_properties()
    d = c.ungroup_leg(axis=1, leg_order=lo2)
    e = d.ungroup_leg(axis=0, leg_order=lo1)

    assert pytest.approx(a.norm_diff(e)) == 0
    assert a.is_independent(c)
    assert a.is_independent(d)
    assert c.is_independent(d)
    assert c.is_symmetric()
    assert d.is_symmetric()
    e.show_properties()


def test_group1_sparse():
    a = tensor.Tensor(settings=settings_U1_R, s=(1, -1, 1, -1))
    a.set_block(ts=(0, 0, 1, 1), Ds=(1, 1, 2, 2), val='rand')
    a.set_block(ts=(1, 1, 0, 0), Ds=(2, 2, 1, 1), val='rand')
    a.show_properties()
    group_ungroup_sparse(a)


if __name__ == '__main__':
    # test_group0()
    # test_group1()
    # test_group2()
    test_group1_sparse()
