import yamps.yast as yast
import numpy as np
import config_dense_R
import config_U1_R
import config_Z2_U1_R

def run_moveaxis(a, ad, source, destination, result):
    newa = a.moveaxis(source=source, destination=destination)
    assert newa.to_dense().shape == result
    assert newa.get_shape_all() == result
    assert np.moveaxis(ad, source=source, destination=destination).shape == result
    assert newa.is_consistent()
    assert a.is_independent(newa)


def run_transpose(a, ad, axes, result):
    newa = a.transpose(axes=axes)
    assert newa.to_dense().shape == result
    assert newa.get_shape_all() == result
    assert np.transpose(ad, axes=axes).shape == result
    assert newa.is_consistent()
    assert a.is_independent(newa)


def test_transpose_0():
    a = yast.ones(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    assert a.get_shape_all() == (2, 3, 4, 5)
    ad = a.to_dense()
    run_transpose(a, ad, axes=(1, 3, 2, 0), result=(3, 5, 4, 2))
    run_moveaxis(a, ad, source=1, destination=-1, result=(2, 4, 5, 3))
    run_moveaxis(a, ad, source=(1, 3), destination=(1, 0), result=(5, 3, 2, 4))
    run_moveaxis(a, ad, source=(3, 1), destination=(0, 1), result=(5, 3, 2, 4))
    run_moveaxis(a, ad, source=(3, 1), destination=(1, 0), result=(3, 5, 2, 4))
    run_moveaxis(a, ad, source=(1, 3), destination=(0, 1), result=(3, 5, 2, 4))


def test_transpose_1():
    a = yast.ones(config=config_U1_R, s=(-1, -1, -1, 1, 1, 1),
                    t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                    D=[(2, 3), (4, 5), (6, 7), (6, 5), (4, 3), (2, 1)])
    ad = a.to_dense()
    assert a.get_shape_all() == (5, 9, 13, 11, 7, 3)
    
    run_transpose(a, ad, axes=(1, 2, 3, 0, 5, 4), result=(9, 13, 11, 5, 3, 7))
    run_moveaxis(a, ad, source=1, destination=4, result=(5, 13, 11, 7, 9, 3))
    run_moveaxis(a, ad, source=(2, 0), destination=(0, 2), result=(13, 9, 5, 11, 7, 3))
    run_moveaxis(a, ad, source=(2, -1, 0), destination=(-1, 2, -2), result=(9, 11, 3, 7, 5, 13))


def test_transpose_2():
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2_U1_R, s=(-1, -1, 1, 1),
                    t=[t1, t1, t1, t1],
                    D=[(7, 3, 4, 5), (5, 4, 3, 2), (3, 4, 5, 6), (1, 2, 3, 4)])
    assert a.get_shape_all() == (19, 14, 18, 10)
    ad = a.to_dense()
    run_transpose(a, ad, axes=(1, 2, 3, 0), result=(14, 18, 10, 19))
    run_moveaxis(a, ad, source=-1, destination=-3, result=(19, 10, 14, 18))


def test_transpose_inplace():
    a = yast.ones(config=config_U1_R, s=(-1, -1, -1, 1, 1, 1),
                    t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                    D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    assert a.get_shape_all() == (3, 5, 7, 9, 11, 13)

    a.transpose(axes=(5, 4, 3, 2, 1, 0), inplace=True)
    assert a.get_shape_all() == (13, 11, 9, 7, 5, 3)
    a.moveaxis(source=2, destination=3, inplace=True)
    assert a.get_shape_all() == (13, 11, 7, 9, 5, 3)
    

if __name__ == '__main__':
    test_transpose_0()
    test_transpose_1()
    test_transpose_2()
    test_transpose_inplace()
