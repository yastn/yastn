""" 
Test: fill_tensor (which is called in: rand, randR, zeros, ones), to_numpy, match_legs, norm_diff 
"""

import yamps.yast as yast
import config_dense_R
import config_U1_R
import config_Z2_U1_R
from math import isclose
import numpy as np

tol = 1e-12

def test_fill_0():
    # print('3d tensor:')
    a = yast.ones(config=config_dense_R, s=(-1, 1, 1), D=(1, 2, 3))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (1, 2, 3)
    assert a.tset.shape == (1, 3, 0)
    assert a.Dset.shape == (1, 3)
    assert a.is_consistent()
    #assert not ((npa is a.A[()]) or (npa.base is a.A[()]) or (npa is a.A[()].base))

    b = yast.match_legs(tensors=[a, a, a], legs=[0, 1, 2], conjs=[1, 1, 1])
    assert isclose(a.norm_diff(b), 0, rel_tol=tol, abs_tol=tol)
    assert b.is_consistent()

    # print('0d tensor:')
    a = yast.ones(config=config_dense_R)  # s=() D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.tset.shape == (1, 0, 0)
    assert a.Dset.shape == (1, 0)
    assert isclose(a.to_number(), 1, rel_tol=tol, abs_tol=tol)
    assert a.is_consistent()

    # print('1d tensor:')
    a = yast.zeros(config=config_dense_R, s=1, D=5)  # s=(1,)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5,)
    assert a.tset.shape == (1, 1, 0)
    assert a.Dset.shape == (1, 1)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a], legs=[0], conjs=[1], val='zeros')
    assert isclose(a.norm_diff(b), 0, rel_tol=tol, abs_tol=tol)
    assert b.is_consistent()

    # print('diagonal tensor:')
    a = yast.rand(config=config_dense_R, isdiag=True, D=5)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5, 5)
    assert a.tset.shape == (1, 2, 0)
    assert a.Dset.shape == (1, 2)
    assert a.is_consistent()

    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa), 0, rel_tol=tol, abs_tol=tol)


def test_fill_1():
    # print('4d tensor: ')
    a = yast.ones(config=config_U1_R, s=(-1, 1, 1, 1),
                    t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                    D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (6, 3, 6, 1)
    assert a.tset.shape == (5, 4, 1)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a, a, a, a], legs=[0, 1, 2, 3], conjs=[1, 1, 1, 1], val='ones')
    assert isclose(a.norm_diff(b), 0, rel_tol=tol, abs_tol=tol)

    # print('0d tensor:')
    a = yast.ones(config=config_U1_R)  # s=()  # t=(), D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.tset.shape == (1, 0, 1)
    assert a.Dset.shape == (1, 0)
    assert isclose(a.to_number(), 1, rel_tol=tol, abs_tol=tol)
    assert a.is_consistent()

    # print('1d tensor:')
    a = yast.ones(config=config_U1_R, s=-1, t=0, D=5)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5,)
    assert a.tset.shape == (1, 1, 1)
    assert a.Dset.shape == (1, 1)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a], legs=[0], conjs=[1], val='ones')
    assert isclose(a.norm_diff(b), 0, rel_tol=tol, abs_tol=tol)

    # print('diagonal tensor:')
    a = yast.rand(config=config_U1_R, isdiag=True, t=0, D=5)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5, 5)
    assert a.tset.shape == (1, 2, 1)
    assert a.Dset.shape == (1, 2)
    
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa), 0, rel_tol=tol, abs_tol=tol)
    assert a.is_consistent()

    # print('diagonal tensor:')
    a = yast.randR(config=config_U1_R, isdiag=True, t=(-1, 0, 1), D=(2, 3, 4))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (9, 9)
    assert a.tset.shape == (3, 2, 1)
    assert a.Dset.shape == (3, 2)

    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa.conj()), 0, rel_tol=tol, abs_tol=tol)
    assert a.is_consistent()

    # print('diagonal tensor:')
    a = yast.eye(config=config_U1_R, t=(-1, 0, 1), D=(2, 3, 4))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (9, 9)
    assert a.tset.shape == (3, 2, 1)
    assert a.Dset.shape == (3, 2)

    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa), 0, rel_tol=tol, abs_tol=tol)
    assert a.is_consistent()


def test_fill_2():
    # print('3d tensor:')
    a = yast.ones(config=config_Z2_U1_R, s=(-1, 1, 1),
                    t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0)]],
                    D=[[1, 2], 3, [1, 2]])
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (3, 3, 3)
    assert a.tset.shape == (2, 3, 2)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a, a, a], legs=[0, 1, 2], conjs=[1, 1, 1], val='ones')
    assert isclose(a.norm_diff(b), 0, rel_tol=tol, abs_tol=tol)

    # print('1d tensor:')
    a = yast.ones(config=config_Z2_U1_R, s=1,
                    t=[[(0, 0)]],
                    D=[[2]])
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (2,)
    assert a.tset.shape == (1, 1, 2)
    assert a.Dset.shape == (1, 1)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a], legs=[0], conjs=[1], val='ones')
    assert isclose(a.norm_diff(b), 0, rel_tol=tol, abs_tol=tol)

    # print('diagonal tensor:')
    a = yast.rand(config=config_Z2_U1_R, isdiag=True,
                    t=[[(0, 0), (1, 1), (0, 2)]],
                    D=[[2, 2, 2]])
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (6, 6)
    assert a.tset.shape == (3, 2, 2)
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa), 0, rel_tol=tol, abs_tol=tol)
    assert a.is_consistent()

    # print('diagonal tensor:')
    a = yast.randR(config=config_Z2_U1_R, isdiag=True,
                     t=[[(0, 1), (-1, 0)], [(0, 1), (2, 0)]],
                     D=[[2, 5], [2, 7]])
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (2, 2)
    assert a.tset.shape == (1, 2, 2)

    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa), 0, rel_tol=tol, abs_tol=tol)
    assert a.is_consistent()


if __name__ == '__main__':
    test_fill_0()
    test_fill_1()
    test_fill_2()
