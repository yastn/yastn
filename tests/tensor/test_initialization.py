""" Test: fill_tensor (which is called in: rand, zeros, ones), to_numpy, match_legs """
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_fill_0():
    # 3d tensor
    a = yast.ones(config=config_dense, s=(-1, 1, 1), D=(1, 2, 3))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (1, 2, 3)
    assert a.size == np.sum(npa != 0.)
    assert a.is_consistent()

    # b = yast(legs=([a, 0, 'conj', a, 1, 'conj', {'s'=(1,), (1,): 5}]))

    b = yast.match_legs(tensors=[a, a, a], legs=[0, 1, 2], conjs=[1, 1, 1])
    assert yast.norm(a - b) < tol  # == 0.0
    assert b.is_consistent()

    # 0d tensor
    a = yast.ones(config=config_dense)  # s=() D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.size == np.sum(npa != 0.)
    assert pytest.approx(a.item(), rel=tol) == 1
    assert a.is_consistent()

    # 1d tensor
    a = yast.zeros(config=config_dense, s=1, D=5)  # s=(1,)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5,)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a], legs=[0], conjs=[1], val='zeros')
    assert yast.norm(a - b) < tol  # == 0.0
    assert b.is_consistent()

    # diagonal tensor
    a = yast.rand(config=config_dense, isdiag=True, D=5)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5, 5)
    assert a.size == np.sum(npa != 0.)
    assert a.is_consistent()

    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0


def test_fill_1():
    # 4d tensor: ')
    a = yast.ones(config=config_U1, s=(-1, 1, 1, 1),
                  t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                  D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    a2 = a.to_nonsymmetric()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (6, 3, 6, 1)
    assert a.size == np.sum(npa != 0.)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a, a, a, a], legs=[0, 1, 2, 3], conjs=[1, 1, 1, 1], val='ones')
    assert yast.norm(a - b) < tol  # == 0.0

    # 0d tensor
    a = yast.ones(config=config_U1)  # s=()  # t=(), D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.size == np.sum(npa != 0.)
    assert pytest.approx(a.item(), rel=tol) == 1
    assert a.is_consistent()

    # 1d tensor
    a = yast.ones(config=config_U1, s=-1, t=0, D=5)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5,)
    assert a.size == np.sum(npa != 0.)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a], legs=[0], conjs=[1], val='ones')
    assert yast.norm(a - b) < tol  # == 0.0

    # diagonal tensor
    a = yast.rand(config=config_U1, isdiag=True, t=0, D=5)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5, 5)
    assert a.size == np.sum(npa != 0.)
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0
    assert a.is_consistent()

    # diagonal tensor
    a = yast.rand(config=config_U1, isdiag=True, t=(-1, 0, 1), D=(2, 3, 4), dtype='complex128')
    npa = a.to_numpy()
    assert np.iscomplexobj(npa)
    assert npa.shape == (9, 9)
    assert a.size == np.sum(npa != 0.)
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0
    assert a.is_consistent()

    # diagonal tensor
    a = yast.eye(config=config_U1, t=(-1, 0, 1), D=(2, 3, 4))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (9, 9)
    assert a.size == np.sum(npa != 0.)
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa.conj()) < tol  # == 0.0
    assert a.is_consistent()


def test_fill_2():
    # 3d tensor
    a = yast.ones(config=config_Z2xU1, s=(-1, 1, 1),
                  t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0)]],
                  D=[[1, 2], 3, [1, 2]])
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (3, 3, 3)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a, a, a], legs=[0, 1, 2], conjs=[1, 1, 1], val='ones')
    assert yast.norm(a - b) < tol  # == 0.0

    # 1d tensor
    a = yast.ones(config=config_Z2xU1, s=1,
                  t=[[(0, 0)]], D=[[2]])
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (2,)
    assert a.size == np.sum(npa != 0.)
    assert a.is_consistent()

    b = yast.match_legs(tensors=[a], legs=[0], conjs=[1], val='ones')
    assert yast.norm(a - b) < tol  # == 0.0

    # diagonal tensor
    a = yast.rand(config=config_Z2xU1, isdiag=True,
                  t=[[(0, 0), (1, 1), (0, 2)]],
                  D=[[2, 2, 2]])
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (6, 6)
    assert a.size == np.sum(npa != 0.)
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0
    assert a.is_consistent()

    # diagonal tensor
    a = yast.rand(config=config_Z2xU1, isdiag=True,
                  t=[[(0, 1), (-1, 0)], [(0, 1), (2, 0)]],
                  D=[[2, 5], [2, 7]])
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (2, 2)
    assert a.size == np.sum(npa != 0.)

    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0
    assert a.is_consistent()


if __name__ == '__main__':
    test_fill_0()
    test_fill_1()
    test_fill_2()
