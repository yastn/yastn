""" yast.set_block """
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_set0():
    # print('3d tensor:')
    a = yast.Tensor(config=config_dense, s=(-1, 1, 1))
    a.set_block(Ds=(4, 5, 6), val='randR')
    npa = a.to_numpy()
    # assert np.iscomplexobj(npa)
    assert np.linalg.norm(npa - npa.conj()) < tol  # == 0.0
    assert npa.shape == (4, 5, 6)
    assert a.is_consistent()

    # print('0d tensor:')
    a = yast.Tensor(config=config_dense)  # s=()
    a.set_block(val=3)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert pytest.approx(a.to_number(), rel=tol) == 3
    assert a.is_consistent()

    # print('1d tensor:')
    a = yast.Tensor(config=config_dense, s=1)  # s=(1,)
    a.set_block(Ds=5, val='ones')
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5,)
    assert a.is_consistent()

    # print('diagonal tensor:')
    a = yast.Tensor(config=config_dense, isdiag=True)
    a.set_block(Ds=5, val='ones')
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5, 5)
    assert a.is_consistent()
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0


def test_set1():
    # print('3d tensors ')
    a = yast.Tensor(config=config_U1, s=(-1, 1, 1))
    a.set_block(ts=(1, -1, 2), Ds=(2, 5, 3), val='rand')
    a.set_block(ts=(2, 0, 2), Ds=(3, 6, 3), val='rand')

    b = yast.Tensor(config=config_U1, s=(-1, 1, 1))
    b.set_block(ts=(1, 0, 1), Ds=(2, 6, 2), val='rand')
    b.set_block(ts=(2, 0, 2), Ds=(3, 6, 3), val='rand')

    c1 = yast.tensordot(a, a, axes=((0, 1, 2), (0, 1, 2)), conj=(0, 1))
    c2 = yast.tensordot(b, b, axes=((1, 2), (1, 2)), conj=(0, 1))
    c3 = yast.tensordot(a, b, axes=(0, 2))

    na = a.to_numpy()
    nb = b.to_numpy()
    nc1 = c1.to_numpy()
    nc2 = c2.to_numpy()
    nc3 = c3.to_numpy()

    nnc1 = np.tensordot(na, na.conj(), axes=((0, 1, 2), (0, 1, 2)))
    nnc2 = np.tensordot(nb, nb.conj(), axes=((1, 2), (1, 2)))
    nnc3 = np.tensordot(na.conj(), nb.conj(), axes=(0, 2))

    assert np.linalg.norm(nc1 - nnc1) < tol  # == 0.0
    assert np.linalg.norm(nc2 - nnc2) < tol  # == 0.0
    assert np.linalg.norm(nc3 - nnc3) < tol  # == 0.0
    assert np.linalg.norm(nc1) - c1.norm() < tol  # == 0.0
    assert np.linalg.norm(nc2) - c2.norm() < tol  # == 0.0
    assert np.linalg.norm(nc3) - c3.norm() < tol  # == 0.0
    
    assert na.shape == (5, 11, 3)
    assert nb.shape == (5, 6, 5)
    assert nc1.shape == ()
    assert nc2.shape == (5, 5)
    assert nc3.shape == (11, 3, 5, 6)

    # print('4d tensor: ')
    a = yast.ones(config=config_U1, s=(-1, 1, 1, 1),
                  t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                  D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    a.set_block(ts=(-2, 0, -2, 0), val='randC')
    npa = a.to_numpy()
    assert np.iscomplexobj(npa)
    assert npa.shape == (6, 3, 6, 1)
    assert a.is_consistent()

    # print('0d tensor:')
    a = yast.ones(config=config_U1)  # s=()  # t=(), D=()
    a.set_block(val=2)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert pytest.approx(a.to_number(), rel=tol) == 2
    assert a.is_consistent()

    # print('3d tensor:')
    a = yast.ones(config=config_U1, s=(1, 1, -1), t=((0, 1), (0, 1), (0, 1)), D=((2, 3), (4, 5), (6, 7)))
    b = a.copy()
    with pytest.raises(yast.YastError):
        a.set_block(ts=(0, 0, 0), Ds=(3, 4, 6))  # here (3, ...) is inconsistent bond dimension
    b.set_block(ts=(0, 0, 0))  # here should infer bond dimensions

    # print('diagonal tensor:')
    a = yast.rand(config=config_U1, isdiag=True, t=0, D=5)
    a.set_block(ts=0, val='rand')
    a.set_block(ts=1, val='rand', Ds=4)
    a.set_block(ts=-1, val='randC', Ds=4)
    npa = a.to_numpy()

    assert np.iscomplexobj(npa)
    assert npa.shape == (13, 13)
    assert a.is_consistent()
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0
    a.show_properties()


def test_set2():
    # print('3d tensor: ')
    a = yast.ones(config=config_Z2_U1, s=(-1, 1, 1),
                  t=(((0, 0), (1, 0), (0, 2), (1, 2)), ((0, -2), (0, 2)), ((0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2))),
                  D=((1, 2, 2, 4), (1, 2), (2, 4, 6, 3, 6, 9)))
    a.set_block(ts=(0, 0, 0, 0, 0, 0), Ds=(1, 5, 4), val=np.arange(20))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (9, 8, 30)
    assert a.is_consistent()

    # print('3d tensor:')
    a = yast.ones(config=config_Z2_U1, s=(-1, 1, 1),
                  t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0)]],
                  D=[[1, 2], 3, [1, 2]])
    a.set_block(ts=(0, 1, 0, -2, 0, 3), Ds=(1, 5, 6))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (3, 8, 9)
    assert a.is_consistent()

    # print('diagonal tensor:')
    a = yast.rand(config=config_Z2_U1, isdiag=True,
                  t=[[(0, 0), (1, 1), (0, 2)]],
                  D=[[2, 3, 5]])
    a.set_block(ts=(0, 0), val='ones')
    a.set_block(ts=(1, 1), val='ones')
    a.set_block(ts=(0, 2), val='ones')
    a.set_block(ts=(1, 3), val='ones', Ds=1)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (11, 11)
    assert np.allclose(npa, np.eye(11), rtol=tol, atol=tol)
    assert a.is_consistent()

    b = a.to_nonsymmetric()
    assert b.get_shape() == (11, 11)


def test_dict():
    a = yast.rand(config=config_dense)  # s=()
    d = a.export_to_dict()
    b = yast.import_from_dict(config=config_dense, d=d)
    assert yast.norm_diff(a, b) < tol  # == 0.0
    assert a.is_consistent()
    assert a.are_independent(b)

    a = yast.rand(config=config_U1, isdiag=False, s=(1, -1, 1),
                  t=((0, 1, 2), (0, 1, 3), (-1, 0, 1)),
                  D=((3, 5, 2), (1, 2, 3), (2, 3, 4)))
    d = a.export_to_dict()
    b = yast.import_from_dict(config=config_U1, d=d)
    assert yast.norm_diff(a, b) < tol  # == 0.0
    assert a.is_consistent()
    assert a.are_independent(b)

    a = yast.rand(config=config_U1, isdiag=True, t=(0, 1), D=(3, 5))
    d = a.export_to_dict()
    b = yast.import_from_dict(config=config_U1, d=d)
    assert yast.norm_diff(a, b) < tol  # == 0.0
    assert a.is_consistent()
    assert a.are_independent(b)

    a = yast.ones(config=config_Z2_U1, s=(-1, 1, 1), n=(0, -2),
                  t=(((0, 0), (0, 2), (1, 0), (1, 2)), ((0, -2), (0, 2)), ((0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2))),
                  D=((1, 2, 3, 4), (2, 1), (2, 3, 5, 4, 1, 6)))
    d = a.export_to_dict()
    b = yast.import_from_dict(config=config_Z2_U1, d=d)
    assert yast.norm_diff(a, b) < tol  # == 0.0
    assert b.is_consistent()
    assert a.are_independent(b)


if __name__ == '__main__':
    test_set0()
    test_set1()
    test_set2()
    test_dict()
