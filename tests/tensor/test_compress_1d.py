""" fill_tensor (which is called in: rand, zeros, ones), yast.to_numpy """
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_compress_dense():
    # print('3d tensor:')
    A = yast.rand(config=config_dense, s=(-1, 1, 1), D=(1, 2, 3))
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_dense, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (1, 2, 3)
    assert T.is_consistent()
    assert np.allclose(npa, npt)

    # print('0d tensor:')
    a = yast.ones(config=config_dense)  # s=() D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert pytest.approx(a.item(), rel=tol) == 1
    assert a.is_consistent()

    # print('1d tensor:')
    A = yast.rand(config=config_dense, s=1, D=5)  # s=(1,)
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_dense, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5,)
    assert T.is_consistent()
    assert np.allclose(npa, npt)

    # print('diagonal tensor:')
    A = yast.rand(config=config_dense, isdiag=True, D=5)
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_dense, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5, 5)
    assert T.is_consistent()
    assert np.allclose(npa, npt)


def test_compress_U1():
    # print('4d tensor: ')
    A = yast.ones(config=config_U1, s=(-1, 1, 1, 1),
                  t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                  D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_U1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (6, 3, 6, 1)
    assert T.is_consistent()
    assert np.allclose(npa, npt)

    # print('0d tensor:')
    a = yast.ones(config=config_U1)  # s=()  # t=(), D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert pytest.approx(a.item(), rel=tol) == 1
    assert a.is_consistent()

    # print('1d tensor:')
    A = yast.ones(config=config_U1, s=-1, t=0, D=5)
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_U1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5,)
    assert T.is_consistent()
    assert np.allclose(npa, npt)

    # print('diagonal tensor:')
    A = yast.rand(config=config_U1, isdiag=True, t=0, D=5)
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_U1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5, 5)
    assert T.is_consistent()
    assert np.allclose(npa, npt)

    # print('diagonal tensor:')
    A = yast.rand(config=config_U1, isdiag=True, t=(-1, 0, 1), D=(2, 3, 4), dtype='complex128')
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_U1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.iscomplexobj(npt)
    assert npt.shape == (9, 9)
    assert T.is_consistent()
    assert np.allclose(npa, npt)

    # print('diagonal tensor:')
    A = yast.eye(config=config_U1, t=(-1, 0, 1), D=(2, 3, 4))
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_U1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (9, 9)
    assert T.is_consistent()
    assert np.allclose(npa, npt)


def test_compress_Z2xU1():
    # print('3d tensor:')
    A = yast.ones(config=config_Z2xU1, s=(-1, 1, 1),
                  t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0)]],
                  D=[[1, 2], 3, [1, 2]])
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_Z2xU1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (3, 3, 3)
    assert T.is_consistent()
    assert np.allclose(npa, npt)

    # print('1d tensor:')
    A = yast.ones(config=config_Z2xU1, s=1,
                  t=[[(0, 0)]], D=[[2]])
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_Z2xU1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (2,)
    assert T.is_consistent()
    assert np.allclose(npa, npt)

    # print('diagonal tensor:')
    A = yast.rand(config=config_Z2xU1, isdiag=True,
                  t=[[(0, 0), (1, 1), (0, 2)]], D=[[2, 2, 2]])
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_Z2xU1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (6, 6)
    assert T.is_consistent()
    assert np.allclose(npa, npt)

    # print('diagonal tensor:')
    A = yast.rand(config=config_Z2xU1, isdiag=True,
                  t=[[(0, 1), (1, 0), (1, 1), (0, 0)]], D=[4, 4, 4, 4])
    r1d, meta = A.compress_to_1d()
    T = yast.decompress_from_1d(r1d, config=config_Z2xU1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (16, 16)
    assert T.is_consistent()
    assert np.allclose(npa, npt)


def test_meta_compress():
    A = yast.Tensor(config=config_U1, s=(-1, 1, 1, 1))
    A.set_block(ts=(2, 0, 1, 1), Ds=(1, 2, 3, 4))
    A.set_block(ts=(0, 1, 0, -1), Ds=(5, 6, 7, 8))
    # creates tensor matching A for dot multiplication filling in all possible blocks.
    B = yast.zeros(config=A.config, legs=A.get_legs())
    _, meta = B.compress_to_1d()
    r1d, _ = A.compress_to_1d(meta)
    T = yast.decompress_from_1d(r1d, config=config_U1, meta=meta)
    npa = A.to_numpy()
    npt = T.to_numpy()
    assert np.allclose(npa, npt)
    assert yast.norm(A - T) < tol
    # now adding some blocks with new charges
    A.set_block(ts=(1, 1, 0, 0), Ds=(2, 6, 7, 1))
    with pytest.raises(yast.YastError):
        r1d, _ = A.compress_to_1d(meta)


if __name__ == '__main__':
    test_compress_dense()
    test_compress_U1()
    test_compress_Z2xU1()
    test_meta_compress()
