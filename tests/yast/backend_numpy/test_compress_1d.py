r"""
Test functions: reset_tensor (which is called in: rand, randR, zeros, ones), to_numpy, match_legs, norm_diff

"""
import yamps.tensor.yast as yast
import config_dense_R
import config_U1_R
import config_U1_C
import config_Z2_U1_R
import numpy as np
import pytest


def test_compress_dense():
    # print('3d tensor:')
    A = yast.rand(config=config_dense_R, s=(-1, 1, 1), D=(1, 2, 3))
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_dense_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (1, 2, 3)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

    # print('0d tensor:')
    a = yast.ones(config=config_dense_R)  # s=() D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.tset.shape == (1, 0, 0)
    assert pytest.approx(a.to_number()) == 1
    assert a.is_consistent()

    # print('1d tensor:')
    A = yast.rand(config=config_dense_R, s=1, D=5)  # s=(1,)
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_dense_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5,)
    assert T.tset.shape == (1, 1, 0)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

    # print('diagonal tensor:')
    A = yast.rand(config=config_dense_R, isdiag=True, D=5)
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_dense_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5, 5)
    assert T.tset.shape == (1, 2, 0)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

def test_compress_U1():
    # print('4d tensor: ')
    A = yast.ones(config=config_U1_R, s=(-1, 1, 1, 1),
                    t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                    D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_U1_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (6, 3, 6, 1)
    assert T.tset.shape == (5, 4, 1)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

    # print('0d tensor:')
    a = yast.ones(config=config_U1_R)  # s=()  # t=(), D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.tset.shape == (1, 0, 1)
    assert pytest.approx(a.to_number()) == 1
    assert a.is_consistent()

    # print('1d tensor:')
    A = yast.ones(config=config_U1_R, s=-1, t=0, D=5)
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_U1_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5,)
    assert T.tset.shape == (1, 1, 1)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

    # print('diagonal tensor:')
    A = yast.rand(config=config_U1_R, isdiag=True, t=0, D=5)
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_U1_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5, 5)
    assert T.tset.shape == (1, 2, 1)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

    # print('diagonal tensor:')
    A = yast.randR(config=config_U1_C, isdiag=True, t=(-1, 0, 1), D=(2, 3, 4))
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_U1_C, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.iscomplexobj(npt)
    assert npt.shape == (9, 9)
    assert T.tset.shape == (3, 2, 1)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

    # print('diagonal tensor:')
    A = yast.eye(config=config_U1_C, t=(-1, 0, 1), D=(2, 3, 4))
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_U1_C, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.iscomplexobj(npt)
    assert npt.shape == (9, 9)
    assert T.tset.shape == (3, 2, 1)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

def test_compress_Z2_U1():
    # print('3d tensor:')
    A = yast.ones(config=config_Z2_U1_R, s=(-1, 1, 1),
                    t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0)]],
                    D=[[1, 2], 3, [1, 2]])
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_Z2_U1_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (3, 3, 3)
    assert T.tset.shape == (2, 3, 2)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

    # print('1d tensor:')
    A = yast.ones(config=config_Z2_U1_R, s=1,
                    t=[[(0, 0)]],
                    D=[[2]])
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_Z2_U1_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (2,)
    assert T.tset.shape == (1, 1, 2)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

    # print('diagonal tensor:')
    A = yast.rand(config=config_Z2_U1_R, isdiag=True,
                    t=[[(0, 0), (1, 1), (0, 2)]],
                    D=[[2, 2, 2]])
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_Z2_U1_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (6, 6)
    assert T.tset.shape == (3, 2, 2)
    assert T.is_consistent()
    assert np.allclose(npa,npt)

    # print('diagonal tensor:')
    A = yast.randR(config=config_Z2_U1_R, isdiag=True,
                     t=[[(0, 1), (1, 0), (1, 1), (0, 0)]],
                     D=[4, 4, 4, 4])
    meta, r1d= A.compress_to_1d()
    T= yast.decompress_from_1d(r1d, config=config_Z2_U1_R, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (16, 16)
    assert T.tset.shape == (4, 2, 2)
    assert T.is_consistent()
    assert np.allclose(npa,npt)


if __name__ == '__main__':
    test_compress_dense()
    test_compress_U1()
    test_compress_Z2_U1()