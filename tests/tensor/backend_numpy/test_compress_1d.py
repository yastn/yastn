r"""
Test functions: reset_tensor (which is called in: rand, randR, zeros, ones), to_numpy, match_legs, norm_diff

"""
from types import SimpleNamespace
import yamps.tensor as tensor
from yamps.tensor import decompress_from_1d 
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import numpy as np

def _gen_complex_conf(s):
    return SimpleNamespace(**dict(back= s.back, dtype='complex128', \
            dot_merge=s.dot_merge, sym= s.sym, nsym=s.nsym))

def test_compress_full():
    print('----------')
    print('3d tensor:')
    A = tensor.rand(settings=settings_full, s=(-1, 1, 1), D=(1, 2, 3))
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_full, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (1, 2, 3)
    assert T.tset.shape == (1, 3, 0)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    # print('----------')
    # print('0d tensor:')
    # a= tensor.ones(settings=settings_full)  # s=() D=()
    # a.show_properties()
    # npa = A.to_numpy()
    # assertTnp.isrealobj(npa)
    # assert npa.shape == ()
    # assert a.tset.shape == (1, 0, 0)
    # assert pytest.approx(a.to_number()) == 1
    # assert a.is_symmetric()

    print('----------')
    print('1d tensor:')
    A = tensor.rand(settings=settings_full, s=1, D=5)  # s=(1,)
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_full, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5,)
    assert T.tset.shape == (1, 1, 0)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    print('----------------')
    print('diagonal tensor:')
    A = tensor.rand(settings=settings_full, isdiag=True, D=5)
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_full, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5, 5)
    assert T.tset.shape == (1, 1, 0)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

def test_compress_U1():
    print('----------')
    print('4d tensor: ')
    A = tensor.ones(settings=settings_U1, s=(-1, 1, 1, 1),
                    t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                    D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_U1, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (6, 3, 6, 1)
    assert T.tset.shape == (5, 4, 1)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    # print('----------')
    # print('0d tensor:')
    # a = tensor.ones(settings=settings_U1)  # s=()  # t=(), D=()
    # a.show_properties()
    # npa = A.to_numpy()
    # assertTnp.isrealobj(npa)
    # assert npa.shape == ()
    # assert a.tset.shape == (1, 0, 1)
    # assert pytest.approx(a.to_number()) == 1
    # assert a.is_symmetric()

    print('----------')
    print('1d tensor:')
    A = tensor.ones(settings=settings_U1, s=-1, t=0, D=5)
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_U1, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5,)
    assert T.tset.shape == (1, 1, 1)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    print('----------------')
    print('diagonal tensor:')
    A = tensor.rand(settings=settings_U1, isdiag=True, t=0, D=5)
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_U1, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (5, 5)
    assert T.tset.shape == (1, 1, 1)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    print('----------------')
    print('diagonal tensor:')
    settings_U1_C = _gen_complex_conf(settings_U1)
    A = tensor.randR(settings=settings_U1, isdiag=True, t=(-1, 0, 1), D=(2, 3, 4))
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_U1_C, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.iscomplexobj(npt)
    assert npt.shape == (9, 9)
    assert T.tset.shape == (3, 1, 1)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    print('----------------')
    print('diagonal tensor:')
    A = tensor.eye(settings=settings_U1_C, t=(-1, 0, 1), D=(2, 3, 4))
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_U1_C, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.iscomplexobj(npt)
    assert npt.shape == (9, 9)
    assert T.tset.shape == (3, 1, 1)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

def test_compress_Z2_U1():
    print('----------')
    print('3d tensor: ')
    A = tensor.ones(settings=settings_Z2_U1, s=(-1, 1, 1),
                    t=((0, 1), (0, 2), 0, (-2, 2), (0, 1), (-2, 0, 2)),
                    D=((1, 2), (1, 2), 1, (1, 2), (2, 3), (1, 2, 3)))
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_Z2_U1, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (9, 3, 30)
    assert T.tset.shape == (6, 3, 2)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    print('----------')
    print('3d tensor:')
    A = tensor.ones(settings=settings_Z2_U1, s=(-1, 1, 1),
                    t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0)]],
                    D=[[1, 2], 3, [1, 2]])
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_Z2_U1, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (3, 3, 3)
    assert T.tset.shape == (2, 3, 2)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    print('----------')
    print('1d tensor:')
    A = tensor.ones(settings=settings_Z2_U1, s=1,
                    t=([0], [0]),
                    D=([1], [1]))
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_Z2_U1, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (1,)
    assert T.tset.shape == (1, 1, 2)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    print('----------')
    print('1d tensor:')
    A = tensor.ones(settings=settings_Z2_U1, s=1,
                    t=[[(0, 0)]],
                    D=[[2]])
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_Z2_U1, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (2,)
    assert T.tset.shape == (1, 1, 2)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    print('----------------')
    print('diagonal tensor:')
    A = tensor.rand(settings=settings_Z2_U1, isdiag=True,
                    t=[[(0, 0), (1, 1), (0, 2)]],
                    D=[[2, 2, 2]])
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_Z2_U1, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.isrealobj(npt)
    assert npt.shape == (6, 6)
    assert T.tset.shape == (3, 1, 2)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)

    print('----------------')
    print('diagonal tensor:')
    settings_Z2_U1_C = _gen_complex_conf(settings_Z2_U1)
    A = tensor.randR(settings=settings_Z2_U1_C, isdiag=True,
                     t=[(0, 1), (0, 1)],
                     D=[(2, 2), (2, 2)])
    meta, r1d= A.compress_to_1d()
    T= decompress_from_1d(r1d, settings=settings_Z2_U1_C, d=meta)
    npa= A.to_numpy()
    npt= T.to_numpy()
    assert np.iscomplexobj(npt)
    assert npt.shape == (16, 16)
    assert T.tset.shape == (4, 1, 2)
    assert T.is_symmetric()
    assert np.allclose(npa,npt)


if __name__ == '__main__':
    test_compress_full()
    test_compress_U1()
    test_compress_Z2_U1()