r"""
Test functions: set_block, to_dict, from_dict
"""
from types import SimpleNamespace
import yamps.tensor as tensor
import settings_full_torch as settings_full
import settings_U1_torch as settings_U1
import settings_Z2_U1_torch as settings_Z2_U1
import settings_U1_U1_torch as settings_U1_U1
import numpy as np
from math import isclose

rel_tol=1.0e-14

def _gen_complex_conf(s):
    return SimpleNamespace(**dict(back= s.back, dtype='complex128', \
            dot_merge=s.dot_merge, sym= s.sym, nsym=s.nsym))

def test_set0():
    print('----------')
    print('3d tensor:')
    settings_full_C= _gen_complex_conf(settings_full)
    a = tensor.Tensor(settings=settings_full_C, s=(-1, 1, 1))
    a.set_block(Ds=(4, 5, 6), val='randR')
    npa = a.to_numpy()
    assert np.iscomplexobj(npa)
    assert npa.shape == (4, 5, 6)
    assert a.tset.shape == (1, 3, 0)
    assert a.is_symmetric()

    print('----------')
    print('0d tensor:')
    a = tensor.Tensor(settings=settings_full)  # s=()
    a.set_block(val=3)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.tset.shape == (1, 0, 0)
    assert isclose(a.to_number(),3,rel_tol=rel_tol)
    assert a.is_symmetric()

    print('----------')
    print('1d tensor:')
    a = tensor.Tensor(settings=settings_full, s=1)  # s=(1,)
    a.set_block(Ds=5, val='ones')
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5,)
    assert a.tset.shape == (1, 1, 0)
    assert a.is_symmetric()

    print('----------------')
    print('diagonal tensor:')
    a = tensor.Tensor(settings=settings_full, isdiag=True)
    a.set_block(Ds=5, val='ones')
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5, 5)
    assert a.tset.shape == (1, 1, 0)
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa),0,rel_tol=rel_tol)
    assert a.is_symmetric()

def test_set1():
    print('----------')
    print('4d tensor: ')
    settings_U1_C= _gen_complex_conf(settings_U1)
    a = tensor.ones(settings=settings_U1_C, s=(-1, 1, 1, 1),
                    t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                    D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    a.set_block(ts=(-2, 0, -2, 0), val='randR')
    npa = a.to_numpy()
    assert np.iscomplexobj(npa)
    assert npa.shape == (6, 3, 6, 1)
    assert a.tset.shape == (5, 4, 1)
    assert a.is_symmetric()

    print('----------')
    print('0d tensor:')
    a = tensor.ones(settings=settings_U1)  # s=()  # t=(), D=()
    a.set_block(val=2)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.tset.shape == (1, 0, 1)
    assert isclose(a.to_number(),2,rel_tol=rel_tol)
    assert a.is_symmetric()

    print('----------')
    print('1d tensor:')
    a = tensor.ones(settings=settings_U1, s=-1, t=0, D=5)
    # TODO what is the meaning of this ?
    try:
        a.set_block(ts=0, Ds=4)
        assert 1 == 0
    except tensor.FatalError:
        a.set_block(ts=0, Ds=5)

    print('----------------')
    print('diagonal tensor:')
    a = tensor.rand(settings=settings_U1, isdiag=True, t=0, D=5)
    a.set_block(ts=0, val='randR')
    a.set_block(ts=1, val='randR', Ds=4)
    a.set_block(ts=-1, val='randR', Ds=4)

    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (13, 13)
    assert a.tset.shape == (3, 1, 1)
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa),0,rel_tol=rel_tol)
    assert a.is_symmetric()

def test_set2():
    print('----------')
    print('3d tensor: ')
    a = tensor.ones(settings=settings_Z2_U1, s=(-1, 1, 1),
                    t=((0, 1), (0, 2), 0, (-2, 2), (0, 1), (-2, 0, 2)),
                    D=((1, 2), (1, 2), 1, (1, 2), (2, 3), (1, 2, 3)))
    a.set_block(ts=(0, 0, 0, 0, 0, 0), Ds=(1, 5, 4), val=np.arange(20))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (9, 8, 30)
    assert a.tset.shape == (7, 3, 2)
    assert a.is_symmetric()

    print('----------')
    print('3d tensor:')
    a = tensor.ones(settings=settings_Z2_U1, s=(-1, 1, 1),
                    t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0)]],
                    D=[[1, 2], 3, [1, 2]])
    a.set_block(ts=(0, 1, 0, -2, 0, 3), Ds=(1, 5, 6))
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (3, 8, 9)
    assert a.tset.shape == (3, 3, 2)
    assert a.is_symmetric()

    print('----------------')
    print('diagonal tensor:')
    a = tensor.rand(settings=settings_Z2_U1, isdiag=True,
                    t=[[(0, 0), (1, 1), (0, 2)]],
                    D=[[2, 2, 2]])
    a.set_block(ts=(0, 0), val='ones')
    a.set_block(ts=(1, 1), val='ones')
    a.set_block(ts=(0, 2), val='ones')
    a.set_block(ts=(1, 3), val='ones', Ds=2)
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (8, 8)
    assert a.tset.shape == (4, 1, 2)
    assert np.allclose(npa,np.eye(8),rtol=rel_tol)
    assert a.is_symmetric()

def test_dict():
    a = tensor.ones(settings=settings_Z2_U1, s=(-1, 1, 1),
                    t=((0, 1), (0, 2), 0, (-2, 2), (0, 1), (-2, 0, 2)),
                    D=((1, 2), (1, 2), 1, (1, 2), (2, 3), (1, 2, 3)))
    d = a.to_dict()
    b = tensor.from_dict(settings=settings_Z2_U1, d=d)
    assert isclose(a.norm_diff(b),0,rel_tol=rel_tol)
    assert a.is_symmetric()

    a = tensor.rand(settings=settings_U1, isdiag=True, t=(0, 1), D=(3, 5))
    d = a.to_dict()
    b = tensor.from_dict(settings=settings_U1, d=d)
    assert isclose(a.norm_diff(b),0,rel_tol=rel_tol)
    assert a.is_symmetric()

    a = tensor.randR(settings=settings_full)  # s=()
    d = a.to_dict()
    b = tensor.from_dict(settings=settings_full, d=d)
    assert isclose(a.norm_diff(b),0,rel_tol=rel_tol)
    assert a.is_symmetric()


if __name__ == '__main__':
    test_set0()
    test_set1()
    test_set2()
    test_dict()
