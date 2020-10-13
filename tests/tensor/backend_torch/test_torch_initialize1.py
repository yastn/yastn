r"""
Test functions: reset_tensor (which is called in: rand, randR, zeros, ones), to_numpy, match_legs, norm_diff
"""
from types import SimpleNamespace
import yamps.tensor as tensor
import settings_full_torch as settings_full
import settings_U1_torch as settings_U1
import settings_Z2_U1_torch as settings_Z2_U1
import settings_U1_U1_torch as settings_U1_U1
from math import isclose
import numpy as np

rel_tol=1.0e-14

def _gen_complex_conf(s):
    return SimpleNamespace(**dict(back= s.back, dtype='complex128', \
            dot_merge=s.dot_merge, sym= s.sym, nsym=s.nsym))

def test_reset0():
    print('----------')
    print('3d tensor:')
    a = tensor.ones(settings=settings_full, s=(-1, 1, 1), D=(1, 2, 3))
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (1, 2, 3)
    assert a.tset.shape == (1, 3, 0)
    assert a.is_symmetric()

    b = a.match_legs(tensors=[a, a, a], legs=[0, 1, 2], conjs=[1, 1, 1])
    assert isclose(a.norm_diff(b),0, rel_tol=rel_tol)

    print('----------')
    print('0d tensor:')
    a = tensor.ones(settings=settings_full)  # s=() D=()
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.tset.shape == (1, 0, 0)
    assert isclose(a.to_number(),1, rel_tol=rel_tol)
    assert a.is_symmetric()

    print('----------')
    print('1d tensor:')
    a = tensor.zeros(settings=settings_full, s=1, D=5)  # s=(1,)
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5,)
    assert a.tset.shape == (1, 1, 0)
    b = a.match_legs(tensors=[a], legs=[0], conjs=[1], val='zeros')
    assert isclose(a.norm_diff(b),0, rel_tol=rel_tol)
    assert a.is_symmetric()

    print('----------------')
    print('diagonal tensor:')
    a = tensor.rand(settings=settings_full, isdiag=True, D=5)
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5, 5)
    assert a.tset.shape == (1, 1, 0)
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa), 0, rel_tol=rel_tol)
    assert a.is_symmetric()

def test_reset1():
    print('----------')
    print('4d tensor: ')
    a = tensor.ones(settings=settings_U1, s=(-1, 1, 1, 1),
                    t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                    D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (6, 3, 6, 1)
    assert a.tset.shape == (5, 4, 1)
    assert a.is_symmetric()

    b = a.match_legs(tensors=[a, a, a, a], legs=[0, 1, 2, 3], conjs=[1, 1, 1, 1], val='ones')
    assert isclose(a.norm_diff(b),0,rel_tol=rel_tol)

    print('----------')
    print('0d tensor:')
    a = tensor.ones(settings=settings_U1)  # s=()  # t=(), D=()
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == ()
    assert a.tset.shape == (1, 0, 1)
    assert isclose(a.to_number(),1,rel_tol=rel_tol)
    assert a.is_symmetric()

    print('----------')
    print('1d tensor:')
    a = tensor.ones(settings=settings_U1, s=-1, t=0, D=5)
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5,)
    assert a.tset.shape == (1, 1, 1)
    assert a.is_symmetric()

    b = a.match_legs(tensors=[a], legs=[0], conjs=[1], val='ones')
    assert isclose(a.norm_diff(b),0,rel_tol=rel_tol)

    print('----------------')
    print('diagonal tensor:')
    a = tensor.rand(settings=settings_U1, isdiag=True, t=0, D=5)
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (5, 5)
    assert a.tset.shape == (1, 1, 1)
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa),0,rel_tol=rel_tol)
    assert a.is_symmetric()

    print('----------------')
    print('diagonal tensor:')
    settings_U1_C= _gen_complex_conf(settings_U1)
    a = tensor.randR(settings=settings_U1_C, isdiag=True, t=(-1, 0, 1), D=(2, 3, 4))
    a.show_properties()
    npa = a.to_numpy()
    assert np.iscomplexobj(npa)
    assert npa.shape == (9, 9)
    assert a.tset.shape == (3, 1, 1)
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa),0,rel_tol=rel_tol)
    assert a.is_symmetric()

    print('----------------')
    print('diagonal tensor:')
    a = tensor.eye(settings=settings_U1, t=(-1, 0, 1), D=(2, 3, 4))
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (9, 9)
    assert a.tset.shape == (3, 1, 1)
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa),0,rel_tol=rel_tol)
    assert a.is_symmetric()

def test_reset2():
    print('----------')
    print('3d tensor: ')
    a = tensor.ones(settings=settings_Z2_U1, s=(-1, 1, 1),
                    t=((0, 1), (0, 2), 0, (-2, 2), (0, 1), (-2, 0, 2)),
                    D=((1, 2), (1, 2), 1, (1, 2), (2, 3), (1, 2, 3)))
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (9, 3, 30)
    assert a.tset.shape == (6, 3, 2)
    assert a.is_symmetric()

    b = a.match_legs(tensors=[a, a, a], legs=[0, 1, 2], conjs=[1, 1, 1], val='ones')
    assert isclose(a.norm_diff(b),0,rel_tol=rel_tol)

    print('----------')
    print('3d tensor:')
    a = tensor.ones(settings=settings_Z2_U1, s=(-1, 1, 1),
                    t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0)]],
                    D=[[1, 2], 3, [1, 2]])
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (3, 3, 3)
    assert a.tset.shape == (2, 3, 2)
    assert a.is_symmetric()

    b = a.match_legs(tensors=[a, a, a], legs=[0, 1, 2], conjs=[1, 1, 1], val='ones')
    assert isclose(a.norm_diff(b),0,rel_tol=rel_tol)

    print('----------')
    print('1d tensor:')
    a = tensor.ones(settings=settings_Z2_U1, s=1,
                    t=([0], [0]),
                    D=([1], [1]))
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (1,)
    assert a.tset.shape == (1, 1, 2)
    assert a.is_symmetric()

    b = a.match_legs(tensors=[a], legs=[0], conjs=[1], val='ones')
    assert isclose(a.norm_diff(b),0,rel_tol=rel_tol)

    print('----------')
    print('1d tensor:')
    a = tensor.ones(settings=settings_Z2_U1, s=1,
                    t=[[(0, 0)]],
                    D=[[2]])
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (2,)
    assert a.tset.shape == (1, 1, 2)
    assert a.is_symmetric()

    b = a.match_legs(tensors=[a], legs=[0], conjs=[1], val='ones')
    assert isclose(a.norm_diff(b),0,rel_tol=rel_tol)

    print('----------------')
    print('diagonal tensor:')
    a = tensor.rand(settings=settings_Z2_U1, isdiag=True,
                    t=[[(0, 0), (1, 1), (0, 2)]],
                    D=[[2, 2, 2]])
    a.show_properties()
    npa = a.to_numpy()
    assert np.isrealobj(npa)
    assert npa.shape == (6, 6)
    assert a.tset.shape == (3, 1, 2)
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa),0,rel_tol=rel_tol)
    assert a.is_symmetric()

    print('----------------')
    print('diagonal tensor:')
    settings_Z2_U1_C= _gen_complex_conf(settings_Z2_U1)
    a = tensor.randR(settings=settings_Z2_U1_C, isdiag=True,
                     t=[(0, 1), (0, 1)],
                     D=[(2, 2), (2, 2)])
    a.show_properties()
    npa = a.to_numpy()
    assert np.iscomplexobj(npa)
    assert npa.shape == (16, 16)
    assert a.tset.shape == (4, 1, 2)
    assert isclose(np.linalg.norm(np.diag(np.diag(npa)) - npa),0,rel_tol=rel_tol)
    assert a.is_symmetric()

def test_examples_in_reset_tensor():
    a = tensor.ones(settings=settings_U1, s=(-1, 1, 1),
                    t=[0, (-2, 0), (2, 0)],
                    D=[1, (1, 2), (1, 3)])
    a.show_properties()

    b = tensor.ones(settings=settings_U1_U1, s=(-1, 1, 1),
                    t=[0, 0, (-2, 0), (-2, 0), (2, 0), (2, 0)],
                    D=[1, 1, (1, 2), (1, 2), (1, 3), (1, 3)])
    b.show_properties()

    c = tensor.ones(settings=settings_U1_U1, s=(-1, 1, 1),
                    t=[[(0, 0)], [(-2, -2), (0, 0), (-2, 0), (0, -2)], [(2, 2), (0, 0), (2, 0), (0, 2)]],
                    D=[1, (1, 4, 2, 2), (1, 9, 3, 3)])
    c.show_properties()
    assert isclose(b.norm_diff(c),0,rel_tol=rel_tol)
    assert isclose(c.norm_diff(b),0,rel_tol=rel_tol)

    ta, da = a.get_tD()
    tb, db = b.get_tD()
    tc, dc = c.get_tD()

    a1 = tensor.ones(settings=settings_U1, s=(-1, 1, 1), t=ta, D=da)
    b1 = tensor.ones(settings=settings_U1_U1, s=(-1, 1, 1), t=tb, D=db)
    c1 = tensor.ones(settings=settings_U1_U1, s=(-1, 1, 1), t=tc, D=dc)
    assert isclose(a.norm_diff(a1),0,rel_tol=rel_tol)
    assert isclose(b.norm_diff(b1),0,rel_tol=rel_tol)
    assert isclose(c.norm_diff(c1),0,rel_tol=rel_tol)
    assert a.is_symmetric()
    assert b.is_symmetric()
    assert c.is_symmetric()

if __name__ == '__main__':
    test_reset0()
    test_reset1()
    test_reset2()
    test_examples_in_reset_tensor()
