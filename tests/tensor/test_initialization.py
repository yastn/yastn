"""
Test initialization with yastn.rand, yastn.zeros, yastn.ones, yastn.eye.

Also creating a dense tensor with to_numpy()
"""
import numpy as np
import pytest
import yastn
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_initialize_dense():
    """ initialization of dense tensor with no symmetry """
    # 3-dim tensor
    a = yastn.ones(config=config_dense, s=(-1, 1, 1), D=(1, 2, 3))
    npa = a.to_numpy()
    assert np.isrealobj(npa) == (config_dense.default_dtype == 'float64')
    assert npa.shape == a.get_shape() == (1, 2, 3)
    assert a.size == np.sum(npa != 0.)
    assert a.is_consistent()

    b = yastn.ones(config=config_dense, legs=a.get_legs())
    assert yastn.norm(a - b) < tol  # == 0.0
    assert b.is_consistent()

    # 0-dim tensor
    a = yastn.ones(config=config_dense)  # s=() D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa) == (config_dense.default_dtype == 'float64')
    assert npa.shape == a.get_shape() == ()
    assert a.size == np.sum(npa != 0.)
    assert pytest.approx(a.item(), rel=tol) == 1
    assert a.is_consistent()

    # 1-dim tensor
    a = yastn.zeros(config=config_dense, s=1, D=5)  # s=(1,)
    npa = a.to_numpy()
    assert np.isrealobj(npa) == (config_dense.default_dtype == 'float64')
    assert npa.shape == a.get_shape() == (5,)
    assert a.is_consistent()

    b = yastn.zeros(config=config_dense, legs=a.get_legs())
    assert yastn.norm(a - b) < tol  # == 0.0
    assert a.struct == b.struct

    # diagonal tensor
    a = yastn.rand(config=config_dense, isdiag=True, D=5)
    npa = a.to_numpy()
    assert np.isrealobj(npa) == (config_dense.default_dtype == 'float64')
    assert npa.shape == a.get_shape() == (5, 5)
    assert a.size == np.sum(npa != 0.)
    assert a.is_consistent()
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0


def test_initialize_U1():
    """ initialization of tensor with U1 symmetry """
    # 4-dim tensor
    legs = [yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(0, 2), D=(1, 2)),
            yastn.Leg(config_U1, s=1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(0,), D=(1,))]

    a1 = yastn.ones(config=config_U1, legs=legs)
    a2 = yastn.ones(config=config_U1, s=(-1, 1, 1, 1),
                  t=((-2, 0, 2), (0, 2), (-2, 0, 2), 0),
                  D=((1, 2, 3), (1, 2), (1, 2, 3), 1))
    a3 = yastn.ones(config=config_U1, legs=a2.get_legs())

    assert yastn.norm(a1 - a2) < tol  # == 0.0
    assert yastn.norm(a1 - a3) < tol  # == 0.0

    npa = a1.to_numpy()
    assert np.isrealobj(npa) == (config_U1.default_dtype == 'float64')
    assert npa.shape == a1.get_shape() == (6, 3, 6, 1)
    assert a1.size == np.sum(npa != 0.)
    assert a1.is_consistent()

    # 0-dim tensor
    a = yastn.ones(config=config_U1)  # s=()  # t=(), D=()
    npa = a.to_numpy()
    assert np.isrealobj(npa) == (config_U1.default_dtype == 'float64')
    assert npa.shape == a.get_shape() == ()
    assert a.size == np.sum(npa != 0.)
    assert pytest.approx(a.item(), rel=tol) == 1
    assert a.is_consistent()

    # 1-dim tensor
    a1 = yastn.ones(config=config_U1, s=-1, t=0, D=5)
    a2 = yastn.ones(config=config_U1, legs=[yastn.Leg(config_U1, s=-1, t=[0], D=[5])])
    a3 = yastn.ones(config=config_U1, legs=a2.get_legs())

    assert yastn.norm(a1 - a2) < tol  # == 0.0
    assert yastn.norm(a1 - a3) < tol  # == 0.0

    npa = a1.to_numpy()
    assert np.isrealobj(npa) == (config_U1.default_dtype == 'float64')
    assert npa.shape == a1.get_shape() == (5,)
    assert a1.size == np.sum(npa != 0.)
    assert a1.is_consistent()

    # diagonal tensor
    a1 = yastn.ones(config=config_U1, isdiag=True, t=0, D=5)
    leg = yastn.Leg(config_U1, s=1, t=[0], D=[5])
    a2 = yastn.ones(config=config_U1, isdiag=True, legs=leg) # a2 and a3 are equivalent initializations
    a3 = yastn.ones(config=config_U1, isdiag=True, legs=[leg, leg.conj()])
    assert all(yastn.norm(a1 - x) < tol  for x in (a2, a3))  # == 0.0

    npa = a1.to_numpy()
    assert np.isrealobj(npa) == (config_U1.default_dtype == 'float64')
    assert npa.shape == a1.get_shape() == (5, 5)
    assert a1.size == np.sum(npa != 0.)
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0
    assert a1.is_consistent()

    # diagonal tensor
    a1 = yastn.rand(config=config_U1, isdiag=True, t=(-1, 0, 1), D=(2, 3, 4), dtype='complex128')
    leg = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    a2 = yastn.ones(config=config_U1, isdiag=True, legs=leg, dtype='complex128')
    assert a1.struct == a2.struct

    npa = a1.to_numpy()
    assert np.iscomplexobj(npa)
    assert npa.shape == a1.get_shape() == (9, 9)
    assert a1.size == np.sum(npa != 0.)
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0
    assert a1.is_consistent()

    # diagonal tensor
    a3 = yastn.eye(config=config_U1, t=(-1, 0, 1), D=(2, 3, 4))
    assert a1.struct == a3.struct

    npa = a3.to_numpy()
    assert np.allclose(npa, np.eye(9))

    assert np.isrealobj(npa) == (config_U1.default_dtype == 'float64')
    assert npa.shape == a3.get_shape() == (9, 9)
    assert a3.size == np.sum(npa != 0.)
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa.conj()) < tol  # == 0.0
    assert a3.is_consistent()


def test_initialize_Z2xU1():
    """ initialization of tensor with more complicated symmetry indexed by 2 numbers"""
    # 3-dim tensor
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=[(0, 1), (1, 0)], D=[1, 2]),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, 0)], D=[3]),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, 1), (1, 0)], D=[1, 2])]
    a1 =  yastn.ones(config=config_Z2xU1, legs=legs)
    a2 = yastn.ones(config=config_Z2xU1, s=(-1, 1, 1),
                   t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0)]],
                   D=[[1, 2], 3, [1, 2]])
    a3 = yastn.ones(config=config_Z2xU1, legs=a2.get_legs())

    assert yastn.norm(a1 - a2) < tol  # == 0.0
    assert yastn.norm(a1 - a3) < tol  # == 0.0

    npa = a1.to_numpy()
    assert np.isrealobj(npa) == (config_Z2xU1.default_dtype == 'float64')
    assert npa.shape == a1.get_shape() == (3, 3, 3)
    assert a1.is_consistent()

    # 1-dim tensor
    a1 = yastn.ones(config=config_Z2xU1, legs=[yastn.Leg(config_Z2xU1, s=1, t=[(0, 0)], D=[2])])
    a2 = yastn.ones(config=config_Z2xU1, s=1, t=[[(0, 0)]], D=[[2]])
    a3 = yastn.ones(config=config_Z2xU1, legs=a2.get_legs())

    assert yastn.norm(a1 - a2) < tol  # == 0.0
    assert yastn.norm(a1 - a3) < tol  # == 0.0

    npa = a1.to_numpy()
    assert np.isrealobj(npa) == (config_Z2xU1.default_dtype == 'float64')
    assert npa.shape == a1.get_shape() == (2,)
    assert a1.size == np.sum(npa != 0.)
    assert a1.is_consistent()

    # diagonal tensor
    leg = yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (1, 1), (0, 2)], D=[2, 2, 2])
    a1 = yastn.rand(config=config_Z2xU1, isdiag=True, legs=leg)
    a2 = yastn.rand(config=config_Z2xU1, isdiag=True,
                  t=[[(0, 0), (1, 1), (0, 2)]],
                  D=[[2, 2, 2]])
    assert a1.struct == a2.struct
    npa = a1.to_numpy()
    assert np.isrealobj(npa) == (config_Z2xU1.default_dtype == 'float64')
    assert npa.shape == a1.get_shape() == (6, 6)
    assert a1.size == np.sum(npa != 0.)
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0
    assert a1.is_consistent()

    # diagonal tensor
    a1 = yastn.eye(config=config_Z2xU1,
                  t=[[(0, 1), (1, 2)], [(0, 1), (1, 1)]],
                  D=[[2, 5], [2, 7]])
    legs = [yastn.Leg(config_Z2xU1, s=1, t=[(0, 1), (1, 2)], D=[2, 5]),
            yastn.Leg(config_Z2xU1, s=-1, t=[(0, 1), (1, 1)], D=[2, 7])]
    a2 = yastn.eye(config=config_Z2xU1, legs=legs)  ## only the matching parts are used
    leg = yastn.Leg(config_Z2xU1, s=1, t=[(0, 1)], D=[2])
    a3 = yastn.eye(config=config_Z2xU1, legs=leg) # same as legs=[leg, leg.conj()]

    assert yastn.norm(a1 - a2) < tol  # == 0.0
    assert yastn.norm(a1 - a3) < tol  # == 0.0

    npa = a1.to_numpy()
    assert np.isrealobj(npa) == (config_Z2xU1.default_dtype == 'float64')
    assert npa.shape == a1.get_shape() == (2, 2)
    assert a1.size == np.sum(npa != 0.)

    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0
    assert a1.is_consistent()


def test_initialize_exceptions():
    """ test raise YaseError by fill_tensor()"""
    with pytest.raises(yastn.YastError):
        a = yastn.ones(config=config_dense, s=(1, 1), D=[(1,), (1,), (1,)])
        # Number of elements in D does not match tensor rank.
    with pytest.raises(yastn.YastError):
        a = yastn.ones(config=config_U1, s=(1, 1), t=[(0,), (0,)], D=[(1,), (1,), (1,)])
        # Number of elements in D does not match tensor rank
    with pytest.raises(yastn.YastError):
        a = yastn.ones(config=config_U1, s=(1, 1), t=[(0,), (0,), (0,)], D=[(1,), (1,)])
        # Number of elements in t does not match tensor rank.
    with pytest.raises(yastn.YastError):
        a = yastn.ones(config=config_U1, s=(1, 1), t=[(0,), (0,)], D=[(1, 2), (1,)])
        # Elements of t and D do not match
    with pytest.raises(yastn.YastError):
        a = yastn.eye(config=config_U1, t=[(0,), (0,)], D=[(1,), (2,)])
        # Diagonal tensor requires the same bond dimensions on both legs.
    with pytest.raises(yastn.YastError):
        _ = yastn.Tensor(config=config_U1, n=(0, 0))
        # n does not match the number of symmetry sectors
    with pytest.raises(yastn.YastError):
        _ = yastn.Tensor(config=config_U1, isdiag=True, s=(1, 1))
        # Diagonal tensor should have s equal (1, -1) or (-1, 1)
    with pytest.raises(yastn.YastError):
        _ = yastn.Tensor(config=config_U1, isdiag=True, n=1)
        # Tensor charge of a diagonal tensor should be 0


if __name__ == '__main__':
    test_initialize_dense()
    test_initialize_U1()
    test_initialize_Z2xU1()
    test_initialize_exceptions()
