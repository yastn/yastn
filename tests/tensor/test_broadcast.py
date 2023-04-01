"""Test yast.broadcast """
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_broadcast_dense():
    """ test broadcast on dense tensors """
    # a is a diagonal tensor to be broadcasted
    a = yast.rand(config=config_dense, s=(1, -1), isdiag=True, D=5)
    a1 = a.diag()

    # broadcast on tensor b
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))

    r1 = a.broadcast(b, axes=1)
    r2 = a1.tensordot(b, axes=(1, 1)).transpose((1, 0, 2, 3))
    r3 = a.tensordot(b, axes=(1, 1)).transpose((1, 0, 2, 3))
    r4 = b.tensordot(a, axes=(1, 1)).transpose((0, 3, 1, 2))
    assert all(yast.norm(r1 - x) < tol for x in (r2, r3, r4))


    # broadcast with conj
    a = yast.randC(config=config_dense, s=(-1, 1), D=(5, 5))
    a = a.diag()  # 5x5 isdiag == True
    a1 = a.diag()  # 5x5 isdiag == False
    b = yast.randC(config=config_dense, s=(1, -1, 1, -1), D=(2, 5, 2, 5))

    r1 = a.conj().broadcast(b, axes=1)
    r2 = a.broadcast(b.conj(), axes=1).conj()
    r3 = a1.tensordot(b, axes=(0, 1), conj=(1, 0)).transpose((1, 0, 2, 3))
    r5 = a.tensordot(b, axes=(0, 1), conj=(1, 0)).transpose((1, 0, 2, 3))
    r4 = b.tensordot(a, axes=(1, 0), conj=(0, 1)).transpose((0, 3, 1, 2))
    assert all(yast.norm(r1 - x) < tol for x in [r2, r3, r4, r5])
    assert all(x.is_consistent() for x in [r1, r2, r3, r4, r5])

    # broadcast and trace
    a = yast.rand(config=config_dense, isdiag=True, D=5)
    a1 = a.diag()  # 5x5 isdiag=False
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 5, 3, 5))
    r1 = a1.tensordot(b, axes=((1, 0), (1, 3)))
    r2 = a.tensordot(b, axes=((0, 1), (3, 1)))
    r3 = b.tensordot(a, axes=((1, 3), (1, 0)))
    assert all(yast.norm(r1 - x) < tol for x in [r2, r3])


def test_broadcast_U1():
    """ test broadcast on U1 tensors """
    leg0 = yast.Leg(config_U1, s=1, t=(-1, 1), D=(7, 8))
    leg1 = yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(1, 2, 3))
    leg2 = yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6))
    leg3 = yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg4 = yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(10, 11, 12))

    a = yast.rand(config=config_U1, isdiag=True, legs=(leg0, leg0.conj()))
    a1 = a.diag()
    assert a.get_shape() == a1.get_shape() == (15, 15)

    b = yast.rand(config=config_U1, legs=[leg1.conj(), leg2, leg3, leg4.conj()])
    assert b.get_shape() == (6, 15, 24, 33)

    # broadcast
    r1 = a.broadcast(b, axes=2)
    r2 = a.tensordot(b, axes=(1, 2)).transpose((1, 2, 0, 3))
    r3 = b.tensordot(a1, axes=(2, 1)).transpose((0, 1, 3, 2))
    r4 = b.tensordot(a, axes=(2, 1)).transpose((0, 1, 3, 2))
    assert all(x.is_consistent() for x in [r1, r2, r3, r4])
    assert all(yast.norm(r1 - x) < tol for x in [r2, r3, r4])

    c = yast.rand(config=config_U1, legs=[leg1.conj(), leg2, leg3, leg3.conj()])

    # broadcast with trace
    r1 = a1.tensordot(c, axes=((0, 1), (3, 2)))
    r2 = a.tensordot(c, axes=((0, 1), (3, 2)))
    r3 = c.tensordot(a, axes=((2, 3), (1, 0)))
    assert all(x.is_consistent() for x in [r1, r2, r3])
    assert all(yast.norm(r1 - x) < tol for x in [r2, r3])


def test_broadcast_Z2xU1():
    """ test broadcast on Z2xU1 tensors """
    leg0a = yast.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=[6, 3, 9, 6])
    leg0b = yast.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2), (1, 3)], D=[6, 3, 9, 6, 5])
    leg0c = yast.Leg(config_Z2xU1, s=1, t=[(0, 2), (0, 3)], D=[3, 4])
    leg1 = yast.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2)], D=[2, 3])
    leg2 = yast.Leg(config_Z2xU1, s=1, t=[(0, 1), (1, 0), (0, 0), (1, 1)], D=[4, 5, 6, 3])
    leg3 = yast.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2)], D=[3, 2])

    a = yast.rand(config=config_Z2xU1, legs=[leg0a.conj(), leg1.conj(), leg2, leg3])
    b = yast.eye(config=config_Z2xU1, legs=[leg0b, leg0b.conj()])
    c = yast.eye(config=config_Z2xU1, legs=[leg0c, leg0c.conj()])

    # broadcast
    r1 = b.broadcast(a, axes=0)
    r2 = b.tensordot(a, axes=(0, 0))
    r3 = a.tensordot(b, axes=(0, 0)).transpose((3, 0, 1, 2))
    assert all(x.is_consistent() for x in [r1, r2, r3])
    assert all(yast.norm(a - x) < tol for x in [r1, r2, r3])
    assert not any((r1.isdiag, r2.isdiag, r3.isdiag))

    # broadcast on diagonal
    r4 = c.broadcast(b, axes=1)
    r5 = b.broadcast(c, axes=1)
    assert r4.is_consistent()
    assert r4.isdiag
    nr4 = r4.to_numpy()
    assert nr4.shape == (3, 3)
    assert np.trace(nr4) == 3.
    assert (r4 - r5).norm() < tol

    # broadcast tensor b over multiple tensors in single call
    r1p, r5p = b.broadcast(a, c, axes=(0, 1))
    assert (r1 - r1p).norm() < tol
    assert (r5 - r5p).norm() < tol


def test_broadcast_exceptions():
    """ test broadcast raising errors """
    a = yast.rand(config=config_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    a_nondiag = a.diag()
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    with pytest.raises(yast.YastError):
        _ = a_nondiag.broadcast(b, axes=2)
        # First tensor should be diagonal.
    with pytest.raises(yast.YastError):
        bmf = b.fuse_legs(axes=(0, (1, 2), 3), mode='meta')
        _ = a.broadcast(bmf, axes=1)  
        # Second tensor`s leg specified by axes cannot be fused.
    with pytest.raises(yast.YastError):
        bhf = b.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
        _ = a.broadcast(bhf, axes=1)
        # Second tensor`s leg specified by axes cannot be fused.
    with pytest.raises(yast.YastError):
        a.broadcast(b, axes=1)  # Bond dimensions do not match.
    with pytest.raises(yast.YastError):
        _, _ = a.broadcast(b, b, axes=(1, 1, 1))
        # There should be exactly one axes for each tensor to be projected.


if __name__ == '__main__':
    test_broadcast_dense()
    test_broadcast_U1()
    test_broadcast_Z2xU1()
    test_broadcast_exceptions()
