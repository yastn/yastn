"""Test yaps.broadcast """
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_broadcast_0():
    a = yast.rand(config=config_dense, s=(1, -1), isdiag=True, D=5)
    a1 = a.diag()
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))

    r1 = a.broadcast(b, axis=1)
    r2 = a1.tensordot(b, axes=(1, 1)).transpose((1, 0, 2, 3))
    r3 = a.tensordot(b, axes=(1, 1)).transpose((1, 0, 2, 3))
    r4 = b.tensordot(a, axes=(1, 1)).transpose((0, 3, 1, 2))
    assert all(yast.norm(r1 - x) < tol for x in (r2, r3, r4))

    a = yast.randC(config=config_dense, s=(-1, 1), D=(5, 5))
    a = a.diag()  # 5x5 isdiag=True
    a1 = a.diag()
    b = yast.randC(config=config_dense, s=(1, -1, 1, -1), D=(2, 5, 2, 5))

    r1 = a.broadcast(b, axis=1, conj=(1, 0))
    r2 = a.broadcast(b, axis=1, conj=(0, 1)).conj()
    r3 = a1.tensordot(b, axes=(0, 1), conj=(1, 0)).transpose((1, 0, 2, 3))
    r5 = a.tensordot(b, axes=(0, 1), conj=(1, 0)).transpose((1, 0, 2, 3))
    r4 = b.tensordot(a, axes=(1, 0), conj=(0, 1)).transpose((0, 3, 1, 2))
    assert all(yast.norm(r1 - x) < tol for x in [r2, r3, r4, r5])
    assert all(x.is_consistent() for x in [r1, r2, r3, r4, r5])

    a = yast.rand(config=config_dense, isdiag=True, D=5)
    a1 = a.diag()
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 5, 3, 5))

    r1 = a1.tensordot(b, axes=((1, 0), (1, 3)))
    r2 = a.tensordot(b, axes=((0, 1), (3, 1)))
    r3 = b.tensordot(a, axes=((1, 3), (1, 0)))
    assert all(yast.norm(r1 - x) < tol for x in [r2, r3])


def test_broadcast_1():
    a = yast.rand(config=config_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    a1 = a.diag()
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    r1 = a.broadcast(b, axis=2)
    r2 = a.tensordot(b, axes=(1, 2)).transpose((1, 2, 0, 3))
    r3 = b.tensordot(a1, axes=(2, 1)).transpose((0, 1, 3, 2))
    r4 = b.tensordot(a, axes=(2, 1)).transpose((0, 1, 3, 2))
    assert all(x.is_consistent() for x in [r1, r2, r3, r4])
    assert all(yast.norm(r1 - x) < tol for x in [r2, r3, r4])

    a = yast.rand(config=config_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    a1 = a.diag()
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (7, 8, 9)))

    r1 = a1.tensordot(b, axes=((0, 1), (3, 2)))
    r2 = a.tensordot(b, axes=((0, 1), (3, 2)))
    r3 = b.tensordot(a, axes=((2, 3), (1, 0)))
    assert all(x.is_consistent() for x in [r1, r2, r3])
    assert all(yast.norm(r1 - x) < tol for x in [r2, r3])


def test_broadcast_2():
    a = yast.rand(config=config_Z2xU1, s=(-1, -1, 1, 1),
                    t=[((0, 0), (0, 2), (1, 0), (1, 2)),
                       ((0, 0), (0, 2)),
                       ((0, 1), (1, 0), (0, 0), (1, 1)),
                       ((0, 0), (0, 2))],
                    D=[(6, 3, 9, 6), (3, 2), (4, 5, 6, 3), (2, 3)])
    b = yast.eye(config=config_Z2xU1,
                   t=[[(0, 0), (0, 2), (1, 0), (1, 2), (1, 3)]],
                   D=[[6, 3, 9, 6, 5]])
    c = yast.eye(config=config_Z2xU1,
                   t=[[(0, 2), (0, 3)]],
                   D=[[3, 4]])

    r1 = b.broadcast(a, axis=0)
    r2 = b.tensordot(a, axes=(0, 0))
    r3 = a.tensordot(b, axes=(0, 0)).transpose((3, 0, 1, 2))
    assert all(x.is_consistent() for x in [r1, r2, r3])
    assert all(yast.norm(a - x) < tol for x in [r1, r2, r3])
    assert not any((r1.isdiag, r2.isdiag, r3.isdiag))

    r4 = c.broadcast(b, axis=1)
    r5 = b.broadcast(c, axis=1)
    assert r4.is_consistent()
    assert r4.isdiag
    nr4 = r4.to_numpy()
    assert nr4.shape == (3, 3)
    assert np.trace(nr4) == 3.
    assert (r4 - r5).norm() < tol

    r1p, r5p = b.broadcast(a, c, axis=(0, 1))
    assert (r1 - r1p).norm() < tol
    assert (r5 - r5p).norm() < tol


def test_broadcast_exceptions():
    a = yast.rand(config=config_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    a_nondiag = a.diag()
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    with pytest.raises(yast.YastError):
        a_nondiag.broadcast(b, axis=2)  # Error in broadcast/mask: tensor b should be diagonal.
    with pytest.raises(yast.YastError):
        bmf = b.fuse_legs(axes=(0, (1, 2), 3), mode='meta')
        a.broadcast(bmf, axis=1)  # Error in broadcast/mask: leg of tensor a specified by axis cannot be fused.
    with pytest.raises(yast.YastError):
        bhf = b.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
        a.broadcast(bhf, axis=1)  # Error in broadcast: leg of tensor a specified by axis cannot be fused.
    with pytest.raises(yast.YastError):
        a.broadcast(b, axis=1)  # Error in broadcast: bond dimensions do not match.


if __name__ == '__main__':
    test_broadcast_0()
    test_broadcast_1()
    test_broadcast_2()
    test_broadcast_exceptions()
