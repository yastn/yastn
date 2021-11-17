import numpy as np
import yast
try:
    from .configs import config_dense, config_U1, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_broadcast_diag0():
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    b = yast.rand(config=config_dense, s=(1, -1), isdiag=True, D=5)
    b1 = b.diag()

    r1 = a.broadcast_diag(b, axes=(1, 1))
    r2 = a.tensordot(b1, axes=(1, 1)).transpose((0, 3, 1, 2))
    r3 = a.tensordot(b, axes=(1, 1)).transpose((0, 3, 1, 2))
    r4 = b.tensordot(a, axes=(1, 1)).transpose((1, 0, 2, 3))

    assert r1.norm_diff(r2) < tol
    assert r1.norm_diff(r3) < tol
    assert r1.norm_diff(r4) < tol

    a = yast.randR(config=config_dense, s=(1, -1, 1, -1), D=(2, 5, 2, 5))
    b = yast.randR(config=config_dense, s=(-1, 1), D=(5, 5))
    b = b.diag()  # 5x5 diagonal
    b1 = b.diag()

    r1 = a.broadcast_diag(b, axes=(1, 0), conj=(1, 0))
    r2 = a.broadcast_diag(b, axes=(1, 0), conj=(0, 1)).conj()
    r3 = b1.tensordot(a, axes=(0, 1), conj=(0, 1)).transpose((1, 0, 2, 3))
    r4 = b.tensordot(a, axes=(0, 1), conj=(0, 1)).transpose((1, 0, 2, 3))
    r5 = a.tensordot(b, axes=(1, 0), conj=(1, 0)).transpose((0, 3, 1, 2))

    assert r1.norm_diff(r2) < tol
    assert r1.norm_diff(r3) < tol
    assert r1.norm_diff(r4) < tol
    assert r1.norm_diff(r5) < tol

    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 5, 3, 5))
    b = yast.rand(config=config_dense, isdiag=True, D=5)
    b1 = b.diag()

    r1 = a.tensordot(b1, axes=((1, 3), (1, 0)))
    r2 = a.tensordot(b, axes=((1, 3), (1, 0)))
    r3 = b.tensordot(a, axes=((0, 1), (3, 1)))

    assert r1.norm_diff(r2) < tol
    assert r1.norm_diff(r3) < tol



def test_broadcast_diag1():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    b = yast.rand(config=config_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    b1 = b.diag()

    r1 = a.broadcast_diag(b, axes=(2, 1))
    r2 = a.tensordot(b1, axes=(2, 1)).transpose((0, 1, 3, 2))
    r3 = a.tensordot(b, axes=(2, 1)).transpose((0, 1, 3, 2))
    r4 = b.tensordot(a, axes=(1, 2)).transpose((1, 2, 0, 3))

    assert r1.norm_diff(r2) < tol
    assert r1.norm_diff(r3) < tol
    assert r1.norm_diff(r4) < tol

    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (7, 8, 9)))
    b = yast.rand(config=config_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    b1 = b.diag()

    r1 = a.tensordot(b1, axes=((2, 3), (1, 0)))
    r2 = a.tensordot(b, axes=((2, 3), (1, 0)))
    r3 = b.tensordot(a, axes=((0, 1), (3, 2)))
    assert r1.norm_diff(r2) < tol
    assert r1.norm_diff(r3) < tol

    bb = b.copy()
    bb.A[(-1, -1)] = bb.A[(-1, -1)] < -2
    bb.A[(1, 1)] = bb.A[(1, 1)] > 0

    bd = bb.trace(axes=(0, 1)).item()
    c = a.mask(bb, axis=2)
    ls = c.get_leg_structure(axis=2)
    assert len(ls) == 1
    assert (1,) in ls and ls[(1,)] == bd

    d1 = b.mask(bb, axis=-1)
    ls = d1.get_leg_structure(axis=1)
    assert len(ls) == 1 and (1,) in ls and ls[(1,)] == bd

    d0 = b.mask(bb, axis=0)
    ls = d0.get_leg_structure(axis=1)
    assert len(ls) == 1 and (1,) in ls and ls[(1,)] == bd
    assert yast.norm_diff(d0, d1) < tol


def test_broadcast_diag2():
    a = yast.rand(config=config_Z2_U1, s=(-1, -1, 1, 1),
                    t=[((0, 0), (0, 2), (1, 0), (1, 2)),
                       ((0, 0), (0, 2)),
                       ((0, 1), (1, 0), (0, 0), (1, 1)),
                       ((0, 0), (0, 2))],
                    D=[(6, 3, 9, 6), (3, 2), (4, 5, 6, 3), (2, 3)])
    b = yast.eye(config=config_Z2_U1,
                   t=[[(0, 0), (0, 2), (1, 0), (1, 2), (1, 3)]],
                   D=[[6, 3, 9, 6, 5]])
    c = yast.eye(config=config_Z2_U1,
                   t=[[(0, 2), (0, 3)]],
                   D=[[3, 4]])

    r1 = a.broadcast_diag(b, axes=(0, 0))
    r2 = a.tensordot(b, axes=(0, 0)).transpose((3, 0, 1, 2))
    r3 = b.tensordot(a, axes=(0, 0))

    assert r1.norm_diff(a) < tol
    assert r2.norm_diff(a) < tol
    assert r3.norm_diff(a) < tol
    assert not any((r1.isdiag, r2.isdiag, r3.isdiag))

    r4 = b.broadcast_diag(c, axes=(1, 0))
    assert r4.isdiag

    r5 = r4.to_numpy()
    assert r5.shape == (3, 3)
    assert np.trace(r5) == 3.


if __name__ == '__main__':
    test_broadcast_diag0()
    test_broadcast_diag1()
    test_broadcast_diag2()


