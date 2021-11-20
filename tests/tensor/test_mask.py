
import numpy as np
import yast
try:
    from .configs import config_dense, config_U1, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_mask1():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b0 = yast.rand(config=config_U1, isdiag=True, t=(-1, 1), D=(7, 8))

    b = b0.copy()
    b.A[(-1, -1)] = b.A[(-1, -1)] < -2
    b.A[(1, 1)] = b.A[(1, 1)] > 0

    bd = b.trace(axes=(0, 1)).item()
    c = a.mask(b, axis=2)
    ls = c.get_leg_structure(axis=2)
    assert len(ls) == 1
    assert (1,) in ls and ls[(1,)] == bd

    d1 = b0.mask(b, axis=-1)
    ls = d1.get_leg_structure(axis=1)
    assert len(ls) == 1 and (1,) in ls and ls[(1,)] == bd

    d0 = b0.mask(b, axis=0)
    ls = d0.get_leg_structure(axis=1)
    assert len(ls) == 1 and (1,) in ls and ls[(1,)] == bd
    assert yast.norm_diff(d0, d1) < tol


def test_mask2():
    a = yast.rand(config=config_Z2_U1, s=(-1, -1, 1, 1),
                    t=[((0, 0), (0, 2), (1, 0), (1, 2)),
                       ((0, 0), (0, 2)),
                       ((0, 1), (1, 0), (0, 0), (1, 1)),
                       ((0, 0), (0, 2))],
                    D=[(6, 3, 9, 6), (3, 2), (4, 5, 6, 3), (2, 3)])
    b = yast.rand(config=config_Z2_U1, isdiag=True,
                   t=[[(0, 0), (0, 2), (1, 0), (1, 2)]],
                   D=[[6, 3, 9, 6]])
    b.A[(0, 0, 0, 0)] *= 0

    bgt = b > 0
    blt = b < 0
    bge = b >= 0
    ble = b <= 0

    assert bgt.trace().item() + ble.trace().item() == blt.trace().item() + bge.trace().item() == a.get_shape(axes=0)

    def mask_and_shape(aa, bb):
        bd = bb.trace(axes=(0, 1)).item()
        c = aa.mask(bb, axis=0)
        ls = c.get_leg_structure(axis=0)
        assert sum(ls.values()) == bd

    for bb in [bgt, blt, bge, ble]:
        mask_and_shape(a, bb)

    assert yast.mask(bge, blt, axis=0).trace() < tol
    assert yast.mask(bgt, ble, axis=1).trace() < tol


if __name__ == '__main__':
    test_mask1()
    test_mask2()
