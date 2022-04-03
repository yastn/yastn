""" Test yaps.mask """
import pytest
import yast
try:
    from .configs import config_U1, config_Z2xU1
except ImportError:
    from configs import config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_mask_1():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b0 = yast.rand(config=config_U1, isdiag=True, t=(-1, 1), D=(7, 8))

    b = b0.copy()
    b[(-1, -1)] = b[(-1, -1)] < -2
    b[(1, 1)] = b[(1, 1)] > 0

    bd = b.trace(axes=(0, 1)).item()
    c = b.mask_apply(a, axis=2)
    ls = c.get_leg_structure(axis=2)
    assert len(ls) == 1
    assert (1,) in ls and ls[(1,)] == bd

    d1 = b.mask_apply(b0, axis=-1)
    ls = d1.get_leg_structure(axis=1)
    assert len(ls) == 1 and (1,) in ls and ls[(1,)] == bd

    d0 = b.mask_apply(b0, axis=0)
    ls = d0.get_leg_structure(axis=1)
    assert len(ls) == 1 and (1,) in ls and ls[(1,)] == bd
    assert yast.norm(d0 - d1) < tol


def test_mask_2():
    a = yast.rand(config=config_Z2xU1, s=(-1, -1, 1, 1),
                    t=[((0, 0), (0, 2), (1, 0), (1, 2)),
                       ((0, 0), (0, 2)),
                       ((0, 1), (1, 0), (0, 0), (1, 1)),
                       ((0, 0), (0, 2))],
                    D=[(6, 3, 9, 6), (3, 2), (4, 5, 6, 3), (2, 3)])
    b = yast.rand(config=config_Z2xU1, isdiag=True,
                   t=[[(0, 0), (0, 2), (1, 0), (1, 2)]],
                   D=[[6, 3, 9, 6]])
    b[(0, 0, 0, 0)] *= 0

    bgt = b > 0
    blt = b < 0
    bge = b >= 0
    ble = b <= 0

    assert bgt.trace().item() + ble.trace().item() == blt.trace().item() + bge.trace().item() == a.get_shape(axes=0)

    def mask_and_shape(aa, bb):
        bd = bb.trace(axes=(0, 1)).item()
        c = bb.mask_apply(aa, axis=0)
        ls = c.get_leg_structure(axis=0)
        assert sum(ls.values()) == bd

    for bb in [bgt, blt, bge, ble]:
        mask_and_shape(a, bb)

    assert blt.mask_apply(bge, axis=0).trace() < tol
    assert ble.mask_apply(bgt, axis=1).trace() < tol


def test_mask_exceptions():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = yast.rand(config=config_U1, isdiag=True, t=(-1, 1), D=(7, 8))
    b_nondiag = b.diag()
    with pytest.raises(yast.YastError):
        b_nondiag.mask_apply(a, axis=2)  # Error in broadcast/mask: tensor b should be diagonal.
    with pytest.raises(yast.YastError):
        b.mask_apply(a, axis=(1,))  # Error in broadcast/mask: axis should be an int.
    with pytest.raises(yast.YastError):
        amf = a.fuse_legs(axes=(0, (1, 2), 3), mode='meta')
        b.mask_apply(amf, axis=1)  # Error in broadcast/mask: leg of tensor a specified by axis cannot be fused.
    with pytest.raises(yast.YastError):
        ahf = a.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
        b.mask_apply(ahf, axis=1)  # Error in mask: leg of tensor a specified by axis cannot be fused.
    with pytest.raises(yast.YastError):
        b.mask_apply(a, axis=1)  # Error in mask: bond dimensions do not match.


if __name__ == '__main__':
    test_mask_1()
    test_mask_2()
    test_mask_exceptions()
