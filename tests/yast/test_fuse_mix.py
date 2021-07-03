from context import yast
from context import config_dense, config_U1, config_Z2_U1
import numpy as np

tol = 1e-10


def test_fuse_mix():
    a = yast.ones(config=config_U1, s=(1, -1, 1, 1, -1, 1),
                  t=[(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    # assert a.get_shape() == (3, 5, 7, 9, 11, 13)
    ma = a.fuse_legs(axes=((0, 1), 2, (3, 4), 5), mode='meta')
    assert ma.nlegs == 6
    assert ma.mlegs == 4
    ha = a.fuse_legs(axes=((0, 1), 2, (3, 4), 5), mode='hard')
    assert ha.nlegs == 4
    assert ha.mlegs == 4

    hma = ma.fuse_legs(axes=(0, (3, 2, 1)), mode='hard')
    hha = ha.fuse_legs(axes=(0, (3, 2, 1)), mode='hard')
    assert hma.nlegs == 2
    assert hma.mlegs == 2
    assert hha.nlegs == 2
    assert hha.mlegs == 2
    assert yast.norm_diff(hma, hha) < tol


def test_dot_1_super_sparse():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                t=((0,), (0,), (-1, 0, 1), (-1, 0, 1)),
                D=((2,), (5,), (7, 8, 9), (10, 11, 12)))
    a.set_block(ts=(1, 1, 0, 0), Ds=(3, 6, 8, 11))
    # a.set_block(ts=(-1, -1, 0, 0), Ds=(1, 4, 8, 11))

    b = yast.rand(config=config_U1, s=(1, -1, -1, 1),
                t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1), (-2, 0, 2)),
                D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    ab = yast.tensordot(a, b, axes=((0, 1, 2, 3), (0, 1, 2, 3)))

    fa = yast.fuse_legs(a, axes=(0, (1, 2), 3), mode='hard')
    fb = yast.fuse_legs(b, axes=(0, (1, 2), 3), mode='hard')
    fab = yast.tensordot(fa, fb, axes=((0, 1, 2), (0, 1, 2)))

    ffa = yast.fuse_legs(fa, axes=((0, 1), 2), mode='hard')
    ffb = yast.fuse_legs(fb, axes=((0, 1), 2), mode='hard')
    ffab = yast.tensordot(ffa, ffb, axes=((0, 1), (0, 1)))

    fffa = yast.fuse_legs(ffa, axes=((0, 1),), mode='hard')
    fffb = yast.fuse_legs(ffb, axes=((0, 1),), mode='hard')
    fffab = yast.tensordot(fffa, fffb, axes=((0,), (0,)))

    assert yast.norm_diff(ab, fab) < tol
    assert yast.norm_diff(ab, ffab) < tol
    assert yast.norm_diff(ab, fffab) < tol

    ffa = yast.fuse_legs(fa, axes= ((0, 2), 1), mode='hard')
    ffb = yast.fuse_legs(fb, axes= ((0, 2), 1), mode='hard')
    ffab = yast.tensordot(ffa, ffb, axes=(0, 0))
    ab = yast.tensordot(a, b, axes=((0, 3), (0, 3)))
    uab = yast.unfuse_legs(ffab, axes=(0, 1))
    assert yast.norm_diff(ab, uab) < tol



if __name__ == '__main__':
    test_fuse_mix()
