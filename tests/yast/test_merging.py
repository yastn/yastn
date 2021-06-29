from context import yast
from context import config_dense, config_U1, config_Z2_U1
import numpy as np

tol = 1e-10


def test_merge_trace():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (1, 2, 3), (4, 5, 6)))

    af = yast.fuse_legs_hard(a, axes=((1, 2), (3, 0)))
    tra = yast.trace(a, axes=((1, 2), (3, 0)))
    traf = yast.trace(af, axes=(0, 1))
    assert yast.norm_diff(tra, traf) < tol


def test_merge_split():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))

    af = a.fuse_legs_hard(axes=(0, (2, 1), (3, 4)))
    af.fuse_legs_hard(axes=((0, 1), 2), inplace=True)
    Uf, Sf, Vf = yast.linalg.svd(af, axes=(0, 1))

    U, S, V = yast.linalg.svd(a, axes=((0, 1, 2), (3, 4)))
    U = U.fuse_legs_hard(axes=(0, (2, 1), 3))
    U.fuse_legs_hard(axes=((0, 1), 2), inplace=True)
    V = V.fuse_legs_hard(axes=(0, (1, 2)))

    US = yast.tensordot(U, S, axes=(1, 0))
    a2 = yast.tensordot(US, V, axes=(1, 0))
    assert af.norm_diff(a2) < tol  # == 0.0
    USf = yast.tensordot(Uf, Sf, axes=(1, 0))
    a3 = yast.tensordot(USf, Vf, axes=(1, 0))
    assert af.norm_diff(a3) < tol  # == 0.0
    a3.unfuse_legs_hard(axes=0, inplace=True)
    a3.unfuse_legs_hard(axes=(1, 2), inplace=True)
    a3.moveaxis(source=2, destination=1, inplace=True)
    assert a.norm_diff(a3) < tol  # == 0.0

    Qf, Rf = yast.linalg.qr(af, axes=(0, 1))
    Q, R = yast.linalg.qr(a, axes=((0, 1, 2), (3, 4)))
    Q = Q.fuse_legs_hard(axes=(0, (2, 1), 3))
    Q.fuse_legs_hard(axes=((0, 1), 2), inplace=True)
    assert Q.norm_diff(Qf) < tol  # == 0.0
    Rf.unfuse_legs_hard(axes=1, inplace=True)
    assert R.norm_diff(Rf) < tol  # == 0.0

    aH = yast.tensordot(af, af, axes=(1, 1), conj=(0, 1))
    Vf, Uf = yast.linalg.eigh(aH, axes=(0, 1))
    Uf.unfuse_legs_hard(axes=0, inplace=True)
    UVf = yast.tensordot(Uf, Vf, axes=(2, 0))
    aH2 = yast.tensordot(UVf, Uf, axes=(2, 2), conj=(0, 1))
    aH.unfuse_legs_hard(axes=(0, 1), inplace=True)
    assert aH2.norm_diff(aH) < tol  # == 0.0


def test_merge_transpose():
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    # assert a.get_shape() == (3, 5, 7, 9, 11, 13)
    b = a.fuse_legs_hard(axes=((0, 1), 2, (3, 4), 5))

    c = np.transpose(b, axes=(3, 2, 1, 0))
    assert c.get_shape() == (13, 99, 7, 15)
    c.unfuse_legs_hard(axes=(1, 3), inplace=True)
    assert c.get_shape() == (13, 9, 11, 7, 3, 5)

    c = b.moveaxis(source=1, destination=2)
    assert c.get_shape() == (15, 99, 7, 13)
    c.unfuse_legs_hard(axes=(1, 0), inplace=True)
    assert c.get_shape() == (3, 5, 9, 11, 7, 13)


def test_dot_2():
    t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    t2 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.rand(config=config_Z2_U1, s=(-1, 1, 1, -1, -1),
                  t=(t1, t1, t2, t2, t2),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (7, 8, 9, 10), (5, 6, 7, 8), (1, 2, 2, 4)))
    b = yast.rand(config=config_Z2_U1, s=(1, -1, 1, 1),
                  t=(t1, t1, t2, t2),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (1, 2, 2, 4), (5, 6, 7, 8)))

    aa = yast.fuse_legs_hard(a, axes=((0, 3), (4, 1), 2))
    bb = yast.fuse_legs_hard(b, axes=((0, 3), (2, 1)))

    c = yast.tensordot(a, b, axes=((0, 1, 3, 4), (0, 1, 3, 2)))
    cc = yast.tensordot(aa, bb, axes=((0, 1), (0, 1)))
    assert yast.norm_diff(c, cc) < tol

    aaa = yast.unfuse_legs_hard(aa, axes=(0, 1)).transpose(axes=(0, 3, 4, 1, 2))
    assert yast.norm_diff(a, aaa) < tol
    bbb = yast.unfuse_legs_hard(bb, axes=0)
    bbb = yast.unfuse_legs_hard(bbb, axes=2).transpose(axes=(0, 3, 2, 1))
    assert yast.norm_diff(b, bbb) < tol

    aa = yast.fuse_legs_hard(aa, axes=(0, (1, 2)))
    aa = yast.fuse_legs_hard(aa, axes=[(0, 1)])
    aaa = yast.unfuse_legs_hard(aa, axes=0)
    aaa = yast.unfuse_legs_hard(aaa, axes=1)
    aaa = yast.unfuse_legs_hard(aaa, axes=(0, 1)).transpose(axes=(0, 3, 4, 1, 2))
    assert yast.norm_diff(a, aaa) < tol


def test_dot_1():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = yast.rand(config=config_U1, s=(1, -1, 1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 0, 1)),
                  D=((1, 2, 3), (4, 5, 6), (10, 7, 11)))

    bb = yast.fuse_legs_hard(b, axes=((0, 1), 2))
    aa =  yast.fuse_legs_hard(a, axes=((0, 1), 2, 3))

    aaa = yast.unfuse_legs_hard(aa, axes=0)
    bbb = yast.unfuse_legs_hard(bb, axes=0)

    c = yast.tensordot(a, b, axes=((0, 1), (0, 1)))
    cc = yast.tensordot(aa, bb, axes=(0, 0))


    assert yast.norm_diff(c, cc) < tol
    assert yast.norm_diff(a, aaa) < tol
    assert yast.norm_diff(b, bbb) < tol


def test_dot_1_sparse():
    a = yast.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=-2)
    a.set_block(ts=(2, 1, 0, 1), Ds=(2, 1, 5, 3), val='rand')
    a.set_block(ts=(1, 1, -1, 1), Ds=(1, 1, 6, 3), val='rand')
    a.set_block(ts=(1, 2, -1, 2), Ds=(1, 2, 6, 4), val='rand')

    b = yast.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=1)
    b.set_block(ts=(1, 2, 0, 0), Ds=(1, 2, 4, 4), val='rand')
    b.set_block(ts=(1, 1, 1, 0), Ds=(1, 1, 3, 4), val='rand')
    b.set_block(ts=(2, 2, 1, 0), Ds=(2, 2, 3, 4), val='rand')

    aa = yast.fuse_legs_hard(a, axes=((1, 0), 2, 3))
    bb = yast.fuse_legs_hard(b, axes=((1, 0), 2, 3))
    xx = yast.match_legs([aa, aa], legs=[0, 0], conjs=[1, 0], val='rand')
    yast.tensordot(xx, aa, axes=(1, 0))
    yast.tensordot(xx, aa, axes=(0, 0), conj = (1, 0))

    c = yast.tensordot(a, b, axes=((0, 1), (0, 1)), conj=(1, 0))
    cc = yast.tensordot(aa, bb, axes=(0, 0), conj=(1, 0))
    assert yast.norm_diff(c, cc) < tol

    aaa = yast.unfuse_legs_hard(aa, axes=0).transpose(axes=(1, 0, 2, 3))
    bbb = yast.unfuse_legs_hard(bb, axes=0).transpose(axes=(1, 0, 2, 3))
    assert yast.norm_diff(a, aaa) < tol
    assert yast.norm_diff(b, bbb) < tol


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

    fa = yast.fuse_legs_hard(a, axes=(0, (1, 2), 3))
    fb = yast.fuse_legs_hard(b, axes=(0, (1, 2), 3))
    fab = yast.tensordot(fa, fb, axes=((0, 1, 2), (0, 1, 2)))

    ffa = yast.fuse_legs_hard(fa, axes=((0, 1), 2))
    ffb = yast.fuse_legs_hard(fb, axes=((0, 1), 2))
    ffab = yast.tensordot(ffa, ffb, axes=((0, 1), (0, 1)))

    fffa = yast.fuse_legs_hard(ffa, axes=((0, 1),))
    fffb = yast.fuse_legs_hard(ffb, axes=((0, 1),))
    fffab = yast.tensordot(fffa, fffb, axes=((0,), (0,)))

    assert yast.norm_diff(ab, fab) < tol
    assert yast.norm_diff(ab, ffab) < tol
    assert yast.norm_diff(ab, fffab) < tol

    ffa = yast.fuse_legs_hard(fa, axes= ((0, 2), 1))
    ffb = yast.fuse_legs_hard(fb, axes= ((0, 2), 1))
    ffab = yast.tensordot(ffa, ffb, axes=(0, 0))
    ab = yast.tensordot(a, b, axes=((0, 3), (0, 3)))
    uab = yast.unfuse_legs_hard(ffab, axes=(0, 1))
    assert yast.norm_diff(ab, uab) < tol


if __name__ == '__main__':
    test_merge_trace()
    test_merge_split()
    test_merge_transpose()
    test_dot_1()
    test_dot_2()
    test_dot_1_sparse()
    test_dot_1_super_sparse()
    
