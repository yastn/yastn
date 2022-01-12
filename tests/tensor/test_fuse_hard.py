""" Test elements of fuse_legs(... mode='hard') """
import numpy as np
import pytest
import yast
try:
    from .configs import config_U1, config_dense, config_Z2xU1
except ImportError:
    from configs import config_U1, config_Z2xU1, config_dense

tol = 1e-10  #pylint: disable=invalid-name





def test_hard_split():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))

    af = a.fuse_legs(axes=(0, (2, 1), (3, 4)), mode='hard')
    af.fuse_legs(axes=((0, 1), 2), inplace=True, mode='hard')
    Uf, Sf, Vf = yast.linalg.svd(af, axes=(0, 1))

    U, S, V = yast.linalg.svd(a, axes=((0, 1, 2), (3, 4)))
    U = U.fuse_legs(axes=(0, (2, 1), 3), mode='hard')
    U.fuse_legs(axes=((0, 1), 2), inplace=True, mode='hard')
    V = V.fuse_legs(axes=(0, (1, 2)), mode='hard')

    US = yast.tensordot(U, S, axes=(1, 0))
    a2 = yast.tensordot(US, V, axes=(1, 0))
    assert yast.norm(af - a2) < tol  # == 0.0
    USf = yast.tensordot(Uf, Sf, axes=(1, 0))
    a3 = yast.tensordot(USf, Vf, axes=(1, 0))
    assert yast.norm(af - a3) < tol  # == 0.0
    a3.unfuse_legs(axes=0, inplace=True)
    a3.unfuse_legs(axes=(1, 2), inplace=True)
    a3.move_leg(source=2, destination=1, inplace=True)
    assert yast.norm(a - a3) < tol  # == 0.0

    Qf, Rf = yast.linalg.qr(af, axes=(0, 1))
    Q, R = yast.linalg.qr(a, axes=((0, 1, 2), (3, 4)))
    Q = Q.fuse_legs(axes=(0, (2, 1), 3), mode='hard')
    Q.fuse_legs(axes=((0, 1), 2), inplace=True, mode='hard')
    assert yast.norm(Q - Qf) < tol  # == 0.0
    Rf.unfuse_legs(axes=1, inplace=True)
    assert yast.norm(R - Rf) < tol  # == 0.0

    aH = yast.tensordot(af, af, axes=(1, 1), conj=(0, 1))
    Vf, Uf = yast.linalg.eigh(aH, axes=(0, 1))
    Uf.unfuse_legs(axes=0, inplace=True)
    UVf = yast.tensordot(Uf, Vf, axes=(2, 0))
    aH2 = yast.tensordot(UVf, Uf, axes=(2, 2), conj=(0, 1))
    aH.unfuse_legs(axes=(0, 1), inplace=True)
    assert yast.norm(aH2 - aH) < tol  # == 0.0


def test_hard_transpose():
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    # assert a.get_shape() == (3, 5, 7, 9, 11, 13)
    b = a.fuse_legs(axes=((0, 1), 2, (3, 4), 5), mode='hard')

    c = np.transpose(b, axes=(3, 2, 1, 0))
    assert c.get_shape() == (13, 99, 7, 15)
    c.unfuse_legs(axes=(1, 3), inplace=True)
    assert c.get_shape() == (13, 9, 11, 7, 3, 5)

    c = b.move_leg(source=1, destination=2)
    assert c.get_shape() == (15, 99, 7, 13)
    c.unfuse_legs(axes=(1, 0), inplace=True)
    assert c.get_shape() == (3, 5, 9, 11, 7, 13)


def test_hard_dot_2():
    t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    t2 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.rand(config=config_Z2xU1, s=(-1, 1, 1, -1, -1),
                  t=(t1, t1, t2, t2, t2),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (7, 8, 9, 10), (5, 6, 7, 8), (1, 2, 2, 4)))
    b = yast.rand(config=config_Z2xU1, s=(1, -1, 1, 1),
                  t=(t1, t1, t2, t2),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (1, 2, 2, 4), (5, 6, 7, 8)))

    aa = yast.fuse_legs(a, axes=((0, 3), (4, 1), 2), mode='hard')
    bb = yast.fuse_legs(b, axes=((0, 3), (2, 1)), mode='hard')

    c = yast.tensordot(a, b, axes=((0, 1, 3, 4), (0, 1, 3, 2)))
    cc = yast.tensordot(aa, bb, axes=((0, 1), (0, 1)))
    assert yast.norm(c -  cc) < tol

    aaa = yast.unfuse_legs(aa, axes=(0, 1)).transpose(axes=(0, 3, 4, 1, 2))
    assert yast.norm(a - aaa) < tol
    bbb = yast.unfuse_legs(bb, axes=0)
    bbb = yast.unfuse_legs(bbb, axes=2).transpose(axes=(0, 3, 2, 1))
    assert yast.norm(b - bbb) < tol

    aa = yast.fuse_legs(aa, axes=(0, (1, 2)), mode='hard')
    aa = yast.fuse_legs(aa, axes=[(0, 1)], mode='hard')
    aaa = yast.unfuse_legs(aa, axes=0)
    aaa = yast.unfuse_legs(aaa, axes=1)
    aaa = yast.unfuse_legs(aaa, axes=(0, 1)).transpose(axes=(0, 3, 4, 1, 2))
    assert yast.norm(a - aaa) < tol


def test_hard_dot_1():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = yast.rand(config=config_U1, s=(1, -1, 1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 0, 1)),
                  D=((1, 2, 3), (4, 5, 6), (10, 7, 11)))

    bb = yast.fuse_legs(b, axes=((0, 1), 2), mode='hard')
    aa =  yast.fuse_legs(a, axes=((0, 1), 2, 3), mode='hard')

    aaa = yast.unfuse_legs(aa, axes=0)
    bbb = yast.unfuse_legs(bb, axes=0)

    c = yast.tensordot(a, b, axes=((0, 1), (0, 1)))
    cc = yast.tensordot(aa, bb, axes=(0, 0))

    assert yast.norm(c -  cc) < tol
    assert yast.norm(a - aaa) < tol
    assert yast.norm(b - bbb) < tol


def test_hard_dot_1_sparse():
    a = yast.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=-2)
    a.set_block(ts=(2, 1, 0, 1), Ds=(2, 1, 5, 3), val='rand')
    a.set_block(ts=(1, 1, -1, 1), Ds=(1, 1, 6, 3), val='rand')
    a.set_block(ts=(1, 2, -1, 2), Ds=(1, 2, 6, 4), val='rand')

    b = yast.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=1)
    b.set_block(ts=(1, 2, 0, 0), Ds=(1, 2, 4, 4), val='rand')
    b.set_block(ts=(1, 1, 1, 0), Ds=(1, 1, 3, 4), val='rand')
    b.set_block(ts=(2, 2, 1, 0), Ds=(2, 2, 3, 4), val='rand')

    aa = yast.fuse_legs(a, axes=((1, 0), 2, 3), mode='hard')
    bb = yast.fuse_legs(b, axes=((1, 0), 2, 3), mode='hard')
    xx = yast.match_legs([aa, aa], legs=[0, 0], conjs=[1, 0], val='rand')
    yast.tensordot(xx, aa, axes=(1, 0))
    yast.tensordot(xx, aa, axes=(0, 0), conj = (1, 0))

    c = yast.tensordot(a, b, axes=((0, 1), (0, 1)), conj=(1, 0))
    cc = yast.tensordot(aa, bb, axes=(0, 0), conj=(1, 0))
    assert yast.norm(c -  cc) < tol

    aat = aa.fuse_legs(axes=((1, 2), 0), mode='hard').conj()
    bbt = bb.fuse_legs(axes=(0, (1, 2)), mode='hard')
    aat.show_properties()
    bbt.show_properties()
    ccc = yast.tensordot(aat, bbt, axes=(1, 0))
    assert yast.norm(c -  ccc.unfuse_legs(axes=(0, 1))) < tol

    aaa = yast.unfuse_legs(aa, axes=0).transpose(axes=(1, 0, 2, 3))
    bbb = yast.unfuse_legs(bb, axes=0).transpose(axes=(1, 0, 2, 3))
    assert yast.norm(a - aaa) < tol
    assert yast.norm(b - bbb) < tol




def _test_fuse_mix(a):
    ma = a.fuse_legs(axes=((0, 1), (2, 3), (4, 5)), mode='meta')
    assert (ma.ndim_n, ma.ndim) == (6, 3)
    ha = a.fuse_legs(axes=((0, 1), (2, 3), (4, 5)), mode='hard')
    assert (ha.ndim_n, ha.ndim) == (3, 3)

    hma = ma.fuse_legs(axes=((2, 0), 1), mode='hard')
    assert (hma.ndim_n, hma.ndim) == (2, 2)
    hha = ha.fuse_legs(axes=((2, 0), 1), mode='hard')
    assert (hha.ndim_n, hha.ndim) == (2, 2)
    mma = ma.fuse_legs(axes=((2, 0), 1), mode='meta')
    assert (mma.ndim_n, mma.ndim) == (6, 2)
    mha = ha.fuse_legs(axes=((2, 0), 1), mode='meta')
    assert (mha.ndim_n, mha.ndim) == (3, 2)

    assert yast.norm(hma - hha) < tol

    fmma = yast.fuse_meta_to_hard(mma)
    fmha = yast.fuse_meta_to_hard(mha)
    fhha = yast.fuse_meta_to_hard(hha)
    assert yast.norm(fmma - hha) < tol
    assert yast.norm(fmha - hha) < tol
    assert yast.norm(fhha - hha) < tol


def test_fuse_mix():
    a = yast.randR(config=config_U1, s=(1, -1, 1, 1, -1, 1),
                    t=[(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3)],
                    D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    _test_fuse_mix(a)

    a = yast.Tensor(config=config_U1, s=(1, -1, 1, 1, -1, 1))
    a.set_block(ts=(1, 2, -1, 2, 0, 0), Ds=(1, 2, 3, 4, 5, 6), val='randR')
    a.set_block(ts=(2, 1, 1, -2, 1, 1), Ds=(6, 5, 4, 3, 2, 1), val='randR')
    _test_fuse_mix(a)


def test_auxliary_merging_functions():
    mf1 = (1,)
    nt = yast.tensor._merging._mf_to_ntree(mf1)
    mfx = yast.tensor._merging._ntree_to_mf(nt)
    assert mf1 == mfx
    yast.tensor._merging._ntree_eliminate_lowest(nt)
    new_mf1 = tuple(yast.tensor._merging._ntree_to_mf(nt))
    assert new_mf1 == (1,)

    mf2 = (9, 5, 1, 3, 2, 1, 1, 1, 1, 4, 1, 3, 1, 1, 1)
    nt = yast.tensor._merging._mf_to_ntree(mf2)
    mfx = yast.tensor._merging._ntree_to_mf(nt)
    assert mf2 == mfx
    yast.tensor._merging._ntree_eliminate_lowest(nt)
    new_mf2 = tuple(yast.tensor._merging._ntree_to_mf(nt))
    assert new_mf2 == (6, 4, 1, 2, 1, 1, 1, 2, 1, 1)

    mf3 = (8, 2, 1, 1, 1, 5, 3, 1, 1, 1, 1, 1)
    nt = yast.tensor._merging._mf_to_ntree(mf3)
    mfx = yast.tensor._merging._ntree_to_mf(nt)
    assert mf3 == mfx
    yast.tensor._merging._ntree_eliminate_lowest(nt)
    new_mf3 = tuple(yast.tensor._merging._ntree_to_mf(nt))
    assert new_mf3 == (5, 1, 1, 3, 1, 1, 1)

    axes, new_mfs = yast.tensor._merging._consume_mfs_lowest((mf1, mf2, mf3))
    assert axes == ((0,), (1,), (2, 3), (4,), (5,), (6,), (7, 8, 9), (10, 11), (12,), (13, 14, 15), (16,), (17,))
    assert (new_mf1, new_mf2, new_mf3) == new_mfs


def test_fuse_hard_dense():
    # for dense
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(6, 2, 6, 2), dtype='float64')
    af = yast.fuse_legs(a, axes=((1, 2), (3, 0)), mode='hard')
    tra = yast.trace(a, axes=((1, 2), (3, 0)))
    traf = yast.trace(af, axes=(0, 1))
    assert yast.norm(tra - traf) < tol


if __name__ == '__main__':
    test_fuse_hard_dense()
    test_hard_split()
    test_hard_transpose()
    test_hard_dot_1()
    test_hard_dot_2()
    test_hard_dot_1_sparse()
    test_fuse_mix()
    test_auxliary_merging_functions()
