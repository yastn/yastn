""" Test elements of logical leg's fusion. """

import yamps.yast as yast
import config_U1_R
import pytest
import numpy as np


def test_fuse():
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1, 1,),
                    t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                    D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))
    b = a.fuse_legs(axes=(0, 1, (2, 3, 4)))
    c = b.fuse_legs(axes=(1, (0, 2)))
    c.unfuse_legs(axes=1, inplace=True)
    c.unfuse_legs(axes=2, inplace=True)
    d = c.moveaxis(source=1, destination=0)
    assert pytest.approx(a.norm_diff(d)) == 0.

def test_fuse_dot():
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1, 1,),
                    t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                    D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))
    b = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1, 1,),
                    t=((-1, 0, 1), (1,), (-1, 1), (0, 1), (0, 1, 2)),
                    D=((2, 1, 2), (4,), (4, 6), (7, 8), (9, 10, 11)))
    
    af = a.fuse_legs(axes=(0, (2, 1), (3, 4)))
    bf = b.fuse_legs(axes=(0, (2, 1), (3, 4)))
    af.fuse_legs(axes=((0, 1), 2), inplace=True)
    bf.fuse_legs(axes=((0, 1), 2), inplace=True)

    r1 = a.dot(b, axes=((0, 1, 2), (0, 1, 2)), conj=(0, 1))
    r1f = af.dot(bf, axes=(0, 0), conj=(0, 1))
    r1uf = r1f.unfuse_legs(axes=(0, 1))
    pytest.approx(r1.norm_diff(r1uf)) == 0.

def test_fuse_split():
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1, 1,),
                    t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                    D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))
    
    af = a.fuse_legs(axes=(0, (2, 1), (3, 4)))
    af.fuse_legs(axes=((0, 1), 2), inplace=True)
    Uf, Sf, Vf = af.split_svd(axes=(0, 1))

    U, S, V = a.split_svd(axes=((0, 1, 2), (3, 4)))
    U = U.fuse_legs(axes=(0, (2, 1), 3))
    U.fuse_legs(axes=((0, 1), 2), inplace=True)
    V = V.fuse_legs(axes=(0, (1, 2)))

    a2 = U.dot(S, axes=(1, 0)).dot(V, axes=(1, 0))
    assert pytest.approx(af.norm_diff(a2)) == 0.
    a3 = Uf.dot(Sf, axes=(1, 0)).dot(Vf, axes=(1, 0))
    assert pytest.approx(af.norm_diff(a3)) == 0.
    a3.unfuse_legs(axes=0, inplace=True)
    a3.unfuse_legs(axes=(1, 2), inplace=True)
    a3.moveaxis(source=2, destination=1, inplace=True)
    assert pytest.approx(a.norm_diff(a3)) == 0.

    Qf, Rf = af.split_qr(axes=(0, 1))
    Q, R = a.split_qr(axes=((0, 1, 2), (3, 4)))
    Q = Q.fuse_legs(axes=(0, (2, 1), 3))
    Q.fuse_legs(axes=((0, 1), 2), inplace=True)
    assert pytest.approx(Q.norm_diff(Qf)) == 0.
    Rf.unfuse_legs(axes=1, inplace=True)
    assert pytest.approx(R.norm_diff(Rf)) == 0.

    aH = af.dot(af, axes=(1, 1), conj=(0, 1))
    Vf, Uf = aH.split_eigh(axes=(0, 1))
    Uf.unfuse_legs(axes=0, inplace = True)
    aH2 = (Uf.dot(Vf, axes=(2, 0))).dot(Uf, axes=(2, 2), conj=(0, 1))
    aH.unfuse_legs(axes=(0, 1), inplace=True)
    assert pytest.approx(aH2.norm_diff(aH), abs=1e-10) == 0.

if __name__ == '__main__':
    test_fuse()
    test_fuse_dot()
    test_fuse_split()
