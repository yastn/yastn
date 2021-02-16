""" 
Test elements of logical leg's fusion. 
"""

tol = 1e-10

from math import isclose
import yamps.yast as yast
import config_U1_R
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
    assert a.norm_diff(d) < tol  # == 0.0

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
    r1.norm_diff(r1uf) < tol  # == 0.0

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
    assert af.norm_diff(a2) < tol  # == 0.0
    a3 = Uf.dot(Sf, axes=(1, 0)).dot(Vf, axes=(1, 0))
    assert af.norm_diff(a3) < tol  # == 0.0
    a3.unfuse_legs(axes=0, inplace=True)
    a3.unfuse_legs(axes=(1, 2), inplace=True)
    a3.moveaxis(source=2, destination=1, inplace=True)
    assert a.norm_diff(a3) < tol  # == 0.0

    Qf, Rf = af.split_qr(axes=(0, 1))
    Q, R = a.split_qr(axes=((0, 1, 2), (3, 4)))
    Q = Q.fuse_legs(axes=(0, (2, 1), 3))
    Q.fuse_legs(axes=((0, 1), 2), inplace=True)
    assert Q.norm_diff(Qf) < tol  # == 0.0
    Rf.unfuse_legs(axes=1, inplace=True)
    assert R.norm_diff(Rf) < tol  # == 0.0

    aH = af.dot(af, axes=(1, 1), conj=(0, 1))
    Vf, Uf = aH.split_eigh(axes=(0, 1))
    Uf.unfuse_legs(axes=0, inplace = True)
    aH2 = (Uf.dot(Vf, axes=(2, 0))).dot(Uf, axes=(2, 2), conj=(0, 1))
    aH.unfuse_legs(axes=(0, 1), inplace=True)
    assert aH2.norm_diff(aH) < tol  # == 0.0


def test_fuse_transpose():
    a = yast.ones(config=config_U1_R, s=(-1, -1, -1, 1, 1, 1),
                    t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                    D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    # assert a.get_shape() == (3, 5, 7, 9, 11, 13)
    b = a.fuse_legs(axes=((0, 1), 2, (3, 4), 5))

    c = np.transpose(b, axes=(3, 2, 1, 0))
    assert c.get_shape() == (13, 99, 7, 15)
    c.unfuse_legs(axes=(1, 3), inplace=True)
    assert c.get_shape() == (13, 9, 11, 7, 3, 5)

    c = b.moveaxis(source=1, destination=2)
    assert c.get_shape() == (15, 99, 7, 13)
    c.unfuse_legs(axes=(1, 0), inplace=True)
    assert c.get_shape() == (3, 5, 9, 11, 7, 13)


def test_get_shapes():
    a = yast.ones(config=config_U1_R, s=(-1, -1, -1, 1, 1, 1),
                    t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                    D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])

    assert a.get_shape() == (3, 5, 7, 9, 11, 13)
    assert a.get_signature() == (-1, -1, -1, 1, 1, 1)
    assert a.to_numpy().shape == (3, 5, 7, 9, 11, 13)
    b = a.to_nonsymmetric()
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)

    a.fuse_legs(axes=[0, 1, (2, 3), (4, 5)], inplace=True)
    assert a.get_shape() == (3, 5, 63, 143)
    assert a.get_signature() == (-1, -1, -1, 1)
    assert a.to_numpy().shape == (3, 5, 63, 143)
    b = a.to_nonsymmetric()
    assert b.get_shape() == (3, 5, 63, 143)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)

    a.fuse_legs(axes=[0, (1, 2, 3)], inplace=True)
    assert a.get_shape() == (3, 28389)
    assert a.get_signature() == (-1, -1)
    assert a.to_numpy().shape == (3, 28389)
    b = a.to_nonsymmetric()
    assert b.get_shape() == (3, 28389)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)

    a.fuse_legs(axes=[(0, 1)], inplace=True)
    assert a.get_shape() == (a.get_size(), )
    assert a.get_signature() == (-1,)
    assert a.to_numpy().shape == (a.get_size(), )
    b = a.to_nonsymmetric()
    assert b.get_shape() == (a.get_size(), )
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)




def test_fuse_match_legs():
    a = yast.ones(config=config_U1_R, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    b = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1, 1,),
                  t=((-1, 0, 1), (1,), (-1, 1), (0, 1), (0, 1, 2)),
                  D=((2, 1, 2), (4,), (4, 6), (7, 8), (9, 10, 11)))
    af = a.fuse_legs(axes=((0, 1), (2, 3, 4), 5))
    bf = b.fuse_legs(axes=(0, (1, 2), 3, 4))
    bff = bf.fuse_legs(axes=(0, (1, 2), 3))
    
    c1 = yast.match_legs(tensors=[a, a, a, b, b, b], legs=[2, 3, 4, 1, 2, 3], conjs=[0, 0, 0, 1, 1, 1], val='ones')
    r1 = yast.ncon([a, b, c1], [[-1, -2, 1, 2, 3, -3], [-4, 4, 5, 6, -5], [1, 2, 3, 4, 5, 6]],  [0, 1, 0])

    c2 = yast.match_legs(tensors=[af, bff], legs=[1, 1], conjs=[0, 1], val='ones')
    r2 = yast.ncon([af, bff, c2], [[-1, 1, -2], [-3, 2, -4], [1, 2]],  [0, 1, 0])
    
    c3 = c2.unfuse_legs(axes=1)  # partial unfuse
    r3 = yast.ncon([af, bf, c3], [[-1, 1, -2], [-3, 2, 3, -4], [1, 2, 3]],  [0, 1, 0])
    
    assert r3.norm_diff(r2) < tol  # == 0.0
    r2.unfuse_legs(axes=0, inplace=True)
    assert r1.norm_diff(r2) < tol  # == 0.0


def test_fuse_block():
    l1 = yast.rand(config=config_U1_R, s=(1, 1), t=[(0, 1), (0, 1)], D=[(1, 2), (2, 3)])
    l2 = yast.rand(config=config_U1_R, s=(1, 1), t=[(0, 1), (0, 1)], D=[(2, 3), (3, 4)])
    c1 = yast.rand(config=config_U1_R, s=(-1, -1, 1, 1),
                   t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                   D=[(1, 2), (2, 3), (3, 4), (4, 5)])
    c2 = yast.rand(config=config_U1_R, s=(-1, -1, 1, 1),
                   t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                   D=[(2, 3), (3, 4), (4, 5), (5, 6)])
    r1 = yast.rand(config=config_U1_R, s=(-1, -1), t=[(0, 1), (0, 1)], D=[(3, 4), (4, 5)])
    r2 = yast.rand(config=config_U1_R, s=(-1, -1), t=[(0, 1), (0, 1)], D=[(4, 5), (5, 6)])

    s1 = yast.ncon([l1, c1, r1], [[1, 2], [1, 2, 3, 4], [3, 4]]) +\
         yast.ncon([l2, c2, r2], [[1, 2], [1, 2, 3, 4], [3, 4]])
    l1.fuse_legs(axes=[(0, 1)], inplace=True)
    l2.fuse_legs(axes=[(0, 1)], inplace=True)
    c1.fuse_legs(axes=((0, 1), (2, 3)), inplace=True)
    c2.fuse_legs(axes=((0, 1), (2, 3)), inplace=True)
    r1.fuse_legs(axes=[(0, 1)], inplace=True)
    r2.fuse_legs(axes=[(0, 1)], inplace=True)
    bl = yast.block({1: l1, 2: l2})
    bc = yast.block({(1, 1): c1, (2, 2): c2})
    br = yast.block({1: r1, 2: r2})
    s2 = yast.ncon([bl, bc, br], [[1], [1, 2], [2]])
    assert isclose(s1, s2, rel_tol=tol)


if __name__ == '__main__':
    test_fuse()
    test_fuse_dot()
    test_fuse_split()
    test_fuse_transpose()
    test_get_shapes()
    test_fuse_match_legs()
    test_fuse_block()
