""" Test elements of fuse_legs(... mode='meta') """
import numpy as np
import pytest
import yast
try:
    from .configs import config_U1, config_U1_force
except ImportError:
    from configs import config_U1, config_U1_force

tol = 1e-10  #pylint: disable=invalid-name


def test_fuse():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))
    b = a.fuse_legs(axes=(0, 1, (2, 3, 4)), mode='meta')
    c = b.fuse_legs(axes=(1, (0, 2)), mode='meta')
    c = c.unfuse_legs(axes=1)
    c = c.unfuse_legs(axes=2)
    d = c.move_leg(source=1, destination=0)
    assert yast.norm(a - d) < tol  # == 0.0

    e = yast.rand(config=config_U1_force, s=(-1, 1),
                  t=((0, 1), (0, 1)), D=((1, 2), (3, 4)))
    e = e.fuse_legs(axes=(0, 1), mode='meta')




def test_fuse_split():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))

    af = a.fuse_legs(axes=(0, (2, 1), (3, 4)), mode='meta')
    af = af.fuse_legs(axes=((0, 1), 2), mode='meta')
    Uf, Sf, Vf = yast.linalg.svd(af, axes=(0, 1))

    U, S, V = yast.linalg.svd(a, axes=((0, 1, 2), (3, 4)))
    U = U.fuse_legs(axes=(0, (2, 1), 3), mode='meta')
    U = U.fuse_legs(axes=((0, 1), 2), mode='meta')
    V = V.fuse_legs(axes=(0, (1, 2)), mode='meta')

    US = yast.tensordot(U, S, axes=(1, 0))
    a2 = yast.tensordot(US, V, axes=(1, 0))
    assert yast.norm(af - a2) < tol  # == 0.0
    USf = yast.tensordot(Uf, Sf, axes=(1, 0))
    a3 = yast.tensordot(USf, Vf, axes=(1, 0))
    assert yast.norm(af - a3) < tol  # == 0.0
    a3 = a3.unfuse_legs(axes=0)
    a3 = a3.unfuse_legs(axes=(1, 2)).move_leg(source=2, destination=1)
    assert yast.norm(a - a3) < tol  # == 0.0

    Qf, Rf = yast.linalg.qr(af, axes=(0, 1))
    Q, R = yast.linalg.qr(a, axes=((0, 1, 2), (3, 4)))
    Q = Q.fuse_legs(axes=(0, (2, 1), 3), mode='meta')
    Q = Q.fuse_legs(axes=((0, 1), 2), mode='meta')
    assert yast.norm(Q - Qf) < tol  # == 0.0
    Rf = Rf.unfuse_legs(axes=1)
    assert yast.norm(R - Rf) < tol  # == 0.0

    aH = yast.tensordot(af, af, axes=(1, 1), conj=(0, 1))
    Vf, Uf = yast.linalg.eigh(aH, axes=(0, 1))
    Uf = Uf.unfuse_legs(axes=0)
    UVf = yast.tensordot(Uf, Vf, axes=(2, 0))
    aH2 = yast.tensordot(UVf, Uf, axes=(2, 2), conj=(0, 1))
    aH = aH.unfuse_legs(axes=(0, 1))
    assert yast.norm(aH2 - aH) < tol  # == 0.0


def test_fuse_transpose():
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    # assert a.get_shape() == (3, 5, 7, 9, 11, 13)
    b = a.fuse_legs(axes=((0, 1), 2, (3, 4), 5), mode='meta')

    c = np.transpose(b, axes=(3, 2, 1, 0))
    assert c.get_shape() == (13, 99, 7, 15)
    c = c.unfuse_legs(axes=(1, 3))
    assert c.get_shape() == (13, 9, 11, 7, 3, 5)

    c = b.move_leg(source=1, destination=2)
    assert c.get_shape() == (15, 99, 7, 13)
    c = c.unfuse_legs(axes=(1, 0))
    assert c.get_shape() == (3, 5, 9, 11, 7, 13)


def test_get_shapes():
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])

    assert a.get_shape() == (3, 5, 7, 9, 11, 13)
    assert a.get_signature() == (-1, -1, -1, 1, 1, 1)
    assert a.to_numpy().shape == (3, 5, 7, 9, 11, 13)
    b = a.to_nonsymmetric()
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)

    a = a.fuse_legs(axes=[0, 1, (2, 3), (4, 5)], mode='meta')
    assert a.get_shape() == (3, 5, 63, 143)
    assert a.get_signature() == (-1, -1, -1, 1)
    assert a.to_numpy().shape == (3, 5, 63, 143)
    b = a.to_nonsymmetric()
    assert b.get_shape() == (3, 5, 63, 143)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)

    a = a.fuse_legs(axes=[0, (1, 2, 3)], mode='meta')
    assert a.get_shape() == (3, 28389)
    assert a.get_signature() == (-1, -1)
    assert a.to_numpy().shape == (3, 28389)
    b = a.to_nonsymmetric()
    assert b.get_shape() == (3, 28389)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)

    a = a.fuse_legs(axes=[(0, 1)], mode='meta')
    assert a.get_shape() == (a.size, )
    assert a.get_signature() == (-1,)
    assert a.to_numpy().shape == (a.size,)
    b = a.to_nonsymmetric()
    assert b.get_shape() == (a.size,)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)


def test_fuse_match_legs():
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((-1, 0, 1), (1,), (-1, 1), (0, 1), (0, 1, 2)),
                  D=((2, 1, 2), (4,), (4, 6), (7, 8), (9, 10, 11)))
    af = a.fuse_legs(axes=((0, 1), (2, 3, 4), 5), mode='meta')
    bf = b.fuse_legs(axes=(0, (1, 2), 3, 4), mode='meta')
    bff = bf.fuse_legs(axes=(0, (1, 2), 3), mode='meta')

    c1 = yast.match_legs(tensors=[a, a, a, b, b, b], legs=[2, 3, 4, 1, 2, 3], conjs=[0, 0, 0, 1, 1, 1], val='ones')
    r1 = yast.ncon([a, b, c1], [[-1, -2, 1, 2, 3, -3], [-4, 4, 5, 6, -5], [1, 2, 3, 4, 5, 6]], [0, 1, 0])

    c2 = yast.match_legs(tensors=[af, bff], legs=[1, 1], conjs=[0, 1], val='ones')
    r2 = yast.ncon([af, bff, c2], [[-1, 1, -2], [-3, 2, -4], [1, 2]], [0, 1, 0])

    c3 = c2.unfuse_legs(axes=1)  # partial unfuse
    r3 = yast.ncon([af, bf, c3], [[-1, 1, -2], [-3, 2, 3, -4], [1, 2, 3]], [0, 1, 0])

    assert yast.norm(r3 - r2) < tol  # == 0.0
    r2 = r2.unfuse_legs(axes=0)
    assert yast.norm(r1 - r2) < tol  # == 0.0


def test_fuse_block():
    l1 = yast.rand(config=config_U1, s=(1, 1), t=[(0, 1), (0, 1)], D=[(1, 2), (2, 3)])
    l2 = yast.rand(config=config_U1, s=(1, 1), t=[(0, 1), (0, 1)], D=[(2, 3), (3, 4)])
    c1 = yast.rand(config=config_U1, s=(-1, -1, 1, 1),
                   t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                   D=[(1, 2), (2, 3), (3, 4), (4, 5)])
    c2 = yast.rand(config=config_U1, s=(-1, -1, 1, 1),
                   t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                   D=[(2, 3), (3, 4), (4, 5), (5, 6)])
    r1 = yast.rand(config=config_U1, s=(-1, -1), t=[(0, 1), (0, 1)], D=[(3, 4), (4, 5)])
    r2 = yast.rand(config=config_U1, s=(-1, -1), t=[(0, 1), (0, 1)], D=[(4, 5), (5, 6)])

    s1 = yast.ncon([l1, c1, r1], [[1, 2], [1, 2, 3, 4], [3, 4]])
    s1 = s1 + yast.ncon([l2, c2, r2], [[1, 2], [1, 2, 3, 4], [3, 4]])
    l1 = l1.fuse_legs(axes=[(0, 1)], mode='meta')
    l2 = l2.fuse_legs(axes=[(0, 1)], mode='meta')
    c1 = c1.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
    c2 = c2.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
    r1 = r1.fuse_legs(axes=[(0, 1)], mode='meta')
    r2 = r2.fuse_legs(axes=[(0, 1)], mode='meta')
    bl = yast.block({1: l1, 2: l2})
    bc = yast.block({(1, 1): c1, (2, 2): c2})
    br = yast.block({1: r1, 2: r2})
    s2 = yast.ncon([bl, bc, br], [[1], [1, 2], [2]])
    assert yast.norm(s1 - s2) < tol
    assert pytest.approx(s1.item(), rel=tol) == s2.item()


def test_fuse_legs_exceptions():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))
    b = yast.rand(config=config_U1, isdiag=True, t=(0, 1), D=(1, 2))
    with pytest.raises(yast.YastError):
        b.fuse_legs(axes=((0, 1),), mode='meta')
        # Cannot fuse legs of a diagonal tensor.
    with pytest.raises(yast.YastError):
        b.unfuse_legs(axes=0)
        # Cannot unfuse legs of a diagonal tensor.
    with pytest.raises(yast.YastError):
        a.fuse_legs(axes=((0, 1, 2, 3, 4),), mode='wrong')
    # mode not in (`meta`, `hard`). Mode can be specified in config file.


if __name__ == '__main__':
    test_fuse()
    test_fuse_split()
    test_fuse_transpose()
    test_get_shapes()
    test_fuse_match_legs()
    test_fuse_block()
    test_fuse_legs_exceptions()
