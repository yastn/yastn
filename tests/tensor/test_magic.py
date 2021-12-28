import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12  #pylint: disable=invalid-name


def magic_vs_numpy(f, a, b):
    """ f is lambda expresion using magic methods on a and b tensors """
    tDsa = {ia: a.get_leg_structure(ia) for ia in range(a.ndim)}
    tDsb = {ib: b.get_leg_structure(ib) for ib in range(b.ndim)}
    na = a.to_numpy(leg_structures=tDsb)
    nb = b.to_numpy(leg_structures=tDsa) # make sure nparrays are consistent
    nc = f(na, nb)

    c = f(a, b)
    assert c.is_consistent()
    assert all(c.are_independent(x) for x in (a, b))

    cn = c.to_numpy()
    assert np.linalg.norm(cn - nc) < tol
    return c


def test_add_0():
    """ test magic methods on dense tensor """
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='float64')
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='float64')
    c = yast.rand(config=config_dense, isdiag=True, D=5, dtype='float64')
    d = c.copy()
    e = c.clone()

    r1 = magic_vs_numpy(lambda x, y: x + 2 * y, a, b)
    r2 = a.apxb(b, 2)
    r3 = magic_vs_numpy(lambda x, y: 2 * x + y, b, a)
    assert yast.norm_diff(r1, r2) < tol  # == 0.0
    assert yast.norm(r1 - r2) < tol  # == 0.0
    assert yast.norm_diff(r1, r3) < tol  # == 0.0
    assert yast.norm(r1 - r3) < tol  # == 0.0

    r4 = magic_vs_numpy(lambda x, y: 2. * x - (y + y), c, d)
    r5 = magic_vs_numpy(lambda x, y: 2. * x - y - y, c, e)
    assert r4.norm() < tol  # == 0.0
    assert r5.norm() < tol  # == 0.0
    assert all(yast.are_independent(c, x) for x in (d, e))


def test_add_1():
    """ test magic methods on U1 tensors """
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), dtype='float64')

    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), dtype='float64')

    r1 = magic_vs_numpy(lambda x, y: x + 2 * y, a, b)
    r2 = a.apxb(b, 2)
    r3 = magic_vs_numpy(lambda x, y: 2 * x + y, b, a)
    assert yast.norm_diff(r1, r2) < tol  # == 0.0
    assert yast.norm(r1 - r2) < tol  # == 0.0
    assert yast.norm_diff(r1, r3) < tol  # == 0.0
    assert yast.norm(r1 - r3) < tol  # == 0.0

    c = yast.eye(config=config_U1, t=1, D=5)
    d = yast.eye(config=config_U1, t=2, D=5)

    r4 = magic_vs_numpy(lambda x, y: 2. * x + y, c, d)
    r5 = magic_vs_numpy(lambda x, y: x - (2 * y) ** 2 + 2 * y, c, d)
    r6 = magic_vs_numpy(lambda x, y: x - y / 0.5, d, c)
    assert all(pytest.approx(x.norm().item(), rel=tol) == 5 for x in (r4, r5, r6))


def test_add_2():
    """ test magic methods on more complicated symmetries"""
    a = yast.randC(config=config_Z2_U1, s=(-1, 1, 1, 1),
        t=[[(0, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 2), (1, -2), (1, 0), (1, 2)], [(0, 0), (0, 2)]],
        D=((1, 2, 4), (2, 3), (2, 6, 3, 6, 9), (4, 7)))
    b = yast.randC(config=config_Z2_U1, s=(-1, 1, 1, 1),
        t=[[(0, 0), (1, 0), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)], [(0, 0)]],
        D=((1, 2, 4), (2, 3), (2, 4, 6, 3, 6, 9), 4))

    r1 = magic_vs_numpy(lambda x, y: x + 2 * y, a, b)
    r2 = a.apxb(b, 2)
    r3 = magic_vs_numpy(lambda x, y:  x / 0.5 + y, b, a)
    assert yast.norm_diff(r1, r2) < tol  # == 0.0
    assert yast.norm(r1 - r2) < tol  # == 0.0
    assert yast.norm_diff(r1, r3) < tol  # == 0.0
    assert yast.norm(r1 - r3) < tol  # == 0.0


def magic_hf_vs_numpy(f, a, b):
    """
    Test operations on a and b 4-legs tensors, combined with hard fusion.

    f is lambda expresion using magic methods on a and b tensors
    """
    fa = yast.fuse_legs(a, axes=(0, (3, 2), 1), mode='hard')
    fb = yast.fuse_legs(b, axes=(0, (3, 2), 1), mode='hard')
    ffa = yast.fuse_legs(fa, axes=((0, 2), 1), mode='hard')
    ffb = yast.fuse_legs(fb, axes=((0, 2), 1), mode='hard')
    fffa = yast.fuse_legs(ffa, axes=[(0, 1)], mode='hard')
    fffb = yast.fuse_legs(ffb, axes=[(0, 1)], mode='hard')

    c = magic_vs_numpy(f, a, b)
    cf = yast.fuse_legs(c, axes=(0, (3, 2), 1), mode='hard')
    fc = f(fa, fb)
    fcf = yast.fuse_legs(fc, axes=((0, 2), 1), mode='hard')
    ffc = f(ffa, ffb)
    ffcf = yast.fuse_legs(ffc, axes=[(0, 1)], mode='hard')
    fffc = f(fffa, fffb)
    assert all(yast.norm_diff(x, y) < tol for x, y in zip((fc, ffc, fffc), (cf, fcf, ffcf)))

    uffc = fffc.unfuse_legs(axes=0)
    uufc = uffc.unfuse_legs(axes=0).transpose(axes=(0, 2, 1))
    uuuc = uufc.unfuse_legs(axes=1).transpose(axes=(0, 3, 2, 1))
    assert all(yast.norm_diff(x, y) < tol for x, y in zip((ffc, fc, c), (uffc, uufc, uuuc)))
    assert all(x.is_consistent() for x in (fc, fcf, ffc, ffcf, fffc, uffc, uufc, uuuc))


def test_add_fuse_hard():
    """ execute tests of additions after hard fusion for several tensors. """
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                t=((0,), (0,), (-1, 0, 1), (-1, 0, 1)),
                D=((2,), (5,), (7, 8, 9), (10, 11, 12)))
    a.set_block(ts=(1, 1, 0, 0), Ds=(3, 6, 8, 11))
    b = yast.rand(config=config_U1, s=(1, -1, -1, 1),
                t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1), (-2, 0, 2)),
                D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    c = yast.rand(config=config_U1, s=(1, -1, -1, 1),
                t=((1,), (1,), (0, 1), (0, 1)),
                D=((3,), (6,), (8, 9), (11, 12)))

    magic_hf_vs_numpy(lambda x, y: x / 0.5 + y * 3, b, c)
    magic_hf_vs_numpy(lambda x, y: x - y ** 2, a.conj(), c)
    magic_hf_vs_numpy(lambda x, y: 0.5 * x + y * 3, a, b.conj())

    t1, t2 = ((0, -1), (0, 1), (1, -1), (1, 1)), ((0, 0), (0, 1), (1, 1))
    D1, D2 = (1, 2, 3, 4), (5, 2, 4)
    a = yast.rand(config=config_Z2_U1, s=(-1, 1, 1, -1),
                  t=(t2, t2, t1, t1), D=(D2, D2, D1, D1))
    b = yast.rand(config=config_Z2_U1, s=(1, -1, -1, 1),
                  t=(t1, t1, t2, t2), D=(D1, D1, D2, D2))

    magic_hf_vs_numpy(lambda x, y: x / 0.5 + y * 3, a, b.conj())

    a.set_block(ts=(((1, 2), (1, 2), (1, 2), (1, 2))), Ds=(6, 6, 6, 6), val='rand')
    a.set_block(ts=(((1, -1), (1, -1), (1, -1), (1, -1))), Ds=(3, 3, 3, 3), val='rand')
    magic_hf_vs_numpy(lambda x, y: x - y ** 3, a.conj(), b)


def test_add_exceptions():
    """ test handling exceptions """
    t1 = (-1, 0, 1)
    D1, D2 = (2, 3, 4), (2, 3, 5)
    t3, D3 = (-1, 0), (2, 4)
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        b = yast.rand(config=config_U1, s=(1, -1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        _ = a + b  # Error in add: tensor signatures do not match.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        b = yast.rand(config=config_U1, s=(-1, 1, 1), t=(t1, t1, t1), D=(D1, D2, D1))
        _ = a + b  # Error in add: tensor signatures do not match.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t1, t1), D=(D1, D1, D1))
        b = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t1, t1), D=(D1, D2, D1))
        _ = a + b  # Error in addition: bond dimensions do not match.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t1, t1), D=(D1, D1, D1))
        b = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t3, t1), D=(D1, D3, D1))
        _ = a + b  # Error in addition: bond dimensions do not match.
    with pytest.raises(yast.YastError):
        # Here, individual blocks between a na b are consistent, but cannot form consistent sum.
        a = yast.Tensor(config=config_U1, s=(1, -1, 1, -1))
        a.set_block(ts=(1, 1, 0, 0), Ds=(2, 2, 1, 1), val='rand')
        b = yast.Tensor(config=config_U1, s=(1, -1, 1, -1))
        b.set_block(ts=(1, 1, 1, 1), Ds=(1, 1, 1, 1), val='rand')
        _ = a + b  # Bond dimensions related to some charge are not consistent.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        b = yast.rand(config=config_U1, s=(1, -1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        a = a.fuse_legs(axes=(0, 1, (2, 3)), mode='meta')
        b = b.fuse_legs(axes=((0, 1), 2, 3), mode='meta')
        _ = a + b  # Error in add: fusion trees do not match.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t1, t1), D=(D1, D2, D1), n=1)
        b = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t1, t1), D=(D1, D2, D1), n=0)
        _ = a + b  # Error in add: tensor charges do not match.


if __name__ == '__main__':
    test_add_0()
    test_add_1()
    test_add_2()
    test_add_fuse_hard()
    test_add_exceptions()
