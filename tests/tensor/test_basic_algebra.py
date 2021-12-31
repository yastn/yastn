""" test algebraic expresions using magic methods like: a + 2. * b,  a / 2. - c * 3.,  a + b ** 2"""
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2, config_Z2_U1

tol = 1e-12  #pylint: disable=invalid-name


def algebra_vs_numpy(f, a, b):
    """
    f is lambda expresion using magic methods on a and b tensors
    e.g. f = lambda x, y: x + y
    """
    tDsa = {ia: a.get_leg_structure(ia) for ia in range(a.ndim)}
    tDsb = {ib: b.get_leg_structure(ib) for ib in range(b.ndim)}
    na = a.to_numpy(leg_structures=tDsb) # makes sure nparrays have consistent shapes
    nb = b.to_numpy(leg_structures=tDsa) # makes sure nparrays have consistent shapes
    nc = f(na, nb)

    c = f(a, b)
    assert c.is_consistent()
    assert all(c.are_independent(x) for x in (a, b))

    cn = c.to_numpy()
    assert np.linalg.norm(cn - nc) < tol
    return c


def combine_tests(a, b):
    """ some standard set of tests """
    r1 = algebra_vs_numpy(lambda x, y: x + 2 * y, a, b)
    r2 = a.apxb(b, 2)
    r3 = algebra_vs_numpy(lambda x, y: 2 * x + y, b, a)
    assert all(yast.norm(r1 - x, p=p) < tol for x in (r2, r3) for p in ('fro', 'inf'))  # == 0.0
    # additionally tests norm

def test_algebra_basic():
    """ test basic algebra for various symmetries """
    # dense
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='float64')
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='float64')
    combine_tests(a, b)

    c = yast.rand(config=config_dense, isdiag=True, D=5, dtype='float64')
    d, e = c.copy(), c.clone()
    r4 = algebra_vs_numpy(lambda x, y: 2. * x - (y + y), c, d)
    r5 = algebra_vs_numpy(lambda x, y: 2. * x - y - y, c, e)
    assert r4.norm() < tol  # == 0.0
    assert r5.norm() < tol  # == 0.0
    assert all(yast.are_independent(c, x) for x in (d, e))

    # U1
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), dtype='float64')

    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), dtype='float64')
    combine_tests(a, b)

    c = yast.eye(config=config_U1, t=1, D=5)
    d = yast.eye(config=config_U1, t=2, D=5)
    r4 = algebra_vs_numpy(lambda x, y: 2. * x + y, c, d)
    r5 = algebra_vs_numpy(lambda x, y: x - (2 * y) ** 2 + 2 * y, c, d)
    r6 = algebra_vs_numpy(lambda x, y: x - y / 0.5, d, c)
    assert all(pytest.approx(x.norm().item(), rel=tol) == 5 for x in (r4, r5, r6))
    assert all(pytest.approx(x.norm(p='inf').item(), rel=tol) == 2 for x in (r4, r5, r6))

    e = yast.ones(config=config_U1, s=(1, -1), n=1, t=(1, 0), D=(5, 5))
    f = yast.ones(config=config_U1, s=(1, -1), n=1, t=(2, 1), D=(5, 5))
    r7 = algebra_vs_numpy(lambda x, y: x - y / 0.5, e, f)
    assert pytest.approx(r7.norm().item(), rel=tol) == 5 * np.sqrt(5)
    assert pytest.approx(r7.norm(p='inf').item(), rel=tol) == 2

    # Z2xU1
    a = yast.randC(config=config_Z2_U1, s=(-1, 1, 1, 1),
        t=[[(0, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 2), (1, -2), (1, 0), (1, 2)], [(0, 0), (0, 2)]],
        D=((1, 2, 4), (2, 3), (2, 6, 3, 6, 9), (4, 7)))
    b = yast.randC(config=config_Z2_U1, s=(-1, 1, 1, 1),
        t=[[(0, 0), (1, 0), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)], [(0, 0)]],
        D=((1, 2, 4), (2, 3), (2, 4, 6, 3, 6, 9), 4))
    combine_tests(a, b)


def test_algebra_fuse_meta():
    """ test basic algebra on meta-fused tensor. """
    a = yast.rand(config=config_Z2, s=(-1, 1, 1, -1),
                  t=((0, 1), (0,), (0, 1), (1,)), D=((1, 2), (3,), (4, 5), (7, )))
    b = yast.rand(config=config_Z2, s=(-1, 1, 1, -1),
                  t=((0,), (0, 1), (0, 1), (0, 1)), D=((1,), (3, 4), (4, 5), (6, 7)))
    ma = a.fuse_legs(axes=((0, 3), (2, 1)), mode='meta')
    mb = b.fuse_legs(axes=((0, 3), (2, 1)), mode='meta')
    mc = algebra_vs_numpy(lambda x, y: x + y, ma, mb)
    uc = mc.unfuse_legs(axes=(0, 1)).transpose(axes=(0, 3, 2, 1))
    assert yast.norm(uc - a - b) < tol


def algebra_hf(f, a, b, hf_axes1=(0, (1, 2), 3)):
    """
    Test operations on a and b combined with application of fuse_legs(..., mode='hard').

    f is lambda expresion using magic methods on a and b tensors.
    hf_axes1 are axes of first hard fusion resulting in 3 legs; without transpose.
    """
    fa = yast.fuse_legs(a, axes=hf_axes1, mode='hard')
    fb = yast.fuse_legs(b, axes=hf_axes1, mode='hard')
    ffa = yast.fuse_legs(fa, axes=(1, (2, 0)), mode='hard')
    ffb = yast.fuse_legs(fb, axes=(1, (2, 0)), mode='hard')
    fffa = yast.fuse_legs(ffa, axes=[(0, 1)], mode='hard')
    fffb = yast.fuse_legs(ffb, axes=[(0, 1)], mode='hard')

    c = algebra_vs_numpy(f, a, b)
    cf = yast.fuse_legs(c, axes=hf_axes1, mode='hard')
    fc = f(fa, fb)
    fcf = yast.fuse_legs(fc, axes=(1, (2, 0)), mode='hard')
    ffc = f(ffa, ffb)
    ffcf = yast.fuse_legs(ffc, axes=[(0, 1)], mode='hard')
    fffc = f(fffa, fffb)
    assert all(yast.norm(x - y) < tol for x, y in zip((fc, ffc, fffc), (cf, fcf, ffcf)))

    uffc = fffc.unfuse_legs(axes=0)
    uufc = uffc.unfuse_legs(axes=1).transpose(axes=(2, 0, 1))
    uf_axes = tuple(i for i, a in enumerate(hf_axes1) if not isinstance(a, int))
    uuuc = uufc.unfuse_legs(axes=uf_axes)
    assert all(yast.norm(x - y) < tol for x, y in zip((ffc, fc, c), (uffc, uufc, uuuc)))
    assert all(x.is_consistent() for x in (fc, fcf, ffc, ffcf, fffc, uffc, uufc, uuuc))


def test_algebra_fuse_hard():
    """ execute tests of additions after hard fusion for several tensors. """
    # U1 with 4 legs
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

    algebra_hf(lambda x, y: x / 0.5 + y * 3, b, c)
    algebra_hf(lambda x, y: x - y ** 2, a.conj(), c)
    algebra_hf(lambda x, y: 0.5 * x + y * 3, a, b.conj())

    # U1 with 6 legs
    t1, t2, t3 = (-1, 0, 1), (-2, 0, 2), (-3, 0, 3)
    D1, D2, D3 = (1, 3, 2), (3, 3, 4), (5, 3, 6)
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t1, t1, t2, t2, t3, t3), D=(D1, D2, D2, D1, D1, D2))
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t2, t2, t3, t3, t1, t1), D=(D2, D3, D1, D3, D1, D2))
    algebra_hf(lambda x, y: x / 0.5 + y * 3, a, b, hf_axes1=((0, 1), (2, 3), (4, 5)))
    b.set_block(ts=(2, 2, 1, -2, -3, 0), Ds=(4, 6, 1, 1, 1, 3), val='randR')
    algebra_hf(lambda x, y: x - 3 * y, a, b, hf_axes1=((0, 1), (2, 3), (4, 5)))

    # Z2xU1 with 4 legs
    t1, t2 = ((0, -1), (0, 1), (1, -1), (1, 1)), ((0, 0), (0, 1), (1, 1))
    D1, D2 = (1, 2, 3, 4), (5, 2, 4)
    a = yast.rand(config=config_Z2_U1, s=(-1, 1, 1, -1),
                  t=(t2, t2, t1, t1), D=(D2, D2, D1, D1))
    b = yast.rand(config=config_Z2_U1, s=(1, -1, -1, 1),
                  t=(t1, t1, t2, t2), D=(D1, D1, D2, D2))

    algebra_hf(lambda x, y: x / 0.5 + y * 3, a, b.conj())

    a.set_block(ts=(((1, 2), (1, 2), (1, 2), (1, 2))), Ds=(6, 6, 6, 6), val='rand')
    a.set_block(ts=(((1, -1), (1, -1), (1, -1), (1, -1))), Ds=(3, 3, 3, 3), val='rand')
    algebra_hf(lambda x, y: x - y ** 3, a.conj(), b)


def test_algebra_exceptions():
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


def test_hf_union_exceptions():
    """ exceptions happening in resolving hard-fusion mismatches. """
    t1, t2 = (-1, 0, 1), (-2, 0, 2)
    D1, D2 = (2, 3, 2), (2, 5, 2)
    with pytest.raises(yast.YastError):
        a = yast.Tensor(config=config_U1, s=(1, -1, 1, -1))
        a.set_block(ts=(1, 1, 0, 0), Ds=(2, 2, 1, 1), val='rand')
        b = yast.Tensor(config=config_U1, s=(1, -1, 1, -1))
        b.set_block(ts=(1, 1, 1, 1), Ds=(1, 1, 1, 1), val='rand')
        a.fuse_legs(axes=[(0, 1, 2, 3)], inplace=True, mode='hard')
        b.fuse_legs(axes=[(0, 1, 2, 3)], inplace=True, mode='hard')
        _ = a + b  # Error in union: mismatch of native bond dimensions of fused legs.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t1, t1), D=(D1, D1, D1))
        b = yast.rand(config=config_U1, s=(1, -1, 1), t=(t2, t2, t2), D=(D1, D2, D1))
        a.fuse_legs(axes=((0, 2), 1), inplace=True, mode='hard')
        b.fuse_legs(axes=((0, 2), 1), inplace=True, mode='hard')
        _ = a + b  # Error in union: mismatch of bond dimensions of unfused legs.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1, -1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        b = yast.rand(config=config_U1, s=(-1, -1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        a.fuse_legs(axes=((0, 1, 2), 3), inplace=True, mode='hard')
        b.fuse_legs(axes=((0, 1, 2), 3), inplace=True, mode='hard')
        _ = a + b  # Error in union: mismatch in native signatures of fused legs.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1, -1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        b = yast.rand(config=config_U1, s=(-1, 1, -1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        a.fuse_legs(axes=((0, 1), (2, 3)), inplace=True, mode='hard')
        b.fuse_legs(axes=((0, 1, 2), 3), inplace=True, mode='hard')
        _ = a + b  # Error in union: mismatch in number of fused legs or fusion order.


if __name__ == '__main__':
    test_algebra_basic()
    test_algebra_fuse_meta()
    test_algebra_fuse_hard()
    test_algebra_exceptions()
    test_hf_union_exceptions()