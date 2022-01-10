""" yast.vdot """
import pytest
import yast
try:
    from .configs import config_dense, config_U1
except ImportError:
    from configs import config_dense, config_U1

tol = 1e-12  #pylint: disable=invalid-name


def vdot_vs_numpy(a, b):
    """ test vdot vs numpy """
    tDsa = {ii: b.get_leg_structure(ii) for ii in range(b.ndim)}
    tDsb = {ii: a.get_leg_structure(ii) for ii in range(a.ndim)}
    na = a.to_numpy(tDsa)  # makes sure nparrays have consistent shapes
    nb = b.to_numpy(tDsb)  # makes sure nparrays have consistent shapes
    ns = na.conj().reshape(-1) @ nb.reshape(-1)
    bc = b.conj()
    ac = a.conj()
    sab = yast.vdot(a, b)
    sab2 = yast.vdot(ac, b, conj=(0, 0))
    sab3 = yast.vdot(ac, bc, conj=(0, 1))
    sab4 = yast.vdot(a, bc, conj=(1, 1))
    sba = b.vdot(a)
    assert all(abs(ns - x.item()) < tol for x in (sab, sab2, sab3, sab4, sba.conj()))
    return ns


def test_vdot_basic():
    """ basic tests for various symmetries. """
    # dense
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='complex128')
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='complex128')
    vdot_vs_numpy(a, b)

    # U1
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 0, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 2, 12)))
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)))
    c = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=(1, -1, 2, 0), D=(2, 4, 9, 2))
    vdot_vs_numpy(a, b)
    vdot_vs_numpy(a, c)
    vdot_vs_numpy(c, b)

    # U1 complex
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 0, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 2, 12)), dtype='complex128')
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)), dtype='complex128')
    c = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=(1, -1, 2, 0), D=(2, 4, 9, 2), dtype='complex128')
    vdot_vs_numpy(a, b)
    vdot_vs_numpy(a, c)
    vdot_vs_numpy(c, b)


def test_vdot_fuse_hard():
    t1, t2, t3 = (-1, 0, 1), (-2, 0, 2), (-3, 0, 3)
    D1, D2, D3 = (1, 3, 2), (3, 3, 4), (5, 3, 6)
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t1, t1, t2, t2, t3, t3), D=(D1, D2, D2, D1, D1, D2))
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t2, t2, t3, t3, t1, t1), D=(D2, D3, D1, D3, D1, D2))
    vdot_hf(a, b, hf_axes1=((0, 1), (2, 3), (4, 5)))
    vdot_hf(a, b, hf_axes1=(0, (4, 3, 1), (5, 2)))

    b.set_block(ts=(1, 1, -2, -2, -3, 3), Ds=(2, 4, 3, 1, 1, 4), val='randR')
    vdot_hf(a, b, hf_axes1=((0, 1), (2, 3), (4, 5)))


    # a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
    #             t=((0,), (0,), (-1, 0, 1), (-1, 0, 1)), D=((2,), (5,), (7, 8, 9), (10, 11, 12)))
    # a.set_block(ts=(1, 1, 0, 0), Ds=(3, 6, 8, 11))
    # b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
    #             t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1), (-2, 0, 2)),
    #             D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    # c = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
    #             t=((1,), (1,), (0, 1), (0, 1)), D=((3,), (6,), (8, 9), (11, 12)))

    # vdot_hf(a, b)
    # vdot_hf(a, c)
    # vdot_hf(b.conj(), c.conj())

    # t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    # D1 = (1, 2, 3, 4)
    # t2 = [(0, 0), (0, 1), (1, 1)]
    # D2 = (5, 2, 4)
    # a2 = yast.rand(config=config_Z2xU1, s=(-1, 1, 1, -1),
    #               t=(t2, t2, t1, t1),
    #               D=(D2, D2, D1, D1))
    # b2 = yast.rand(config=config_Z2xU1, s=(1, -1, -1, 1),
    #               t=(t1, t1, t2, t2),
    #               D=(D1, D1, D2, D2))
    # vdot_hf(a2, b2)

    # a2.set_block(ts=((1, 2, 1, 2, 1, 2, 1, 2)), Ds=(6, 6, 6, 6), val='rand')
    # a2.set_block(ts=((1, -1, 1, -1, 1, -1, 1, -1)), Ds=(3, 3, 3, 3), val='rand')
    # vdot_hf(a2, b2)



def vdot_hf(a, b, hf_axes1=(0, (2, 3), 1)):
    """ Test vdot of a and b combined with application of fuse_legs(..., mode='hard'). """
    fa = yast.fuse_legs(a, axes=hf_axes1, mode='hard')
    fb = yast.fuse_legs(b, axes=hf_axes1, mode='hard')
    ffa = yast.fuse_legs(fa, axes=(1, (2, 0)), mode='hard')
    ffb = yast.fuse_legs(fb, axes=(1, (2, 0)), mode='hard')
    fffa = yast.fuse_legs(ffa, axes=[(0, 1)], mode='hard')
    fffb = yast.fuse_legs(ffb, axes=[(0, 1)], mode='hard')
    s = vdot_vs_numpy(a, b)
    fs = yast.vdot(fa, fb)
    ffs = yast.vdot(ffa, ffb)
    fffs = yast.vdot(fffa, fffb)
    fs2 = yast.vdot(fb, fa)
    ffs2 = yast.vdot(ffb, ffa)
    fffs2 = yast.vdot(fffb, fffa)
    assert all(abs(s - x.item()) < tol for x in (fs, ffs, fffs, fs2.conj(), ffs2.conj(), fffs2.conj()))


def test_vdot_exceptions():
    """ special cases and exceptions"""
    a = yast.Tensor(config=config_U1, s=(), dtype='complex128')
    b = yast.Tensor(config=config_U1, s=(), dtype='complex128')
    vdot_vs_numpy(a, b)  # == 0 for empty tensors

    t1, D1 = (-1, 1), (3, 4)
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1), n=1,
                  t=(t1, t1, t1, t1), D=(D1, D1, D1, D1), dtype='complex128')
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1), n=-1,
                  t=(t1, t1, t1, t1), D=(D1, D1, D1, D1), dtype='complex128')
    c = yast.rand(config=config_U1, s=(1, -1, -1, 1), n=1,
                  t=(t1, t1, t1, t1), D=(D1, D1, D1, D1), dtype='complex128')

    assert abs(a.vdot(b)) < tol  # == 0 as charges do not match
    assert abs(a.vdot(c, conj=(0, 0))) < tol  # == 0 as charges do not match for that conj
    assert abs(a.vdot(c, conj=(1, 1))) < tol  # == 0 as charges do not match for that conj


    c = yast.rand(config=config_U1, s=(-1, -1, 1, -1), n=1,
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)), dtype='complex128')

    d = yast.rand(config=config_U1, s=(-1, 1, 1, -1), n=1,
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)), dtype='complex128')
    d = d.fuse_legs(axes=(0, (1, 2), 3))

    with pytest.raises(yast.YastError):
        d = yast.rand(config=config_U1, s=(-1, -1, 1, -1), n=1,
                    t=(t1, t1, t1, t1), D=(D1, D1, D1, D1))
        a.vdot(d)  # Error in vdot: signatures do not match.
    with pytest.raises(yast.YastError):
        d = yast.rand(config=config_U1, s=(-1, -1, 1), n=1,
                    t=(t1, t1, t1), D=(D1, D1, D1))
        a.vdot(d)  # Error in vdot: mismatch in number of legs.
    with pytest.raises(yast.YastError):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
        bf = b.fuse_legs(axes=(0, (1, 2, 3)), mode='meta')
        af.vdot(bf)  # Error in vdot: mismatch in number of fused legs or fusion order.

    t1, t2 = (-1, 0, 1), (-1, 0, 2)
    D1, D2 = (2, 3, 4), (2, 4, 5)
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t1, t1), D=(D1, D1, D1))
        b = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t2, t1), D=(D1, D2, D1))
        yast.vdot(a, b)  # Bond dimensions do not match.


def test_hf_intersect_exceptions():
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
        yast.vdot(a, b)  # Error in intersect: mismatch of native bond dimensions of fused legs.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1), t=(t1, t1, t1), D=(D1, D1, D1))
        b = yast.rand(config=config_U1, s=(1, -1, 1), t=(t2, t2, t2), D=(D1, D2, D1))
        a.fuse_legs(axes=((0, 2), 1), inplace=True, mode='hard')
        b.fuse_legs(axes=((0, 2), 1), inplace=True, mode='hard')
        yast.vdot(a, b)  # Error in union: mismatch of bond dimensions of unfused legs.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1, -1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        b = yast.rand(config=config_U1, s=(-1, -1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        a.fuse_legs(axes=((0, 1, 2), 3), inplace=True, mode='hard')
        b.fuse_legs(axes=((0, 1, 2), 3), inplace=True, mode='hard')
        yast.vdot(a, b)  # Error in vdot: signatures of fused legs do not match.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1, -1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        b = yast.rand(config=config_U1, s=(-1, 1, -1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        a.fuse_legs(axes=((0, 1), (2, 3)), inplace=True, mode='hard')
        b.fuse_legs(axes=((0, 1, 2), 3), inplace=True, mode='hard')
        yast.vdot(a, b)  # Error in vdot: signatures of fused legs do not match.


if __name__ == '__main__':
    test_vdot_basic()
    test_vdot_fuse_hard()
    test_vdot_exceptions()
    test_hf_intersect_exceptions()
