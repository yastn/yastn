""" yastn.vdot """
import pytest
import yastn
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def vdot_vs_numpy(a, b):
    """ test vdot vs numpy """
    legs_for_a = {ii: leg for ii, leg in enumerate(b.get_legs())}
    legs_for_b = {ii: leg for ii, leg in enumerate(a.get_legs())}
    na = a.to_numpy(legs=legs_for_a)  # makes sure nparrays have consistent shapes
    nb = b.to_numpy(legs=legs_for_b)  # makes sure nparrays have consistent shapes
    ns = na.conj().reshape(-1) @ nb.reshape(-1)
    bc = b.conj()
    ac = a.conj()
    sab = yastn.vdot(a, b)
    sab2 = yastn.vdot(ac, b, conj=(0, 0))
    sab3 = yastn.vdot(ac, bc, conj=(0, 1))
    sab4 = yastn.vdot(a, bc, conj=(1, 1))
    sba = b.vdot(a)
    assert all(abs(ns - x.item()) < tol for x in (sab, sab2, sab3, sab4, sba.conj()))
    return ns


def test_vdot_basic():
    """ basic tests for various symmetries. """
    # dense
    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='complex128')
    b = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='complex128')
    vdot_vs_numpy(a, b)

    # U1
    legs_a = [yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(1, 2, 3)),
              yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
              yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
              yastn.Leg(config_U1, s=-1, t=(-1, 0, 2), D=(10, 2, 12))]
    legs_b = [yastn.Leg(config_U1, s=-1, t=(-1, 2), D=(1, 3)),
              yastn.Leg(config_U1, s=1, t=(1, 2), D=(5, 6)),
              yastn.Leg(config_U1, s=1, t=(-1, 1), D=(7, 8)),
              yastn.Leg(config_U1, s=-1, t=(-1, 0, 1, 2), D=(10, 2, 11, 12))]
    legs_c = [yastn.Leg(config_U1, s=-1, t=[1], D=[2]),
              yastn.Leg(config_U1, s=1, t=[-1], D=[4]),
              yastn.Leg(config_U1, s=1, t=[1], D=[9]),
              yastn.Leg(config_U1, s=-1, t=[2], D=[12])]
    a = yastn.rand(config=config_U1, legs=legs_a)
    b = yastn.rand(config=config_U1, legs=legs_b)
    c = yastn.rand(config=config_U1, legs=legs_c)
    vdot_vs_numpy(a, b)
    vdot_vs_numpy(a, c)
    vdot_vs_numpy(c, b)

    # U1 complex
    a = yastn.rand(config=config_U1, legs=legs_a, dtype='complex128')
    b = yastn.rand(config=config_U1, legs=legs_b, dtype='complex128')
    c = yastn.rand(config=config_U1, legs=legs_c, dtype='complex128')
    vdot_vs_numpy(a, b)
    vdot_vs_numpy(a, c)
    vdot_vs_numpy(c, b)


def test_vdot_fuse_hard():
    t1, t2, t3 = (-1, 0, 1), (-2, 0, 2), (-3, 0, 3)
    D1, D2, D3 = (1, 3, 2), (3, 3, 4), (5, 3, 6)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t1, t1, t2, t2, t3, t3), D=(D1, D2, D2, D1, D1, D2))
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t2, t2, t3, t3, t1, t1), D=(D2, D3, D1, D3, D1, D2))
    vdot_hf(a, b, hf_axes1=((0, 1), (2, 3), (4, 5)))
    vdot_hf(a, b, hf_axes1=(0, (4, 3, 1), (5, 2)))

    b.set_block(ts=(1, 1, -2, -2, -3, 3), Ds=(2, 4, 3, 1, 1, 4), val='rand')
    vdot_hf(a, b, hf_axes1=((0, 1), (2, 3), (4, 5)))

    t1, t2 = [(0, -1), (0, 1), (1, -1), (1, 1)],  [(0, 0), (0, 1), (1, 1)]
    D1, D2 = (1, 2, 3, 4), (5, 2, 4)
    a2 = yastn.rand(config=config_Z2xU1, s=(-1, 1, 1, -1), t=(t2, t2, t1, t1), D=(D2, D2, D1, D1))
    b2 = yastn.rand(config=config_Z2xU1, s=(-1, 1, 1, -1), t=(t1, t1, t2, t2), D=(D1, D1, D2, D2))
    vdot_hf(a2, b2, hf_axes1=(0, (2, 3), 1))
    a2.set_block(ts=(((1, 2), (1, 2), (1, 2), (1, 2))), Ds=(6, 6, 6, 6), val='rand')
    a2.set_block(ts=(((1, -1), (1, -1), (1, -1), (1, -1))), Ds=(3, 3, 3, 3), val='rand')
    vdot_hf(a2, b2, hf_axes1=(0, (2, 3), 1))



def vdot_hf(a, b, hf_axes1=(0, (2, 3), 1)):
    """ Test vdot of a and b combined with application of fuse_legs(..., mode='hard'). """
    fa = yastn.fuse_legs(a, axes=hf_axes1, mode='hard')
    fb = yastn.fuse_legs(b, axes=hf_axes1, mode='hard')
    ffa = yastn.fuse_legs(fa, axes=(1, (2, 0)), mode='hard')
    ffb = yastn.fuse_legs(fb, axes=(1, (2, 0)), mode='hard')
    fffa = yastn.fuse_legs(ffa, axes=[(0, 1)], mode='hard')
    fffb = yastn.fuse_legs(ffb, axes=[(0, 1)], mode='hard')
    s = vdot_vs_numpy(a, b)
    fs = yastn.vdot(fa, fb)
    ffs = yastn.vdot(ffa, ffb)
    fffs = yastn.vdot(fffa, fffb)
    fs2 = yastn.vdot(fb, fa)
    ffs2 = yastn.vdot(ffb, ffa)
    fffs2 = yastn.vdot(fffb, fffa)
    assert all(abs(s - x.item()) < tol for x in (fs, ffs, fffs, fs2.conj(), ffs2.conj(), fffs2.conj()))


def test_vdot_exceptions():
    """ special cases and exceptions"""
    a = yastn.Tensor(config=config_U1, s=(), dtype='complex128')
    b = yastn.Tensor(config=config_U1, s=(), dtype='complex128')
    vdot_vs_numpy(a, b)  # == 0 for empty tensors

    leg = yastn.Leg(config_U1, s=1, t=(-1, 1), D=(3, 4))
    a = yastn.randC(config=config_U1, legs=[leg.conj(), leg, leg, leg.conj()], n=1)
    b = yastn.randC(config=config_U1, legs=[leg.conj(), leg, leg, leg.conj()], n=-1)
    c = yastn.randC(config=config_U1, legs=[leg, leg.conj(), leg.conj(), leg], n=1)

    assert abs(a.vdot(b)) < tol  # == 0 as charges do not match
    assert abs(a.vdot(c, conj=(0, 0))) < tol  # == 0 as charges do not match for that conj
    assert abs(a.vdot(c, conj=(1, 1))) < tol  # == 0 as charges do not match for that conj

    with pytest.raises(yastn.YastError):
        d = yastn.rand(config=config_U1, legs=[leg.conj(), leg.conj(), leg, leg.conj()], n=1)
        a.vdot(d)  # Error in vdot: signatures do not match.
    with pytest.raises(yastn.YastError):
        d = yastn.rand(config=config_U1, egs=[leg.conj(), leg, leg], n=1)
        a.vdot(d)  # Error in vdot: mismatch in number of legs.
    with pytest.raises(yastn.YastError):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
        bf = b.fuse_legs(axes=(0, (1, 2, 3)), mode='meta')
        af.vdot(bf)  # Error in vdot: mismatch in number of fused legs or fusion order.
    with pytest.raises(yastn.YastError):
        a = yastn.rand(config=config_U1, legs=[leg, leg.conj(), leg, leg])
        leg2 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(3, 4, 5))
        b = yastn.rand(config=config_U1, legs=[leg, leg.conj(), leg, leg2])
        yastn.vdot(a, b)  # Bond dimensions do not match.


def test_hf_intersect_exceptions():
    """ exceptions happening in resolving hard-fusion mismatches. """
    t1, t2 = (-1, 0, 1), (-2, 0, 2)
    D1, D2 = (2, 3, 2), (2, 5, 2)
    with pytest.raises(yastn.YastError):
        a = yastn.Tensor(config=config_U1, s=(1, -1, 1, -1))
        a.set_block(ts=(1, 1, 0, 0), Ds=(2, 2, 1, 1), val='rand')
        b = yastn.Tensor(config=config_U1, s=(1, -1, 1, -1))
        b.set_block(ts=(1, 1, 1, 1), Ds=(1, 1, 1, 1), val='rand')
        a = a.fuse_legs(axes=[(0, 1, 2, 3)], mode='hard')
        b = b.fuse_legs(axes=[(0, 1, 2, 3)], mode='hard')
        yastn.vdot(a, b)  # Error in intersect: mismatch of native bond dimensions of fused legs.
    with pytest.raises(yastn.YastError):
        a = yastn.rand(config=config_U1, s=(1, -1, 1), t=(t1, t1, t1), D=(D1, D1, D1))
        b = yastn.rand(config=config_U1, s=(1, -1, 1), t=(t2, t2, t2), D=(D1, D2, D1))
        a = a.fuse_legs(axes=((0, 2), 1), mode='hard')
        b = b.fuse_legs(axes=((0, 2), 1), mode='hard')
        yastn.vdot(a, b)  # Error in union: mismatch of bond dimensions of unfused legs.
    with pytest.raises(yastn.YastError):
        a = yastn.rand(config=config_U1, s=(-1, 1, -1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        b = yastn.rand(config=config_U1, s=(-1, -1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        a = a.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        b = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        yastn.vdot(a, b)  # Error in vdot: signatures of fused legs do not match.
    with pytest.raises(yastn.YastError):
        a = yastn.rand(config=config_U1, s=(-1, 1, -1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        b = yastn.rand(config=config_U1, s=(-1, 1, -1, -1), t=(t1, t1, t1, t1), D=(D1, D2, D1, D2))
        a = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        b = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        yastn.vdot(a, b)  # Error in vdot: signatures of fused legs do not match.


if __name__ == '__main__':
    test_vdot_basic()
    test_vdot_fuse_hard()
    test_vdot_exceptions()
    test_hf_intersect_exceptions()
