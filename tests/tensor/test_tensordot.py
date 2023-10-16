import numpy as np
import pytest
import yastn
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def tensordot_vs_numpy(a, b, axes, conj):
    outa = tuple(ii for ii in range(a.ndim) if ii not in axes[0])
    outb = tuple(ii for ii in range(b.ndim) if ii not in axes[1])

    legs_a = a.get_legs()
    legs_b = b.get_legs()

    legs_a_out = {nn: legs_a[ii] for nn, ii in enumerate(outa)}
    legs_b_out = {nn + len(outa): legs_b[ii] for nn, ii in enumerate(outb)}

    conj_in = False if conj[0] + conj[1] == 1 else True
    na = a.to_numpy(legs={ia: legs_b[ib].conj() if conj_in else legs_b[ib] for ia, ib in zip(*axes)})
    nb = b.to_numpy(legs={ib: legs_a[ia].conj() if conj_in else legs_a[ia] for ia, ib in zip(*axes)})
    if conj[0]:
        na = na.conj()
        legs_a_out = {nn: leg.conj() for nn, leg in legs_a_out.items()}
    if conj[1]:
        nb = nb.conj()
        legs_b_out = {nn: leg.conj() for nn, leg in legs_b_out.items()}
    nab = np.tensordot(na, nb, axes)

    c = yastn.tensordot(a, b, axes, conj)

    nc = c.to_numpy(legs={**legs_a_out, **legs_b_out})
    assert c.is_consistent()
    assert a.are_independent(c)
    assert c.are_independent(b)
    assert np.linalg.norm(nc - nab) < tol  # == 0.0
    return c


def test_dot_basic():
    """ test tensordot for different symmetries. """
    # dense
    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='complex128')
    b = yastn.rand(config=config_dense, s=(1, -1, 1), D=(2, 3, 5), dtype='complex128')
    c1 = tensordot_vs_numpy(a, b, axes=((0, 3), (0, 2)), conj=(0, 0))
    c2 = tensordot_vs_numpy(b, a, axes=((2, 0), (3, 0)), conj=(1, 1))
    assert yastn.norm(c1.conj() - c2.transpose(axes=(1, 2, 0)))

    # U1
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    b = yastn.rand(config=config_U1, s=(1, -1, 1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 0, 1)),
                  D=((1, 2, 3), (4, 5, 6), (10, 7, 11)))
    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 0))
    tensordot_vs_numpy(a, b, axes=((1, 3), (1, 2)), conj=(0, 0))

    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=-2)
    a.set_block(ts=(2, 1, 0, 1), Ds=(2, 1, 10, 1), val='rand')
    b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=1)
    b.set_block(ts=(1, 2, 0, 0), Ds=(1, 2, 10, 10), val='rand')
    c = tensordot_vs_numpy(a, b, axes=((2, 1), (1, 2)), conj=(1, 0))
    assert c.struct.n == (3,)
    a.set_block(ts=(1, 1, -1, 1), Ds=(1, 1, 11, 1), val='rand')
    a.set_block(ts=(2, 2, -1, 1), Ds=(2, 2, 11, 1), val='rand')
    a.set_block(ts=(3, 3, -1, 1), Ds=(3, 3, 11, 1), val='rand')
    b.set_block(ts=(1, 1, 1, 0), Ds=(1, 1, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 1, 0), Ds=(3, 3, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 2, 1), Ds=(3, 3, 2, 1), val='rand')
    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 1))
    tensordot_vs_numpy(a, b, axes=((0, 3, 1), (1, 2, 0)), conj=(0, 0))

    # Z2xU1
    t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    t2 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yastn.rand(config=config_Z2xU1, s=(-1, 1, 1, -1),
                  t=(t1, t1, t1, t1),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8), (7, 8, 9, 10)))
    b = yastn.rand(config=config_Z2xU1, s=(1, -1, 1),
                  t=(t1, t1, t2),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8,)))

    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 0))
    tensordot_vs_numpy(b, a, axes=((1, 0), (1, 0)), conj=(0, 0))


def test_tensordot_diag():
    t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    D1 = (1, 2, 2, 4)

    t2 = [(0, -1), (0, 1), (1, -1), (0, 0)]
    D2 = (1, 2, 2, 5)

    a = yastn.rand(config=config_Z2xU1, s=(-1, 1, 1, -1),
                  t=(t1, t1, t1, t1),
                  D=(D1, (9, 4, 3, 2), (5, 6, 7, 8), (7, 8, 9, 10)))
    b = yastn.rand(config=config_Z2xU1, s=(1, -1), t = [t2, t2], D=[D2, D2], isdiag=True)
    b2 = b.diag()

    c1 = b.broadcast(a, axes=0)
    c2 = b.conj().broadcast(a, axes=0)
    c3 = b2.tensordot(a, axes=(0, 0))

    assert(yastn.norm(c1 - c2)) < tol
    assert(yastn.norm(c1 - c3)) < tol
    assert c3.get_shape() == (5, 18, 26, 34)


def tensordot_hf(a, b, hf_axes1):
    """ Test vdot of a and b combined with application of fuse_legs(..., mode='hard'). """
    fa = yastn.fuse_legs(a, axes=hf_axes1, mode='hard')
    fb = yastn.fuse_legs(b, axes=hf_axes1, mode='hard')
    ffa = yastn.fuse_legs(fa, axes=(0, (2, 1)), mode='hard')
    ffb = yastn.fuse_legs(fb, axes=(0, (2, 1)), mode='hard')
    c = tensordot_vs_numpy(a, b, axes=((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)), conj=(1, 0))
    fc = yastn.tensordot(fa, fb, axes=((1, 2), (1, 2)), conj=(1, 0))
    ffc = yastn.tensordot(ffa, ffb, axes=(1, 1), conj=(1, 0))
    assert all(yastn.norm(c - x) < tol for x in (fc, ffc))


def test_tensordot_fuse_hard():
    """ test tensordot combined with hard-fusion."""
    t1, t2, t3 = (-1, 0, 1), (-2, 0, 2), (-3, 0, 3)
    D1, D2, D3 = (1, 3, 2), (3, 3, 4), (5, 3, 6)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t1, t1, t2, t2, t3, t3), D=(D1, D2, D2, D1, D1, D2))
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t2, t2, t3, t3, t1, t1), D=(D2, D3, D1, D3, D1, D2))
    tensordot_hf(a, b, hf_axes1=(0, (4, 3, 1), (5, 2)))
    tensordot_hf(a, b, hf_axes1=(0, (4, 3, 1, 5), 2))


def test_tensordot_fuse_meta():
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (3, 4)))
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1),
                  t=((-1, 0, 1), (1,), (-1, 1), (0, 1), (0, 1, 2)),
                  D=((2, 1, 2), (4,), (4, 6), (7, 8), (3, 4, 5)))

    c = tensordot_vs_numpy(a, b, axes=((0, 3, 4), (0, 3, 4)), conj=(0, 1))
    fa = a.fuse_legs(axes=(0, (1, 2), (4, 3)), mode='meta')
    fb = b.fuse_legs(axes=(0, (1, 2), (4, 3)), mode='meta')
    fc = tensordot_vs_numpy(fa, fb, axes=((2, 0), (2, 0)), conj=(0, 1))
    fc = fc.unfuse_legs(axes=(0, 1))
    fa = fa.fuse_legs(axes=((0, 2), 1), mode='meta')
    fb = fb.fuse_legs(axes=((0, 2), 1), mode='meta')
    ffc = tensordot_vs_numpy(fa, fb, axes=((0,), (0,)), conj=(0, 1))
    ffc = ffc.unfuse_legs(axes=(0, 1))
    assert all(yastn.norm(c - x) < tol for x in (fc, ffc))


def test_tensordot_exceptions():
    """ special cases and exceptions"""
    t1, t2 = (-1, 0, 1), (-1, 0, 2)
    D1, D2 = (2, 3, 4), (2, 4, 5)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=(t1, t1, t1, t1), D=(D1, D1, D1, D1))
    b = yastn.rand(config=config_U1, s=(-1, -1, 1, -1),
                    t=(t1, t1, t2, t1), D=(D1, D1, D1, D2))
    with pytest.raises(yastn.YastnError):
        _ = yastn.tensordot(a, b, axes=((0, 1, 2), (0, 1, 2)), conj=(0, 1))
        # Signatures do not match.
    with pytest.raises(yastn.YastnError):
        _ = yastn.tensordot(a, b, axes=((0, 1, 2), (0, 1)), conj=(1, 0))
        # axes[0] and axes[1] indicated different number of legs.
    with pytest.raises(yastn.YastnError):
        _ = yastn.tensordot(a, b, axes=((0, 1), (0, 0)), conj=(1, 0))
        # Repeated axis in axes[0] or axes[1].
    with pytest.raises(yastn.YastnError):
        _ = yastn.tensordot(a, b, axes=((2, 3), (2, 3)), conj=(1, 0))
        # Bond dimensions do not match.
    with pytest.raises(yastn.YastnError):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
        bf = b.fuse_legs(axes=(0, (1, 2, 3)), mode='meta')
        _ = yastn.tensordot(af, bf, axes=(0, 0), conj=(1, 0))
        # Indicated axes of two tensors have different number of meta-fused legs or sub-fusions order.
    with pytest.raises(yastn.YastnError):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        bf = b.fuse_legs(axes=(0, (1, 2, 3)), mode='hard')
        _ = yastn.tensordot(af, bf, axes=(0, 0), conj=(1, 0))
        # Indicated axes of two tensors have different number of meta-fused legs or sub-fusions order.
    with pytest.raises(yastn.YastnError):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        bf = b.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _ = yastn.tensordot(af, bf, axes=(0, 0), conj=(1, 0))
        # Signatures of hard-fused legs do not match
    with pytest.raises(yastn.YastnError):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        bf = b.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _ = yastn.tensordot(af, bf, axes=(1, 1), conj=(1, 0))
        # Bond dimensions of fused legs do not match.
    with pytest.raises(yastn.YastnError):
        af = a.fuse_legs(axes=(1, 3, (0, 2)), mode='hard')
        bf = b.fuse_legs(axes=(1, 3, (0, 2)), mode='hard')
        _ = yastn.tensordot(af, bf, axes=((1, 2), (1, 2)), conj=(1, 0))
        # Bond dimensions do not match.
    with pytest.raises(yastn.YastnError):
        c = yastn.rand(config=config_U1, isdiag=True, t=(-1, 0, 1), D=(1, 2, 3))
        _ = yastn.tensordot(c, a, axes=((),()))
        # Outer product with diagonal tensor not supported. Use yastn.diag() first.


@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_tensordot_fuse_hard_backward():
    import torch
    # U1
    t1, t2, t3 = (-1, 0, 1), (-2, 0, 2), (-3, 0, 3)
    D1, D2, D3 = (1, 2, 2), (2, 2, 2), (2, 2, 2)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t1, t1, t2, t2, t3, t3), D=(D1, D2, D2, D1, D1, D2), dtype='complex128')
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t2, t2, t3, t3, t1, t1), D=(D2, D3, D1, D3, D1, D2), dtype='complex128')
    fb = yastn.fuse_legs(b, axes=(0, (4, 3, 1), (5, 2)), mode='hard')
    ffb = yastn.fuse_legs(fb, axes=(0, (2, 1)), mode='hard')

    target_block = (0,0,0,0,0,0)
    target_block_size = a[target_block].size()

    def test_f(block):
        a.set_block(ts=target_block, val=block)
        fa = yastn.fuse_legs(a, axes=(0, (4, 3, 1), (5, 2)), mode='hard')
        ffa = yastn.fuse_legs(fa, axes=(0, (2, 1)), mode='hard')
        ffab = yastn.tensordot(ffa, ffb, axes=(1, 1), conj=(1, 0))
        ffab = ffab.norm()
        return ffab

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
    assert test

@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_tensordot_backward():
    import torch
    # U1
    a = yastn.rand(config=config_U1, s=(-1, -1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(2, 3), (4, 5), (4, 3), (2, 1)], dtype='complex128')
    b = yastn.rand(config=config_U1, s=(1, 1, -1, -1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(2, 3), (4, 5), (4, 3), (2, 1)], dtype='complex128')
    # b = a.clone()
    target_block = (0, 1, 1, 0)
    target_block_size = a[target_block].size()

    def test_f(block):
        a.set_block(ts=target_block, val=block)
        ab = yastn.tensordot(a, b, axes=((1, 2), (1, 2)))
        ab = ab.norm()
        return ab

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
    assert test

if __name__ == '__main__':
    test_dot_basic()
    test_tensordot_fuse_hard()
    test_tensordot_diag()
    test_tensordot_fuse_meta()
    test_tensordot_exceptions()
    test_tensordot_backward()
    test_tensordot_fuse_hard_backward()
