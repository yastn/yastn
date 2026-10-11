import pytest
import yastn


tol = 1e-10


def _config(config_kwargs, sym):
    opts = {**config_kwargs, 'lazy_threshold': 0}
    return yastn.make_config(sym=sym, **opts)


def _charges(sym):
    if sym == 'SU2':
        return ((0,), (1,), (2,))
    return ((0, 0), (1, 0), (2, 0))


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('axes', [((1, 0), (3, 2)), ((3, 0), (2, 1))])
@pytest.mark.parametrize('transpose', [False, True])
def test_ordered_disjoint_fusion(config_kwargs, sym, axes, transpose):
    config = _config(config_kwargs, sym)
    charges = ((0,), (1,)) if sym == 'SU2' else ((0, 0), (1, 1))
    leg = yastn.Leg(config, t=charges, D=(1, 2))
    a = yastn.rand(config, legs=(leg, leg.conj(), leg, leg.conj()))
    if transpose:
        a = a.transpose((2, 0, 3, 1))
    order = tuple(i for group in axes for i in group)
    fused = a.fuse_legs(axes, mode='hard')
    restored = fused.unfuse_legs((0, 1))
    assert restored.get_legs() == a.transpose(order).get_legs()
    assert (restored - a.transpose(order)).norm() < tol


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_multiblock_multisignature_fuse_tree(config_kwargs, sym):
    """Three disjoint fusion subtrees retain every channel and signature."""
    config = _config(config_kwargs, sym)
    leg = yastn.Leg(config, s=1, t=_charges(sym), D=(1, 2, 1))
    legs = (leg, leg.conj(), leg.conj(), leg, leg, leg.conj())
    a = yastn.rand(config=config, legs=legs)
    assert a.nblocks > 10

    order = (0, 3, 1, 4, 2, 5)
    fused = a.fuse_legs(axes=((0, 3), (1, 4), (2, 5)), mode='hard')
    restored = fused.unfuse_legs(axes=(0, 1, 2))
    assert (restored - a.transpose(order)).norm() < tol
    assert abs(restored.norm() - a.norm()) < tol


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_multiblock_arbitrary_fmove_roundtrip(config_kwargs, sym):
    config = _config(config_kwargs, sym)
    leg = yastn.Leg(config, s=1, t=_charges(sym), D=(1, 2, 1))
    a = yastn.rand(config=config,
                   legs=(leg, leg.conj(), leg, leg.conj(), leg, leg.conj()))
    order = (4, 1, 3, 0, 5, 2)
    inverse = tuple(order.index(i) for i in range(len(order)))
    b = a.transpose(order).consume_transpose()
    c = b.transpose(inverse).consume_transpose()
    assert (c - a).norm() < tol
    assert abs(b.norm() - a.norm()) < tol


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_multiblock_mixed_signature_svd_qr(config_kwargs, sym):
    config = _config(config_kwargs, sym)
    leg = yastn.Leg(config, s=1, t=_charges(sym), D=(1, 2, 1))
    a = yastn.rand(config=config, legs=(leg, leg, leg.conj(), leg.conj()))
    axes = ((0, 2), (3, 1))
    target = a.transpose((0, 2, 3, 1)).consume_transpose()

    U, S, V = a.svd(axes=axes)
    assert (U @ S @ V - target).norm() < tol
    gram_u = yastn.tensordot(U.conj(), U, axes=((0, 1), (0, 1)))
    eye_u = yastn.eye(config, legs=U.get_legs(-1).conj(), isdiag=False)
    assert (gram_u - eye_u).norm() < tol

    Q, R = a.qr(axes=axes)
    assert (Q @ R - target).norm() < tol
    gram_q = yastn.tensordot(Q.conj(), Q, axes=((0, 1), (0, 1)))
    eye_q = yastn.eye(config, legs=Q.get_legs(-1).conj(), isdiag=False)
    assert (gram_q - eye_q).norm() < tol


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_multiblock_eigh_reconstruction(config_kwargs, sym):
    config = _config(config_kwargs, sym)
    leg = yastn.Leg(config, s=1, t=_charges(sym), D=(2, 3, 2))
    m = yastn.rand(config=config, legs=(leg.conj(), leg))
    h = m + m.H
    S, U = h.eigh(axes=(0, 1))
    assert (U @ S @ U.H - h).norm() < tol
    gram = yastn.tensordot(U.conj(), U, axes=((0,), (0,)))
    eye = yastn.eye(config, legs=U.get_legs(-1).conj(), isdiag=False)
    assert (gram - eye).norm() < tol
