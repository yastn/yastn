"""Exact boundary slicing for 2x2 measurements, including graded SU2 tensors."""
import pytest

import yastn
from yastn.tn import fpeps
from yastn.tn.fpeps.envs import _env_ctm_measure as measure


def _random_env(config_kwargs, sym='dense', single_layer=False):
    if sym in ('dense', 'SU2'):
        ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
        charges = ((),) if sym == 'dense' else ((0,), (1,))
        boundary_charges = ((),) if sym == 'dense' else ((0,), (2,))
    elif sym == 'U1':
        ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
        charges = ((0,), (1,))
        boundary_charges = ((0,), (1,))
    else:
        ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
        charges = ((0, 0), (1, 1), (1, -1))
        boundary_charges = ((0, 0), (2, 0))
    config = ops.config
    config.backend.random_seed(seed=42)
    leg = yastn.Leg(config, s=1, t=charges, D=(2,) if sym == 'dense' else (1,) * len(charges))
    legs = (leg, leg, leg.conj(), leg.conj())
    if not single_layer:
        legs += (ops.I().get_legs(0),)
    tensor = yastn.rand(config, legs=legs, dtype='complex128')
    psi = fpeps.Peps(fpeps.SquareLattice((2, 2), boundary='infinite'), tensors=tensor)
    boundary = yastn.Leg(config, s=1, t=boundary_charges,
                         D=(3,) if sym == 'dense' else (2, 1))
    env = fpeps.EnvCTM(psi, init='rand', leg=boundary)
    return env, ops, tensor


@pytest.mark.parametrize('sym', ['dense', 'U1', 'SU2', 'SU2xU1'])
def test_measure_2x2_chunked(config_kwargs, sym):
    env, ops, tensor = _random_env(config_kwargs, sym)
    original = tensor.clone()
    if sym == 'dense':
        operators = ops.sz(), ops.sx()
    elif sym == 'U1':
        operators = ops.cp(), ops.c()
    elif sym == 'SU2':
        operators = ops.S(), ops.S()
    else:
        operators = ops.cp_spinor(), ops.c_spinor()
    checked = 0
    for sites in (((0, 0), (1, 1)), ((1, 1), (0, 0)), ((0, 1), (0, 1))):
        reference = env.measure_2x2(*operators, sites=sites)
        assert abs(reference.item()) > 1e-10
        for chunk_size in (1, 2, 10):
            actual = env.measure_2x2(*operators, sites=sites, chunk_size=chunk_size)
            assert actual.item() == pytest.approx(reference.item(), rel=2e-10, abs=2e-11)
            checked += 1
    assert checked == 9
    assert (tensor - original).norm().item() == 0


@pytest.mark.parametrize('single_layer', [False, True])
@pytest.mark.parametrize('sym', ['dense', 'SU2'])
def test_measure_2x2_chunked_fused_boundary(config_kwargs, single_layer, sym):
    env, ops, tensor = _random_env(config_kwargs, sym, single_layer=single_layer)
    env.reset_(init='dl')
    assert env[0, 0].l.get_legs(0).is_fused()
    sites = ((0, 0), (1, 1))
    local = ops.S() if sym == 'SU2' else ops.sz()
    operators = (tensor * 0.7, tensor * 1.3) if single_layer else (local, local)
    reference = env.measure_2x2(*operators, sites=sites)
    actual = env.measure_2x2(*operators, sites=sites, chunk_size=2)
    assert actual.item() == pytest.approx(reference.item(), rel=2e-10, abs=2e-11)


def test_measure_2x2_chunks_reach_corner_contractions(config_kwargs, monkeypatch):
    env, ops, _ = _random_env(config_kwargs)
    original = measure.tensordot
    sizes = []

    def record(a, b, *args, **kwargs):
        result = original(a, b, *args, **kwargs)
        if isinstance(b, fpeps.DoublePepsTensor):
            sizes.append((a.shape[0], a.shape[3], result.size))
        return result

    monkeypatch.setattr(measure, 'tensordot', record)
    reference = env.measure_2x2(ops.sz(), ops.sz(), sites=((0, 0), (1, 1)))
    full_sizes = sizes[:]
    sizes.clear()
    actual = env.measure_2x2(ops.sz(), ops.sz(), sites=((0, 0), (1, 1)), chunk_size=1)
    assert actual.item() == pytest.approx(reference.item(), rel=1e-11)
    assert full_sizes and len(sizes) > len(full_sizes)
    assert all(min(left, right) == 1 for left, right, _ in sizes)
    assert max(size for _, _, size in sizes) < max(size for _, _, size in full_sizes)


@pytest.mark.parametrize('sym', ['U1', 'SU2'])
def test_boundary_chunks_align_fused_subspaces(config_kwargs, sym):
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=14)
    full = yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 1, 1))
    partial = yastn.Leg(config, s=-1, t=(1, 2), D=(1, 1))
    a = yastn.rand(config, legs=(full, full.conj())).fuse_legs(axes=((0, 1),))
    b = yastn.rand(config, legs=(partial, full)).fuse_legs(axes=((0, 1),))
    reference = yastn.vdot(a, b, conj=(0, 0))
    assert abs(reference.item()) > 1e-10
    chunks = list(measure._2x2_boundary_chunks(a, b, (0, 0), 1))
    assert len(chunks) > 1
    actual = sum(yastn.vdot(x, y, conj=(0, 0)) for x, y in chunks)
    assert actual.item() == pytest.approx(reference.item(), abs=1e-12)


@pytest.mark.parametrize('chunk_size', [0, -1, 1.5, True, '2'])
def test_measure_2x2_invalid_chunk_size(config_kwargs, chunk_size):
    env, ops, _ = _random_env(config_kwargs)
    with pytest.raises(yastn.YastnError, match='chunk_size must be a positive integer or None'):
        env.measure_2x2(ops.sz(), sites=((0, 0),), chunk_size=chunk_size)


@pytest.mark.parametrize('sym', ['dense', 'SU2'])
def test_measure_2x2_chunked_gradient(config_kwargs, sym):
    if config_kwargs['backend'] not in ('torch', 'torch_cutensor'):
        pytest.skip('Torch autograd test.')
    import torch
    env, ops, tensor = _random_env(config_kwargs, sym)
    tensor.requires_grad_()
    edge = env[0, 0].l
    edge.requires_grad_()
    sites = ((0, 0), (1, 1))
    local = ops.S() if sym == 'SU2' else ops.sz()
    reference = env.measure_2x2(local, local, sites=sites)
    ref_grad = torch.autograd.grad(reference.real, (tensor._data, edge._data))
    actual = env.measure_2x2(local, local, sites=sites, chunk_size=2)
    gradients = torch.autograd.grad(actual.real, (tensor._data, edge._data))
    assert actual.item() == pytest.approx(reference.item(), rel=1e-11)
    for found, expected in zip(gradients, ref_grad):
        assert expected.norm().item() > 1e-10
        torch.testing.assert_close(found, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('cache_bytes', [0, 64 << 20])
def test_measure_2x2_reuses_corners(config_kwargs, monkeypatch, cache_bytes):
    env, ops, _ = _random_env(config_kwargs)
    reference = env.measure_2x2(ops.sz(), ops.sz(), sites=((0, 0), (1, 1)))
    original = measure.tensordot
    contractions = []

    def record(a, b, *args, **kwargs):
        if isinstance(b, fpeps.DoublePepsTensor):
            contractions.append(b.has_operator_or_swap())
        return original(a, b, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Known boundary intervals must not rediscover mask nonzeros')

    monkeypatch.setattr(measure, 'tensordot', record)
    monkeypatch.setattr(measure, '_2X2_CORNER_CACHE_BYTES', cache_bytes)
    monkeypatch.setattr(ops.config.backend, 'to_mask', forbidden)
    actual = env.measure_2x2(ops.sz(), ops.sz(), sites=((0, 0), (1, 1)), chunk_size=1)
    assert actual.item() == pytest.approx(reference.item(), rel=1e-11)
    # Three chunks per side; only tl and br carry an operator.
    assert len(contractions) == (18 if cache_bytes else 36)
    assert sum(bool(c) for c in contractions) == (6 if cache_bytes else 12)
