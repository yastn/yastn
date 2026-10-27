"""Contractions recouple requested blocks without full input intermediates."""
import pytest
import yastn
from yastn.tensor import _contractions as contractions

torch = pytest.importorskip('torch')
from yastn.backend import backend_torch
from yastn.backend._backend_torch_backwards import _MatrixBlocks


@pytest.fixture(autouse=True)
def streamed_contractions(monkeypatch):
    monkeypatch.setattr(contractions, '_NONABELIAN_RECOUPLE_MIN_SIZE', 0)


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('dtypes', [('float64', 'float64'), ('complex128', 'complex128'),
                                    ('float64', 'complex128'), ('complex128', 'float64')])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('axes', [((0, 2, 3), (1, 3, 0)), ((0, 3), (1, 2)),
                                  ((0, 1, 2, 3), (1, 0, 3, 2))])
def test_recoupled_contraction(config_kwargs, sym, dtypes, masked, axes, monkeypatch):
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym=sym)
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    leg = yastn.Leg(config, t=charges, D=(1, 2, 1))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2, dtype=dtypes[0])
    b = yastn.rand(config, legs=(leg, leg.conj()) * 2, dtype=dtypes[1])
    if masked:
        a = a.remove_random_blocks(number=a.nblocks // 3, keep_legs=True)
        b = b.remove_random_blocks(number=b.nblocks // 4, keep_legs=True)
    a.requires_grad_()
    b.requires_grad_()
    with monkeypatch.context() as patch:
        patch.setattr(backend_torch, 'recouple_dot_sum', None)
        expected = yastn.tensordot(a, b, axes)
    assert expected.size > 0 and expected.norm().item() > 0

    def forbidden(*args, **kwargs):
        raise AssertionError('Contraction materialized a full recoupled input')

    monkeypatch.setattr(backend_torch, 'recouple', forbidden)
    monkeypatch.setattr(backend_torch, 'recouple_sequence', forbidden)
    monkeypatch.setattr(_MatrixBlocks, 'max_bytes', 64)
    found = yastn.tensordot(a, b, axes)
    assert found.struct == expected.struct and found.hfs == expected.hfs
    torch.testing.assert_close(found.data, expected.data, atol=1e-11, rtol=1e-12)
    gradient = torch.randn(found.size * 2, dtype=found.data.dtype, device=found.data.device)[::2]
    expected_grad = torch.autograd.grad(expected.data, (a.data, b.data), gradient)
    found_grad = torch.autograd.grad(found.data, (a.data, b.data), gradient)
    for actual, reference in zip(found_grad, expected_grad):
        torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-12)
    operation = lambda x, y: yastn.tensordot(a._replace(data=x), b._replace(data=y), axes).data
    assert torch.autograd.gradgradcheck(operation, (a.data, b.data), fast_mode=True)


@pytest.mark.parametrize('needs', [(True, True), (True, False), (False, True)])
def test_recoupled_saved_inputs(config_kwargs, needs):
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym='SU2')
    leg = yastn.Leg(config, t=(1,), D=(2,))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2, dtype='complex128')
    b = yastn.rand_like(a)
    a.requires_grad_(needs[0])
    b.requires_grad_(needs[1])
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        found = yastn.tensordot(a, b, ((0, 3), (1, 2)))
    assert len(saved) == sum(needs) and all(t is a.data or t is b.data for t in saved)
    inputs = tuple(t.data for t, need in zip((a, b), needs) if need)
    gradients = torch.autograd.grad(found.data.real.sum(), inputs)
    assert all(gradient.numel() > 0 for gradient in gradients)


@pytest.mark.parametrize('dtypes', [('float32', 'float32'), ('complex64', 'complex64'),
                                    ('float32', 'float64'), ('float32', 'complex128'),
                                    ('complex64', 'complex128')])
def test_recoupled_low_precision(config_kwargs, dtypes, monkeypatch):
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym='SU2')
    leg = yastn.Leg(config, t=(0, 1, 2), D=(1, 2, 1))
    a, b = (yastn.rand(config, legs=(leg, leg.conj()) * 2, dtype=dtype)
            for dtype in dtypes)
    a.requires_grad_()
    b.requires_grad_()
    axes = ((0, 3), (1, 2))
    with monkeypatch.context() as patch:
        patch.setattr(backend_torch, 'recouple_dot_sum', None)
        expected = yastn.tensordot(a, b, axes)
    found = yastn.tensordot(a, b, axes)
    assert expected.size > 0 and expected.norm().item() > 0
    # Recoupling now accumulates in the promoted dtype, so mixed inputs need
    # tolerances appropriate to the original single-precision recoupling.
    torch.testing.assert_close(found.data, expected.data, atol=2e-5, rtol=2e-5)
    gradient = torch.randn_like(found.data)
    expected_grad = torch.autograd.grad(expected.data, (a.data, b.data), gradient)
    found_grad = torch.autograd.grad(found.data, (a.data, b.data), gradient)
    for actual, reference in zip(found_grad, expected_grad):
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('conjugate', [False, True])
@pytest.mark.parametrize('charged', [False, True])
def test_recoupled_pending_transpose(config_kwargs, sym, conjugate, charged, monkeypatch):
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym=sym)
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    total = (2 * charged,) if sym == 'SU2' else (2 * charged, 0)
    leg = yastn.Leg(config, t=charges, D=(1, 2, 1))
    tensors = []
    for side, order in enumerate(((2, 0, 3, 1), (3, 1, 2, 0))):
        charge = total if side == 0 else (0,) * len(total)
        tensor = yastn.rand(config, legs=(leg, leg.conj()) * 2, n=charge, dtype='complex128')
        storage = tensor.data.new_empty(tensor.size * 2)
        storage[::2] = tensor.data
        data = storage[::2].detach().requires_grad_()
        tensors.append(tensor._replace(data=data).transpose(order))
    a, b = tensors
    assert not a.data.is_contiguous() and not b.data.is_contiguous()
    axes, conj = ((0, 3), (1, 2)), (conjugate, conjugate)
    with monkeypatch.context() as patch:
        patch.setattr(backend_torch, 'recouple_dot_sum', None)
        expected = yastn.tensordot(a, b, axes, conj=conj)
    calls = []
    original = backend_torch.recouple_dot_sum

    def tracked(*args):
        result = original(*args)
        assert result is not None
        calls.append(result.numel())
        return result

    monkeypatch.setattr(backend_torch, 'recouple_dot_sum', tracked)
    found = yastn.tensordot(a, b, axes, conj=conj)
    assert calls and min(calls) > 0 and expected.norm().item() > 0
    assert (found.struct, found.hfs, found.trans) == (expected.struct, expected.hfs, expected.trans)
    torch.testing.assert_close(found.data, expected.data)
    gradient = torch.randn_like(found.data)
    expected_grad = torch.autograd.grad(expected.data, (a.data, b.data), gradient)
    found_grad = torch.autograd.grad(found.data, (a.data, b.data), gradient)
    for actual, reference in zip(found_grad, expected_grad):
        torch.testing.assert_close(actual, reference)


def test_recoupled_composition_fallback(config_kwargs, monkeypatch):
    from yastn.backend import _recoupling
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym='SU2')
    leg = yastn.Leg(config, t=(0, 1, 2), D=(1, 2, 1))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2)
    b = yastn.rand_like(a)
    axes = ((0, 3), (1, 2))
    expected = yastn.tensordot(a, b, axes)
    monkeypatch.setattr(_recoupling, '_UNFUSE_COMPOSITION_MAX_WORK_RATIO', 0)
    calls = []
    original = backend_torch.recouple_dot_sum

    def tracked(*args):
        result = original(*args)
        assert result is None
        calls.append(True)
        return result

    monkeypatch.setattr(backend_torch, 'recouple_dot_sum', tracked)
    found = yastn.tensordot(a, b, axes)
    assert calls and found.size > 0 and expected.norm().item() > 0
    torch.testing.assert_close(found.data, expected.data)


def test_recoupled_allocation_bound(monkeypatch):
    config = yastn.make_config(backend='torch', sym='SU2')
    leg = yastn.Leg(config, t=(0, 1, 2), D=(3, 3, 1))  # D=12.
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2)
    b = yastn.rand_like(a)
    axes = ((0, 2, 3), (1, 3, 0))
    warm = yastn.tensordot(a, b, axes)
    assert warm.size > 0
    del warm
    with torch.profiler.profile(profile_memory=True) as profile:
        found = yastn.tensordot(a, b, axes)
    allocations = [event.self_cpu_memory_usage for event in profile.events()
                   if event.self_cpu_memory_usage > 0]
    assert allocations and found.size > 0
    # No allocation is large enough to hold either complete recoupled input.
    assert max(allocations) < min(a.size, b.size) * a.data.element_size()
