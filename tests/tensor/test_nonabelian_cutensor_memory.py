"""Batched cuTENSOR packing, original-input retention and allocation recovery."""
import sys
import weakref
from types import SimpleNamespace

import pytest
import yastn

torch = pytest.importorskip('torch')
from yastn.backend import backend_torch
from yastn.tensor import _contractions, _merging
from .test_nonabelian_cutensor import oracle_backend, accelerated


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('recoupled', [False, True])
@pytest.mark.parametrize('needs', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('dtypes', [('float64', 'float64'), ('complex128', 'complex128'),
                                  ('float32', 'complex128'), ('complex128', 'float32')])
def test_batched_values_gradients_and_storage(accelerated, monkeypatch, sym, recoupled, needs, dtypes):
    backend, device, calls = accelerated
    implementation = sys.modules[backend.__name__]
    config = yastn.make_config(sym=sym, backend=backend, default_device=device)
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    leg = yastn.Leg(config, t=charges, D=(1, 2, 1))
    a, b = (yastn.rand(config, legs=(leg, leg.conj()) * 2, dtype=dtype)
            for dtype in dtypes)
    a = a.remove_random_blocks(number=a.nblocks // 4, keep_legs=True)
    b = b.remove_random_blocks(number=b.nblocks // 3, keep_legs=True)
    a.requires_grad_(needs[0])
    b.requires_grad_(needs[1])
    axes = ((0, 3), (1, 2)) if recoupled else ((2, 3), (1, 0))
    reference_config = config._replace(backend=backend_torch)
    expected = yastn.tensordot(a._replace(config=reference_config), b._replace(config=reference_config), axes)
    assert expected.size > 0 and expected.norm().item() > 0
    monkeypatch.setattr(_contractions, '_NONABELIAN_RECOUPLE_MIN_SIZE', 0)
    monkeypatch.setenv('YASTN_CUTENSOR_PACK_CHUNK_BYTES', '1024')
    packed_refs, batch_sizes = [], []
    original_pack, original_product = implementation._pack_operand, implementation._packed_product

    def pack(*args):
        result = original_pack(*args)
        packed_refs.append(weakref.ref(result))
        return result

    def product(x, y, plan, order, axes):
        size = implementation._packed_elements(plan) * x.element_size()
        assert 0 < size <= 1024
        batch_sizes.append(size)
        return original_product(x, y, plan, order, axes)

    monkeypatch.setattr(implementation, '_pack_operand', pack)
    monkeypatch.setattr(implementation, '_packed_product', product)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        actual = yastn.tensordot(a, b, axes)
    assert len(calls) > 1 and len(batch_sizes) == len(calls)
    assert len(saved) == sum(needs)
    assert all(t is a.data or t is b.data for t in saved)
    assert all(ref() is None for ref in packed_refs)
    assert (actual.struct, actual.hfs) == (expected.struct, expected.hfs)
    tolerance = 2e-5 if 'float32' in dtypes else 1e-10
    torch.testing.assert_close(actual.data, expected.data, atol=tolerance, rtol=tolerance)
    inputs = tuple(t.data for t, need in zip((a, b), needs) if need)
    gradient = torch.randn(actual.size * 2, dtype=actual.data.dtype, device=device)[::2]
    found = torch.autograd.grad(actual.data, inputs, gradient)
    wanted = torch.autograd.grad(expected.data, inputs, gradient)
    for x, y in zip(found, wanted):
        torch.testing.assert_close(x, y, atol=tolerance, rtol=tolerance)
    assert all(ref() is None for ref in packed_refs)


@pytest.mark.parametrize('budget', ['1', '128'])
@pytest.mark.parametrize('recoupled', [False, True])
def test_batched_higher_derivatives(oracle_backend, monkeypatch, budget, recoupled):
    from .test_nonabelian_cutensor import _dense_descriptor_oracle
    backend = SimpleNamespace(**vars(oracle_backend))
    backend.is_cpu_device = lambda device: False
    monkeypatch.setattr(oracle_backend, 'tensordot_bs_v2', _dense_descriptor_oracle)
    monkeypatch.setattr(_contractions, '_NONABELIAN_RECOUPLE_MIN_SIZE', 0)
    monkeypatch.setenv('YASTN_CUTENSOR_PACK_CHUNK_BYTES', budget)
    config = yastn.make_config(sym='SU2', backend=backend)
    leg = yastn.Leg(config, t=(1,), D=(1,))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2, dtype='complex128')
    b = yastn.rand_like(a)
    axes = ((0, 3), (1, 2)) if recoupled else ((2, 3), (1, 0))

    def operation(x, y):
        return yastn.tensordot(a._replace(data=x), b._replace(data=y), axes).data

    a.requires_grad_()
    b.requires_grad_()
    assert operation(a.data, b.data).numel() > 0
    assert torch.autograd.gradcheck(operation, (a.data, b.data), fast_mode=True)
    assert torch.autograd.gradgradcheck(operation, (a.data, b.data), fast_mode=True)


@pytest.mark.parametrize('stage', ['forward', 'backward'])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('error', [torch.OutOfMemoryError('CUDA out of memory'),
                                 RuntimeError('torch_call_dispatcher( "aten::empty", "memory_format", '
                                              'stack.data(), TORCH_ABI_VERSION) API call failed at ops.h, line 431')])
def test_allocation_recovery(accelerated, monkeypatch, stage, error, batched):
    backend, device, calls = accelerated
    implementation = sys.modules[backend.__name__]
    config = yastn.make_config(sym='SU2', backend=backend, default_device=device)
    leg = yastn.Leg(config, t=(0, 1), D=(1, 2))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2, dtype='complex128')
    b = yastn.rand_like(a)
    a.requires_grad_()
    b.requires_grad_()
    axes = ((0, 3), (1, 2))
    reference_config = config._replace(backend=backend_torch)
    expected = yastn.tensordot(a._replace(config=reference_config), b._replace(config=reference_config), axes)
    assert expected.norm().item() > 0
    monkeypatch.setattr(_contractions, '_NONABELIAN_RECOUPLE_MIN_SIZE', 0)
    if batched:
        monkeypatch.setenv('YASTN_CUTENSOR_PACK_CHUNK_BYTES', '1024')
    original = implementation.tensordot_bs_v2
    failures, packed_refs, attempts = [], [], []

    def failing(x, y, ax, bx, *args):
        if ((ax, bx) == ([1], [0])) == (stage == 'forward'):
            attempts.append(True)
            if len(attempts) == (2 if batched else 1):
                failures.append(True)
                packed_refs.extend((weakref.ref(x), weakref.ref(y)))
                # A fresh exception prevents the test from retaining a traceback.
                raise type(error)(str(error))
        return original(x, y, ax, bx, *args)

    monkeypatch.setattr(implementation, 'tensordot_bs_v2', failing)
    with pytest.warns(RuntimeWarning, match='allocation failed'):
        actual = yastn.tensordot(a, b, axes)
        gradient = torch.randn_like(actual.data)
        found = torch.autograd.grad(actual.data, (a.data, b.data), gradient)
    assert failures and all(ref() is None for ref in packed_refs)
    torch.testing.assert_close(actual.data, expected.data)
    wanted = torch.autograd.grad(expected.data, (a.data, b.data), gradient)
    for x, y in zip(found, wanted):
        torch.testing.assert_close(x, y)


def test_nonallocation_errors_propagate(accelerated, monkeypatch):
    backend, device, _ = accelerated
    config = yastn.make_config(sym='SU2', backend=backend, default_device=device)
    leg = yastn.Leg(config, t=(1,), D=(1,))
    a = yastn.ones(config, legs=(leg, leg.conj()))

    def fail(*args):
        raise RuntimeError('Invalid descriptor')

    monkeypatch.setattr(sys.modules[backend.__name__], 'tensordot_bs_v2', fail)
    with pytest.raises(RuntimeError, match='Invalid descriptor'):
        a @ a


def test_inherited_optimizations(oracle_backend, monkeypatch):
    # Public Torch kernels, including QR/SVD, mask streaming and fusion, are
    # imported unchanged; the only cuTENSOR overrides are contractions.
    names = tuple(name for name in backend_torch.__all__ if callable(getattr(backend_torch, name)))
    assert len(names) > 50
    assert all(getattr(oracle_backend, name) is getattr(backend_torch, name) for name in names)
    backend = SimpleNamespace(**vars(oracle_backend))
    config = yastn.make_config(sym='SU2', backend=backend)
    leg = yastn.Leg(config, t=(0, 1, 2), D=(1, 2, 1))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2)
    expected = a.fuse_legs(((0, 3), (1, 2)), mode='hard')
    monkeypatch.setattr(_merging, '_USE_STREAMED_NONABELIAN_FUSION', True)
    monkeypatch.setattr(_merging, '_NONABELIAN_FUSION_MIN_SIZE', 0)
    calls = []
    original = backend.recouple_and_fuse

    def tracked(*args):
        result = original(*args)
        assert result is not None
        calls.append(True)
        return result

    monkeypatch.setattr(backend, 'recouple_and_fuse', tracked)
    actual = a.fuse_legs(((0, 3), (1, 2)), mode='hard')
    assert calls and actual.size > 0
    torch.testing.assert_close(actual.data, expected.data)
    restored = actual.unfuse_legs((0, 1))
    assert (restored - a.transpose((0, 3, 1, 2))).norm().item() < 1e-11


def test_batched_mixed_precision_cancellation(oracle_backend, monkeypatch):
    from yastn.tensor._contractions_cutensor import _build_nonabelian_cutensor_plan
    from .test_nonabelian_cutensor import _dense_descriptor_oracle
    products = tuple(((2, 3), (1, 1, 1), (0, 1), (1, 1), (i, i + 1), (1, 1), 1.)
                     for i in range(3))
    plan = _build_nonabelian_cutensor_plan(None, 5, products)
    a = torch.ones(1, dtype=torch.float32, requires_grad=True)
    b = torch.tensor([1e8, 1., -1e8], dtype=torch.float64, requires_grad=True)
    monkeypatch.setattr(oracle_backend, 'tensordot_bs_v2', _dense_descriptor_oracle)
    monkeypatch.setenv('YASTN_CUTENSOR_PACK_CHUNK_BYTES', '32')
    actual = oracle_backend.tensordot_nonabelian(a, b, plan)
    torch.testing.assert_close(actual, b.new_tensor([0., 0., 1., 0., 0.]))
    ga, gb = torch.autograd.grad(actual.sum(), (a, b))
    torch.testing.assert_close(ga, torch.ones_like(a))
    torch.testing.assert_close(gb, torch.ones_like(b))


def test_forward_live_packing_bound(oracle_backend, monkeypatch):
    from yastn.tensor._contractions_cutensor import _meta_tensordot_nonabelian_cutensor
    from .test_nonabelian_cutensor import _dense_descriptor_oracle
    config = yastn.make_config(sym='SU2', backend='torch')
    leg = yastn.Leg(config, t=(0, 1, 2), D=(3, 3, 1))  # D=12.
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2)
    b = yastn.rand(config, legs=(leg.conj(), leg) * 2)
    plan = _meta_tensordot_nonabelian_cutensor(config.sym, a.struct, b.struct,
                                             (0, 1), (2, 3), (0, 1), (2, 3), 0.5)
    budget = 1 << 14
    assert oracle_backend._packed_elements(plan) * 8 > 2 * budget
    monkeypatch.setenv('YASTN_CUTENSOR_PACK_CHUNK_BYTES', str(budget))
    refs, peaks, batch_count = [], [], []
    original_pack = oracle_backend._pack_operand

    def observe(tensor):
        refs.append(weakref.ref(tensor))
        live = [ref() for ref in refs if ref() is not None]
        peaks.append(sum(t.numel() * t.element_size() for t in live))

    def pack(*args):
        tensor = original_pack(*args)
        observe(tensor)
        return tensor

    def product(*args):
        batch_count.append(True)
        result = _dense_descriptor_oracle(*args)
        observe(result)
        return result

    monkeypatch.setattr(oracle_backend, '_pack_operand', pack)
    monkeypatch.setattr(oracle_backend, 'tensordot_bs_v2', product)
    actual = oracle_backend.tensordot_nonabelian(a.data, b.data, plan)
    expected = yastn.tensordot(a, b, ((2, 3), (0, 1)))
    assert len(batch_count) > 1 and actual.numel() > 0
    assert 0 < max(peaks) <= budget
    assert all(ref() is None for ref in refs)
    torch.testing.assert_close(actual, expected.data)


def test_adaptive_packing_budget(oracle_backend, monkeypatch):
    data = SimpleNamespace(is_cuda=True, device='cuda:0')
    plan = SimpleNamespace(slices_a=((0, 1 << 24),), slices_b=((0, 1 << 24),),
                           slices_c=((0, 1 << 20),), size=1 << 20)
    monkeypatch.setenv('YASTN_CUTENSOR_PACK_CHUNK_BYTES', str(1 << 30))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (64 << 20, 1 << 30))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 8 << 20)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 4 << 20)
    budget = oracle_backend._nonabelian_pack_budget(data, plan, torch.float64)
    available = (64 + 8 - 4 - 8) << 20
    assert 0 < budget * 8 <= available // 8
    assert budget & (budget - 1) == 0


@pytest.mark.parametrize('setting', ['0', '-1', 'invalid'])
def test_invalid_packing_budget(oracle_backend, monkeypatch, setting):
    monkeypatch.setenv('YASTN_CUTENSOR_PACK_CHUNK_BYTES', setting)
    with pytest.raises(ValueError, match='expected a positive integer'):
        oracle_backend._nonabelian_pack_budget(torch.ones(1), None, torch.float64)


@pytest.mark.parametrize('expanded', [False, True])
def test_contiguous_descriptor_storage(oracle_backend, expanded):
    data = torch.randn(1 if expanded else 20, dtype=torch.float64, requires_grad=True)
    view = data.expand(10) if expanded else data[::2]
    packed = oracle_backend._pack(view, ((0, 4), (4, 10)))
    assert packed.numel() == 10 and packed.is_contiguous()
    torch.testing.assert_close(packed, view)
    actual, = torch.autograd.grad(packed.sum(), data, retain_graph=True)
    expected, = torch.autograd.grad(view.sum(), data)
    torch.testing.assert_close(actual, expected)
