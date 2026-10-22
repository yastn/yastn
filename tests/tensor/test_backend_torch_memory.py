"""Contraction regressions for bounded block storage and selective gradients."""
import pytest
import yastn

torch = pytest.importorskip('torch')

from yastn.backend._backend_torch_backwards import (
    _MatrixBlocks, kernel_dot, kernel_transpose_dot_sum,
)
from yastn.backend import backend_torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_scaled_merge_gradients_and_storage(dtype, monkeypatch, request):
    """Reordered, partial blocks scale correctly without saving expanded weights."""
    device = request.config.getoption('--device')
    data = torch.randn(34, dtype=dtype, device=device, requires_grad=True)
    original = data.detach().clone()
    order = (1, 0, 2)
    # Both source blocks land in a strided subview of one output block.
    # Other input blocks and output padding are intentionally unused.
    meta = ((slice(2, 47), (5, 9), slice(20, 32), (2, 3, 2),
             (slice(1, 4), slice(1, 5)), (3, 4)),
            (slice(2, 47), (5, 9), slice(3, 15), (2, 3, 2),
             (slice(1, 4), slice(5, 9)), (3, 4)))
    weights, lengths = (1., -0.5, 1., 2. ** 0.5, 1.), (3, 12, 5, 12, 2)
    reference = backend_torch.transpose_and_merge(
        backend_torch.scale_blocks(data, weights, lengths), order, meta, 50)
    # Compute the reference before forbidding repeat_interleave: its CUDA
    # scatter backward legitimately uses that operation to rebuild indices.
    gradient = torch.randn(100, dtype=dtype, device=device)[::2]
    reference_grad = torch.autograd.grad(reference, data, gradient)[0]

    def forbidden(*args, **kwargs):
        raise AssertionError('Weighted merge must not expand weights or scale the full source')

    monkeypatch.setattr(backend_torch, 'scale_blocks', forbidden)
    monkeypatch.setattr(torch, 'repeat_interleave', forbidden)
    operation = lambda x: backend_torch.transpose_and_merge_scaled(
        x, order, meta, 50, weights, lengths)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x) or x, lambda x: x):
        actual = operation(data)
    assert saved and sum(x.numel() for x in saved) == len(weights)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(data, original, rtol=0, atol=0)
    # A noncontiguous incoming gradient must also work.
    actual_grad = torch.autograd.grad(actual, data, gradient)[0]
    torch.testing.assert_close(actual_grad, reference_grad)
    if dtype in (torch.float64, torch.complex128):
        assert torch.autograd.gradcheck(operation, (data,), fast_mode=True)
        assert torch.autograd.gradgradcheck(operation, (data,), fast_mode=True)


@pytest.mark.parametrize('size_in,size_out', [(0, 0), (3, 5)])
def test_scaled_merge_empty_metadata(size_in, size_out, request):
    device = request.config.getoption('--device')
    data = torch.randn(size_in, dtype=torch.float64, device=device, requires_grad=True)
    weights, lengths = ((1.,), (size_in,)) if size_in else ((), ())
    result = backend_torch.transpose_and_merge_scaled(data, (), (), size_out, weights, lengths)
    torch.testing.assert_close(result, data.new_zeros(size_out))
    gradient = torch.autograd.grad(result.sum(), data, create_graph=True)[0]
    torch.testing.assert_close(gradient, data.new_zeros(size_in))


def test_scaled_merge_no_block_copy():
    """A permuted block is multiplied directly into output, with no block-sized scratch."""
    data = torch.randn(12 ** 3, dtype=torch.float64, requires_grad=True)
    meta = ((slice(0, data.numel()), (12, 144), slice(0, data.numel()),
             (12, 12, 12), (slice(0, 12), slice(0, 144)), (12, 144)),)
    with torch.profiler.profile(profile_memory=True) as profile:
        result = backend_torch.transpose_and_merge_scaled(data, (1, 0, 2), meta,
                                                         data.numel(), (0.5,), (data.numel(),))
        gradient = torch.autograd.grad(result.sum(), data)[0]
    events = profile.events()
    assert not any(event.name in ('aten::clone', 'aten::repeat_interleave') for event in events)
    multiplies = [event for event in events if event.name == 'aten::mul']
    assert len(multiplies) == 2
    assert all(event.cpu_memory_usage == 0 for event in multiplies)
    torch.testing.assert_close(result.view(12, 12, 12), data.view(12, 12, 12).permute(1, 0, 2) * 0.5)
    torch.testing.assert_close(gradient, torch.full_like(data, 0.5))


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('kind', ['legs', 'blocks'])
def test_fusion_uses_scaled_merge(sym, kind, monkeypatch, request):
    """Both fusion entry points match the old scale-then-merge path and its gradient."""
    from yastn.tensor._merging import _fuse_blocks

    config = yastn.make_config(sym=sym, backend='torch',
                               default_device=request.config.getoption('--device'))
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    leg = yastn.Leg(config, t=charges, D=(1, 2, 1))
    a = yastn.rand(config, legs=(leg, leg.conj(), leg, leg.conj()), dtype='complex128')
    a.requires_grad_()
    assert a.nblocks > 1

    def operation():
        if kind == 'legs':
            return a.fuse_legs(((0, 2), (1, 3)), mode='hard').data
        return _fuse_blocks(config, a.data, a.struct, ((0, 1), (2, 3)))[0]

    merged = backend_torch.transpose_and_merge_scaled
    with monkeypatch.context() as reference_patch:
        reference_patch.setattr(backend_torch, 'transpose_and_merge_scaled', None)
        reference = operation()
    calls = []

    def probe(*args):
        calls.append(len(args[2]))
        return merged(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError('Hard fusion must not create a scaled source buffer')

    monkeypatch.setattr(backend_torch, 'transpose_and_merge_scaled', probe)
    monkeypatch.setattr(backend_torch, 'scale_blocks', forbidden)
    actual = operation()
    assert calls and all(count > 0 for count in calls)
    torch.testing.assert_close(actual, reference)
    gradient = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, a.data, gradient)[0]
    reference_grad = torch.autograd.grad(reference, a.data, gradient)[0]
    torch.testing.assert_close(actual_grad, reference_grad)


@pytest.mark.parametrize('kind', ['dot', 'transpose'])
@pytest.mark.parametrize('dtypes', [(torch.float64, torch.float64),
                                    (torch.complex128, torch.complex128),
                                    (torch.float64, torch.complex128),
                                    (torch.complex128, torch.float64)])
@pytest.mark.parametrize('needs', [(True, True), (True, False), (False, True)])
def test_contraction_block_gradients(kind, dtypes, needs, monkeypatch, request):
    device = request.config.getoption('--device')
    # Force eviction on small correctness inputs; one block exceeds this budget.
    monkeypatch.setattr(_MatrixBlocks, 'max_bytes', 64)
    a = torch.randn(42, dtype=dtypes[0], device=device, requires_grad=needs[0])
    b = torch.randn(42, dtype=dtypes[1], device=device, requires_grad=needs[1])
    slices = (slice(1, 13), slice(15, 27), slice(29, 41))
    ra = tuple((sl, (2, 3, 2), 3, 4) for sl in slices)
    rb = tuple((sl, (2, 3, 2), 4, 3) for sl in slices)
    oa, ob = (1, 0, 2), (0, 2, 1)
    records = ((slice(1, 10), (3, 3), 0, 0, 0.5),
               (slice(13, 22), (3, 3), 1, 0, -2.),
               (slice(24, 33), (3, 3), 0, 1, 1.))
    if kind == 'transpose':
        # Two products accumulate into the same output block.
        records += ((slice(1, 10), (3, 3), 1, 1, 3.),)
        operation = lambda x, y: kernel_transpose_dot_sum.apply(x, y, records, ra, rb, oa, ob, 35)
    else:
        meta = tuple((slc, shape, slices[ia], (3, 4), slices[ib], (4, 3))
                     for slc, shape, ia, ib, _ in records)
        operation = lambda x, y: kernel_dot.apply(x, y, meta, 35)

    dtype = torch.promote_types(*dtypes)
    expected = torch.zeros(35, dtype=dtype, device=device)
    for slc, shape, ia, ib, coefficient in records:
        if kind == 'transpose':
            left = a[slices[ia]].view(2, 3, 2).permute(oa).reshape(3, 4)
            right = b[slices[ib]].view(2, 3, 2).permute(ob).reshape(4, 3)
        else:
            left, right = a[slices[ia]].view(3, 4), b[slices[ib]].view(4, 3)
            coefficient = 1.
        expected[slc] += (coefficient * (left.to(dtype) @ right.to(dtype))).reshape(-1)

    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x) or x, lambda x: x):
        actual = operation(a, b)
    assert len(saved) == sum(needs)
    assert all(t is a or t is b for t in saved)
    torch.testing.assert_close(actual, expected)
    inputs = tuple(x for x in (a, b) if x.requires_grad)
    weight = torch.randn_like(actual)
    found = torch.autograd.grad(actual, inputs, weight)
    wanted = torch.autograd.grad(expected, inputs, weight)
    for x, y in zip(found, wanted):
        torch.testing.assert_close(x, y)
    assert torch.autograd.gradcheck(operation, (a, b), fast_mode=True)
    assert torch.autograd.gradgradcheck(operation, (a, b), fast_mode=True)


def test_matrix_block_cache_bound(monkeypatch):
    monkeypatch.setattr(_MatrixBlocks, 'max_bytes', 2 * 12 * 8)
    data = torch.arange(48, dtype=torch.float64)
    reshape = tuple((slice(i, i + 12), (2, 3, 2), 3, 4) for i in range(0, 48, 12))
    blocks = _MatrixBlocks(data, reshape, (1, 0, 2), data.dtype)
    for i in (0, 1, 0, 2, 3, 1):
        expected = data[reshape[i][0]].view(2, 3, 2).permute(1, 0, 2).reshape(3, 4)
        torch.testing.assert_close(blocks[i], expected)
        assert 0 < blocks.nbytes <= blocks.max_bytes
        assert len(blocks.cache) <= 2


def test_dot_mixed_precision_accumulation(request):
    device = request.config.getoption('--device')
    a = torch.ones(1, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor([1e8, 1., -1e8], dtype=torch.float64, device=device)
    meta = tuple((slice(i, i + 1), (1, 1), slice(0, 1), (1, 1),
                  slice(i, i + 1), (1, 1)) for i in range(3))
    out = kernel_dot.apply(a, b, meta, 3)
    torch.testing.assert_close(torch.autograd.grad(out.sum(), a)[0], torch.ones_like(a))
