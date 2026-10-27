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
    recoupled = backend_torch.recouple_and_merge
    fused = backend_torch.recouple_and_fuse
    with monkeypatch.context() as reference_patch:
        reference_patch.setattr(backend_torch, 'transpose_and_merge_scaled', None)
        reference_patch.setattr(backend_torch, 'recouple_and_merge', None)
        reference_patch.setattr(backend_torch, 'recouple_and_fuse', None)
        reference = operation()
    calls = []

    def probe(*args):
        calls.append(len(args[2]))
        return merged(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError('Hard fusion must not create a scaled source buffer')

    def recoupled_probe(*args, **kwargs):
        calls.append(len(args[2]))
        return recoupled(*args, **kwargs)

    def fused_probe(*args, **kwargs):
        result = fused(*args, **kwargs)
        if result is not None:
            calls.append(len(args[2]))
        return result

    monkeypatch.setattr(backend_torch, 'transpose_and_merge_scaled', probe)
    monkeypatch.setattr(backend_torch, 'recouple_and_merge', recoupled_probe)
    monkeypatch.setattr(backend_torch, 'recouple_and_fuse', fused_probe)
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


@pytest.mark.parametrize('tile', [127, 1 << 20])
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_mask_tiles_and_adjoint(axis, dtype, tile, monkeypatch, request):
    """Noncontiguous selections span tiles and preserve first/second derivatives."""
    import numpy as np

    device = request.config.getoption('--device')
    shape = (12, 13, 14)
    selected_shape = list(shape)
    selected_shape[axis] = 5
    count, selected_count = int(np.prod(shape)), int(np.prod(selected_shape))
    meta_dtype = [('sln', np.int64, (2,)), ('Dn', np.int64, (3,)),
                  ('sla', np.int64, (2,)), ('Da', np.int64, (3,)), ('tm', np.int64, (1,))]
    meta = np.array([((i * selected_count, (i + 1) * selected_count), selected_shape,
                      (i * count, (i + 1) * count), shape, (i,)) for i in range(2)], dtype=meta_dtype)
    indices = torch.tensor([0, 3, 5, 8, 11], device=device)
    mask = {(0,): indices, (1,): indices.flip(0)}
    data = torch.randn(2 * count, dtype=dtype, device=device, requires_grad=True)
    expected = torch.cat([data[i * count:(i + 1) * count].view(shape)
                          .index_select(axis, mask[(i,)]).reshape(-1) for i in range(2)])
    calls = []
    original = backend_torch._mask_indices

    def probe(plan, lo, hi):
        calls.append(hi - lo)
        return original(plan, lo, hi)

    monkeypatch.setattr(backend_torch, '_MASK_TILE', tile)
    monkeypatch.setattr(backend_torch, '_mask_indices', probe)
    operation = lambda x: backend_torch.apply_mask(x, mask, meta, 2 * selected_count, axis, 3)
    actual = operation(data)
    torch.testing.assert_close(actual, expected)
    assert sum(p.numel() for p in actual.grad_fn.parameters if torch.is_tensor(p)) < selected_count
    weight = torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad(actual, data, weight)[0],
                               torch.autograd.grad(expected, data, weight)[0])
    # Embedding uses the reversed metadata and must be the adjoint of gathering.
    reverse = meta.copy()
    reverse['sln'], reverse['sla'] = meta['sla'], meta['sln']
    reverse['Dn'], reverse['Da'] = meta['Da'], meta['Dn']
    embed = lambda x: backend_torch.embed_mask(x, mask, reverse, data.numel(), axis, 3)
    small = weight.detach().requires_grad_()
    torch.testing.assert_close(embed(small), torch.autograd.grad(operation(data), data, weight)[0])
    assert torch.autograd.gradcheck(operation, (data,), fast_mode=True)
    assert torch.autograd.gradgradcheck(operation, (data,), fast_mode=True)
    assert torch.autograd.gradgradcheck(embed, (small,), fast_mode=True)
    if device == 'cpu' and tile >= selected_count:
        assert not calls and actual.grad_fn.parameters[4]
    else:
        assert len(calls) > 2 and 0 < min(calls) <= max(calls) <= tile


@pytest.mark.parametrize('kind', ['merge', 'unmerge'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_hybrid_streams_maps(kind, dtype, monkeypatch, request):
    """Mixed blocks with gaps use bounded maps and retain only block metadata."""
    from yastn.backend import _backend_torch_backwards as kernels

    device = request.config.getoption('--device')
    shapes = ((12, 12, 12), (12, 24, 24), (12, 12, 12))
    order = (1, 0, 2)
    meta, source_size, output_size = [], 5, 3
    for shape in shapes:
        count = shape[0] * shape[1] * shape[2]
        fused = (shape[1], shape[0] * shape[2])
        meta.append((slice(output_size, output_size + count), fused,
                     slice(source_size, source_size + count), shape,
                     tuple(slice(0, d) for d in fused), fused))
        source_size += count + 7
        output_size += count + 3
    calls = []
    original = kernels._build_dest

    def probe(parameters, positions, sentinel=None):
        calls.append(positions.numel())
        return original(parameters, positions, sentinel)

    monkeypatch.setattr(kernels, '_build_dest', probe)
    if kind == 'merge':
        parameters = kernels.device_merge_parameters(order, (meta[0], meta[2]), source_size, device)
        operation = lambda x: kernels.kernel_transpose_and_merge_hybrid.apply(
            x, order, (meta[1],), parameters, output_size, 257)
        reference = lambda x: kernels.kernel_transpose_and_merge.apply(x, order, meta, output_size)
        size = source_size
    else:
        unmeta = [(slo, Dns, sln, Dn, ssln) for sln, Dn, slo, Do, ssln, Dns in meta]
        relabeled = tuple((slo, Do, sln, Dn, sslo, Dn)
                          for sln, Dn, slo, Do, sslo in (unmeta[0], unmeta[2]))
        parameters = kernels.device_merge_parameters((0, 1), relabeled, source_size, device)
        operation = lambda x: kernels.kernel_unmerge_hybrid.apply(
            x, (unmeta[1],), parameters, source_size, 257)
        reference = lambda x: kernels.kernel_unmerge.apply(x, unmeta, source_size)
        size = output_size
    data = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
    actual, expected = operation(data), reference(data)
    torch.testing.assert_close(actual, expected)
    assert sum(p.numel() for p in parameters.values() if torch.is_tensor(p)) < 100
    weight = torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad(actual, data, weight)[0],
                               torch.autograd.grad(expected, data, weight)[0])
    assert torch.autograd.gradcheck(operation, (data,), fast_mode=True)
    assert torch.autograd.gradgradcheck(operation, (data,), fast_mode=True)
    assert len(calls) > 2 and 0 < min(calls) <= max(calls) <= 257


def test_unscaled_merge_no_block_copy():
    """The unscaled path must also write a permuted D=12 block directly."""
    from yastn.backend._backend_torch_backwards import kernel_transpose_and_merge

    data = torch.randn(12 ** 3, dtype=torch.float64, requires_grad=True)
    meta = ((slice(0, data.numel()), (12, 144), slice(0, data.numel()),
             (12, 12, 12), (slice(0, 12), slice(0, 144)), (12, 144)),)
    with torch.profiler.profile(profile_memory=True) as profile:
        actual = kernel_transpose_and_merge.apply(data, (1, 0, 2), meta, data.numel())
    copies = [e for e in profile.events() if e.name == 'aten::copy_']
    assert copies and all(e.cpu_memory_usage == 0 for e in copies)
    assert not any(e.name == 'aten::clone' for e in profile.events())
    torch.testing.assert_close(actual.view(12, 12, 12), data.view(12, 12, 12).permute(1, 0, 2))


@pytest.mark.parametrize('start', [0, 2 ** 31])
def test_mask_index_address_width(start, monkeypatch, request):
    """Large offsets need int64 even when the selected output is tiny."""
    import numpy as np

    device = torch.device(request.config.getoption('--device'))
    dtype = [('sln', np.int64, (2,)), ('Dn', np.int64, (1,)),
             ('sla', np.int64, (2,)), ('Da', np.int64, (1,)), ('tm', np.int64, (1,))]
    meta = np.array([((0, 5), (5,), (start, start + 12), (12,), (0,))], dtype=dtype)
    mask = {(0,): torch.tensor([0, 3, 5, 8, 11], device=device)}
    monkeypatch.setattr(backend_torch, '_MASK_TILE', 1)
    parameters = backend_torch._mask_parameters(mask, meta, 0, device, True)
    assert parameters[0].dtype == (torch.int64 if start else torch.int32)
    actual = backend_torch._mask_indices(parameters, 0, 5)
    torch.testing.assert_close(actual, mask[(0,)] + start)


def test_adaptive_index_tile_memory_budget(monkeypatch):
    """Large workloads grow tiles; output and live memory constrain that growth."""
    import numpy as np
    from yastn.backend._backend_torch_backwards import adaptive_index_tile

    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (16 << 30, 24 << 30))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 4 << 30)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 2 << 30)
    assert adaptive_index_tile('cuda', 1 << 27, np.int64(1 << 30)) == 1 << 23
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (256 << 20, 24 << 30))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 2 << 30)
    tile = adaptive_index_tile('cuda', 1 << 27, 128 << 20)
    assert tile < 1 << 20
    assert tile * 80 <= (128 << 20) // 8

    def forbidden(*args):
        raise AssertionError('Small operations must not query driver memory')

    monkeypatch.setattr(torch.cuda, 'mem_get_info', forbidden)
    assert adaptive_index_tile('cuda', 123, 0) == 123
    assert adaptive_index_tile('cuda', 0, 0) == 1
