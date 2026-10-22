"""Contraction regressions for bounded block storage and selective gradients."""
import pytest

torch = pytest.importorskip('torch')

from yastn.backend._backend_torch_backwards import (
    _MatrixBlocks, kernel_dot, kernel_transpose_dot_sum,
)


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
