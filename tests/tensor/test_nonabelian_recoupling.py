"""Cached fusion plans and the vectorized reduced-block transformation."""
import numpy as np
import pytest
import yastn


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_fusion_tree_cache_reuses_structure(config_kwargs, sym):
    config = yastn.make_config(sym=sym, **config_kwargs)
    charge = (1,) if sym == 'SU2' else (1, 1)
    leg = yastn.Leg(config, s=1, t=(charge,), D=(2,))
    a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()))
    yastn.clear_cache()
    order = (3, 0, 2, 1)
    first = a.transpose(order).consume_transpose()
    second = (a * 2).transpose(order).consume_transpose()
    assert (second - first * 2).norm() < 1e-12
    assert yastn.get_cache_info()['fusion_tree_adjacent_swap'].hits > 0
    yastn.clear_cache()
    assert yastn.get_cache_info()['fusion_tree_adjacent_swap'].currsize == 0


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_recoupling_kernel_and_adjoint(config_kwargs, dtype):
    torch = pytest.importorskip('torch')
    from yastn.backend._backend_torch_nonabelian import _Recouple, _recoupling_plan
    # Two input channels, unequal axes, both mixed into two output channels.
    shape, order = (2, 3, 2), (2, 0, 1)
    meta = ((0, 0, shape, 0.6), (0, 12, shape, 0.8),
            (12, 0, shape, -0.8), (12, 12, shape, 0.6))
    data = torch.randn(24, dtype=getattr(torch, dtype), device=config_kwargs['default_device'], requires_grad=True)
    plan = _recoupling_plan(order, meta, data.real.dtype, data.device)

    def evaluate(x):
        return _Recouple.apply(x, plan, 24)

    channels = data.reshape(2, *shape).permute(0, *(i + 1 for i in order)).reshape(2, -1)
    rotation = data.new_tensor([[0.6, 0.8], [-0.8, 0.6]])
    expected = (rotation @ channels).reshape(-1)
    torch.testing.assert_close(evaluate(data), expected)
    assert torch.autograd.gradcheck(evaluate, (data,))


def test_recoupling_tiled_and_unused_entries(config_kwargs):
    torch = pytest.importorskip('torch')
    from yastn.backend._backend_torch_nonabelian import _Recouple, _recoupling_plan
    # Cross the tile boundary, with uncovered input and output entries.
    shape = (1025, 1024)
    size = int(np.prod(shape))
    data = torch.arange(size + 3, dtype=torch.float64, device=config_kwargs['default_device'], requires_grad=True)
    meta = ((2, 1, shape, -0.5),)
    plan = _recoupling_plan((1, 0), meta, data.dtype, data.device)
    result = _Recouple.apply(data, plan, size + 5)
    torch.testing.assert_close(result[2:2 + size], -0.5 * data[1:1 + size].reshape(shape).T.reshape(-1))
    assert result[:2].count_nonzero() == 0 and result[2 + size:].count_nonzero() == 0
    gradient, = torch.autograd.grad(result.sum(), (data,))
    assert torch.all(gradient[1:1 + size] == -0.5)
    assert gradient[0] == 0 and gradient[-2:].count_nonzero() == 0


def test_recoupling_gpu_matches_magnetic_basis(config_kwargs, monkeypatch):
    """Run the vectorized kernel on CPU too, using the independent CG oracle."""
    pytest.importorskip('torch')
    from yastn.backend import backend_torch
    from yastn.backend._backend_torch_nonabelian import _Recouple, _recoupling_plan
    from .test_nonabelian_metadata import test_nonabelian_adjacent_swaps_against_magnetic_basis

    def vectorized(data, order, meta, size):
        if not meta:
            return data[:0]
        return _Recouple.apply(data, _recoupling_plan(order, meta, data.real.dtype, data.device), size)

    monkeypatch.setattr(backend_torch, 'recouple', vectorized)
    for sym in ('SU2', 'SU2xU1'):
        test_nonabelian_adjacent_swaps_against_magnetic_basis(
            {**config_kwargs, 'backend': 'torch'}, sym)
