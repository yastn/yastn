"""R-only QR and truncation before singular-vector unfusion."""
import pytest

import yastn
from yastn.tensor import linalg


def _tensor(config_kwargs, sym, dtype):
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=17)
    if sym == 'dense':
        return yastn.rand(config, s=(1, 1, -1, -1), D=(3, 4, 4, 3), dtype=dtype)
    charges = (0, 1) if sym != 'SU2xU1' else ((0, 0), (1, 1))
    leg = yastn.Leg(config, s=1, t=charges, D=(2, 2))
    return yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()),
                      n=1 if sym == 'U1' else None, dtype=dtype)


@pytest.mark.parametrize('sym', ['dense', 'U1', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_r_only_qr(config_kwargs, sym, dtype):
    a = _tensor(config_kwargs, sym, dtype)
    for axes in (((0, 1), (2, 3)), ((2, 0), (3, 1))):
        _, reference = a.qr(axes=axes, sQ=-1, Raxis=-1)
        actual = a.qr(axes=axes, sQ=-1, Raxis=-1, compute_q=False)
        assert actual.size > 0 and actual.struct == reference.struct
        assert actual.is_consistent()
        assert (actual - reference).norm().item() < 1e-11
    empty = yastn.Tensor(a.config, s=(1, -1))
    assert empty.qr(compute_q=False).size == 0


def _alternating_mask(s):
    indices = s.config.backend.arange(0, s.size, device=s.device)
    return s._replace(data=indices % 2 == 1)


@pytest.mark.parametrize('sym', ['dense', 'U1', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
@pytest.mark.parametrize('truncation', [dict(D_total=5), dict(tol=0.3, D_block=2),
                                       dict(D_total=0), dict(mask_f=_alternating_mask),
                                       dict(D_total=5, eps_multiplet=1e-8)])
def test_truncate_before_unfuse(config_kwargs, sym, dtype, truncation, monkeypatch):
    a = _tensor(config_kwargs, sym, dtype)
    axes = ((2, 0), (3, 1)) if 'mask_f' in truncation else ((0, 1), (2, 3))
    options = dict(axes=axes, sU=-1, nU=False, fix_signs=True)
    u, s, v = a.svd(**options)
    mask = yastn.truncation_mask(s, **truncation)
    reference = mask.apply_mask(u, s, v, axes=(-1, 0, 0))
    observed = []
    original = linalg._unfuse_blocks

    def record(config, data, struct, axes, hfs):
        observed.append(data.size if config.backend.BACKEND_ID == 'np' else data.numel())
        return original(config, data, struct, axes, hfs)

    monkeypatch.setattr(linalg, '_unfuse_blocks', record)
    actual = a.svd_with_truncation(**options, **truncation)
    assert len(observed) == 2
    assert observed == [actual[0].size, actual[2].size]
    for found, wanted in zip(actual, reference):
        assert found.struct == wanted.struct and found.is_consistent()
        assert (found - wanted).norm().item() < 2e-10
    if truncation.get('D_total') != 0:
        assert actual[1].size > 0
    assert actual[0].size < u.size


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_r_only_qr_grad_and_mode(config_kwargs, dtype, monkeypatch):
    if 'torch' not in config_kwargs['backend']:
        pytest.skip('Torch QR dispatch and derivatives')
    import torch

    config = yastn.make_config(sym='dense', **config_kwargs)
    a = yastn.rand(config, s=(1, -1), D=(12, 7), dtype=dtype)
    a.requires_grad_()
    modes = []
    original = torch.linalg.qr

    def record(data, mode='reduced', **kwargs):
        modes.append(mode)
        return original(data, mode=mode, **kwargs)

    monkeypatch.setattr(torch.linalg, 'qr', record)
    with torch.no_grad():
        actual = a.qr(compute_q=False)
        assert modes and set(modes) == {'r'}
        modes.clear()
        reference = a.qr()[1]
        torch.testing.assert_close(actual.data, reference.data)
    modes.clear()
    actual = a.qr(compute_q=False)
    reference = a.qr()[1]
    assert set(modes) == {'reduced'}
    weight = torch.randn_like(actual.data)
    g = torch.autograd.grad(actual.data, a.data, weight)[0]
    ref = torch.autograd.grad(reference.data, a.data, weight)[0]
    torch.testing.assert_close(g, ref)
    assert torch.autograd.gradcheck(lambda x: a._replace(data=x).qr(compute_q=False).data,
                                   (a.data,), fast_mode=True)


@pytest.mark.parametrize('sym', ['dense', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_early_svd_truncation_gradient(config_kwargs, sym, dtype):
    if 'torch' not in config_kwargs['backend']:
        pytest.skip('Torch derivatives')
    import torch

    a = _tensor(config_kwargs, sym, dtype)
    a.requires_grad_()
    options = dict(axes=((0, 1), (2, 3)), fix_signs=True)
    actual = a.svd_with_truncation(**options, D_total=3)
    u, s, v = a.svd(**options)
    reference = yastn.truncation_mask(s, D_total=3).apply_mask(u, s, v, axes=(-1, 0, 0))
    weights = [torch.randn_like(t.data) for t in actual]
    assert actual[1].size > 0
    g = torch.autograd.grad([t.data for t in actual], a.data, weights)[0]
    ref = torch.autograd.grad([t.data for t in reference], a.data, weights)[0]
    torch.testing.assert_close(g, ref, rtol=1e-9, atol=1e-10)
