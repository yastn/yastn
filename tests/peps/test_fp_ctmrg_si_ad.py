# Copyright 2026 The YASTN Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Fixed-point autograd tests for recycled SI-CTMRG."""

from dataclasses import replace
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.fpeps.envs._env_ctm as env_ctm_module

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    "config.getoption('--backend') not in ('torch', 'torch_cutensor')",
    reason="Backend with AD support is required: [torch, torch_cutensor]")


def _dense_product_env(config):
    """Seeded dense PEPS used to inspect CTMRG differentiation graphs."""
    leg = yastn.Leg(config, s=1, D=(2,))
    physical = yastn.Leg(config, s=1, D=(2,))
    tensor = yastn.zeros(
        config, legs=(leg, leg, leg.conj(), leg.conj(), physical))
    values = np.sin(np.arange(1, 33, dtype=float)).reshape((2,) * 5)
    tensor.set_block(val=values)
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(geometry, tensors={(0, 0): tensor})
    return fpeps.EnvCTM(psi, init='eye')


def _fp_basis_policy_case(config_kwargs, monkeypatch, *, errors=None,
                          mode='si_skip', method='2x2 corner', moves='hv',
                          use_autograd=False, post_gauge_error=None):
    """Exercise real FP linearization with a controlled convergence snapshot.

    Gauge fitting and the convergence loop are independent of the basis policy:
    use an already warmed environment and identity gauges, but retain actual CTM
    updates, projector construction, and either production VJP implementation.
    """
    if config_kwargs['backend'] == 'torch_cutensor' and not use_autograd:
        pytest.skip('torch.func.vjp does not support torch_cutensor custom ops')
    from yastn.tn.fpeps.envs import make_ctm_opts, make_fixed_point_opts
    from yastn.tn.fpeps.envs import fixed_pt
    from yastn.tn.fpeps.envs import _env_ctm_dist_mp_AD as dispatch

    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=91)
    env = _dense_product_env(config)
    source = env.psi.ket[(0, 0)]
    source.requires_grad_(True)
    warmup_opts = make_ctm_opts(
        opts_svd={'D_total': 1, 'tol': 0, 'fix_signs': True},
        method=method, moves='hv', use_qr=False,
        opts_si={'enabled': True, 'oversampling': 1, 'niter': 2})
    with torch.no_grad():
        env.update_(opts=warmup_opts)
        env.update_(opts=warmup_opts)
    for key, state in tuple(env._si_age.items()):
        error = (errors or {}).get(key[1], 0.)
        if error is None:
            del env._si_age[key]
        else:
            env._si_age[key] = state._replace(error=error)
    si_opts = replace(warmup_opts.opts_si, enabled=mode != 'full_svd',
                      skip_SI_update=mode == 'si_skip', tol=1e-3,
                      warmup=20, redistribute_sectors=True)
    fp_opts = replace(warmup_opts, opts_si=si_opts, moves=moves,
                      max_sweeps=1, corner_tol=1e-10)
    opts = make_fixed_point_opts(
        fwd=warmup_opts, fp=fp_opts,
        devices=(config.default_device,) if use_autograd else None)
    calls, contexts, linearization_opts = [], [], []

    def converged_env(current, *_args, **_kwargs):
        return current, True, [], 0., 0.

    def identity_gauge(saved, updated, **_kwargs):
        gauge = fixed_pt.EnvGauge(saved.geometry)
        phases = {}
        for site in saved.sites():
            for direction in ('t', 'l', 'b', 'r'):
                leg = getattr(saved[site], direction).get_legs(0)
                setattr(gauge[site], direction,
                        yastn.eye(config, legs=leg, isdiag=False))
            for direction in ('tl', 'bl', 'br', 'tr', 't', 'l', 'b', 'r'):
                phases[saved.site2index(site), direction] = torch.zeros(
                    (), dtype=torch.float64, device=config.default_device)
        if post_gauge_error is not None:
            for key, state in tuple(updated._si_age.items()):
                updated._si_age[key] = state._replace(error=post_gauge_error)
        return gauge, phases

    original_projectors = env_ctm_module.si_proj_corners

    def record_projectors(*args, **kwargs):
        calls.append((torch.is_grad_enabled(), kwargs['niter'],
                      kwargs['redistribute']))
        return original_projectors(*args, **kwargs)

    original_forward = fixed_pt.FixedPoint.forward

    def record_forward(ctx, *args):
        result = original_forward(ctx, *args)
        contexts.append(ctx)
        return result

    original_iter = fixed_pt.FixedPoint.fixed_point_iter

    def record_iter(gauge, phases, current_opts, *args):
        linearization_opts.append(current_opts)
        return original_iter(gauge, phases, current_opts, *args)

    monkeypatch.setattr(fixed_pt.FixedPoint, 'get_converged_env',
                        staticmethod(converged_env))
    monkeypatch.setattr(fixed_pt, 'find_gauge_multi_sites', identity_gauge)
    monkeypatch.setattr(env_ctm_module, 'si_proj_corners', record_projectors)
    monkeypatch.setattr(fixed_pt.FixedPoint, 'forward',
                        staticmethod(record_forward))
    monkeypatch.setattr(fixed_pt.FixedPoint, 'fixed_point_iter',
                        staticmethod(record_iter))
    # Select the real autograd.grad branch without starting worker processes;
    # the distributed projector implementation does not support SI yet.
    if use_autograd:
        monkeypatch.setattr(dispatch, 'fp_update_',
                            lambda current, current_opts, **_kwargs:
                            current.update_(opts=current_opts))
        monkeypatch.setattr(dispatch, 'release_pool_cache', lambda: None)

    with warnings.catch_warnings(record=True) as forward_warnings:
        warnings.simplefilter('always')
        result = fixed_pt.fp_ctmrg(env, opts=opts)
    assert not [warning for warning in forward_warnings
                if 'unconverged X,Y projectors' in str(warning.message)]
    loss = sum(tensor._data.sum()
               for site in result.sites()
               for tensor in result[site].__dict__.values()
               if tensor is not None)
    return SimpleNamespace(loss=loss, source=source, calls=calls,
                           contexts=contexts, linearization_opts=linearization_opts,
                           opts=opts,
                           original_si=si_opts, original_fp=fp_opts,
                           original_fwd=warmup_opts)


@pytest.mark.parametrize('use_autograd', [False, True],
                         ids=('func_vjp', 'autograd_grad'))
@pytest.mark.parametrize('error', [0., 1e-3, 2e-3, float('inf'),
                                  float('nan'), None],
                         ids=('converged', 'equal_tolerance', 'above_tolerance',
                              'infinite', 'nan', 'missing'))
def test_fp_si_skip_freezes_bases_and_warns_only_in_backward(
        config_kwargs, monkeypatch, use_autograd, error):
    """Stored SI errors govern diagnostics; every skip-mode VJP freezes X/Y."""
    case = _fp_basis_policy_case(
        config_kwargs, monkeypatch, errors={'hlb': error},
        use_autograd=use_autograd,
        post_gauge_error=2e-3 if error == 0. else 0.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        case.loss.backward()
    warnings_fp = [warning for warning in caught
                   if 'unconverged X,Y projectors' in str(warning.message)]
    assert len(warnings_fp) == (0 if error == 0. else 1)
    if warnings_fp:
        assert issubclass(warnings_fp[0].category, RuntimeWarning)
        assert 'hlb' in str(warnings_fp[0].message)
        assert '0.001' in str(warnings_fp[0].message)

    key = next(key for key in case.contexts[0].si_errors if key[1] == 'hlb')
    saved_error = case.contexts[0].si_errors[key]
    if error is None:
        assert saved_error == float('inf')
    elif np.isnan(error):
        assert np.isnan(saved_error)
    else:
        assert saved_error == error
    tracked_calls = [call for call in case.calls if call[0]]
    assert tracked_calls and all(call[1:] == (0, False)
                                 for call in tracked_calls)
    assert any(niter == 2 and redistribute
               for tracked, niter, redistribute in case.calls if not tracked)
    assert case.source.grad() is not None
    assert np.isfinite(float(case.source.grad().norm()))
    assert case.opts.fwd is case.original_fwd
    assert case.opts.fp is case.original_fp
    assert case.opts.fp.opts_si is case.original_si
    assert case.opts.fp.opts_si.niter == 2
    assert case.opts.fp.opts_si.redistribute_sectors
    assert len(case.linearization_opts) == 1
    assert case.linearization_opts[0].fp.opts_si.niter == 0
    assert not case.linearization_opts[0].fp.opts_si.redistribute_sectors


def test_fp_si_skip_aggregates_unconverged_pairs(config_kwargs, monkeypatch):
    """One warning identifies every bad active pair and excludes converged pairs."""
    case = _fp_basis_policy_case(
        config_kwargs, monkeypatch, errors={'hlb': 2e-3, 'hrb': float('nan')})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        case.loss.backward()
    warnings_fp = [warning for warning in caught
                   if 'unconverged X,Y projectors' in str(warning.message)]
    assert len(warnings_fp) == 1
    message = str(warnings_fp[0].message)
    assert 'hlb' in message and 'hrb' in message
    assert 'vtr' not in message and 'vbr' not in message


@pytest.mark.parametrize('method', ['2x2 corner', '1x2'])
def test_fp_si_skip_ignores_inactive_pairs(config_kwargs, monkeypatch, method):
    """A horizontal FP step does not warn about stale vertical SI records."""
    case = _fp_basis_policy_case(
        config_kwargs, monkeypatch, method=method, moves='h',
        errors={'vtr': float('inf'), 'vbr': float('nan')})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        case.loss.backward()
    assert not [warning for warning in caught
                if 'unconverged X,Y projectors' in str(warning.message)]
    assert {key[1] for key in case.contexts[0].si_errors} == {'hlb', 'hrb'}
    assert all(call[1:] == (0, False) for call in case.calls if call[0])


@pytest.mark.parametrize('mode', ['si', 'full_svd'])
@pytest.mark.parametrize('use_autograd', [False, True],
                         ids=('func_vjp', 'autograd_grad'))
def test_fp_regular_si_and_full_svd_preserve_options(
        config_kwargs, monkeypatch, mode, use_autograd):
    """Freezing and its warning apply only to SI with skip_SI_update enabled."""
    case = _fp_basis_policy_case(
        config_kwargs, monkeypatch, mode=mode, use_autograd=use_autograd,
        errors={'hlb': float('inf')})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        case.loss.backward()
    assert not [warning for warning in caught
                if 'unconverged X,Y projectors' in str(warning.message)]
    assert case.linearization_opts == [case.opts]
    if mode == 'si':
        tracked_calls = [call for call in case.calls if call[0]]
        assert tracked_calls and all(call[1:] == (2, True)
                                     for call in tracked_calls)
    else:
        assert not case.calls
    assert case.source.grad() is not None
    assert np.isfinite(float(case.source.grad().norm()))


def test_fp_frozen_si_vjp_matches_finite_difference(config_kwargs, monkeypatch):
    """The linearized one-sweep map differentiates through frozen-basis projectors."""
    from yastn.tn.fpeps.envs import fixed_pt

    case = _fp_basis_policy_case(config_kwargs, monkeypatch)
    case.loss.backward(retain_graph=True)
    ctx = case.contexts[0]
    env_dict = yastn.combine_data_and_meta(
        ctx.saved_tensors[:ctx.env_data_num], ctx.env_meta)
    gauge_dict = yastn.combine_data_and_meta(
        ctx.saved_tensors[ctx.env_data_num:], ctx.g_meta)
    gauge = fixed_pt.EnvGauge.from_dict(gauge_dict)
    env_data, env_meta = yastn.split_data_and_meta(env_dict['env'])
    flat_env, env_slices = fixed_pt._concat_data(env_data)
    psi_data, psi_meta = yastn.split_data_and_meta(env_dict['psi'])
    flat_env = flat_env.detach()
    parameters = tuple(data.detach().clone() for data in psi_data)
    probe = torch.cos(torch.arange(flat_env.numel(), dtype=flat_env.dtype,
                                   device=flat_env.device) + .3)
    frozen_opts = case.linearization_opts[0]

    def objective(current_parameters):
        output, = fixed_pt.FixedPoint.fixed_point_iter(
            gauge, ctx.phase_dict, frozen_opts, env_dict, env_meta,
            env_slices, psi_meta, flat_env, current_parameters)
        return torch.dot(probe, output)

    _, vjp = torch.func.vjp(objective, parameters)
    gradient, = vjp(torch.ones((), dtype=flat_env.dtype,
                             device=flat_env.device))
    assert torch.linalg.vector_norm(gradient[0]) > 1e-10
    eps = 2e-6
    for index in (0, 3, 9, 15, 22):
        plus = tuple(data.clone() for data in parameters)
        minus = tuple(data.clone() for data in parameters)
        plus[0][index] += eps
        minus[0][index] -= eps
        with torch.no_grad():
            finite_difference = (objective(plus) - objective(minus)) / (2 * eps)
        assert np.isclose(gradient[0][index].item(), finite_difference.item(),
                          rtol=1e-4, atol=2e-7)
