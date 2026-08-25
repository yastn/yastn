# Copyright 2026 The YASTN Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Autograd and checkpointing tests for recycled SI-CTMRG."""

import numpy as np
import pytest

import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.fpeps.envs._env_ctm as env_ctm_module


def _differentiable_ising_peps(config, beta):
    """One-site Ising PEPS whose bond weights retain beta's torch graph."""
    back = config.backend.torch
    leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    vertex = yastn.ones(
        config, legs=(leg, leg, leg.conj(), leg.conj()), n=0)
    spin_vertex = yastn.ones(
        config, legs=(leg, leg, leg.conj(), leg.conj()), n=1)
    bond = yastn.zeros(config, legs=(leg, leg.conj()))
    bond.set_block(ts=(0, 0), val=back.cosh(beta))
    bond.set_block(ts=(1, 1), val=back.sinh(beta))
    site = yastn.ncon(
        (vertex, bond, bond), ((-0, -1, 2, 3), (2, -2), (3, -3)))
    spin = yastn.ncon(
        (spin_vertex, bond, bond), ((-0, -1, 2, 3),
                                   (2, -2), (3, -3)))
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    return fpeps.Peps(geometry, tensors={(0, 0): site}), spin


def _ising_nn_objective(config_kwargs, beta, method, recycled=False,
                        return_env=False):
    """Evaluate one horizontal correlator with a fixed CTMRG update count."""
    import torch

    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=73)
    beta = (beta if isinstance(beta, torch.Tensor)
            else config.backend.to_tensor(beta, dtype='float64'))
    opts_svd = {'D_total': 2, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 2, 'niter': 4,
               'tol': 1e-10, 'warmup': 20, 'recycle_grad': False}
    update_kwargs = {'opts_svd': opts_svd, 'moves': 'hv',
                     'method': '2x2 corner'}
    if method == 'si':
        update_kwargs['opts_si'] = opts_si

    if recycled:
        # Freeze the recycled environment at the differentiation point.  This
        # is essential for a finite-difference oracle of recycle_grad=False:
        # rebuilding the warmup at beta +/- eps would also differentiate the
        # deliberately detached history.
        warmup_beta = config.backend.to_tensor(0.37, dtype='float64')
        warmup_psi, _ = _differentiable_ising_peps(config, warmup_beta)
        env = fpeps.EnvCTM(warmup_psi, init='eye')
        # These sweeps construct a realistic recycled environment but are not
        # part of the differentiated history.  The following tracked update
        # still depends on beta through the PEPS and through detached X/Y.
        with torch.no_grad():
            env.update_(**update_kwargs)
            env.update_(**update_kwargs)
        psi, spin = _differentiable_ising_peps(config, beta)
        # Keep C/T and recycled SI state, but attach the perturbed PEPS used by
        # the single differentiated update and observable contraction.
        env.psi = fpeps.EnvCTM(psi, init=None).psi
        if method == 'si':
            assert env.X and env.Y
            assert all(not x.requires_grad for x in env.X.values())
            assert all(not y.requires_grad for y in env.Y.values())
    else:
        psi, spin = _differentiable_ising_peps(config, beta)
        env = fpeps.EnvCTM(psi, init='eye')

    env.update_(**update_kwargs)
    value = env.measure_nn(spin, spin)[((0, 0), (0, 1))].real
    return (value, env) if return_env else value


@pytest.fixture
def torch_config(config_kwargs):
    """Use the torch backend independently of pytest's global default."""
    return {**config_kwargs, 'backend': 'torch'}


# ---------------------------------------------------------------------------
# Physical-observable differentiation oracles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('recycled', [False, True],
                         ids=('fresh_bases', 'recycled_detached_bases'))
def test_si_gradient_matches_central_finite_difference(torch_config,
                                                       recycled):
    """AD of a physical correlator agrees with a deterministic central FD."""
    import torch

    beta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value = _ising_nn_objective(
        torch_config, beta, method='si', recycled=recycled)
    value.backward()
    gradient_ad = beta.grad.item()

    eps = 2e-6
    plus = _ising_nn_objective(
        torch_config, beta.item() + eps, method='si', recycled=recycled)
    minus = _ising_nn_objective(
        torch_config, beta.item() - eps, method='si', recycled=recycled)
    gradient_fd = ((plus - minus) / (2 * eps)).item()

    # Symmetry-related singular values can be nearly degenerate.  The physical
    # derivative remains stable, but decomposition gauges limit tighter tests.
    assert np.isclose(gradient_ad, gradient_fd, rtol=5e-3, atol=5e-5)


@pytest.mark.parametrize('recycled', [False, True],
                         ids=('fresh_bases', 'recycled_detached_bases'))
def test_si_gradient_matches_full_svd_ctmrg(torch_config, recycled):
    """SI and full-SVD CTMRG differentiate the same physical correlator."""
    import torch

    beta_si = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value_si = _ising_nn_objective(
        torch_config, beta_si, method='si', recycled=recycled)
    value_si.backward()

    beta_full = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value_full = _ising_nn_objective(
        torch_config, beta_full, method='full', recycled=recycled)
    value_full.backward()

    assert np.isclose(value_si.item(), value_full.item(),
                      rtol=2e-5, atol=2e-7)
    assert np.isclose(beta_si.grad.item(), beta_full.grad.item(),
                      rtol=7e-3, atol=7e-5)


def test_recycle_grad_false_detaches_only_basis_history(torch_config):
    """Detached X/Y participate in forward SI without retaining their graph."""
    import torch

    beta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value, env = _ising_nn_objective(
        torch_config, beta, method='si', recycled=True, return_env=True)
    assert env.X and env.Y
    assert all(not x.requires_grad for x in env.X.values())
    assert all(not y.requires_grad for y in env.Y.values())
    assert value.requires_grad
    value.backward()
    assert beta.grad is not None
    assert np.isfinite(beta.grad.item())


# ---------------------------------------------------------------------------
# Recycled-basis graph policy and checkpointed CTMRG updates
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize('recycle_grad', [False, True])
def test_si_autograd_recycle_policy(torch_config, recycle_grad):
    """The recycle_grad option controls whether X/Y retain their graph."""
    config = yastn.make_config(sym='none', **torch_config)
    env = _dense_product_env(config)
    source = env.psi.ket[(0, 0)]
    source.requires_grad_(True)
    env.update_(
        opts_svd={'D_total': 1}, moves='h', method='2x2 corner',
        opts_si={'enabled': True, 'oversampling': 0, 'niter': 2,
                 'recycle_grad': recycle_grad})
    assert all(x.requires_grad == recycle_grad for x in env.X.values())
    assert all(y.requires_grad == recycle_grad for y in env.Y.values())
    loss = sum(tensor.norm()
               for site in env.sites()
               for tensor in env[site].__dict__.values()
               if tensor is not None)
    loss.backward()
    gradient = source.grad()
    assert gradient is not None
    assert np.isfinite(float(gradient.norm()))
    assert float(gradient.norm()) > 1e-12


def test_recycle_grad_true_backpropagates_through_second_update(
        torch_config, monkeypatch):
    """AI-generated test: a second update consumes graph-connected bases."""
    config = yastn.make_config(sym='none', **torch_config)
    env = _dense_product_env(config)
    source = env.psi.ket[(0, 0)]
    source.requires_grad_(True)
    opts_svd = {'D_total': 1}
    opts_si = {'enabled': True, 'oversampling': 0, 'niter': 2,
               'warmup': 20, 'recycle_grad': True}

    env.update_(opts_svd, moves='h', method='2x2 corner', opts_si=opts_si)
    assert env.X and all(x.requires_grad for x in env.X.values())
    assert all(y.requires_grad for y in env.Y.values())

    recycled_inputs = []
    original_proj_corners = env_ctm_module.proj_corners

    def recording_proj_corners(*args, **kwargs):
        recycled_inputs.append((kwargs.get('X'), kwargs.get('Y')))
        return original_proj_corners(*args, **kwargs)

    monkeypatch.setattr(
        env_ctm_module, 'proj_corners', recording_proj_corners)
    env.update_(opts_svd, moves='h', method='2x2 corner', opts_si=opts_si)

    assert recycled_inputs
    assert all(X is not None and Y is not None for X, Y in recycled_inputs)
    assert all(age == 2 for age in env._si_age.values())
    assert all(x.requires_grad for x in env.X.values())
    assert all(y.requires_grad for y in env.Y.values())

    loss = sum(tensor.norm()
               for site in env.sites()
               for tensor in env[site].__dict__.values()
               if tensor is not None)
    loss.backward()
    gradient = source.grad()
    assert gradient is not None
    assert np.isfinite(float(gradient.norm()))
    assert float(gradient.norm()) > 1e-12


@pytest.mark.parametrize('checkpoint_move', ['reentrant', 'nonreentrant'])
def test_si_checkpoint_move(torch_config, checkpoint_move, monkeypatch):
    """Both torch checkpoint modes preserve SI state and gradients."""
    config = yastn.make_config(sym='none', **torch_config)
    env = _dense_product_env(config)
    source = env.psi.ket[(0, 0)]
    source.requires_grad_(True)
    checkpoint_calls = []
    original_checkpoint = config.backend.checkpoint

    def recording_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs.get('use_reentrant'))
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(config.backend, 'checkpoint', recording_checkpoint)
    env.update_(
        opts_svd={'D_total': 1}, moves='h', method='2x2 corner',
        checkpoint_move=checkpoint_move,
        opts_si={'enabled': True, 'oversampling': 0, 'niter': 2})
    assert env.is_consistent()
    assert env.X
    assert checkpoint_calls == [checkpoint_move == 'reentrant']
    loss = sum(tensor.norm()
               for site in env.sites()
               for tensor in env[site].__dict__.values()
               if tensor is not None)
    loss.backward()
    gradient = source.grad()
    assert gradient is not None
    assert np.isfinite(float(gradient.norm()))
    assert float(gradient.norm()) > 1e-12
