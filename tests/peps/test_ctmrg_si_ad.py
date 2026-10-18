# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Autograd and checkpointing tests for recycled SI-CTMRG."""

import numpy as np
import pytest

import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.fpeps.envs._env_ctm_SI_projectors as si_module
from yastn.tn.fpeps.envs._env_ctm_SI_projectors import (
    initialize_si_bases, si_proj_corners)

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    "config.getoption('--backend') not in ('torch', 'torch_cutensor')",
    reason="Backend with AD support is required: [torch, torch_cutensor]")


def _si_bases(env, container):
    """Assigned recycled bases of ``env.si_X``/``env.si_Y``, keyed like ``env._si_age``."""
    return {(env.site2index(site), name): getattr(projectors, name)
            for site, projectors in container.items()
            for name in projectors.fields()
            if getattr(projectors, name) is not None}


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
            assert _si_bases(env, env.si_X) and _si_bases(env, env.si_Y)
            assert all(not x.requires_grad
                       for x in _si_bases(env, env.si_X).values())
            assert all(not y.requires_grad
                       for y in _si_bases(env, env.si_Y).values())
    else:
        psi, spin = _differentiable_ising_peps(config, beta)
        env = fpeps.EnvCTM(psi, init='eye')

    env.update_(**update_kwargs)
    value = env.measure_nn(spin, spin)[((0, 0), (0, 1))].real
    return (value, env) if return_env else value


# ---------------------------------------------------------------------------
# Physical-observable differentiation oracles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('recycled', [False, True],
                         ids=('fresh_bases', 'recycled_detached_bases'))
def test_si_gradient_matches_central_finite_difference(config_kwargs,
                                                       recycled):
    """AD of a physical correlator agrees with a deterministic central FD."""
    beta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value = _ising_nn_objective(
        config_kwargs, beta, method='si', recycled=recycled)
    value.backward()
    gradient_ad = beta.grad.item()

    eps = 2e-6
    plus = _ising_nn_objective(
        config_kwargs, beta.item() + eps, method='si', recycled=recycled)
    minus = _ising_nn_objective(
        config_kwargs, beta.item() - eps, method='si', recycled=recycled)
    gradient_fd = ((plus - minus) / (2 * eps)).item()

    # Symmetry-related singular values can be nearly degenerate.  The physical
    # derivative remains stable, but decomposition gauges limit tighter tests.
    assert np.isclose(gradient_ad, gradient_fd, rtol=5e-3, atol=5e-5)


@pytest.mark.parametrize('recycled', [False, True],
                         ids=('fresh_bases', 'recycled_detached_bases'))
def test_si_gradient_matches_full_svd_ctmrg(config_kwargs, recycled):
    """SI and full-SVD CTMRG differentiate the same physical correlator."""
    beta_si = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value_si = _ising_nn_objective(
        config_kwargs, beta_si, method='si', recycled=recycled)
    value_si.backward()

    beta_full = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value_full = _ising_nn_objective(
        config_kwargs, beta_full, method='full', recycled=recycled)
    value_full.backward()

    assert np.isclose(value_si.item(), value_full.item(),
                      rtol=2e-5, atol=2e-7)
    assert np.isclose(beta_si.grad.item(), beta_full.grad.item(),
                      rtol=7e-3, atol=7e-5)


def test_recycle_grad_false_detaches_only_basis_history(config_kwargs):
    """Detached X/Y participate in forward SI without retaining their graph."""
    beta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    value, env = _ising_nn_objective(
        config_kwargs, beta, method='si', recycled=True, return_env=True)
    assert _si_bases(env, env.si_X) and _si_bases(env, env.si_Y)
    assert all(not x.requires_grad for x in _si_bases(env, env.si_X).values())
    assert all(not y.requires_grad for y in _si_bases(env, env.si_Y).values())
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
def test_si_autograd_recycle_policy(config_kwargs, recycle_grad):
    """The recycle_grad option controls whether X/Y retain their graph."""
    config = yastn.make_config(sym='none', **config_kwargs)
    env = _dense_product_env(config)
    source = env.psi.ket[(0, 0)]
    source.requires_grad_(True)
    env.update_(
        opts_svd={'D_total': 1}, moves='h', method='2x2 corner',
        opts_si={'enabled': True, 'oversampling': 0, 'niter': 2,
                 'recycle_grad': recycle_grad})
    assert all(x.requires_grad == recycle_grad
               for x in _si_bases(env, env.si_X).values())
    assert all(y.requires_grad == recycle_grad
               for y in _si_bases(env, env.si_Y).values())
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
        config_kwargs, monkeypatch):
    """AI-generated test: a second update consumes graph-connected bases."""
    config = yastn.make_config(sym='none', **config_kwargs)
    env = _dense_product_env(config)
    source = env.psi.ket[(0, 0)]
    source.requires_grad_(True)
    opts_svd = {'D_total': 1}
    opts_si = {'enabled': True, 'oversampling': 0, 'niter': 2,
               'warmup': 20, 'recycle_grad': True}

    env.update_(opts_svd, moves='h', method='2x2 corner', opts_si=opts_si)
    assert _si_bases(env, env.si_X)
    assert all(x.requires_grad for x in _si_bases(env, env.si_X).values())
    assert all(y.requires_grad for y in _si_bases(env, env.si_Y).values())

    recycled_inputs = []
    original_si_proj_corners = si_module.si_proj_corners

    def recording_si_proj_corners(*args, **kwargs):
        recycled_inputs.append((kwargs.get('X'), kwargs.get('Y')))
        return original_si_proj_corners(*args, **kwargs)

    monkeypatch.setattr(
        si_module, 'si_proj_corners', recording_si_proj_corners)
    env.update_(opts_svd, moves='h', method='2x2 corner', opts_si=opts_si)

    assert recycled_inputs
    assert all(X is not None and Y is not None for X, Y in recycled_inputs)
    assert all(si_state.age == 2 for si_state in env._si_age.values())
    assert all(x.requires_grad for x in _si_bases(env, env.si_X).values())
    assert all(y.requires_grad for y in _si_bases(env, env.si_Y).values())

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
def test_si_checkpoint_move(config_kwargs, checkpoint_move, monkeypatch):
    """Both torch checkpoint modes preserve SI state and gradients."""
    config = yastn.make_config(sym='none', **config_kwargs)
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
    assert _si_bases(env, env.si_X)
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


# ---------------------------------------------------------------------------
# Functorch transforms, as used by the fixed-point backward
# ---------------------------------------------------------------------------


@pytest.fixture
def forbid_functorch_host_reads(monkeypatch):
    """Fail on ``.tolist()`` and ``.numpy()`` of a functorch-wrapped tensor.

    The fixed-point backward differentiates a CTM step with ``torch.func.vjp``,
    whose tensors have no storage: ``.tolist()`` of a CUDA tensor and
    ``.numpy()`` of any tensor fail there. On CPU ``.tolist()`` happens to
    work, so this guard makes such a read fail on every device. ``.item()``
    works under the transform and stays allowed.
    """
    is_wrapped = torch._C._functorch.is_functorch_wrapped_tensor

    def guarded(method):
        def read(self, *args, **kwargs):
            if is_wrapped(self):
                raise AssertionError(
                    f"Tensor.{method.__name__}() of a functorch-wrapped tensor.")
            return method(self, *args, **kwargs)
        return read

    monkeypatch.setattr(torch.Tensor, 'tolist', guarded(torch.Tensor.tolist))
    monkeypatch.setattr(torch.Tensor, 'numpy', guarded(torch.Tensor.numpy))


def _skip_torch_cutensor(config_kwargs):
    if config_kwargs['backend'] == 'torch_cutensor':
        pytest.skip('torch.func.vjp rejects the custom ops of torch_cutensor')


@pytest.mark.parametrize('recycled', [False, True],
                         ids=('fresh_bases', 'recycled_detached_bases'))
def test_si_gradient_under_functorch_vjp_matches_autograd(
        config_kwargs, recycled, forbid_functorch_host_reads):
    """torch.func.vjp of an SI CTM step agrees with ordinary autograd."""
    _skip_torch_cutensor(config_kwargs)
    beta = torch.tensor(0.37, dtype=torch.float64)
    value, vjp = torch.func.vjp(
        lambda b: _ising_nn_objective(
            config_kwargs, b, method='si', recycled=recycled), beta)
    gradient_vjp, = vjp(torch.ones_like(value))

    beta_ad = beta.clone().requires_grad_(True)
    _ising_nn_objective(
        config_kwargs, beta_ad, method='si', recycled=recycled).backward()

    assert np.isclose(gradient_vjp.item(), beta_ad.grad.item(),
                      rtol=1e-10, atol=0)


@pytest.mark.parametrize('refinement', ('per_sector_oversampling',
                                        'adaptive_spectrum',
                                        'sector_dimensions'))
def test_si_refinements_under_functorch_vjp(
        config_kwargs, refinement, forbid_functorch_host_reads):
    """Sector refinement and SI projectors differentiate under torch.func.vjp."""
    _skip_torch_cutensor(config_kwargs)
    from tests.peps.test_ctmrg_si_symmetry_oracle import _biased_z2_corners

    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _biased_z2_corners(config)
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 2, 'niter': 3, 'tol': 0,
               'refinement': refinement}

    def loss(data0, data1):
        # Refinement draws random probes; reseed so that both passes agree.
        config.backend.random_seed(seed=37)
        X, Y = initialize_si_bases(r0, r1, rank=6,
                                   charges={(0,): 3, (1,): 3})
        p0, p1, _, _, _ = si_proj_corners(
            r0._replace(data=data0), r1._replace(data=data1),
            opts_svd, opts_si, X=X, Y=Y, redistribute=True)
        return p0.norm() ** 2 + p1.norm() ** 2

    value, vjp = torch.func.vjp(loss, r0.data, r1.data)
    gradients_vjp = vjp(torch.ones_like(value))

    data = [x.detach().clone().requires_grad_(True) for x in (r0.data, r1.data)]
    gradients_ad = torch.autograd.grad(loss(*data), data)

    for gradient_vjp, gradient_ad in zip(gradients_vjp, gradients_ad):
        assert torch.isfinite(gradient_vjp).all()
        assert torch.allclose(gradient_vjp, gradient_ad, rtol=1e-10, atol=1e-12)
