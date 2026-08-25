# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Regression tests for recycled subspace-iteration CTM projectors."""

import numpy as np
import pytest

import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.fpeps.envs._env_ctm as env_ctm_module
from yastn.tn.fpeps._geometry import Site
from yastn.tn.fpeps.envs._env_ctm import (
    initialize_si_bases,
    proj_corners,
    si_bases_compatible,
    si_projector_svd,
    svd_charge_sector_values,
)


def _sector_legs(config, sym):
    """Return matching external legs and a common contracted leg."""
    if sym == 'none':
        left = yastn.Leg(config, s=1, D=(7,))
        bond0 = yastn.Leg(config, s=-1, D=(11,))
        bond1 = yastn.Leg(config, s=1, D=(11,))
    elif sym == 'U1':
        charges = (-1, 0, 1)
        left = yastn.Leg(config, s=1, t=charges, D=(2, 3, 2))
        bond0 = yastn.Leg(config, s=-1, t=charges, D=(4, 5, 4))
        bond1 = yastn.Leg(config, s=1, t=charges, D=(4, 5, 4))
    else:  # Z2
        charges = (0, 1)
        left = yastn.Leg(config, s=1, t=charges, D=(3, 4))
        bond0 = yastn.Leg(config, s=-1, t=charges, D=(6, 5))
        bond1 = yastn.Leg(config, s=1, t=charges, D=(6, 5))
    right = left.conj()
    return left, right, bond0, bond1


def _ctm_corner_pair(config, sym):
    """Make rank-2 corners whose two legs retain unfusion histories."""
    left, right, bond0, bond1 = _sector_legs(config, sym)
    trivial_p = yastn.Leg(config, s=1, t=(0,), D=(1,)) if sym != 'none' \
        else yastn.Leg(config, s=1, D=(1,))
    trivial_m = trivial_p.conj()

    r0 = yastn.rand(config, legs=(trivial_p, left, bond0, trivial_p))
    r1 = yastn.rand(config, legs=(trivial_m, right, bond1, trivial_m))
    r0 = r0.fuse_legs(axes=((0, 1), (2, 3)))
    r1 = r1.fuse_legs(axes=((0, 1), (2, 3)))
    return r0, r1


def _matrix_projector(projector):
    return projector.fuse_legs(axes=((0, 1), 2))


def _projector_subspace_error(reference, approximate, rank_tol=1e-12):
    """Mean squared sine of the principal angles between projector ranges."""
    pref = _matrix_projector(reference).to_numpy()
    psi = _matrix_projector(approximate).to_numpy()

    uref, sref, _ = np.linalg.svd(pref, full_matrices=False)
    usi, ssi, _ = np.linalg.svd(psi, full_matrices=False)
    scale_ref = sref[0] if sref.size else 0
    scale_si = ssi[0] if ssi.size else 0
    rank_ref = np.count_nonzero(sref > rank_tol * scale_ref)
    rank_si = np.count_nonzero(ssi > rank_tol * scale_si)
    assert rank_ref == rank_si and rank_ref > 0
    print("rank_ref, rank_si", rank_ref, rank_si)
    qref = uref[:, :rank_ref]
    qsi = usi[:, :rank_si]
    overlap = qref.conj().T @ qsi
    error = 1.0 - np.linalg.norm(overlap, ord='fro') ** 2 / rank_ref
    return float(np.clip(error, 0.0, 1.0))


def _assert_projectors_equivalent(reference, approximate, tol=2e-8):
    """Compare projector ranges using their gauge-invariant principal angles."""
    for index, (pref, psi) in enumerate(zip(reference, approximate)):
        pref_matrix = _matrix_projector(pref)
        psi_matrix = _matrix_projector(psi)
        ref_sectors = pref_matrix.get_legs(1).tD
        si_sectors = psi_matrix.get_legs(1).tD
        error = _projector_subspace_error(pref, psi)
        print(f"Projector {index}: full-SVD sectors={ref_sectors}, "
              f"SI sectors={si_sectors}")
        print(f"Projector {index}: mean squared sine of principal angles="
              f"{error:.3e}")
        assert ref_sectors == si_sectors
        assert error < tol


@pytest.mark.parametrize('sym', ['none', 'U1', 'Z2'])
def test_si_projectors_match_full_svd(config_kwargs, sym):
    """SI and full SVD produce the same projector maps on CTM corner halves."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=10)
    r0, r1 = _ctm_corner_pair(config, sym)
    # chi + p = 6, strictly below the corner-leg rank of 7.
    opts_svd = {'D_total': 5, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 1,
               'niter': 12, 'tol': 1e-5}
    full = proj_corners(r0, r1, opts_svd=opts_svd)
    p0, p1, X, Y = proj_corners(
        r0, r1, opts_svd=opts_svd, opts_si=opts_si,
        return_si_state=True)
    print("X, Y",X.shape, Y.shape)
    assert X.get_shape(axes=1) == 6
    assert Y.get_shape(axes=0) == 6
    assert X.get_shape(axes=1) < min(r0.get_shape(axes=0),
                                     r1.get_shape(axes=0))
    assert si_bases_compatible(r0, r1, X, Y)
    _assert_projectors_equivalent(full, (p0, p1))


@pytest.mark.parametrize('sym', ['none', 'U1', 'Z2'])
def test_si_spectrum_matches_full_svd(config_kwargs, sym):
    """SI iterations recover the leading spectrum from a strict subspace."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=11)
    r0, r1 = _ctm_corner_pair(config, sym)
    opts_svd = {'D_total': 5, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 24, 'tol': 1e-12}
    X, Y = initialize_si_bases(r0, r1, rank=6)
    assert X.get_shape(axes=1) < min(r0.get_shape(axes=0),
                                     r1.get_shape(axes=0))

    # Guard against accidentally testing a full-rank change of basis: the
    # unrefined random sketch must not already reproduce the reference.
    _, s_initial, _, _, _, _ = si_projector_svd(
        r0, r1, X, Y, opts_svd, {**opts_si, 'niter': 0},
        return_spectrum=True)
    _, s_si, _, _, _, _ = si_projector_svd(
        r0, r1, X, Y, opts_svd, opts_si, return_spectrum=True)

    rr = yastn.tensordot(r0, r1, axes=(1, 1))
    _, s_ref, _ = rr.svd_with_truncation(
        axes=(0, 1), sU=r0.s[1], **opts_svd)
    ref = svd_charge_sector_values(s_ref)
    initial = svd_charge_sector_values(s_initial)
    actual = svd_charge_sector_values(s_si)
    print("ref, initial, actual", ref, initial, actual)

    assert ref.keys() == actual.keys()
    initial_error = 0.0
    final_error = 0.0
    for charge in ref:
        initial_error += np.linalg.norm(
            np.asarray(ref[charge]) - np.asarray(initial[charge])) ** 2
        final_error += np.linalg.norm(
            np.asarray(ref[charge]) - np.asarray(actual[charge])) ** 2
        assert np.allclose(ref[charge], actual[charge], rtol=2e-8, atol=2e-10)
    initial_error = np.sqrt(initial_error)
    final_error = np.sqrt(final_error)
    assert initial_error > 1e-6
    assert final_error < 1e-4 * initial_error


def test_si_rejects_insufficient_corner_capacity(config_kwargs):
    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _ctm_corner_pair(config, 'Z2')
    with pytest.raises(yastn.YastnError, match='exceeds CTM corner-leg capacity'):
        initialize_si_bases(r0, r1, rank=8)


def test_si_rejects_mismatched_ctm_corner_halves(config_kwargs):
    """Both closures of the CTM corner loop must match sector by sector."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    r0, r1 = _ctm_corner_pair(config, 'U1')
    bad_external = yastn.Leg(
        config, s=-1, t=(-1, 0, 1), D=(3, 2, 2))
    bad_contracted = yastn.Leg(
        config, s=1, t=(-1, 0, 1), D=(5, 4, 4))
    malformed_pairs = (
        (r0, yastn.rand(config, legs=(bad_external, r1.get_legs(1)))),
        (r0, yastn.rand(config, legs=(r1.get_legs(0), bad_contracted))),
    )
    message = 'matching dimensions in every shared charge sector'
    for r0_bad, r1_bad in malformed_pairs:
        with pytest.raises(yastn.YastnError, match=message):
            initialize_si_bases(r0_bad, r1_bad, rank=3)
        with pytest.raises(yastn.YastnError, match=message):
            proj_corners(r0_bad, r1_bad, opts_svd={'D_total': 3})


def test_si_reinitializes_recycled_bases_after_leg_change(config_kwargs,
                                                          monkeypatch):
    """
    TODO create an actual recycle functionality
    check if X0,X1 are compatible with r0,r1
    """
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=12)
    r0, r1 = _ctm_corner_pair(config, 'none')
    opts_svd = {'D_total': 4, 'tol': 0}
    opts_si = {'enabled': True, 'oversampling': 2, 'niter': 4}
    _, _, X0, Y0 = proj_corners(
        r0, r1, opts_svd, opts_si=opts_si, return_si_state=True)

    # Change the external spaces while leaving the contracted corner leg valid.
    one = yastn.Leg(config, s=1, D=(1,))
    left = yastn.Leg(config, s=1, D=(8,))
    right = left.conj()
    bond0 = yastn.Leg(config, s=-1, D=(11,))
    bond1 = bond0.conj()
    r0_new = yastn.rand(config, legs=(one, left, bond0, one))
    r1_new = yastn.rand(config, legs=(one.conj(), right, bond1, one.conj()))
    r0_new = r0_new.fuse_legs(axes=((0, 1), (2, 3)))
    r1_new = r1_new.fuse_legs(axes=((0, 1), (2, 3)))

    assert not si_bases_compatible(r0_new, r1_new, X0, Y0)

    initialize_calls = 0
    original_initialize = env_ctm_module.initialize_si_bases

    def counting_initialize(*args, **kwargs):
        nonlocal initialize_calls
        initialize_calls += 1
        return original_initialize(*args, **kwargs)

    monkeypatch.setattr(env_ctm_module, 'initialize_si_bases',
                        counting_initialize)
    _, _, X1, Y1 = proj_corners(
        r0_new, r1_new, opts_svd, opts_si=opts_si, X=X0, Y=Y0,
        return_si_state=True)
    assert si_bases_compatible(r0_new, r1_new, X1, Y1)
    assert X1.get_shape(axes=1) == 6
    assert initialize_calls == 1


def test_si_rebuilds_recycled_basis_after_fusion_history_change(
        config_kwargs, monkeypatch):
    """An incompatible fused recycled basis is rebuilt transparently."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _ctm_corner_pair(config, 'Z2')
    X, Y = initialize_si_bases(r0, r1, rank=4)

    external = r1.get_legs(0).drop_history()
    contracted = r1.get_legs(1)
    r1_with_new_history = yastn.rand(config, legs=(external, contracted))

    assert external.tD == r1.get_legs(0).tD
    assert external.hf != r1.get_legs(0).hf
    # Aggregate compatibility cannot see the changed hard-fusion layout.  The
    # first contraction must fail and exercise proj_corners' retry path.
    assert si_bases_compatible(r0, r1_with_new_history, X, Y)
    initialize_calls = 0
    original_initialize = env_ctm_module.initialize_si_bases

    def counting_initialize(*args, **kwargs):
        nonlocal initialize_calls
        initialize_calls += 1
        return original_initialize(*args, **kwargs)

    monkeypatch.setattr(env_ctm_module, 'initialize_si_bases',
                        counting_initialize)
    opts_svd = {'D_total': 4, 'tol': 0}
    opts_si = {'enabled': True, 'oversampling': 0, 'niter': 2}
    p0, p1, X_new, Y_new = proj_corners(
        r0, r1_with_new_history, opts_svd, opts_si=opts_si, X=X, Y=Y,
        return_si_state=True)

    assert p0 is not None and p1 is not None
    assert si_bases_compatible(r0, r1_with_new_history, X_new, Y_new)
    assert initialize_calls == 1


def _dense_product_env(config):
    """Seeded nontrivial dense PEPS used by the CTM integration tests."""
    leg = yastn.Leg(config, s=1, D=(2,))
    physical = yastn.Leg(config, s=1, D=(2,))
    tensor = yastn.zeros(
        config, legs=(leg, leg, leg.conj(), leg.conj(), physical))
    values = np.sin(np.arange(1, 33, dtype=float)).reshape((2,) * 5)
    tensor.set_block(val=values)
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(geometry, tensors={(0, 0): tensor})
    return fpeps.EnvCTM(psi, init='eye')


def test_si_state_copy_clone_detach_to_and_serialization(config_kwargs):
    """
    checks if SI isometries actually stored in environment
    """
    config = yastn.make_config(sym='none', **config_kwargs)
    env = _dense_product_env(config)
    r0, r1 = _ctm_corner_pair(config, 'none')
    key = (Site(0, 0), 'vtr')
    env.X[key], env.Y[key] = initialize_si_bases(r0, r1, rank=3)
    env._si_age[key] = 4

    variants = (
        env.copy(), env.clone(), env.detach(),
        env.to(dtype=config.default_dtype),
        fpeps.EnvCTM.from_dict(env.to_dict()),
    )
    for other in variants:
        assert other._si_age == env._si_age
        assert yastn.allclose(other.X[key], env.X[key])
        assert yastn.allclose(other.Y[key], env.Y[key])
        assert other.X is not env.X and other.Y is not env.Y

    # Copy, clone, and deserialization promise independent tensor objects.
    for other in (variants[0], variants[1], variants[4]):
        assert other.X[key] is not env.X[key]
        assert other.Y[key] is not env.Y[key]

    # Replacing state in the source must not mutate any copied mapping.
    old_x = variants[0].X[key]
    env.X[key] = 2 * env.X[key]
    assert yastn.allclose(variants[0].X[key], old_x)
    assert not yastn.allclose(env.X[key], variants[0].X[key])

    env.detach_()
    assert yastn.allclose(env.X[key], 2 * variants[0].X[key])
    assert yastn.allclose(env.Y[key], variants[0].Y[key])


@pytest.mark.parametrize('method', ['1x2 corner', '2x2 corner'])
def test_si_ctm_update_methods(config_kwargs, method):
    """
    Checks if projectors can be computed for both environments
    """
    config = yastn.make_config(sym='none', **config_kwargs)
    env = _dense_product_env(config)
    # Grow the eye environment first; 1x2 updates cannot increase chi and
    # would otherwise exercise only rank-one corner spaces.
    env.update_(opts_svd={'D_total': 2}, moves='hv', method='2x2 corner')
    assert env.effective_chi() == 2
    env.update_(
        opts_svd={'D_total': 2}, moves='hv', method=method,
        opts_si={'enabled': True, 'oversampling': 1, 'niter': 3})
    assert env.is_consistent()
    assert env.X
    assert env.X.keys() == env.Y.keys() == env._si_age.keys()
    assert all(age == 1 for age in env._si_age.values())
    assert env.effective_chi() == 2
    assert any(x.get_shape(axes=1) > 1 for x in env.X.values())


@pytest.mark.parametrize('recycle_grad', [False, True])
def test_si_autograd_recycle_policy(config_kwargs, recycle_grad):
    """
    Checks if grad can travel through the environment
    """
    if config_kwargs['backend'] != 'torch':
        pytest.skip('torch backend is required')
    config = yastn.make_config(sym='none', **config_kwargs)
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


@pytest.mark.parametrize('checkpoint_move', ['reentrant', 'nonreentrant'])
def test_si_checkpoint_move(config_kwargs, checkpoint_move, monkeypatch):
    """
    checks if reentrant works
    """
    if config_kwargs['backend'] != 'torch':
        pytest.skip('torch backend is required')
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

if __name__ == '__main__':
    config_kwargs = {
        'backend': 'np',
        'default_device': 'cpu',
        'default_fusion': 'hard',
        'tensordot_policy': 'fuse_to_matrix',
    }
    for sym in ('none', 'U1', 'Z2'):
        test_si_spectrum_matches_full_svd(config_kwargs, sym)
        print(f'PASSED: test_si_spectrum_matches_full_svd[{sym}]')
