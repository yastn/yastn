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
from yastn.tn.fpeps._geometry import Site
from yastn.tn.fpeps.envs._env_ctm import (
    initialize_si_bases,
    proj_corners,
    si_bases_compatible,
    si_projector_svd,
    svd_charge_sector_values,
)


def _sector_legs(config, sym):
    """Return unequal external legs and a common contracted leg."""
    if sym == 'none':
        left = yastn.Leg(config, s=1, D=(7,))
        right = yastn.Leg(config, s=-1, D=(9,))
        bond0 = yastn.Leg(config, s=-1, D=(11,))
        bond1 = yastn.Leg(config, s=1, D=(11,))
    elif sym == 'U1':
        charges = (-1, 0, 1)
        left = yastn.Leg(config, s=1, t=charges, D=(2, 3, 2))
        right = yastn.Leg(config, s=-1, t=charges, D=(4, 3, 3))
        bond0 = yastn.Leg(config, s=-1, t=charges, D=(4, 5, 4))
        bond1 = yastn.Leg(config, s=1, t=charges, D=(4, 5, 4))
    else:  # Z2
        charges = (0, 1)
        left = yastn.Leg(config, s=1, t=charges, D=(3, 4))
        right = yastn.Leg(config, s=-1, t=charges, D=(5, 4))
        bond0 = yastn.Leg(config, s=-1, t=charges, D=(6, 5))
        bond1 = yastn.Leg(config, s=1, t=charges, D=(6, 5))
    return left, right, bond0, bond1


def _rectangular_corners(config, sym):
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


def _assert_projectors_equivalent(reference, approximate, tol=2e-8):
    """Compare projector maps modulo rotations of their retained SVD leg."""
    for pref, psi in zip(reference, approximate):
        pref = _matrix_projector(pref)
        psi = _matrix_projector(psi)
        gram_ref = pref @ pref.H
        gram_si = psi @ psi.H
        error = (gram_ref - gram_si).norm() / gram_ref.norm()
        assert error < tol


@pytest.mark.parametrize('sym', ['none', 'U1', 'Z2'])
def test_si_projectors_match_full_svd_rectangular(config_kwargs, sym):
    """SI and full SVD produce the same projector maps on unequal legs."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=10)
    r0, r1 = _rectangular_corners(config, sym)

    # chi + p = 7, the full shared rank on the smaller external space.
    opts_svd = {'D_total': 5, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 2,
               'niter': 8, 'tol': 1e-12}
    full = proj_corners(r0, r1, opts_svd=opts_svd)
    p0, p1, X, Y = proj_corners(
        r0, r1, opts_svd=opts_svd, opts_si=opts_si,
        return_si_state=True)

    assert X.get_shape(axes=1) == 7
    assert Y.get_shape(axes=0) == 7
    assert si_bases_compatible(r0, r1, X, Y)
    _assert_projectors_equivalent(full, (p0, p1))


@pytest.mark.parametrize('sym', ['none', 'U1', 'Z2'])
def test_si_spectrum_matches_full_svd(config_kwargs, sym):
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=11)
    r0, r1 = _rectangular_corners(config, sym)
    opts_svd = {'D_total': 5, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 2, 'niter': 8, 'tol': 1e-12}
    X, Y = initialize_si_bases(r0, r1, rank=7)
    _, s_si, _, _, _, _ = si_projector_svd(
        r0, r1, X, Y, opts_svd, opts_si, return_spectrum=True)

    rr = yastn.tensordot(r0, r1, axes=(1, 1))
    _, s_ref, _ = rr.svd_with_truncation(
        axes=(0, 1), sU=r0.s[1], **opts_svd)
    ref = svd_charge_sector_values(s_ref)
    actual = svd_charge_sector_values(s_si)
    assert ref.keys() == actual.keys()
    for charge in ref:
        assert np.allclose(ref[charge], actual[charge], rtol=2e-8, atol=2e-10)


def test_si_rejects_insufficient_shared_capacity(config_kwargs):
    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _rectangular_corners(config, 'Z2')
    with pytest.raises(yastn.YastnError, match='exceeds total shared capacity'):
        initialize_si_bases(r0, r1, rank=8)


def test_si_reinitializes_recycled_bases_after_leg_change(config_kwargs):
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=12)
    r0, r1 = _rectangular_corners(config, 'none')
    opts_svd = {'D_total': 4, 'tol': 0}
    opts_si = {'enabled': True, 'oversampling': 2, 'niter': 4}
    _, _, X0, Y0 = proj_corners(
        r0, r1, opts_svd, opts_si=opts_si, return_si_state=True)

    # Change the external spaces while leaving the contracted corner leg valid.
    one = yastn.Leg(config, s=1, D=(1,))
    left = yastn.Leg(config, s=1, D=(8,))
    right = yastn.Leg(config, s=-1, D=(10,))
    bond0 = yastn.Leg(config, s=-1, D=(11,))
    bond1 = bond0.conj()
    r0_new = yastn.rand(config, legs=(one, left, bond0, one))
    r1_new = yastn.rand(config, legs=(one.conj(), right, bond1, one.conj()))
    r0_new = r0_new.fuse_legs(axes=((0, 1), (2, 3)))
    r1_new = r1_new.fuse_legs(axes=((0, 1), (2, 3)))

    assert not si_bases_compatible(r0_new, r1_new, X0, Y0)
    _, _, X1, Y1 = proj_corners(
        r0_new, r1_new, opts_svd, opts_si=opts_si, X=X0, Y=Y0,
        return_si_state=True)
    assert si_bases_compatible(r0_new, r1_new, X1, Y1)
    assert X1.get_shape(axes=1) == 6


def _dense_product_env(config):
    leg = yastn.Leg(config, s=1, D=(1,))
    physical = yastn.Leg(config, s=1, D=(1,))
    tensor = yastn.ones(
        config, legs=(leg, leg, leg.conj(), leg.conj(), physical))
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(geometry, tensors={(0, 0): tensor})
    return fpeps.EnvCTM(psi, init='eye')


def test_si_state_copy_clone_detach_to_and_serialization(config_kwargs):
    config = yastn.make_config(sym='none', **config_kwargs)
    env = _dense_product_env(config)
    r0, r1 = _rectangular_corners(config, 'none')
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

    env.detach_()
    assert yastn.allclose(env.X[key], variants[0].X[key])
    assert yastn.allclose(env.Y[key], variants[0].Y[key])


@pytest.mark.parametrize('method', ['1x2 corner', '2x2 corner'])
def test_si_ctm_update_methods(config_kwargs, method):
    config = yastn.make_config(sym='none', **config_kwargs)
    env = _dense_product_env(config)
    env.update_(
        opts_svd={'D_total': 1}, moves='hv', method=method,
        opts_si={'enabled': True, 'oversampling': 0, 'niter': 2})
    assert env.is_consistent()
    assert env.X
    assert env.X.keys() == env.Y.keys() == env._si_age.keys()


@pytest.mark.parametrize('recycle_grad', [False, True])
def test_si_autograd_recycle_policy(config_kwargs, recycle_grad):
    if config_kwargs['backend'] != 'torch':
        pytest.skip('torch backend is required')
    config = yastn.make_config(sym='none', **config_kwargs)
    env = _dense_product_env(config)
    env.psi.ket[(0, 0)].requires_grad_(True)
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
    assert env.psi.ket[(0, 0)].grad() is not None


@pytest.mark.parametrize('checkpoint_move', ['reentrant', 'nonreentrant'])
def test_si_checkpoint_move(config_kwargs, checkpoint_move):
    if config_kwargs['backend'] != 'torch':
        pytest.skip('torch backend is required')
    config = yastn.make_config(sym='none', **config_kwargs)
    env = _dense_product_env(config)
    env.update_(
        opts_svd={'D_total': 1}, moves='h', method='2x2 corner',
        checkpoint_move=checkpoint_move,
        opts_si={'enabled': True, 'oversampling': 0, 'niter': 2})
    assert env.is_consistent()
    assert env.X
