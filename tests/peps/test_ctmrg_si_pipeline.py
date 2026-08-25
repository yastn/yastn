# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Generic projector and end-to-end tests for recycled SI-CTMRG.

The final acceptance test uses physical CTMRG output as the oracle.  Testing
the reduced SVD alone does not detect errors in basis recycling, projector
routing, or environment updates.
"""

import numpy as np
import pytest

import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.fpeps.envs._env_ctm as env_ctm_module
from yastn.tn.fpeps.envs._env_ctm import (
    initialize_si_bases,
    proj_corners,
    si_projector_svd,
    svd_charge_sector_values,
)


def _classical_ising_peps(config, beta=0.5):
    """Nontrivial infinite PEPS for the square-lattice Ising partition sum."""
    leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    vertex = yastn.ones(
        config, legs=(leg, leg, leg.conj(), leg.conj()), n=0)
    spin_vertex = yastn.ones(
        config, legs=(leg, leg, leg.conj(), leg.conj()), n=1)
    bond = yastn.zeros(config, legs=(leg, leg.conj()))
    bond.set_block(ts=(0, 0), val=np.cosh(beta))
    bond.set_block(ts=(1, 1), val=np.sinh(beta))

    site = yastn.ncon(
        (vertex, bond, bond), ((-0, -1, 2, 3), (2, -2), (3, -3)))
    spin = yastn.ncon(
        (spin_vertex, bond, bond), ((-0, -1, 2, 3), (2, -2), (3, -3)))
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    return fpeps.Peps(geometry, tensors={(0, 0): site}), spin


def _normalized_corner_spectra(env):
    spectra = env.calculate_corner_svd()
    return {
        key: {
            charge: np.sort(np.abs(np.asarray(
                value[charge + charge]).reshape(-1)))[::-1]
            for charge in value.get_legs(0).t
        }
        for key, value in spectra.items()
    }


def _dense_matrix_pair_with_spectrum(singular_values, seed):
    """Return dense real matrices whose product has a set spectrum."""
    singular_values = np.asarray(singular_values, dtype=float)
    dimension = singular_values.size
    rng = np.random.default_rng(seed)
    q_left, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    q_right, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    q_shared, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    r0 = q_left @ q_shared.T
    r1 = q_right @ np.diag(singular_values) @ q_shared.T
    return r0, r1


def _dense_corners_with_spectrum(config, singular_values):
    """Build real corners with a fused CTM leg and a prescribed spectrum."""
    singular_values = np.asarray(singular_values, dtype=float)
    dimension = singular_values.size
    matrix_r0, matrix_r1 = _dense_matrix_pair_with_spectrum(
        singular_values, seed=41)
    r0 = yastn.Tensor(config=config, s=(1, 1, -1, 1))
    r1 = yastn.Tensor(config=config, s=(-1, -1, 1, -1))
    r0.set_block(
        Ds=(1, dimension, dimension, 1),
        val=matrix_r0.reshape(1, dimension, dimension, 1))
    r1.set_block(
        Ds=(1, dimension, dimension, 1),
        val=matrix_r1.reshape(1, dimension, dimension, 1))
    return (r0.fuse_legs(axes=((0, 1), (2, 3))),
            r1.fuse_legs(axes=((0, 1), (2, 3))))


def _projector_matrix(projector):
    return projector.fuse_legs(axes=((0, 1), 2)).to_numpy()


def _projector_range_error(reference, approximate, rank_tol=1e-12):
    """Return the largest gauge-invariant projector-range error."""
    errors = []
    for pref, psi in zip(reference, approximate):
        qref, sref, _ = np.linalg.svd(
            _projector_matrix(pref), full_matrices=False)
        qsi, ssi, _ = np.linalg.svd(
            _projector_matrix(psi), full_matrices=False)
        rank_ref = np.count_nonzero(sref > rank_tol * sref[0])
        rank_si = np.count_nonzero(ssi > rank_tol * ssi[0])
        assert rank_ref == rank_si and rank_ref > 0
        overlap = qref[:, :rank_ref].conj().T @ qsi[:, :rank_si]
        error = 1 - np.linalg.norm(overlap, ord='fro') ** 2 / rank_ref
        errors.append(float(np.clip(error, 0., 1.)))
    return max(errors)


# ---------------------------------------------------------------------------
# Dense projector numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('singular_values', [
    (1., 1e-2, 1e-4, 1e-8, 1e-12, 1e-14),
    (1., .5, .1, 0., 0., 0.),
], ids=('ill_conditioned', 'rank_deficient'))
def test_si_projector_identity_and_optimal_residual(config_kwargs,
                                                    singular_values):
    """SI projectors obey Pl Pr=I and attain the optimal rank-chi error."""
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=21)
    r0, r1 = _dense_corners_with_spectrum(config, singular_values)
    chi = 3
    opts_svd = {'D_total': chi, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 2,
               'niter': 12, 'tol': 1e-14}

    # Keep the sampled subspace smaller than the full matrix. Otherwise the
    # test would reduce to an exact SVD and would not exercise SI convergence.
    assert chi + opts_si['oversampling'] < len(singular_values)

    p_left, p_right, X, Y = proj_corners(
        r0, r1, opts_svd, opts_si=opts_si, return_si_state=True)
    pl = _projector_matrix(p_left)
    pr = _projector_matrix(p_right)

    # The left and right projectors are biorthogonal on the retained space.
    identity_residual = np.linalg.norm(pr.T @ pl - np.eye(chi))
    assert identity_residual < 2e-9

    # Reconstruct the rank-chi environment obtained from the recycled SI
    # subspaces and compare it with the best dense rank-chi approximation.
    u, s, v, _, _, s_all = si_projector_svd(
        r0, r1, X, Y, opts_svd, opts_si, return_spectrum=True)
    approximation = (u @ s @ v).to_numpy()
    effective_environment = yastn.tensordot(
        r0, r1, axes=(1, 1)).to_numpy()
    reconstruction_residual = np.linalg.norm(
        effective_environment - approximation)

    u_ref, values_ref, vh_ref = np.linalg.svd(
        effective_environment, full_matrices=False)
    optimal = ((u_ref[:, :chi] * values_ref[:chi])
               @ vh_ref[:chi, :])
    optimal_residual = np.linalg.norm(effective_environment - optimal)

    # Allow only scale-aware floating-point slack above the theoretical
    # optimum. This also covers the rank-deficient case, whose optimum is zero.
    numerical_slack = 1e-11 * np.linalg.norm(effective_environment)
    assert reconstruction_residual <= optimal_residual + numerical_slack

    # By the Eckart-Young theorem, the optimal Frobenius residual equals the
    # Euclidean norm of the singular values discarded beyond chi.
    discarded_weight = np.linalg.norm(values_ref[chi:])
    assert np.isclose(optimal_residual, discarded_weight,
                      rtol=1e-12, atol=1e-14)

    # Check the spectrum itself in addition to the reconstruction error, which
    # alone would not identify incorrectly ordered retained singular values.
    si_values = np.concatenate(tuple(
        np.asarray(values) for values in
        svd_charge_sector_values(s_all).values()))
    assert np.allclose(np.sort(si_values)[::-1][:chi], values_ref[:chi],
                       rtol=2e-9, atol=2e-12)


def test_projectors_remain_finite_when_cutoff_removes_null_space(
        config_kwargs):
    """Pseudo-inverse cutoff must not produce NaN/Inf for a null spectrum."""
    config = yastn.make_config(sym='none', **config_kwargs)
    r0, r1 = _dense_corners_with_spectrum(
        config, (1., 1e-4, 1e-10, 0., 0., 0.))
    projectors = proj_corners(
        r0, r1, opts_svd={'D_total': 6, 'tol': 0, 'fix_signs': True},
        cutoff=1e-8)
    for projector in projectors:
        assert np.isfinite(_projector_matrix(projector)).all()


def test_si_public_path_is_matrix_free_and_uses_reduced_svd(
        config_kwargs, monkeypatch):
    """AI-generated test: SI cannot fall through to full construction."""
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=92)
    dimension = 12
    rank = 5
    r0, r1 = _dense_corners_with_spectrum(
        config, np.geomspace(1., 1e-4, dimension))
    matrix_shapes = []
    svd_shapes = []
    si_calls = 0
    original_tensordot = env_ctm_module.tensordot
    original_svd = yastn.Tensor.svd
    original_si = env_ctm_module.si_projector_svd

    def recording_tensordot(*args, **kwargs):
        result = original_tensordot(*args, **kwargs)
        if result.ndim == 2:
            matrix_shapes.append(tuple(result.get_shape()))
        return result

    def recording_svd(self, *args, **kwargs):
        svd_shapes.append(tuple(self.get_shape()))
        return original_svd(self, *args, **kwargs)

    def recording_si(*args, **kwargs):
        nonlocal si_calls
        si_calls += 1
        return original_si(*args, **kwargs)

    def forbidden_full_svd(*args, **kwargs):
        pytest.fail("SI path called full svd_with_truncation")

    monkeypatch.setattr(env_ctm_module, 'tensordot', recording_tensordot)
    monkeypatch.setattr(yastn.Tensor, 'svd', recording_svd)
    monkeypatch.setattr(env_ctm_module, 'si_projector_svd', recording_si)
    monkeypatch.setattr(
        yastn.Tensor, 'svd_with_truncation', forbidden_full_svd)

    p0, p1, X, Y = proj_corners(
        r0, r1, opts_svd={'D_total': 3, 'tol': 0},
        opts_si={'enabled': True, 'oversampling': 2,
                 'niter': 1, 'tol': 0},
        return_si_state=True)

    assert p0 is not None and p1 is not None
    assert X is not None and Y is not None
    assert si_calls == 1
    assert (dimension, dimension) not in matrix_shapes
    assert svd_shapes and all(max(shape) <= rank for shape in svd_shapes)
    assert max(np.prod(shape) for shape in matrix_shapes) <= dimension * rank


def test_public_si_starts_approximate_then_converges(config_kwargs):
    """AI-generated test: strict SI starts approximate, then converges."""
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=93)
    r0, r1 = _dense_corners_with_spectrum(
        config, (1., .8, .6, .4, .25, .15, .08, .03))
    opts_svd = {'D_total': 2, 'tol': 0, 'fix_signs': True}
    reference = proj_corners(r0, r1, opts_svd)

    initial = proj_corners(
        r0, r1, opts_svd,
        opts_si={'enabled': True, 'oversampling': 1,
                 'niter': 0, 'tol': 0})
    refined = proj_corners(
        r0, r1, opts_svd,
        opts_si={'enabled': True, 'oversampling': 1,
                 'niter': 24, 'tol': 1e-13})

    initial_error = _projector_range_error(reference, initial)
    refined_error = _projector_range_error(reference, refined)
    assert initial_error > 1e-4
    assert refined_error < 1e-8
    assert refined_error < 1e-4 * initial_error


# ---------------------------------------------------------------------------
# Recycling lifecycle, scheduling, fallback, and environment isolation
# ---------------------------------------------------------------------------


def _assert_si_bases_are_orthonormal(env, atol=1e-10):
    for key in env.X:
        x_overlap = (env.X[key].H @ env.X[key]).to_numpy()
        y_overlap = (env.Y[key] @ env.Y[key].H).to_numpy()
        assert np.allclose(x_overlap, np.eye(x_overlap.shape[0]), atol=atol)
        assert np.allclose(y_overlap, np.eye(y_overlap.shape[0]), atol=atol)


def test_si_recycling_state_machine_across_updates(config_kwargs,
                                                   monkeypatch):
    """Check the lifecycle of recycled subspace-iteration (SI) bases.

    A first CTMRG update initializes orthonormal X/Y bases under matching
    site-and-projector-pair keys and gives each state age one.  After the
    eye-initialized environment has grown to the requested SI rank, the next
    update must pass every stored basis back to ``proj_corners``, preserve the
    set of state keys, increment every age exactly once, and leave the returned
    bases orthonormal.  This exercises SI state management, not CTMRG
    convergence or projector accuracy.
    """
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=41)
    psi, _ = _classical_ising_peps(config)
    env = fpeps.EnvCTM(psi, init='eye')
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 1, 'niter': 2,
               'tol': 1e-8, 'warmup': 20}

    env.update_(opts_svd, moves='h', method='2x2 corner', opts_si=opts_si)
    assert env.X.keys() == env.Y.keys() == env._si_age.keys()
    assert env.X
    assert all(env.site2index(site) in env.sites()
               and pair in {'hlb', 'hrb', 'vtr', 'vbr'}
               for site, pair in env.X)
    assert all(age == 1 for age in env._si_age.values())
    _assert_si_bases_are_orthonormal(env)

    # Eye initialization grows the CTM progressively.  Bases from this phase
    # are intentionally invalidated as their external legs enlarge.  Wait
    # until chi + p is available before checking object-level recycling.
    target_rank = opts_svd['D_total'] + opts_si['oversampling']
    for _ in range(5):
        if all(x.get_shape(axes=1) == target_rank for x in env.X.values()):
            break
        env.update_(opts_svd, moves='h', method='2x2 corner',
                    opts_si=opts_si)
    assert all(x.get_shape(axes=1) == target_rank for x in env.X.values())

    recycled_ids = {id(x) for x in env.X.values()} | {
        id(y) for y in env.Y.values()}
    consumed_ids = set()
    original = env_ctm_module.proj_corners

    def recording_proj_corners(*args, **kwargs):
        X = kwargs.get('X')
        Y = kwargs.get('Y')
        if X is not None and Y is not None:
            consumed_ids.update((id(X), id(Y)))
        return original(*args, **kwargs)

    monkeypatch.setattr(env_ctm_module, 'proj_corners',
                        recording_proj_corners)
    ages_before = dict(env._si_age)
    env.update_(opts_svd, moves='h', method='2x2 corner', opts_si=opts_si)

    assert recycled_ids <= consumed_ids
    assert env._si_age.keys() == ages_before.keys()
    assert all(env._si_age[key] == ages_before[key] + 1
               for key in ages_before)
    _assert_si_bases_are_orthonormal(env)


def test_si_warmup_and_periodic_correction_schedule(config_kwargs,
                                                    monkeypatch):
    """Correction runs at warmup age and then at the requested frequency."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=42)
    psi, _ = _classical_ising_peps(config)
    env = fpeps.EnvCTM(psi, init='eye')
    opts_svd = {'D_total': 2, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 0, 'niter': 2,
               'warmup': 2, 'correction_frequency': 2}
    correction_ages = []
    original = env_ctm_module.si_refinement

    def recording_correction(*args, **kwargs):
        recycled_x = args[2]
        key = next(key for key, value in env.X.items()
                   if value is recycled_x)
        correction_ages.append(env._si_age[key])
        return original(*args, **kwargs)

    monkeypatch.setattr(env_ctm_module, 'si_refinement',
                        recording_correction)
    for _ in range(5):
        env.update_(opts_svd, moves='h', method='2x2 corner', opts_si=opts_si)

    assert set(correction_ages) == {2, 4}


def test_si_disabled_path_matches_full_svd(config_kwargs):
    """Explicitly disabling SI selects the unchanged full-SVD path."""
    config = yastn.make_config(sym='none', **config_kwargs)
    r0, r1 = _dense_corners_with_spectrum(
        config, (1., .5, .1, 1e-3, 0., 0.))
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    reference = proj_corners(r0, r1, opts_svd)
    disabled = proj_corners(
        r0, r1, opts_svd, opts_si={'enabled': False})
    for actual, expected in zip(disabled, reference):
        assert yastn.allclose(actual, expected)


def test_new_environment_starts_without_si_recycling_state(config_kwargs):
    """SI dictionaries are not implicitly shared with another environment."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    one_site_psi, _ = _classical_ising_peps(config)
    psi_tensor = one_site_psi[(0, 0)]
    geometry = fpeps.SquareLattice(dims=(2, 1), boundary='infinite')
    two_site_psi = fpeps.Peps(
        geometry, tensors={(0, 0): psi_tensor, (1, 0): psi_tensor})
    other_env = fpeps.EnvCTM(two_site_psi, init='eye')
    assert not other_env.X and not other_env.Y and not other_env._si_age


# ---------------------------------------------------------------------------
# End-to-end CTMRG convergence and physical observables
# ---------------------------------------------------------------------------


def test_si_ctmrg_matches_full_svd_on_ising_peps(config_kwargs):
    """SI and full-SVD CTMRG must give the same fixed-point physics.

    This covers a complete sequence of random SI initialization, power/QR
    updates, small SVD, gauge rotation, recycling, projector application, and
    convergence of the environment.  The Ising PEPS is the same nontrivial
    analytic network used by the standard CTMRG acceptance test.
    """
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=2026)
    psi, spin = _classical_ising_peps(config)

    chi = 12
    opts_svd = {'D_total': chi, 'tol': 0, 'fix_signs': True}
    # SI uses a finite subspace tolerance and therefore approaches the fixed
    # point with small stochastic fluctuations.  A 1e-9 corner-spectrum gate
    # is already substantially tighter than the observable checks below.
    common = dict(opts_svd=opts_svd, max_sweeps=100, corner_tol=1e-9,
                  method='2x2 corner')

    env_full = fpeps.EnvCTM(psi, init='eye')
    info_full = env_full.ctmrg_(**common)

    env_si = fpeps.EnvCTM(psi, init='eye')
    info_si = env_si.ctmrg_(
        **common,
        opts_si={'enabled': True, 'oversampling': 4, 'niter': 1,
                 'tol': 1e-3, 'warmup': 5})

    assert info_full.converged
    assert info_si.converged
    assert env_si.X.keys() == env_si.Y.keys() == env_si._si_age.keys()
    assert env_si.X
    assert min(env_si._si_age.values()) >= 5
    assert all(x.get_shape(axes=1) == chi + 4 for x in env_si.X.values())

    # Gauge-independent fixed-point data.
    spectra_full = _normalized_corner_spectra(env_full)
    spectra_si = _normalized_corner_spectra(env_si)
    assert spectra_full.keys() == spectra_si.keys()
    for key in spectra_full:
        assert spectra_full[key].keys() == spectra_si[key].keys()
        for charge in spectra_full[key]:
            assert np.allclose(
                spectra_si[key][charge], spectra_full[key][charge],
                rtol=2e-5, atol=2e-8)

    one_full = env_full.measure_1site(spin)[(0, 0)]
    one_si = env_si.measure_1site(spin)[(0, 0)]
    nn_full = env_full.measure_nn(spin, spin)
    nn_si = env_si.measure_nn(spin, spin)
    assert abs(one_si - one_full) < 2e-8
    assert nn_full.keys() == nn_si.keys()
    for bond in nn_full:
        assert abs(nn_si[bond] - nn_full[bond]) < 2e-6

    # At beta=0.5 the exact nearest-neighbour correlator is 0.872783.
    assert abs(nn_si[((0, 0), (0, 1))] - 0.872783) < 2e-5
    assert abs(nn_si[((0, 0), (1, 0))] - 0.872783) < 2e-5
