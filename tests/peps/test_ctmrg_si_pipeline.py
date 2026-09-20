# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Generic projector and end-to-end tests for recycled SI-CTMRG.

The final acceptance test uses physical CTMRG output as the oracle.  Testing
the reduced SVD alone does not detect errors in basis recycling, projector
routing, or environment updates.
"""

import json
import os

import numpy as np
import pytest

import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.fpeps.envs._env_ctm as env_ctm_module
import yastn.tn.fpeps.envs._env_ctm_SI_projectors as si_module
from yastn.tn.fpeps._geometry import Site
from yastn.tn.fpeps.envs._env_ctm import proj_corners
from yastn.tn.fpeps.envs._env_ctm_SI_projectors import (
    si_proj_corners,
    si_projector_svd,
    svd_charge_sector_values,
)
from yastn.tn.fpeps.envs.rdm import rdm1x1


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


def _load_peps_ad(config, filename):
    """PEPS stored under ``inputs/`` in the PepsAD JSON layout of peps-torch.

    ``config`` has to match the stored ``sym`` and ``fermionic``, which
    ``from_dict`` validates.  Complex entries are stored as
    ``{"real": ..., "imag": ...}``, and a float64 config would drop their
    imaginary parts, so ``default_dtype`` has to be set explicitly as well.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', filename)

    def complex_decoder(dct):
        if 'real' in dct and 'imag' in dct:
            return complex(dct['real'], dct['imag'])
        return dct

    with open(path) as f:
        d = json.load(f, object_hook=complex_decoder)

    geometry = fpeps.RectangularUnitcell(**d['geometry'])
    tensors = {tuple(d['parameters_key_to_id'][coord]):
               yastn.from_dict(tensor, config=config)
               for coord, tensor in d['parameters'].items()}
    return fpeps.Peps(geometry, tensors=tensors)


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


def _dense_matrix_pair_with_spectrum(singular_values, seed, dtype='float64'):
    """Return dense real or complex matrices whose product has a set spectrum."""
    singular_values = np.asarray(singular_values, dtype=float)
    dimension = singular_values.size
    rng = np.random.default_rng(seed)

    def gaussian():
        sample = rng.standard_normal((dimension, dimension))
        if dtype == 'complex128':
            sample = sample + 1j * rng.standard_normal((dimension, dimension))
        return sample

    q_left, _ = np.linalg.qr(gaussian())
    q_right, _ = np.linalg.qr(gaussian())
    q_shared, _ = np.linalg.qr(gaussian())
    # r0 @ r1.T = q_left @ diag(s) @ q_right.T requires q_shared.T @ conj(q_shared) = I.
    r0 = q_left @ q_shared.T
    r1 = q_right @ np.diag(singular_values) @ q_shared.conj().T
    return r0, r1


def _dense_corners_with_spectrum(config, singular_values):
    """Build corners of ``config.default_dtype`` with a fused CTM leg and a prescribed spectrum."""
    singular_values = np.asarray(singular_values, dtype=float)
    dimension = singular_values.size
    matrix_r0, matrix_r1 = _dense_matrix_pair_with_spectrum(
        singular_values, seed=41, dtype=config.default_dtype)
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


def _record_si_updates(monkeypatch):
    """Record the number of power updates made by every ``_si_reduced_svd`` call.

    Each update orthonormalizes both bases, i.e., makes two QR decompositions.
    """
    updates = []
    qr_calls = 0
    original_qr = si_module.qr
    original_reduced_svd = si_module._si_reduced_svd

    def counting_qr(*args, **kwargs):
        nonlocal qr_calls
        qr_calls += 1
        return original_qr(*args, **kwargs)

    def recording_reduced_svd(*args, **kwargs):
        start = qr_calls
        result = original_reduced_svd(*args, **kwargs)
        updates.append((qr_calls - start) // 2)
        return result

    monkeypatch.setattr(si_module, 'qr', counting_qr)
    monkeypatch.setattr(si_module, '_si_reduced_svd', recording_reduced_svd)
    return updates


# ---------------------------------------------------------------------------
# Dense projector numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('singular_values', [
    (1., 1e-2, 1e-4, 1e-8, 1e-12, 1e-14),
    (1., .5, .1, 0., 0., 0.),
], ids=('ill_conditioned', 'rank_deficient'))
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_si_projector_identity_and_optimal_residual(config_kwargs,
                                                    singular_values, dtype):
    """SI projectors obey Pl Pr=I and attain the optimal rank-chi error."""
    config = yastn.make_config(sym='none', default_dtype=dtype, **config_kwargs)
    config.backend.random_seed(seed=21)
    r0, r1 = _dense_corners_with_spectrum(config, singular_values)
    chi = 3
    opts_svd = {'D_total': chi, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 2,
               'niter': 12, 'tol': 1e-14}

    # Keep the sampled subspace smaller than the full matrix. Otherwise the
    # test would reduce to an exact SVD and would not exercise SI convergence.
    assert chi + opts_si['oversampling'] < len(singular_values)

    p_left, p_right, X, Y, info = si_proj_corners(r0, r1, opts_svd, opts_si)
    pl = _projector_matrix(p_left)
    pr = _projector_matrix(p_right)

    # The left and right projectors are biorthogonal on the retained space.
    identity_residual = np.linalg.norm(pr.T @ pl - np.eye(chi))
    assert identity_residual < 2e-9

    # SI reports how far it got.  It does not have to reach ``tol`` here: the
    # oversampled bases carry surplus columns of roundoff whose directions keep
    # the reported error above it.
    assert 1 <= info['niter'] <= opts_si['niter']
    assert 0. <= info['error'] <= 1.

    # Reconstruct the rank-chi environment obtained from the recycled SI
    # subspaces and compare it with the best dense rank-chi approximation.
    # The spectrum is requested separately: in spectrum mode the projectors are
    # not built, so the two modes cannot be had from a single call.
    u, s, v, _, _, _ = si_projector_svd(r0, r1, X, Y, opts_svd, opts_si)
    _, s_all, _, _, _, _ = si_projector_svd(
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
    original_tensordot = si_module.tensordot
    original_svd = yastn.Tensor.svd
    original_si = si_module.si_projector_svd

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
    monkeypatch.setattr(si_module, 'tensordot', recording_tensordot)
    monkeypatch.setattr(yastn.Tensor, 'svd', recording_svd)
    monkeypatch.setattr(si_module, 'si_projector_svd', recording_si)
    monkeypatch.setattr(
        yastn.Tensor, 'svd_with_truncation', forbidden_full_svd)

    p0, p1, X, Y, _ = si_proj_corners(
        r0, r1, {'D_total': 3, 'tol': 0},
        {'enabled': True, 'oversampling': 2, 'niter': 1, 'tol': 0})

    assert p0 is not None and p1 is not None
    assert X is not None and Y is not None
    assert si_calls == 1
    assert (dimension, dimension) not in matrix_shapes
    assert svd_shapes and all(max(shape) <= rank for shape in svd_shapes)
    assert max(np.prod(shape) for shape in matrix_shapes) <= dimension * rank


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_public_si_starts_approximate_then_converges(config_kwargs, dtype):
    """AI-generated test: strict SI starts approximate, then converges."""
    config = yastn.make_config(sym='none', default_dtype=dtype, **config_kwargs)
    config.backend.random_seed(seed=93)
    r0, r1 = _dense_corners_with_spectrum(
        config, (1., .8, .6, .4, .25, .15, .08, .03))
    opts_svd = {'D_total': 2, 'tol': 0, 'fix_signs': True}
    reference = proj_corners(r0, r1, opts_svd)

    initial = si_proj_corners(
        r0, r1, opts_svd,
        {'enabled': True, 'oversampling': 1, 'niter': 0, 'tol': 0})[:2]
    refined = si_proj_corners(
        r0, r1, opts_svd,
        {'enabled': True, 'oversampling': 1, 'niter': 24, 'tol': 1e-13})[:2]

    initial_error = _projector_range_error(reference, initial)
    refined_error = _projector_range_error(reference, refined)
    assert initial_error > 1e-4
    assert refined_error < 1e-8
    assert refined_error < 1e-4 * initial_error


def test_si_reports_an_unconverged_budget_of_zero_updates(config_kwargs):
    """``niter=0`` makes no update, and has to report that instead of failing.

    The bases are then used exactly as they come in, so there is no pair of
    successive subspaces to compare and no error to report.
    """
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=93)
    r0, r1 = _dense_corners_with_spectrum(config, (1., .8, .6, .4, .25, .15))
    opts_svd = {'D_total': 2, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 1, 'niter': 0, 'tol': 0}

    *_, info = si_proj_corners(r0, r1, opts_svd, opts_si)
    assert info == {'niter': 0, 'error': float('inf')}

    # Every entry point has to survive an empty budget, spectrum mode included.
    X, Y = si_module.initialize_si_bases(r0, r1, 3)
    assert si_projector_svd(r0, r1, X, Y, opts_svd, opts_si)[-1] == info
    assert si_projector_svd(r0, r1, X, Y, opts_svd, opts_si,
                            return_spectrum=True)[-1] == info


def test_si_spectrum_mode_returns_the_spectrum_alone(config_kwargs):
    """``return_spectrum`` builds no projectors and nothing recyclable."""
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=93)
    r0, r1 = _dense_corners_with_spectrum(config, (1., .8, .6, .4, .25, .15))
    opts_svd = {'D_total': 2, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 1, 'niter': 4, 'tol': 1e-13}
    X, Y = si_module.initialize_si_bases(r0, r1, 3)

    u, s, v, X_new, Y_new, info = si_projector_svd(
        r0, r1, X, Y, opts_svd, opts_si, return_spectrum=True)

    assert (u, v, X_new, Y_new) == (None, None, None, None)
    assert info.keys() == {'niter', 'error'}
    # The spectrum is the untruncated one of the reduced rho, so it keeps every
    # sampled direction rather than the D_total the mask would retain.
    assert s.get_shape(axes=0) == 3


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_si_convergence_ignores_roundoff_directions(config_kwargs, monkeypatch, dtype):
    """SI stops once the directions of a rank-deficient product have converged.

    The oversampled bases exceed the rank of ``r0 @ r1.T``. Their surplus
    columns carry only roundoff, whose directions change at random between
    iterations, so they must not keep SI running until ``niter``.
    """
    config = yastn.make_config(sym='none', default_dtype=dtype, **config_kwargs)
    config.backend.random_seed(seed=94)
    values = (1., .5, .25, .1) + (0.,) * 8
    r0, r1 = _dense_corners_with_spectrum(config, values)
    X, Y = si_module.initialize_si_bases(r0, r1, 6)
    updates = _record_si_updates(monkeypatch)

    s = si_module._si_reduced_svd(
        si_module._Half(r0), si_module._Half(r1), X, Y,
        {'niter': 30, 'tol': 1e-10})[3]

    # One update captures the range of the rank-4 product, a second confirms it.
    assert updates == [2]
    si_values = np.sort(np.diag(s.to_numpy()))[::-1]
    assert np.allclose(si_values[:4], values[:4], rtol=1e-10)
    assert np.all(si_values[4:] < 1e-12)


# ---------------------------------------------------------------------------
# Recycling lifecycle, scheduling, fallback, and environment isolation
# ---------------------------------------------------------------------------


def _si_bases(env, container):
    """Assigned recycled bases of ``env.si_X``/``env.si_Y``, keyed like ``env._si_age``."""
    return {(env.site2index(site), name): getattr(projectors, name)
            for site, projectors in container.items()
            for name in projectors.fields()
            if getattr(projectors, name) is not None}


def _assert_si_bases_are_orthonormal(env, atol=1e-10):
    bases_x, bases_y = _si_bases(env, env.si_X), _si_bases(env, env.si_Y)
    for key in bases_x:
        x_overlap = (bases_x[key].H @ bases_x[key]).to_numpy()
        y_overlap = (bases_y[key] @ bases_y[key].H).to_numpy()
        assert np.allclose(x_overlap, np.eye(x_overlap.shape[0]), atol=atol)
        assert np.allclose(y_overlap, np.eye(y_overlap.shape[0]), atol=atol)


def test_si_recycling_state_machine_across_updates(config_kwargs,
                                                   monkeypatch):
    """Check the lifecycle of recycled subspace-iteration (SI) bases.

    A first CTMRG update initializes orthonormal X/Y bases under matching
    site-and-projector-pair keys and gives each state age one.  After the
    eye-initialized environment has grown to the requested SI rank, the next
    update must pass every stored basis back to ``si_proj_corners``, preserve the
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
    bases_x = _si_bases(env, env.si_X)
    assert bases_x.keys() == _si_bases(env, env.si_Y).keys() == env._si_age.keys()
    assert bases_x
    assert all(pair in {'hlb', 'hrb', 'vtr', 'vbr'} for _, pair in bases_x)
    assert all(si_state.age == 1 for si_state in env._si_age.values())
    _assert_si_bases_are_orthonormal(env)

    # Eye initialization grows the CTM progressively.  Bases from this phase
    # are intentionally invalidated as their external legs enlarge.  Wait
    # until chi + p is available before checking object-level recycling.
    target_rank = opts_svd['D_total'] + opts_si['oversampling']
    for _ in range(5):
        if all(x.get_shape(axes=1) == target_rank
               for x in _si_bases(env, env.si_X).values()):
            break
        env.update_(opts_svd, moves='h', method='2x2 corner',
                    opts_si=opts_si)
    assert all(x.get_shape(axes=1) == target_rank
               for x in _si_bases(env, env.si_X).values())

    recycled_ids = {id(x) for x in _si_bases(env, env.si_X).values()} | {
        id(y) for y in _si_bases(env, env.si_Y).values()}
    consumed_ids = set()
    original = env_ctm_module.si_proj_corners

    def recording_si_proj_corners(*args, **kwargs):
        X = kwargs.get('X')
        Y = kwargs.get('Y')
        if X is not None and Y is not None:
            consumed_ids.update((id(X), id(Y)))
        return original(*args, **kwargs)

    monkeypatch.setattr(env_ctm_module, 'si_proj_corners',
                        recording_si_proj_corners)
    ages_before = dict(env._si_age)
    env.update_(opts_svd, moves='h', method='2x2 corner', opts_si=opts_si)

    assert recycled_ids <= consumed_ids
    assert env._si_age.keys() == ages_before.keys()
    assert all(env._si_age[key].age == ages_before[key].age + 1
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
    original = si_module.si_refinement

    def recording_correction(*args, **kwargs):
        recycled_x = args[2]
        key = next(key for key, value in _si_bases(env, env.si_X).items()
                   if value is recycled_x)
        correction_ages.append(env._si_age[key].age)
        return original(*args, **kwargs)

    monkeypatch.setattr(si_module, 'si_refinement',
                        recording_correction)
    for _ in range(5):
        env.update_(opts_svd, moves='h', method='2x2 corner', opts_si=opts_si)

    assert set(correction_ages) == {2, 4}


def test_si_disabled_path_matches_full_svd(config_kwargs, monkeypatch):
    """Explicitly disabling SI selects the unchanged full-SVD path."""
    config = yastn.make_config(sym='none', **config_kwargs)
    r0, r1 = _dense_corners_with_spectrum(
        config, (1., .5, .1, 1e-3, 0., 0.))
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    reference = proj_corners(r0, r1, opts_svd)

    leg = yastn.Leg(config, s=1, D=(2,))
    psi = fpeps.Peps(
        fpeps.SquareLattice(dims=(1, 1), boundary='infinite'),
        tensors={(0, 0): yastn.rand(
            config, legs=(leg, leg, leg.conj(), leg.conj()))})
    env = fpeps.EnvCTM(psi, init=None)
    site = Site(0, 0)
    site_b = env.nn_site(site, d='b')

    def forbidden_si(*args, **kwargs):
        pytest.fail("Disabled SI called si_proj_corners")

    monkeypatch.setattr(env_ctm_module, 'si_proj_corners', forbidden_si)
    env._set_projector_pair_(site, 'hlb', site_b, 'hlt', r0, r1, opts_svd,
                             opts_si={'enabled': False})
    disabled = (env.proj[site].hlb, env.proj[site_b].hlt)
    for actual, expected in zip(disabled, reference):
        assert yastn.allclose(actual, expected)
    assert not _si_bases(env, env.si_X)
    assert not _si_bases(env, env.si_Y)
    assert not env._si_age


def test_new_environment_starts_without_si_recycling_state(config_kwargs):
    """SI dictionaries are not implicitly shared with another environment."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    one_site_psi, _ = _classical_ising_peps(config)
    psi_tensor = one_site_psi[(0, 0)]
    geometry = fpeps.SquareLattice(dims=(2, 1), boundary='infinite')
    two_site_psi = fpeps.Peps(
        geometry, tensors={(0, 0): psi_tensor, (1, 0): psi_tensor})
    other_env = fpeps.EnvCTM(two_site_psi, init='eye')
    assert not _si_bases(other_env, other_env.si_X)
    assert not _si_bases(other_env, other_env.si_Y)
    assert not other_env._si_age


# ---------------------------------------------------------------------------
# End-to-end CTMRG convergence and physical observables
# ---------------------------------------------------------------------------


def _ising_acceptance_state(config_kwargs):
    """Real-valued classical-Ising PEPS with analytically known correlators."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    psi, spin = _classical_ising_peps(config)

    def check_observables(env_full, env_si):
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

    return config, psi, check_observables


def _complex_rdm1x1_check(psi):
    """Observable check comparing 1x1 RDMs of a complex PEPS in both environments."""

    def check_observables(env_full, env_si):
        # The 1x1 RDM is gauge invariant and, unlike the CTM norm, insensitive
        # to the arbitrary normalization of the converged environment.
        rdm_full, _ = rdm1x1((0, 0), psi, env_full)
        rdm_si, _ = rdm1x1((0, 0), psi, env_si)
        assert (rdm_si - rdm_full).norm() < 1e-8
        assert abs(rdm_full.trace().item() - 1) < 1e-10
        # SI has to carry the complex phases, not merely complex storage.
        assert all(x.yastn_dtype == 'complex128'
                   for x in _si_bases(env_si, env_si.si_X).values())

    return check_observables


def _honeycomb_complex_acceptance_state(config_kwargs):
    """Complex Z2 spinless-fermion honeycomb PEPS.

    The A-B dimer is merged into a single square-lattice tensor, so each virtual
    leg is Z2 D=(1, 1) and the physical leg is a hard fusion of the two sites
    with a dim-1 leg carrying the odd parity.
    """
    config = yastn.make_config(sym='Z2', fermionic=True,
                               default_dtype='complex128', **config_kwargs)
    psi = _load_peps_ad(config, 'D1_1x1_Z2_spinlessf_honeycomb_complex.json')
    return config, psi, _complex_rdm1x1_check(psi)


def _triangular_complex_acceptance_state(config_kwargs):
    """Complex D=3 spin-1/2 PEPS for the triangular J1-J2 model, without symmetry.

    Variational 1-site state at J2=0.05 from peps-torch
    (trglC_j20.05_j40_D3ch27_r0_LS_1SITE_iD3n_C4X4cS_ptol8), converted with
    ``read_ipeps`` and ``PepsAD.from_pt``.  SI runs on a single dense block with
    double-layer bond dimension D^2 = 9.
    """
    config = yastn.make_config(sym='none', default_dtype='complex128', **config_kwargs)
    psi = _load_peps_ad(config, 'D3_1x1_dense_spin-half_triangular_complex.json')
    return config, psi, _complex_rdm1x1_check(psi)


@pytest.mark.parametrize(
    'prepare_state',
    [_ising_acceptance_state,
     _honeycomb_complex_acceptance_state,
     _triangular_complex_acceptance_state],
    ids=('ising', 'honeycomb_complex', 'triangular_complex'))
def test_si_ctmrg_matches_full_svd_on_reference_peps(config_kwargs, prepare_state):
    """SI and full-SVD CTMRG must give the same fixed-point physics.

    This covers a complete sequence of random SI initialization, power/QR
    updates, small SVD, gauge rotation, recycling, projector application, and
    convergence of the environment.  The Ising PEPS is the same nontrivial
    analytic network used by the standard CTMRG acceptance test; the honeycomb
    state additionally drives the whole loop with complex amplitudes, and the
    triangular state does so without symmetry at a larger bond dimension.
    """
    config, psi, check_observables = prepare_state(config_kwargs)
    config.backend.random_seed(seed=2026)

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
    bases_x = _si_bases(env_si, env_si.si_X)
    assert bases_x.keys() == _si_bases(env_si, env_si.si_Y).keys() == env_si._si_age.keys()
    assert bases_x
    assert min(si_state.age for si_state in env_si._si_age.values()) >= 5
    assert all(x.get_shape(axes=1) == chi + 4 for x in bases_x.values())

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

    check_observables(env_full, env_si)


def test_si_updates_do_not_depend_on_tensordot_policy(config_kwargs, monkeypatch):
    """Roundoff of a tensordot policy must not decide how long SI iterates.

    The policies contract corners with different floating-point operations.
    CTMRG amplifies the difference in the gauge of Ising environment directions
    whose singular values are at roundoff level. As long as SI convergence is
    judged on directions above the noise floor only, both policies make the
    same number of power updates in almost every call; judged on all columns,
    only about 60% of the calls agreed.
    """
    opts_si = {'enabled': True, 'oversampling': 4, 'niter': 10,
               'tol': 1e-3, 'warmup': 5}
    updates = []
    for policy in ('fuse_contracted', 'no_fusion'):
        config, psi, _ = _ising_acceptance_state(
            {**config_kwargs, 'tensordot_policy': policy})
        config.backend.random_seed(seed=2026)
        env = fpeps.EnvCTM(psi, init='eye')
        updates.append(_record_si_updates(monkeypatch))
        env.ctmrg_(opts_svd={'D_total': 12, 'tol': 0, 'fix_signs': True},
                   max_sweeps=40, corner_tol=1e-9, method='2x2 corner',
                   opts_si=opts_si)
        monkeypatch.undo()

    assert all(updates)
    agreement = np.mean([a == b for a, b in zip(*updates)])
    assert agreement >= 0.95
