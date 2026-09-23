# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit and environment-state tests for recycled SI-CTM projectors."""

import numpy as np
import pytest

import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.fpeps.envs._env_ctm_SI_projectors as si_module
from yastn.tn.fpeps._geometry import Site
from yastn.tn.fpeps.envs._env_ctm_c4v import EnvCTM_c4v
from yastn.tn.fpeps.envs._env_ctm import SI_state, proj_corners
from yastn.tn.fpeps.envs._env_ctm_SI_projectors import (
    initialize_si_bases,
    isometry_expansion,
    isometry_shrinkage,
    si_bases_compatible,
    si_proj_corners,
    si_projector_svd,
    si_refinement,
    symmetric_isometry_recycle,
    svd_charge_sector_values,
)


def _si_ages(env):
    """Ages alone of ``env._si_age``, dropping the per-update niter and error."""
    return {key: si_state.age for key, si_state in env._si_age.items()}


def _si_bases(env, container):
    """Assigned recycled bases of ``env.si_X``/``env.si_Y``, keyed like ``env._si_age``."""
    return {(env.site2index(site), name): getattr(projectors, name)
            for site, projectors in container.items()
            for name in projectors.fields()
            if getattr(projectors, name) is not None}


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
    qref = uref[:, :rank_ref]
    qsi = usi[:, :rank_si]
    overlap = qref.conj().T @ qsi
    error = 1.0 - np.linalg.norm(overlap, ord='fro') ** 2 / rank_ref
    return float(np.clip(error, 0.0, 1.0))


def _isometry_error(basis):
    leg = basis.get_legs(1)
    identity = yastn.eye(
        basis.config, legs=(leg.conj(), leg), isdiag=False,
        dtype=basis.yastn_dtype, device=basis.device)
    return (basis.H @ basis - identity).norm()


def _dense_isometry(config, rows=7, columns=3):
    """Create a random dense column isometry."""
    left = yastn.Leg(config, s=1, D=(rows,))
    right = yastn.Leg(config, s=-1, D=(columns,))
    return yastn.qr(
        yastn.rand(config, legs=(left, right), distribution='normal'),
        axes=(0, 1), sQ=-1)[0]


def test_dense_isometry_expansion_preserves_existing_basis(config_kwargs):
    """Expansion adds orthonormal columns after the existing basis."""
    config = yastn.make_config(sym='none', **config_kwargs)
    basis = _dense_isometry(config)

    expanded = isometry_expansion(basis, 5)

    assert expanded.get_shape() == (7, 5)
    assert _isometry_error(expanded) < 1e-12
    assert np.allclose(expanded.to_numpy()[:, :3], basis.to_numpy())


@pytest.mark.parametrize('added_dimension', [1, 2])
def test_dense_isometry_expansion_uses_candidate_subspace(
        config_kwargs, added_dimension):
    """One or several candidate columns determine the added subspace."""
    config = yastn.make_config(sym='none', **config_kwargs)
    basis = _dense_isometry(config)
    left = basis.get_legs(0)
    addition_leg = yastn.Leg(config, s=-1, D=(added_dimension,))
    candidate = yastn.rand(
        config, legs=(left, addition_leg), distribution='normal')

    expanded = isometry_expansion(
        basis, 3 + added_dimension, u=candidate)

    assert expanded.get_shape() == (7, 3 + added_dimension)
    assert _isometry_error(expanded) < 1e-12
    assert np.allclose(expanded.to_numpy()[:, :3], basis.to_numpy())

    basis_array = basis.to_numpy()
    candidate_array = candidate.to_numpy()
    residual = candidate_array - basis_array @ (
        basis_array.conj().T @ candidate_array)
    expected, _ = np.linalg.qr(residual, mode='reduced')
    actual = expanded.to_numpy()[:, 3:]
    assert np.allclose(actual @ actual.conj().T,
                       expected @ expected.conj().T)


def test_dense_isometry_shrinkage_keeps_leading_columns(config_kwargs):
    """Shrinkage returns the requested leading part of an isometry."""
    config = yastn.make_config(sym='none', **config_kwargs)
    basis = _dense_isometry(config, columns=5)

    shrunk = isometry_shrinkage(basis, 2)

    assert shrunk.get_shape() == (7, 2)
    assert _isometry_error(shrunk) < 1e-12
    assert np.allclose(shrunk.to_numpy(), basis.to_numpy()[:, :2])
    assert isometry_shrinkage(basis, 5) is basis


def test_dense_isometry_resize_rejects_impossible_dimensions(config_kwargs):
    """Resize operations reject targets outside their mathematical domain."""
    config = yastn.make_config(sym='none', **config_kwargs)
    basis = _dense_isometry(config)

    with pytest.raises(yastn.YastnError):
        isometry_expansion(basis, 3)
    with pytest.raises(yastn.YastnError):
        isometry_expansion(basis, 8)
    with pytest.raises(yastn.YastnError):
        isometry_shrinkage(basis, 4)


def _u1_isometry(config, right_charges=(-1, 0, 1),
                  right_dimensions=(2, 3, 1)):
    """Create a U1 isometry with independently sized charge sectors."""
    left = yastn.Leg(config, s=1, t=(-1, 0, 1), D=(4, 5, 3))
    right = yastn.Leg(
        config, s=-1, t=right_charges, D=right_dimensions)
    return yastn.qr(
        yastn.rand(config, legs=(left, right), distribution='normal'),
        axes=(0, 1), sQ=-1)[0]


def test_symmetric_isometry_recycle_returns_unchanged_basis(config_kwargs):
    """Recycling with the current allocation is a no-op."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(config)

    unchanged = symmetric_isometry_recycle(
        basis, {-1: 2, 0: 3, 1: 1})
    assert unchanged is basis


def test_symmetric_isometry_recycle_changes_sector_dimensions(config_kwargs):
    """Recycling grows and shrinks sectors while retaining prior columns."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(config)
    resized = symmetric_isometry_recycle(
        basis, {-1: 3, 0: 1, 1: 2})

    assert resized.get_legs(0) == basis.get_legs(0)
    assert resized.get_legs(1).tD == {(-1,): 3, (0,): 1, (1,): 2}
    assert _isometry_error(resized) < 1e-12

    for charge, target_dimension in resized.get_legs(1).tD.items():
        old_block = basis[charge + charge]
        new_block = resized[charge + charge]
        retained_dimension = min(old_block.shape[1], target_dimension)
        assert np.allclose(
            basis.config.backend.to_numpy(old_block[:, :retained_dimension]),
            basis.config.backend.to_numpy(new_block[:, :retained_dimension]))


def test_symmetric_isometry_recycle_removes_and_expands_sectors(config_kwargs):
    """A sector can be removed while another retained sector is expanded."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(config)

    resized = symmetric_isometry_recycle(basis, {-1: 2, 1: 2})

    assert resized.get_legs(0).tD == {(-1,): 4, (1,): 3}
    assert resized.get_legs(1).tD == {(-1,): 2, (1,): 2}
    assert _isometry_error(resized) < 1e-12
    for charge in ((-1,), (1,)):
        old_block = basis[charge + charge]
        new_block = resized[charge + charge]
        assert np.allclose(
            basis.config.backend.to_numpy(old_block),
            basis.config.backend.to_numpy(new_block[:, :old_block.shape[1]]))


def test_symmetric_isometry_recycle_rejects_sector_over_capacity(
        config_kwargs):
    """A sector cannot contain more columns than its row space."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(config)

    with pytest.raises(yastn.YastnError, match='row-space capacity'):
        symmetric_isometry_recycle(basis, {-1: 5, 0: 1, 1: 2})


def test_symmetric_isometry_recycle_zero_dimension_removes_sector(
        config_kwargs):
    """A zero target dimension is equivalent to omitting that sector."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(config)

    explicit_zero = symmetric_isometry_recycle(
        basis, {-1: 2, 0: 0, 1: 1})
    omitted = symmetric_isometry_recycle(basis, {-1: 2, 1: 1})

    assert explicit_zero.get_legs() == omitted.get_legs()
    assert np.allclose(explicit_zero.to_numpy(), omitted.to_numpy())
    assert _isometry_error(explicit_zero) < 1e-12


def test_symmetric_isometry_recycle_rejects_unavailable_sector(
        config_kwargs):
    """A new sector requires a matching sector in the row space."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(
        config, right_charges=(-1, 0), right_dimensions=(2, 3))

    with pytest.raises(yastn.YastnError, match='row-space capacity 0'):
        symmetric_isometry_recycle(basis, {-1: 2, 1: 1})


def test_symmetric_isometry_recycle_uses_expanded_left_leg(config_kwargs):
    """A current row leg can seed a sector absent from the old basis."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(
        config, right_charges=(-1, 0), right_dimensions=(2, 3))
    current_left = yastn.Leg(
        config, s=1, t=(-1, 0, 1), D=(4, 5, 3))

    resized = symmetric_isometry_recycle(
        basis, {-1: 2, 1: 2}, left_leg=current_left)

    assert resized.get_legs(0).tD == {(-1,): 4, (1,): 3}
    assert resized.get_legs(1).tD == {(-1,): 2, (1,): 2}
    assert _isometry_error(resized) < 1e-12
    assert np.allclose(
        basis.config.backend.to_numpy(basis[(-1, -1)]),
        basis.config.backend.to_numpy(resized[(-1, -1)]))


@pytest.mark.parametrize(
    'charges, error',
    [
        (None, 'mapping'),
        ({}, 'at least one column'),
        ({-1: -1}, 'nonnegative integers'),
        ({-1: True}, 'nonnegative integers'),
        ({-1: 1.5}, 'nonnegative integers'),
    ],
)
def test_symmetric_isometry_recycle_rejects_invalid_allocations(
        config_kwargs, charges, error):
    """Target allocations must be a nonempty map of nonnegative integers."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(config)

    with pytest.raises(yastn.YastnError, match=error):
        symmetric_isometry_recycle(basis, charges)


def test_symmetric_isometry_recycle_expands_to_sector_capacities(
        config_kwargs):
    """Every charge sector can be expanded to fill its complete row space."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(config)

    resized = symmetric_isometry_recycle(basis, {-1: 4, 0: 5, 1: 3})

    assert resized.get_legs(1).tD == {(-1,): 4, (0,): 5, (1,): 3}
    assert _isometry_error(resized) < 1e-12
    for charge in resized.get_legs(1).tD:
        old_block = basis[charge + charge]
        new_block = resized[charge + charge]
        assert np.allclose(
            basis.config.backend.to_numpy(old_block),
            basis.config.backend.to_numpy(new_block[:, :old_block.shape[1]]))


def test_symmetric_isometry_recycle_supports_complex_tensors(config_kwargs):
    """Resizing preserves complex dtype, old columns, and orthonormality."""
    config = yastn.make_config(
        sym='U1', default_dtype='complex128', **config_kwargs)
    basis = _u1_isometry(config)

    resized = symmetric_isometry_recycle(basis, {-1: 3, 0: 2, 1: 2})

    assert resized.yastn_dtype == basis.yastn_dtype == 'complex128'
    assert _isometry_error(resized) < 1e-12
    for charge, target_dimension in resized.get_legs(1).tD.items():
        old_block = basis[charge + charge]
        new_block = resized[charge + charge]
        retained_dimension = min(old_block.shape[1], target_dimension)
        assert np.allclose(
            basis.config.backend.to_numpy(old_block[:, :retained_dimension]),
            basis.config.backend.to_numpy(new_block[:, :retained_dimension]))


def test_symmetric_isometry_recycle_supports_z2(config_kwargs):
    """Sector recycling is not specific to integer U1 charges."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    left = yastn.Leg(config, s=1, t=(0, 1), D=(4, 5))
    right = yastn.Leg(config, s=-1, t=(0, 1), D=(2, 3))
    basis = yastn.qr(
        yastn.rand(config, legs=(left, right), distribution='normal'),
        axes=(0, 1), sQ=-1)[0]

    resized = symmetric_isometry_recycle(basis, {0: 3, 1: 2})

    assert resized.get_legs(1).tD == {(0,): 3, (1,): 2}
    assert _isometry_error(resized) < 1e-12
    for charge, target_dimension in resized.get_legs(1).tD.items():
        old_block = basis[charge + charge]
        new_block = resized[charge + charge]
        retained_dimension = min(old_block.shape[1], target_dimension)
        assert np.allclose(
            basis.config.backend.to_numpy(old_block[:, :retained_dimension]),
            basis.config.backend.to_numpy(new_block[:, :retained_dimension]))


def test_symmetric_isometry_recycle_can_be_applied_repeatedly(config_kwargs):
    """A recycled basis remains valid input for a later dimension change."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    basis = _u1_isometry(config)
    first = symmetric_isometry_recycle(basis, {-1: 3, 0: 1, 1: 2})

    second = symmetric_isometry_recycle(first, {-1: 2, 0: 4, 1: 1})

    assert second.get_legs(1).tD == {(-1,): 2, (0,): 4, (1,): 1}
    assert _isometry_error(second) < 1e-12
    for charge, target_dimension in second.get_legs(1).tD.items():
        old_block = first[charge + charge]
        new_block = second[charge + charge]
        retained_dimension = min(old_block.shape[1], target_dimension)
        assert np.allclose(
            first.config.backend.to_numpy(old_block[:, :retained_dimension]),
            first.config.backend.to_numpy(new_block[:, :retained_dimension]))


def _assert_projectors_equivalent(reference, approximate, tol=2e-8):
    """Compare projector ranges using their gauge-invariant principal angles."""
    for index, (pref, psi) in enumerate(zip(reference, approximate)):
        pref_matrix = _matrix_projector(pref)
        psi_matrix = _matrix_projector(psi)
        ref_sectors = pref_matrix.get_legs(1).tD
        si_sectors = psi_matrix.get_legs(1).tD
        error = _projector_subspace_error(pref, psi)
        assert ref_sectors == si_sectors
        assert error < tol, f"projector {index} subspace error: {error:.3e}"


# ---------------------------------------------------------------------------
# Projector and reduced-spectrum correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('sym', ['U1', 'Z2'])
def test_si_projectors_match_full_svd(config_kwargs, sym):
    """SI and full SVD produce the same projector maps on CTM corner halves."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=10)
    r0, r1 = _ctm_corner_pair(config, sym)
    # chi + p = 6, strictly below the corner-leg rank of 7.
    opts_svd = {'D_total': 5, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 1,
               'niter': 24, 'tol': 1e-12, 'correct': True}
    full = proj_corners(r0, r1, opts_svd=opts_svd)
    p0, p1, X, Y, _ = si_proj_corners(r0, r1, opts_svd, opts_si)
    assert X.get_shape(axes=1) == 6
    assert Y.get_shape(axes=0) == 6
    assert X.get_shape(axes=1) < min(r0.get_shape(axes=0),
                                     r1.get_shape(axes=0))
    assert si_bases_compatible(r0, r1, X, Y)
    _assert_projectors_equivalent(full, (p0, p1))


def test_si_complex_u1_projectors_match_full_svd(config_kwargs):
    """Complex U1 corners preserve conjugation in recycled projectors."""
    config = yastn.make_config(
        sym='U1', default_dtype='complex128', **config_kwargs)
    config.backend.random_seed(seed=19)
    r0, r1 = _ctm_corner_pair(config, 'U1')
    r0 = (1 + 0.35j) * r0
    r1 = (1 - 0.2j) * r1
    opts_svd = {'D_total': 5, 'tol': 0, 'fix_signs': True}
    full = proj_corners(r0, r1, opts_svd=opts_svd)

    p0, p1, X, Y, _ = si_proj_corners(
        r0, r1, opts_svd,
        {'enabled': True, 'oversampling': 1,
         'niter': 24, 'tol': 1e-12, 'correct': True})

    assert X.dtype == Y.dtype == config.backend.DTYPE['complex128']
    assert si_bases_compatible(r0, r1, X, Y)
    _assert_projectors_equivalent(full, (p0, p1), tol=5e-8)


@pytest.mark.parametrize('sym', ['U1', 'Z2'])
def test_si_spectrum_matches_full_svd(config_kwargs, sym):
    """SI iterations recover the leading spectrum from a strict subspace."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=11)
    r0, r1 = _ctm_corner_pair(config, sym)
    opts_svd = {'D_total': 5, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 24, 'tol': 1e-12}
    X, Y = initialize_si_bases(r0, r1, rank=6)
    X, Y = si_refinement(r0, r1, X, Y, opts_svd, opts_si)
    assert X.get_shape(axes=1) < min(r0.get_shape(axes=0),
                                     r1.get_shape(axes=0))

    # Guard against accidentally testing a full-rank change of basis: the
    # unrefined random sketch must not already reproduce the reference.
    _, s_initial, _, _, _, _ = si_projector_svd(
        r0, r1, X, Y, opts_svd, {**opts_si, 'niter': 0})
    _, s_si, _, _, _, _ = si_projector_svd(
        r0, r1, X, Y, opts_svd, opts_si)

    rr = yastn.tensordot(r0, r1, axes=(1, 1))
    _, s_ref, _ = rr.svd_with_truncation(
        axes=(0, 1), sU=r0.s[1], **opts_svd)
    ref = svd_charge_sector_values(s_ref)
    initial = svd_charge_sector_values(s_initial)
    actual = svd_charge_sector_values(s_si)

    assert ref.keys() == actual.keys()
    initial_error = 0.0
    final_error = 0.0
    for charge in ref:
        ref_values = np.asarray(ref[charge])
        initial_values = np.asarray(initial.get(charge, ()))
        padded_initial = np.zeros_like(ref_values)
        common = min(ref_values.size, initial_values.size)
        padded_initial[:common] = initial_values[:common]
        initial_error += np.linalg.norm(
            ref_values - padded_initial) ** 2
        final_error += np.linalg.norm(
            ref_values - np.asarray(actual[charge])) ** 2
        assert np.allclose(ref[charge], actual[charge], rtol=2e-8, atol=2e-10)
    initial_error = np.sqrt(initial_error)
    final_error = np.sqrt(final_error)
    assert initial_error > 1e-6
    assert final_error < 1e-4 * initial_error


# ---------------------------------------------------------------------------
# Input validation and recycled-basis rebuilding
# ---------------------------------------------------------------------------


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
    opts_si = {'enabled': True, 'oversampling': 1, 'niter': 2}
    for r0_bad, r1_bad in malformed_pairs:
        with pytest.raises(yastn.YastnError, match=message):
            initialize_si_bases(r0_bad, r1_bad, rank=3)
        with pytest.raises(yastn.YastnError, match=message):
            si_proj_corners(r0_bad, r1_bad, {'D_total': 3}, opts_si)

    # The precondition belongs to SI alone: the dense path contracts the same
    # halves happily, and must keep doing so.
    p0, p1 = proj_corners(r0, malformed_pairs[0][1], opts_svd={'D_total': 3})
    assert p0 is not None and p1 is not None


def test_si_recycles_after_leg_dimension_change(config_kwargs):
    """Stale bases do not affect projectors after corner dimensions change."""
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=12)
    r0, r1 = _ctm_corner_pair(config, 'none')
    opts_svd = {'D_total': 4, 'tol': 0}
    opts_si = {'enabled': True, 'oversampling': 2,
               'niter': 24, 'tol': 1e-12, 'correct': True}
    _, _, X0, Y0, _ = si_proj_corners(r0, r1, opts_svd, opts_si)

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

    reference = proj_corners(r0_new, r1_new, opts_svd)
    p0, p1, X1, Y1, _ = si_proj_corners(
        r0_new, r1_new, opts_svd, opts_si, X=X0, Y=Y0)

    assert si_bases_compatible(r0_new, r1_new, X1, Y1)
    assert X1.get_shape(axes=1) == 6
    _assert_projectors_equivalent(reference, (p0, p1))


def test_si_recycles_after_fusion_history_change(config_kwargs):
    """Fusion-history changes do not alter the resulting projectors."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _ctm_corner_pair(config, 'Z2')
    X, Y = initialize_si_bases(r0, r1, rank=4)

    external = r1.get_legs(0).drop_history()
    contracted = r1.get_legs(1)
    r1_with_new_history = yastn.rand(config, legs=(external, contracted))

    assert external.tD == r1.get_legs(0).tD
    assert external.hf != r1.get_legs(0).hf
    opts_svd = {'D_total': 4, 'tol': 1e-10}
    opts_si = {'enabled': True, 'oversampling': 0,
               'niter': 24, 'tol': 1e-12, 'correct': True}
    reference = proj_corners(r0, r1_with_new_history, opts_svd)
    p0, p1, X_new, Y_new, _ = si_proj_corners(
        r0, r1_with_new_history, opts_svd, opts_si, X=X, Y=Y)

    assert si_bases_compatible(r0, r1_with_new_history, X_new, Y_new)
    _assert_projectors_equivalent(reference, (p0, p1))


def _hard_fused_u1_corner_pair(config, Da, Db):
    """U1 corner halves whose external leg is hard-fused from two sub-legs.

    The contracted leg is fused with a trivial leg, as in ``_ctm_corner_pair``,
    so that projectors unfuse to the rank-3 CTM form.
    """
    a = yastn.Leg(config, s=1, t=(0, 1), D=Da)
    b = yastn.Leg(config, s=1, t=(0, 1), D=Db)
    k = yastn.Leg(config, s=-1, t=(0, 1, 2), D=(4, 6, 4))
    one = yastn.Leg(config, s=1, t=(0,), D=(1,))
    r0 = yastn.rand(config, legs=(a, b, k, one))
    r1 = yastn.rand(config, legs=(a.conj(), b.conj(), k.conj(), one.conj()))
    return (r0.fuse_legs(axes=((0, 1), (2, 3))),
            r1.fuse_legs(axes=((0, 1), (2, 3))))


def test_si_rebuilds_basis_after_hard_fused_subleg_change(config_kwargs, monkeypatch):
    """
    Swapped sub-leg dimensions keep the aggregate corner sectors but make old
    bases uncontractible; they are rebuilt before the single SI solve.
    """
    config = yastn.make_config(sym='U1', **config_kwargs)
    config.backend.random_seed(seed=14)
    r0, r1 = _hard_fused_u1_corner_pair(config, (2, 3), (3, 2))
    X, Y = initialize_si_bases(r0, r1, rank=6)
    r0_new, r1_new = _hard_fused_u1_corner_pair(config, (3, 2), (2, 3))
    assert r0_new.get_legs(0).tD == r0.get_legs(0).tD
    assert r1_new.get_legs(0).tD == r1.get_legs(0).tD
    assert not si_bases_compatible(r0_new, r1_new, X, Y)

    calls = []
    original = si_module.si_projector_svd

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(si_module, 'si_projector_svd', counting)
    opts_svd = {'D_total': 4, 'tol': 0}
    opts_si = {'enabled': True, 'oversampling': 2,
               'niter': 24, 'tol': 1e-12, 'correct': True}
    reference = proj_corners(r0_new, r1_new, opts_svd)
    p0, p1, X_new, Y_new, _ = si_proj_corners(
        r0_new, r1_new, opts_svd, opts_si, X=X, Y=Y)

    assert len(calls) == 1
    assert si_bases_compatible(r0_new, r1_new, X_new, Y_new)
    _assert_projectors_equivalent(reference, (p0, p1))


def test_si_solve_errors_are_not_retried(config_kwargs, monkeypatch):
    """An SI solve error on recycled bases propagates without a fresh restart."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    r0, r1 = _ctm_corner_pair(config, 'U1')
    X, Y = initialize_si_bases(r0, r1, rank=4)
    assert si_bases_compatible(r0, r1, X, Y)

    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        raise yastn.YastnError('boom')

    monkeypatch.setattr(si_module, 'si_projector_svd', failing)
    with pytest.raises(yastn.YastnError, match='boom'):
        si_proj_corners(r0, r1, {'D_total': 3},
                        {'enabled': True, 'oversampling': 1}, X=X, Y=Y)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Environment state and CTMRG update integration
# ---------------------------------------------------------------------------


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
    site, name = Site(0, 0), 'vtr'
    X, Y = initialize_si_bases(r0, r1, rank=3)
    setattr(env.si_X[site], name, X)
    setattr(env.si_Y[site], name, Y)
    env._si_age[env.site2index(site), name] = SI_state(age=4, niter=2, error=1e-9)

    def si_x(e):
        return getattr(e.si_X[site], name)

    def si_y(e):
        return getattr(e.si_Y[site], name)

    variants = (
        env.copy(), env.clone(), env.detach(),
        env.to(dtype=config.default_dtype),
        fpeps.EnvCTM.from_dict(env.to_dict()),
    )
    for other in variants:
        # Only the age is serialized: niter and error describe a single past
        # update, so a round trip restores them to their defaults.
        expected = (env._si_age if other is not variants[4] else
                    {key: SI_state(age=si_state.age)
                     for key, si_state in env._si_age.items()})
        assert other._si_age == expected
        assert yastn.allclose(si_x(other), si_x(env))
        assert yastn.allclose(si_y(other), si_y(env))
        assert other.si_X is not env.si_X and other.si_Y is not env.si_Y

    # Copy, clone, and deserialization promise independent tensor objects.
    for other in (variants[0], variants[1], variants[4]):
        assert si_x(other) is not si_x(env)
        assert si_y(other) is not si_y(env)

    # Replacing state in the source must not mutate any copied container.
    old_x = si_x(variants[0])
    setattr(env.si_X[site], name, 2 * si_x(env))
    assert yastn.allclose(si_x(variants[0]), old_x)
    assert not yastn.allclose(si_x(env), si_x(variants[0]))

    env.detach_()
    assert yastn.allclose(si_x(env), 2 * si_x(variants[0]))
    assert yastn.allclose(si_y(env), si_y(variants[0]))


def test_si_state_follows_patch(config_kwargs):
    """
    Patched sites recycle and count their own SI state, and apply_patch commits
    projector, bases and age from the same site, as Lattice does for env.proj.
    """
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=13)
    env = _dense_product_env(config)
    r0, r1 = _ctm_corner_pair(config, 'none')
    opts_svd = {'D_total': 3, 'tol': 0}
    opts_si = {'enabled': True, 'oversampling': 1, 'niter': 2, 'warmup': 100}
    # On the 1x1 infinite lattice all three sites alias one unit-cell index.
    s0, s1, alias = Site(0, 0), Site(0, 1), Site(1, 0)
    name = 'hlb'
    key = (env.site2index(s0), name)

    def update(site):
        env._set_projector_pair_(site, name, env.nn_site(site, d='b'), 'hlt',
                                 r0, r1, opts_svd, opts_si=opts_si)
        return (getattr(env.proj[site], name),
                getattr(env.si_X[site], name), getattr(env.si_Y[site], name))

    committed = update(alias)
    assert _si_ages(env) == {key: 1}

    env.move_to_patch([s0, s1])
    update(s0)
    update(s0)
    # Neither the unpatched alias nor the other patched site sees s0's bases.
    assert getattr(env.si_X[alias], name) is committed[1]
    assert getattr(env.si_X[s1], name) is committed[1]
    assert getattr(env.si_Y[s1], name) is committed[2]
    assert _si_ages(env) == {key: 1}
    last = update(s1)

    env.apply_patch()
    # s1 is patched last: its lineage (1 committed + 1 patched update) wins,
    # not s0's (3) and not a count over all aliases (4).
    assert _si_ages(env) == {key: 2}
    assert not env._si_age_patch
    for site in (s0, s1, alias):
        assert getattr(env.proj[site], name) is last[0]
        assert getattr(env.si_X[site], name) is last[1]
        assert getattr(env.si_Y[site], name) is last[2]


def test_c4v_environment_carries_empty_si_storage(config_kwargs):
    """EnvCTM_c4v inherits EnvCTM's copy/clone/detach/to, which touch SI state."""
    config = yastn.make_config(sym='none', **config_kwargs)
    leg = yastn.Leg(config, s=1, D=(2,))
    psi = fpeps.Peps(
        fpeps.SquareLattice(dims=(1, 1), boundary='infinite'),
        tensors={(0, 0): yastn.rand(config, legs=[leg, leg, leg.conj(),
                                                  leg.conj(), leg])})
    env = EnvCTM_c4v(psi, init='eye')
    for other in (env, env.copy(), env.clone(), env.detach(),
                  env.to(dtype=config.default_dtype)):
        assert not _si_bases(other, other.si_X)
        assert not _si_bases(other, other.si_Y)
        assert not other._si_age


def test_si_ctm_update_1x2_method(config_kwargs):
    """SI projectors support the 1x2 environment update path."""
    config = yastn.make_config(sym='none', **config_kwargs)
    env = _dense_product_env(config)
    # Grow the eye environment first; 1x2 updates cannot increase chi and
    # would otherwise exercise only rank-one corner spaces.
    env.update_(opts_svd={'D_total': 2}, moves='hv', method='2x2 corner')
    assert env.effective_chi() == 2
    env.update_(
        opts_svd={'D_total': 2}, moves='hv', method='1x2 corner',
        opts_si={'enabled': True, 'oversampling': 1, 'niter': 3})
    assert env.is_consistent()
    bases_x = _si_bases(env, env.si_X)
    assert bases_x
    assert bases_x.keys() == _si_bases(env, env.si_Y).keys() == env._si_age.keys()
    assert all(si_state.age == 1 for si_state in env._si_age.values())
    assert env.effective_chi() == 2
    assert any(x.get_shape(axes=1) > 1 for x in bases_x.values())
