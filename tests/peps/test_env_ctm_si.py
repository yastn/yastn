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
from yastn.tn.fpeps._geometry import Site
from yastn.tn.fpeps.envs._env_ctm import (
    initialize_si_bases,
    isometry_expansion,
    isometry_shrinkage,
    proj_corners,
    si_bases_compatible,
    si_projector_svd,
    si_refinement,
    symmetric_isometry_recycle,
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
    p0, p1, X, Y = proj_corners(
        r0, r1, opts_svd=opts_svd, opts_si=opts_si,
        return_si_state=True)
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

    p0, p1, X, Y = proj_corners(
        r0, r1, opts_svd=opts_svd,
        opts_si={'enabled': True, 'oversampling': 1,
                 'niter': 24, 'tol': 1e-12, 'correct': True},
        return_si_state=True)

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
    for r0_bad, r1_bad in malformed_pairs:
        with pytest.raises(yastn.YastnError, match=message):
            initialize_si_bases(r0_bad, r1_bad, rank=3)
        with pytest.raises(yastn.YastnError, match=message):
            proj_corners(r0_bad, r1_bad, opts_svd={'D_total': 3})


def test_si_recycles_after_leg_dimension_change(config_kwargs):
    """Stale bases do not affect projectors after corner dimensions change."""
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=12)
    r0, r1 = _ctm_corner_pair(config, 'none')
    opts_svd = {'D_total': 4, 'tol': 0}
    opts_si = {'enabled': True, 'oversampling': 2,
               'niter': 24, 'tol': 1e-12, 'correct': True}
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

    reference = proj_corners(r0_new, r1_new, opts_svd)
    p0, p1, X1, Y1 = proj_corners(
        r0_new, r1_new, opts_svd, opts_si=opts_si, X=X0, Y=Y0,
        return_si_state=True)

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
    opts_svd = {'D_total': 4, 'tol': 0}
    opts_si = {'enabled': True, 'oversampling': 0,
               'niter': 24, 'tol': 1e-12, 'correct': True}
    reference = proj_corners(r0, r1_with_new_history, opts_svd)
    p0, p1, X_new, Y_new = proj_corners(
        r0, r1_with_new_history, opts_svd, opts_si=opts_si, X=X, Y=Y,
        return_si_state=True)

    assert si_bases_compatible(r0, r1_with_new_history, X_new, Y_new)
    _assert_projectors_equivalent(reference, (p0, p1))


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
    assert env.X
    assert env.X.keys() == env.Y.keys() == env._si_age.keys()
    assert all(age == 1 for age in env._si_age.values())
    assert env.effective_chi() == 2
    assert any(x.get_shape(axes=1) > 1 for x in env.X.values())
