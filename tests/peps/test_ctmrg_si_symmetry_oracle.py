# Copyright 2026 The YASTN Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Block-sparse oracle tests for symmetry-aware SI projector refinement."""

import numpy as np
import pytest

import yastn
from yastn.tn.fpeps.envs._env_ctm import (
    initialize_si_bases,
    si_bases_compatible,
    si_projector_svd,
    si_refinment_cwo,
    svd_charge_sector_dimensions,
)


def _charge_tuple(charge):
    return charge if isinstance(charge, tuple) else (charge,)


def _corners_with_sector_spectra(config, spectra, left=None, right=None):
    """Return r0, r1 with exactly prescribed singular values per sector."""
    r0 = yastn.Tensor(config=config, s=(1, -1))
    r1 = yastn.Tensor(config=config, s=(-1, 1))
    for charge, values in spectra.items():
        charge = _charge_tuple(charge)
        values = np.asarray(values, dtype=float)
        rank = values.size
        dl = rank if left is None else left[charge]
        dr = rank if right is None else right[charge]
        if rank > min(dl, dr):
            raise ValueError("A prescribed spectrum exceeds sector capacity.")
        a = np.zeros((dl, rank))
        b = np.zeros((dr, rank))
        a[:rank, :] = np.eye(rank)
        b[:rank, :] = np.diag(values)
        ts = charge + charge
        r0.set_block(ts=ts, Ds=a.shape, val=a)
        r1.set_block(ts=ts, Ds=b.shape, val=b)
    return r0, r1


def _full_sector_dimensions(r0, r1, opts_svd):
    rho = yastn.tensordot(r0, r1, axes=(1, 1))
    _, s, _ = rho.svd_with_truncation(
        axes=(0, 1), sU=r0.s[1], **opts_svd)
    return svd_charge_sector_dimensions(s)


def _si_sector_dimensions(r0, r1, X, Y, opts_svd, opts_si):
    _, s, _, X, Y = si_projector_svd(
        r0, r1, X, Y, opts_svd, opts_si)
    return svd_charge_sector_dimensions(s), X, Y


@pytest.mark.parametrize(('sym', 'spectra'), [
    ('Z2', {(0,): (10, 9, 8, 7, 6, 5),
            (1,): (4, 3, 2, 1, .5, .25)}),
    ('U1', {(-1,): (3, 2, 1, .5, .25, .1),
            (0,): (12, 11, 10, 9, 8, 7),
            (1,): (6, 5, 4, 3, 2, 1)}),
    ('U1xU1', {(0, 0): (12, 11, 10, 9, 8, 7),
               (1, 0): (6, 5, 4, 3, 2, 1),
               (0, 1): (3, 2, 1, .5, .25, .1)}),
])
def test_cwo_matches_full_svd_for_uneven_multisymmetry_spectra(
        config_kwargs, sym, spectra):
    """CWO finds the exact global sector allocation for all supported symmetries."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=61)
    r0, r1 = _corners_with_sector_spectra(config, spectra)
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 2, 'niter': 10, 'tol': 1e-13}
    charges = {_charge_tuple(q): 2 for q in spectra}
    rank = sum(charges.values())
    # Keep the requested rank equal to chi+p when there are three sectors.
    if rank != 6:
        charges = {charge: 3 for charge in charges}
        rank = 6
    X, Y = initialize_si_bases(r0, r1, rank=rank, charges=charges)
    X, Y = si_refinment_cwo(r0, r1, X, Y, opts_svd, opts_si)
    actual, _, _ = _si_sector_dimensions(
        r0, r1, X, Y, opts_svd, opts_si)
    assert actual == _full_sector_dimensions(r0, r1, opts_svd)


def test_cwo_handles_unequal_left_right_sector_capacities(config_kwargs):
    """The shared auxiliary leg respects min(left capacity, right capacity)."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    spectra = {(0,): (10, 9, 8, 7), (1,): (6, 5, 4)}
    left = {(0,): 6, (1,): 3}
    right = {(0,): 4, (1,): 5}
    r0, r1 = _corners_with_sector_spectra(
        config, spectra, left=left, right=right)
    opts_svd = {'D_total': 5, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 10, 'tol': 1e-13}
    X, Y = initialize_si_bases(
        r0, r1, rank=6, charges={(0,): 3, (1,): 3})
    X, Y = si_refinment_cwo(r0, r1, X, Y, opts_svd, opts_si)
    assert X.get_legs(1).tD[(0,)] <= 4
    assert X.get_legs(1).tD[(1,)] <= 3
    actual, _, _ = _si_sector_dimensions(
        r0, r1, X, Y, opts_svd, opts_si)
    assert actual == _full_sector_dimensions(r0, r1, opts_svd)


def test_sector_change_correction_then_continued_recycling(config_kwargs):
    """A sector can leave and another enter, even at p=0, after correction."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    initial = {(0,): (10, 9, 8, 7), (1,): (4, 3, 2, 1)}
    changed = {(0,): (4, 3, 2, 1), (1,): (10, 9, 8, 7)}
    r0, r1_initial = _corners_with_sector_spectra(config, initial)
    _, r1_changed = _corners_with_sector_spectra(config, changed)
    opts_svd = {'D_total': 3, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 0, 'niter': 10, 'tol': 1e-13}

    # This is the converged allocation before the dominant sector changes.
    X, Y = initialize_si_bases(r0, r1_initial, rank=3,
                               charges={(0,): 3})
    stale, _, _ = _si_sector_dimensions(
        r0, r1_changed, X, Y, opts_svd, opts_si)
    reference = _full_sector_dimensions(r0, r1_changed, opts_svd)
    assert stale != reference  # p=0 cannot discover an absent sector itself.

    X, Y = si_refinment_cwo(
        r0, r1_changed, X, Y, opts_svd, opts_si)
    corrected, X, Y = _si_sector_dimensions(
        r0, r1_changed, X, Y, opts_svd, opts_si)
    assert corrected == reference == {(1,): 3}
    assert si_bases_compatible(r0, r1_changed, X, Y)

    # The corrected allocation remains valid when recycled again.
    recycled, X_next, Y_next = _si_sector_dimensions(
        r0, r1_changed, X, Y, opts_svd, opts_si)
    assert recycled == reference
    assert si_bases_compatible(r0, r1_changed, X_next, Y_next)


def test_conjugate_u1_sectors_and_boundary_degeneracy(config_kwargs):
    """Charge-conjugate sectors tied at the cutoff follow full-SVD allocation."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    spectra = {(-1,): (8, 6, 4), (0,): (10, 9, 1),
               (1,): (8, 6, 4)}
    r0, r1 = _corners_with_sector_spectra(config, spectra)
    opts_svd = {'D_total': 3, 'tol': 0, 'fix_signs': True,
                'truncate_multiplets': True}
    # Multiplet preservation expands the nominal chi=3 truncation to eight
    # states.  A full nine-column basis represents every tied sector exactly.
    opts_si = {'oversampling': 6, 'niter': 10, 'tol': 1e-13}
    X, Y = initialize_si_bases(
        r0, r1, rank=9, charges={(-1,): 3, (0,): 3, (1,): 3})
    X, Y = si_refinment_cwo(r0, r1, X, Y, opts_svd, opts_si)
    actual, _, _ = _si_sector_dimensions(
        r0, r1, X, Y, opts_svd, opts_si)
    assert actual == _full_sector_dimensions(r0, r1, opts_svd)
    assert actual.get((-1,), 0) == actual.get((1,), 0)
