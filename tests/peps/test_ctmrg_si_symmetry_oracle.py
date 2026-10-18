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
"""Block-sparse oracle and dispatch tests for SI projector refinement."""

import numpy as np
import pytest

import yastn
import yastn.tn.fpeps.envs._env_ctm_SI_projectors as si_module
from yastn.tn.fpeps.envs._env_ctm_SI_projectors import (
    _Half,
    initialize_si_bases,
    si_bases_compatible,
    si_proj_corners,
    si_projector_svd,
    si_refinement,
)


def _sector_values(s):
    """Values of a diagonal tensor, by charge sector."""
    return {charge: s.config.backend.to_numpy(s[charge + charge])
            for charge in s.get_legs(0).t}


def _charge_tuple(charge):
    return charge if isinstance(charge, tuple) else (charge,)


def _corners_with_sector_spectra(config, spectra):
    """Return r0, r1 with exactly prescribed singular values per sector."""
    r0 = yastn.Tensor(config=config, s=(1, -1))
    r1 = yastn.Tensor(config=config, s=(-1, 1))
    for charge, values in spectra.items():
        charge = _charge_tuple(charge)
        values = np.asarray(values, dtype=float)
        rank = values.size
        a = np.eye(rank)
        b = np.diag(values)
        ts = charge + charge
        r0.set_block(ts=ts, Ds=a.shape, val=a)
        r1.set_block(ts=ts, Ds=b.shape, val=b)
    return r0, r1


def _global_sector_allocation(spectra, rank):
    """Count the globally largest ``rank`` prescribed sector values."""
    top_values = sorted(
        ((value, _charge_tuple(charge))
         for charge, values in spectra.items() for value in values),
        key=lambda item: item[0], reverse=True)[:rank]
    allocation = {}
    for _, charge in top_values:
        allocation[charge] = allocation.get(charge, 0) + 1
    return allocation


def _full_sector_dimensions(r0, r1, opts_svd):
    rho = yastn.tensordot(r0, r1, axes=(1, 1))
    _, s, _ = rho.svd_with_truncation(
        axes=(0, 1), sU=r0.s[1], **opts_svd)
    return dict(s.get_legs(0).tD)


def _si_sector_dimensions(r0, r1, X, Y, opts_svd, opts_si):
    _, s, _, X, Y, _ = si_projector_svd(
        r0, r1, X, Y, opts_svd, opts_si)
    return dict(s.get_legs(0).tD), X, Y


def _biased_z2_corners(config):
    """Corners whose six globally dominant directions are all Z2-even."""
    return _corners_with_sector_spectra(config, {
        (0,): (10., 9., 8., 7., 6., 5.),
        (1,): (4., 3., 2., 1., .5, .25),
    })


def _assert_refined_si_spectrum(r0, r1, X, Y, opts_svd, opts_si):
    """Check a refined basis against the globally truncated full spectrum."""
    _, s_si, _, _, _, _ = si_projector_svd(
        r0, r1, X, Y, opts_svd, opts_si)
    rho = yastn.tensordot(r0, r1, axes=(1, 1))
    _, s_full, _ = rho.svd_with_truncation(
        axes=(0, 1), sU=r0.s[1], **opts_svd)
    actual = _sector_values(s_si)
    expected = _sector_values(s_full)
    assert actual.keys() == expected.keys()
    for charge in expected:
        assert np.allclose(actual[charge], expected[charge],
                           rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# CWO refinement and dispatch
# ---------------------------------------------------------------------------


def test_si_cwo_refinement_finds_globally_dominant_charge_sector(
        config_kwargs):
    """CWO must move rank out of weak sectors, including oversampling rank."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=31)
    r0, r1 = _biased_z2_corners(config)
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 2, 'niter': 8, 'tol': 1e-12}

    X, Y = initialize_si_bases(r0, r1, rank=6,
                               charges={(0,): 3, (1,): 3})
    X, Y = si_refinement(r0, r1, X, Y, opts_svd, opts_si)

    assert X.get_legs(1).tD == {(0,): 6}
    assert Y.get_legs(0).tD == {(0,): 6}
    _assert_refined_si_spectrum(r0, r1, X, Y, opts_svd, opts_si)


def test_si_refinement_resizes_recycled_bases(config_kwargs, monkeypatch):
    """A changed allocation retains old columns through isometry recycling."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=32)
    r0, r1 = _biased_z2_corners(config)
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 2, 'niter': 8, 'tol': 1e-12}
    X0, Y0 = initialize_si_bases(
        r0, r1, rank=6, charges={(0,): 3, (1,): 3})
    X0_even = config.backend.copy(X0[(0, 0)])
    Yh0_even = config.backend.copy(Y0.H[(0, 0)])
    recycle_calls = 0
    recycle = si_module.symmetric_isometry_recycle

    def counting_recycle(*args, **kwargs):
        nonlocal recycle_calls
        recycle_calls += 1
        return recycle(*args, **kwargs)

    monkeypatch.setattr(
        si_module, 'symmetric_isometry_recycle', counting_recycle)
    X, Y = si_refinement(r0, r1, X0, Y0, opts_svd, opts_si)

    assert recycle_calls == 2
    assert X.get_legs(1).tD == {(0,): 6}
    assert Y.get_legs(0).tD == {(0,): 6}
    assert np.allclose(X0_even, X[(0, 0)][:, :3])
    assert np.allclose(Yh0_even, Y.H[(0, 0)][:, :3])


def test_si_cwo_pipeline_clamps_rank_to_corner_capacity(config_kwargs,
                                                        monkeypatch):
    """Explicit per-sector-oversampling redistribution works while growing below chi + oversampling."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=34)
    r0, r1 = _biased_z2_corners(config)
    opts_svd = {'D_total': 10, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 4, 'niter': 2,
               'tol': 1e-12, 'redistribute_sectors': True, 'refinement': 'per_sector_oversampling'}
    calls = 0
    original = si_module.si_refinement

    def counting_cwo(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(si_module, 'si_refinement', counting_cwo)
    _, _, X, Y, _ = si_proj_corners(r0, r1, opts_svd, opts_si)

    assert calls == 1
    assert X.get_shape(axes=1) == 12
    assert Y.get_shape(axes=0) == 12
    assert X.get_legs(1).tD == {(0,): 6, (1,): 6}
    assert Y.get_legs(0).tD == {(0,): 6, (1,): 6}


def test_si_rejects_unknown_refinement_selector(config_kwargs):
    """An enabled redistribution accepts only the documented refinement names."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _biased_z2_corners(config)
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 1, 'redistribute_sectors': True,
               'refinement': 'unknown'}

    with pytest.raises(yastn.YastnError,
                       match="refinement='unknown' not recognized"):
        si_proj_corners(r0, r1, opts_svd, opts_si)


@pytest.mark.parametrize(('sym', 'spectra'), [
    ('Z2', {(0,): (12, 9, 6, 3, 2, 1),
            (1,): (11, 10, 8, 4, .5, .25)}),
    ('U1', {(-1,): (10, 9, 4, 3, 2, 1),
            (0,): (12, 7, 6, 5, .5, .25),
            (1,): (11, 8, 3, 2, 1, .1)}),
    ('U1xU1', {(0, 0): (12, 7, 6, 5, .5, .25),
               (1, 0): (11, 8, 3, 2, 1, .1),
               (0, 1): (10, 9, 4, 3, 2, 1)}),
])
def test_per_sector_oversampling_matches_full_svd_for_uneven_multisymmetry_spectra(
        config_kwargs, sym, spectra):
    """CWO finds the exact global sector allocation for all supported symmetries."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=61)
    r0, r1 = _corners_with_sector_spectra(config, spectra)
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 10, 'tol': 1e-13}
    charges = {_charge_tuple(q): 2 for q in spectra}
    initial_rank = sum(charges.values())
    # Start from a six-column basis, larger than the target chi+p=5.
    if initial_rank != 6:
        charges = {charge: 3 for charge in charges}
        initial_rank = 6
    assert initial_rank > opts_svd['D_total'] + opts_si['oversampling']
    X, Y = initialize_si_bases(
        r0, r1, rank=initial_rank, charges=charges)
    X, Y = si_refinement(r0, r1, X, Y, opts_svd, opts_si)
    charges = dict(X.get_legs(1).tD)
    refined_rank = X.get_shape(axes=1)
    assert refined_rank == opts_svd['D_total'] + opts_si['oversampling']
    assert len(charges) > 1
    assert charges == _global_sector_allocation(spectra, refined_rank)
    actual, _, _ = _si_sector_dimensions(
        r0, r1, X, Y, opts_svd, opts_si)
    assert len(actual) > 1
    assert actual == _full_sector_dimensions(r0, r1, opts_svd)


@pytest.mark.parametrize('refinement', ('per_sector_oversampling', 'adaptive_spectrum'))
@pytest.mark.parametrize('rank_option', ('D_total', 'D_block'))
@pytest.mark.parametrize('dtype', ('float64', 'complex128'))
def test_refinements_handle_single_dense_sector(
        config_kwargs, refinement, rank_option, dtype):
    """Each spectral refinement supports dense corners and SI rank options."""
    config = yastn.make_config(sym='none', default_dtype=dtype,
                               **config_kwargs)
    r0, r1 = _corners_with_sector_spectra(
        config, {(): (8., 6., 4., 2.)})
    if dtype == 'complex128':
        # Exercise complex arithmetic rather than merely complex storage.
        r0 = (1 + 0.25j) * r0
        r1 = (1 - 0.5j) * r1
    opts_svd = {rank_option: 2, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 4, 'tol': 1e-12,
               'refinement': refinement}
    X, Y = initialize_si_bases(r0, r1, rank=3)

    X, _ = si_refinement(r0, r1, X, Y, opts_svd, opts_si)

    assert X.get_legs(1).tD == {(): 3}


def _corner_pair_halves(config, contracted=(0, 1)):
    """Two halves, each as a pair of corners ``f0 @ f1`` over an inner leg.

    Charges the inner leg cannot reach the contracted leg through are
    annihilated by the product while remaining on the corner legs -- the case
    that a contracted half never shows. Restrict ``contracted`` for that.
    """
    external = yastn.Leg(config, s=1, t=(0, 1), D=(3, 3))
    inner = yastn.Leg(config, s=-1, t=(0, 1), D=(4, 4))
    closing = yastn.Leg(config, s=-1, t=contracted, D=(4,) * len(contracted))
    pair0 = (yastn.rand(config, legs=(external, inner)),
             yastn.rand(config, legs=(inner.conj(), closing)))
    pair1 = (yastn.rand(config, legs=(external.conj(), inner)),
             yastn.rand(config, legs=(inner.conj(), closing.conj())))
    return _Half(pair0), _Half(pair1)


@pytest.mark.parametrize('refinement', ('per_sector_oversampling', 'sector_dimensions'))
def test_si_refinement_accepts_corner_pairs(config_kwargs, refinement):
    """A half given as a pair of corners refines as its contracted product."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=35)
    pair0, pair1 = _corner_pair_halves(config)
    half0, half1 = pair0.contracted(), pair1.contracted()
    opts_svd = {'D_total': 3, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 8, 'tol': 1e-12,
               'refinement': refinement}
    charges = {(0,): 2, (1,): 2}

    X0, Y0 = initialize_si_bases(pair0, pair1, rank=4, charges=charges)
    X_pair, Y_pair = si_refinement(pair0, pair1, X0, Y0, opts_svd, opts_si)
    X_half, Y_half = si_refinement(half0, half1, X0, Y0, opts_svd, opts_si)

    assert X_pair.get_legs(1).tD == X_half.get_legs(1).tD
    assert Y_pair.get_legs(0).tD == Y_half.get_legs(0).tD
    # The pair also reaches the reduced spectrum of its contracted product.
    _, s_pair, _, _, _, _ = si_projector_svd(
        pair0, pair1, X_pair, Y_pair, opts_svd, opts_si)
    _, s_half, _, _, _, _ = si_projector_svd(
        half0, half1, X_half, Y_half, opts_svd, opts_si)
    assert np.allclose(np.sort(s_pair.to_numpy().diagonal()),
                       np.sort(s_half.to_numpy().diagonal()))


def test_si_corner_pairs_keep_recycled_bases(config_kwargs):
    """Bases stay compatible with a pair, so an unchanged allocation is reused."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=36)
    pair0, pair1 = _corner_pair_halves(config)
    opts_svd = {'D_total': 2, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 2, 'tol': 1e-12, 'refinement': 'sector_dimensions'}
    X0, Y0 = initialize_si_bases(pair0, pair1, rank=3)
    assert si_bases_compatible(pair0, pair1, X0, Y0)

    X, Y = si_refinement(pair0, pair1, X0, Y0, opts_svd, opts_si)

    assert X is X0 and Y is Y0


def test_si_corner_pair_drops_sectors_absent_from_the_product(config_kwargs):
    """Capacity of a pair follows its product, not the wider corner legs."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=37)
    # Nothing closes the odd sector, so the product has none.
    pair0, pair1 = _corner_pair_halves(config, contracted=(0,))
    half0, half1 = _Half(pair0.contracted()), _Half(pair1.contracted())

    assert pair0.f0.get_legs(0).tD == {(0,): 3, (1,): 3}
    assert pair0.get_legs(0) == half0.get_legs(0)
    assert (si_module._ctm_shared_sector_capacity(pair0, pair1)
            == si_module._ctm_shared_sector_capacity(half0, half1)
            == {(0,): 3})


@pytest.mark.parametrize('refinement', ('per_sector_oversampling', 'adaptive_spectrum', 'sector_dimensions'))
def test_refinement_reuses_unchanged_dense_bases(config_kwargs, refinement):
    """A redundant single-sector redistribution preserves recycled SI bases."""
    config = yastn.make_config(sym='none', **config_kwargs)
    r0, r1 = _corners_with_sector_spectra(
        config, {(): (8., 6., 4., 2.)})
    opts_svd = {'D_total': 2, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 2, 'tol': 1e-12,
               'refinement': refinement}
    X0, Y0 = initialize_si_bases(r0, r1, rank=3)

    X, Y = si_refinement(r0, r1, X0, Y0, opts_svd, opts_si)

    assert X is X0
    assert Y is Y0


def test_per_sector_oversampling_tied_boundary_preserves_valid_rank(config_kwargs):
    """Either valid allocation of equal boundary values retains both sectors."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    spectra = {(0,): (10., 5., 1.), (1,): (9., 5., 1.)}
    r0, r1 = _corners_with_sector_spectra(config, spectra)
    opts_svd = {'D_total': 2, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 4, 'tol': 1e-12}
    X, Y = initialize_si_bases(
        r0, r1, rank=3, charges={(0,): 2, (1,): 1})

    allocations = []
    for _ in range(3):
        X_refined, _ = si_refinement(
            r0, r1, X, Y, opts_svd, opts_si)
        allocations.append(dict(X_refined.get_legs(1).tD))

    for allocation in allocations:
        assert set(allocation) == {(0,), (1,)}
        assert sorted(allocation.values()) == [1, 2]
        assert sum(allocation.values()) == 3


@pytest.mark.parametrize('spectra', [
    {(0,): (4., 0., 0.), (1,): (3., 0., 0.)},
    {(0,): (0., 0., 0.), (1,): (0., 0., 0.)},
], ids=('rank_deficient', 'all_zero'))
def test_per_sector_oversampling_allocates_structurally_present_null_spectra(
        config_kwargs, spectra):
    """Zero values still represent available SI directions, not empty data."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _corners_with_sector_spectra(config, spectra)
    opts_svd = {'D_total': 3, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 4, 'tol': 1e-12}
    X, Y = initialize_si_bases(
        r0, r1, rank=4, charges={(0,): 2, (1,): 2})

    X, _ = si_refinement(r0, r1, X, Y, opts_svd, opts_si)
    charges = dict(X.get_legs(1).tD)

    assert charges == _global_sector_allocation(spectra, rank=4)
    assert sum(charges.values()) == 4


def test_per_sector_oversampling_rejects_empty_shared_spectrum(config_kwargs):
    """Corners without a shared external sector cannot yield CWO values."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    r0 = yastn.Tensor(config=config, s=(1, -1))
    r1 = yastn.Tensor(config=config, s=(-1, 1))
    r0.set_block(ts=(0, 0), Ds=(1, 1), val=1.)
    r1.set_block(ts=(1, 1), Ds=(1, 1), val=1.)

    with pytest.raises(yastn.YastnError,
                       match='Per-sector-oversampling refinement found no singular values'):
        si_refinement(
            r0, r1, None, None, {'D_total': 1}, {'oversampling': 0})


def test_per_sector_oversampling_validates_corner_pair_and_rank_options(config_kwargs):
    """CWO reports malformed corner input and an unspecified target rank."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _corners_with_sector_spectra(
        config, {(0,): (2., 1.), (1,): (2., 1.)})

    with pytest.raises(yastn.YastnError,
                       match='require an integer D_total or D_block'):
        si_refinement(
            r0, r1, None, None, {'tol': 0}, {'oversampling': 0})


# ---------------------------------------------------------------------------
# ASVR refinement and missing-sector recovery
# ---------------------------------------------------------------------------


def test_adaptive_spectrum_pipeline_recovers_globally_dominant_missing_sector(
        config_kwargs, monkeypatch):
    """The ASVR pipeline recovers a dominant absent sector and confirms it."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=32)
    r0, r1 = _biased_z2_corners(config)
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 2, 'niter': 8,
               'tol': 1e-12, 'redistribute_sectors': True,
               'adaptive_spectrum_iterations': 5, 'refinement': 'adaptive_spectrum'}

    X, Y = initialize_si_bases(r0, r1, rank=6,
                               charges={(1,): 6})
    # ASVR reads only the reduced spectrum while refining; the projectors are
    # built once, afterwards. Count both to see the whole pipeline.
    calls = 0
    original_spectrum = si_module._si_spectrum
    original_projector = si_module.si_projector_svd

    def counting(original):
        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)
        return counted

    monkeypatch.setattr(si_module, '_si_spectrum', counting(original_spectrum))
    monkeypatch.setattr(si_module, 'si_projector_svd',
                        counting(original_projector))
    _, _, X, Y, _ = si_proj_corners(r0, r1, opts_svd, opts_si, X=X, Y=Y)

    assert X.get_legs(1).tD == {(0,): 6}
    assert Y.get_legs(0).tD == {(0,): 6}
    # ASVR needs at least a changed estimate and a confirmation; floating-point
    # convergence may require more estimates before the final projector call.
    assert 3 <= calls <= opts_si['adaptive_spectrum_iterations'] + 1
    _assert_refined_si_spectrum(r0, r1, X, Y, opts_svd, opts_si)


def test_adaptive_spectrum_rejects_rank_too_small_to_probe_every_sector(config_kwargs):
    """ASVR reports when chi+p cannot represent every sector to compare."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    r0, r1 = _corners_with_sector_spectra(
        config, {(-1,): (6., 3.), (0,): (5., 2.), (1,): (4., 1.)})
    opts_svd = {'D_total': 1, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'niter': 2, 'tol': 1e-12,
               'refinement': 'adaptive_spectrum'}
    X, Y = initialize_si_bases(r0, r1, rank=2)

    with pytest.raises(yastn.YastnError,
                       match='cannot probe every shared charge sector'):
        si_refinement(r0, r1, X, Y, opts_svd, opts_si)


def _assert_allocation_within_capacity(X, r0, r1, chip):
    allocation = X.get_legs(1).tD
    capacity = si_module._ctm_shared_sector_capacity(_Half(r0), _Half(r1))
    assert all(dimension <= capacity[charge] for charge, dimension in allocation.items())
    assert sum(allocation.values()) == min(chip, sum(capacity.values()))


@pytest.mark.parametrize(('D_total', 'oversampling', 'expected'), [
    (20, 4, {(-2,): 2, (0,): 5, (2,): 2}),
    # The +2/-2 split of the overflow follows _charge_order (+q before -q).
    (6, 2, {(-2,): 1, (0,): 5, (2,): 2}),
], ids=('chip_above_capacity', 'chip_below_capacity'))
def test_adaptive_spectrum_caps_a_dominant_sector_at_its_capacity(
        config_kwargs, D_total, oversampling, expected):
    """The rank a full dominant sector cannot hold goes to the other sectors."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    config.backend.random_seed(seed=41)
    r0, r1 = _corners_with_sector_spectra(config, {
        (-2,): (.1, .05), (0,): (10., 9., 8., 7., 6.), (2,): (.1, .05)})
    opts_svd = {'D_total': D_total, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': oversampling, 'niter': 4,
               'tol': 1e-12, 'redistribute_sectors': True,
               'refinement': 'adaptive_spectrum'}

    _, _, X, Y, _ = si_proj_corners(r0, r1, opts_svd, opts_si)

    assert X.get_legs(1).tD == Y.get_legs(0).tD == expected
    _assert_allocation_within_capacity(X, r0, r1, D_total + oversampling)


def test_adaptive_spectrum_explores_with_rank_above_capacity(config_kwargs):
    """Seeding missing sectors uses every available direction, not chi + p."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    config.backend.random_seed(seed=42)
    r0, r1 = _corners_with_sector_spectra(config, {
        (-1,): (3., 1.), (0,): (4., 2.), (1,): (5., .5)})
    opts_svd = {'D_total': 8, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 2, 'niter': 4,
               'tol': 1e-12, 'redistribute_sectors': True,
               'refinement': 'adaptive_spectrum'}
    X, Y = initialize_si_bases(r0, r1, rank=4, charges={(0,): 2, (1,): 2})

    _, _, X, Y, _ = si_proj_corners(r0, r1, opts_svd, opts_si, X=X, Y=Y)

    assert X.get_legs(1).tD == {(-1,): 2, (0,): 2, (1,): 2}
    _assert_refined_si_spectrum(r0, r1, X, Y, opts_svd, opts_si)


@pytest.mark.parametrize('seed', range(6))
def test_adaptive_spectrum_respects_capacity_on_random_corners(config_kwargs, seed):
    """Proportional reallocation never exceeds what a sector can hold."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    config.backend.random_seed(seed=seed)
    external = yastn.Leg(config, s=1, t=(-1, 0, 1, 2), D=(1, 2, 3, 2))
    inner = yastn.Leg(config, s=-1, t=(-1, 0, 1, 2), D=(3, 3, 3, 3))
    r0 = yastn.rand(config, legs=(external, inner))
    r1 = yastn.rand(config, legs=(external.conj(), inner.conj()))
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 2, 'niter': 4,
               'tol': 1e-12, 'redistribute_sectors': True,
               'refinement': 'adaptive_spectrum'}

    _, _, X, Y, _ = si_proj_corners(r0, r1, opts_svd, opts_si)

    assert X.get_legs(1).tD == Y.get_legs(0).tD
    _assert_allocation_within_capacity(X, r0, r1, 6)


# ---------------------------------------------------------------------------
# RDS refinement and proportional rank allocation
# ---------------------------------------------------------------------------


def test_sector_dimensions_pipeline_apportions_relative_corner_sector_dimensions(
        config_kwargs):
    """The projector pipeline dispatches RDS proportional allocation."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(seed=33)
    r0, r1 = _corners_with_sector_spectra(
        config, {(0,): (2., 1.), (1,): (6., 5., 4., 3., 2., 1.)})
    opts_svd = {'D_total': 3, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 1,
               'niter': 2, 'tol': 1e-12, 'refinement': 'sector_dimensions'}
    X0, Y0 = initialize_si_bases(
        r0, r1, rank=4, charges={(0,): 2, (1,): 2})
    _, _, X, Y, _ = si_proj_corners(
        r0, r1, opts_svd, {**opts_si, 'redistribute_sectors': True}, X=X0, Y=Y0)
    assert X.get_legs(1).tD == {(0,): 1, (1,): 3}
    assert Y.get_legs(0).tD == {(0,): 1, (1,): 3}


@pytest.mark.parametrize(('sym', 'spectra', 'expected'), [
    ('Z2', {(0,): (4., 3.), (1,): (6., 5., 4., 3., 2., 1.)},
     {(0,): 1, (1,): 4}),
    ('U1', {(-1,): (2., 1.), (0,): (3., 2., 1.),
            (1,): (4., 3., 2., 1.)},
     {(-1,): 1, (0,): 2, (1,): 2}),
    ('U1xU1', {(0, 0): (2., 1.), (1, 0): (3., 2., 1.),
               (0, 1): (4., 3., 2., 1.)},
     {(0, 0): 1, (1, 0): 2, (0, 1): 2}),
])
def test_sector_dimensions_apportions_rank_for_supported_symmetries(
        config_kwargs, sym, spectra, expected):
    """RDS uses largest-remainder apportionment for block-sparse corners."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=63)
    r0, r1 = _corners_with_sector_spectra(config, spectra)
    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 1, 'refinement': 'sector_dimensions'}
    X0, Y0 = initialize_si_bases(r0, r1, rank=5)

    X, Y = si_refinement(r0, r1, X0, Y0, opts_svd, opts_si)

    assert X.get_legs(1).tD == expected
    assert Y.get_legs(0).tD == expected
    assert si_bases_compatible(r0, r1, X, Y)
    assert np.allclose((X.H @ X).to_numpy(), np.eye(5), atol=1e-12)
    assert np.allclose((Y @ Y.H).to_numpy(), np.eye(5), atol=1e-12)


def test_sector_dimensions_clamps_requested_rank_to_total_corner_capacity(config_kwargs):
    """RDS consumes all available directions when chi+p exceeds capacity."""
    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _corners_with_sector_spectra(
        config, {(0,): (5., 4.), (1,): (3., 2., 1.)})
    opts_svd = {'D_total': 8, 'tol': 0, 'fix_signs': True}
    opts_si = {'oversampling': 4, 'refinement': 'sector_dimensions'}
    X0, Y0 = initialize_si_bases(
        r0, r1, rank=2, charges={(0,): 1, (1,): 1})

    X, Y = si_refinement(r0, r1, X0, Y0, opts_svd, opts_si)

    assert X.get_shape(axes=1) == 5
    assert Y.get_shape(axes=0) == 5
    assert X.get_legs(1).tD == {(0,): 2, (1,): 3}
    assert Y.get_legs(0).tD == {(0,): 2, (1,): 3}


# ---------------------------------------------------------------------------
# Recycled-state compatibility and changing symmetry sectors
# ---------------------------------------------------------------------------


def test_si_incompatible_recycled_bases_are_rejected(config_kwargs):
    """Dimension and sector changes invalidate SI bases.

    Mismatched dtype or device is not checked here; contracting such bases
    with the corners fails on its own.
    """
    config = yastn.make_config(sym='Z2', **config_kwargs)
    r0, r1 = _biased_z2_corners(config)
    X, Y = initialize_si_bases(r0, r1, rank=4,
                               charges={(0,): 2, (1,): 2})
    assert si_bases_compatible(r0, r1, X, Y)

    # Bases are rejected if a corner's external-leg dimension changes.
    smaller_r1 = yastn.Tensor(config=config, s=(-1, 1))
    for charge in (0, 1):
        smaller_r1.set_block(ts=(charge, charge), Ds=(5, 6), val='rand')
    assert not si_bases_compatible(r0, smaller_r1, X, Y)

    # Bases are rejected if the available symmetry sectors change.
    one_sector_r1 = yastn.Tensor(config=config, s=(-1, 1))
    one_sector_r1.set_block(ts=(0, 0), Ds=(6, 6), val='rand')
    assert not si_bases_compatible(r0, one_sector_r1, X, Y)


def test_sector_change_redistribution_then_continued_recycling(config_kwargs):
    """A sector can leave and another enter after redistribution."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    initial = {(0,): (10, 9, 8, 7), (1,): (4, 3, 2, 1)}
    changed = {(0,): (4, 3, 2, 1), (1,): (10, 9, 8, 7)}
    r0, r1_initial = _corners_with_sector_spectra(config, initial)
    _, r1_changed = _corners_with_sector_spectra(config, changed)
    opts_svd = {'D_total': 3, 'tol': 0, 'fix_signs': True}
    # p=0 would leave the probe narrower than the sector, so the value deciding
    # the allocation would be its least-converged direction. On random seeds, 
    # the niter must be large enough to converge the least-converged direction to avoid a false positive.
    opts_si = {'oversampling': 1, 'niter': 10, 'tol': 1e-13}

    # This is the converged allocation before the dominant sector changes.
    X, Y = initialize_si_bases(r0, r1_initial, rank=3,
                               charges={(0,): 3})
    stale, _, _ = _si_sector_dimensions(
        r0, r1_changed, X, Y, opts_svd, opts_si)
    reference = _full_sector_dimensions(r0, r1_changed, opts_svd)
    assert stale != reference  # a stale basis cannot report a sector it has no probe in.

    X, Y = si_refinement(
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
    """Several conjugate sectors tied at the cutoff follow full-SVD allocation."""
    config = yastn.make_config(sym='U1', **config_kwargs)
    spectra = {
        (-2,): (12, 8, 4),
        (-1,): (14, 10, 6, 1),
        (0,): (16, 15, 13, 7, 5, .1),
        (1,): (14, 10, 6, 1),
        (2,): (12, 8, 4),
    }
    r0, r1 = _corners_with_sector_spectra(config, spectra)
    opts_svd = {'D_total': 6, 'tol': 0, 'fix_signs': True,
                'largest_gap': True}
    # The nominal cutoff bisects the degenerate pair of 12s in q=-2 and q=2.
    # Multiplet preservation moves it to the largest subsequent gap, between
    # 4 and 1, retaining 17 states with an intentionally uneven allocation.
    expected = {(-2,): 3, (-1,): 3, (0,): 5, (1,): 3, (2,): 3}
    full_basis = {charge: len(values) for charge, values in spectra.items()}
    full_rank = sum(full_basis.values())
    opts_si = {
        'oversampling': full_rank - opts_svd['D_total'],
        'niter': 10,
        'tol': 1e-13,
    }
    X, Y = initialize_si_bases(
        r0, r1, rank=full_rank, charges=full_basis)
    assert all(capacity < full_rank for capacity in full_basis.values())
    X, Y = si_refinement(r0, r1, X, Y, opts_svd, opts_si)
    charges = dict(X.get_legs(1).tD)
    assert charges == full_basis

    actual, X, Y = _si_sector_dimensions(
        r0, r1, X, Y, opts_svd, opts_si)
    reference = _full_sector_dimensions(r0, r1, opts_svd)
    assert actual == reference == expected
    assert sum(actual.values()) == 17 > opts_svd['D_total']
    assert actual[(-1,)] == actual[(1,)]
    assert actual[(-2,)] == actual[(2,)]
    assert si_bases_compatible(r0, r1, X, Y)


@pytest.mark.parametrize(('sym', 'spectra', 'policy', 'expected'), [
    ('none', {(): (10., 5., 5., 1.)},
     {'D_total': 2, 'eps_multiplet': 1e-12}, {(): 1}),
    ('U1', {(-1,): (9., 7.), (0,): (10.,), (1,): (8., 6.)},
     {'D_total': 4, 'hermitian': True},
     {(-1,): 1, (0,): 1, (1,): 1}),
], ids=('eps_multiplet', 'hermitian'))
def test_si_truncation_policies_match_full_svd(
        config_kwargs, sym, spectra, policy, expected):
    """AI-generated test: SI honors symmetry-aware truncation policies."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    r0, r1 = _corners_with_sector_spectra(config, spectra)
    full_rank = sum(len(values) for values in spectra.values())
    opts_svd = {'tol': 0, 'fix_signs': True, **policy}
    opts_si = {'oversampling': full_rank - opts_svd['D_total'],
               'niter': 0, 'tol': 0}
    X, Y = initialize_si_bases(r0, r1, rank=full_rank)

    actual, _, _ = _si_sector_dimensions(
        r0, r1, X, Y, opts_svd, opts_si)
    reference = _full_sector_dimensions(r0, r1, opts_svd)

    assert actual == reference == expected


def test_si_forwards_custom_truncation_mask(config_kwargs):
    """AI-generated test: SI invokes the same custom mask as full SVD."""
    config = yastn.make_config(sym='none', **config_kwargs)
    spectra = {(): (10., 8., 6., 4., 2.)}
    r0, r1 = _corners_with_sector_spectra(config, spectra)
    mask_calls = []

    def keep_two(spectrum):
        mask_calls.append(tuple(spectrum.get_shape()))
        return yastn.truncation_mask(spectrum, D_total=2)

    opts_svd = {'D_total': 4, 'tol': 0, 'fix_signs': True,
                'mask_f': keep_two}
    opts_si = {'oversampling': 1, 'niter': 0, 'tol': 0}
    X, Y = initialize_si_bases(r0, r1, rank=5)

    actual, _, _ = _si_sector_dimensions(
        r0, r1, X, Y, opts_svd, opts_si)
    assert mask_calls == [(5, 5)]
    reference = _full_sector_dimensions(r0, r1, opts_svd)

    assert mask_calls == [(5, 5), (5, 5)]
    assert actual == reference == {(): 2}


# ---------------------------------------------------------------------------
# Halves with equal external signatures
# ---------------------------------------------------------------------------


def _equal_signature_halves(config, charges, dimensions):
    """Halves whose external legs share a signature, as R factors of use_qr do.

    Charge q of ``r1`` then pairs with the conjugate charge of ``r0`` (-q for
    U1), and both legs are consistent under that pairing although not under
    equal labels.
    """
    inner = yastn.Leg(config, s=-1, t=charges, D=(4,) * len(charges))
    external0 = yastn.Leg(config, s=1, t=charges, D=dimensions)
    conjugate = tuple(config.sym.add_charges(q, signatures=(-1,)) for q in charges)
    external1 = yastn.Leg(config, s=1, t=conjugate, D=dimensions)
    r0 = yastn.rand(config, legs=(external0, inner))
    r1 = yastn.rand(config, legs=(external1, inner.conj()))
    return r0, r1


def _sorted_values(s):
    """All values of a diagonal tensor, sorted.

    For equal external signatures, SI labels the truncated bond by the charges
    of ``r1`` and the full SVD by those of ``r0``, which differ in sign, so
    spectra are compared without their labels.
    """
    return np.sort(s.to_numpy().diagonal())


_EQUAL_SIGNATURE_CASES = pytest.mark.parametrize(('sym', 'charges', 'dimensions', 'capacity'), [
    ('U1', ((-1,), (0,), (1,)), (1, 2, 3), {(-1,): 3, (0,): 2, (1,): 1}),
    ('U1', ((0,), (1,), (2,)), (2, 3, 4), {(-2,): 4, (-1,): 3, (0,): 2}),
    ('Z3', ((0,), (1,), (2,)), (1, 2, 3), {(0,): 1, (1,): 3, (2,): 2}),
    ('U1xU1', ((0, 0), (1, 0), (0, 1)), (1, 2, 3), {(-1, 0): 2, (0, -1): 3, (0, 0): 1}),
], ids=('U1_asymmetric_dimensions', 'U1_asymmetric_charges', 'Z3', 'U1xU1'))


@_EQUAL_SIGNATURE_CASES
def test_si_pairs_equal_signature_sectors_by_the_fusion_rule(
        config_kwargs, sym, charges, dimensions, capacity):
    """Every sector of the half product is available, and SI spans it exactly."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=43)
    r0, r1 = _equal_signature_halves(config, charges, dimensions)
    assert si_module._ctm_shared_sector_capacity(r0, r1) == capacity

    # SI bases with every available direction reproduce the full SVD.
    rank = sum(dimensions)
    opts_svd = {'D_total': rank, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 0, 'niter': 2, 'tol': 1e-12}
    _, _, X, Y, _ = si_proj_corners(r0, r1, opts_svd, opts_si)
    _, s_si, _, _, _, _ = si_projector_svd(r0, r1, X, Y, opts_svd, opts_si)
    rho = yastn.tensordot(r0, r1, axes=(1, 1))
    _, s_full, _ = rho.svd_with_truncation(axes=(0, 1), sU=r0.s[1], **opts_svd)

    assert X.get_legs(1).tD == capacity
    assert np.allclose(_sorted_values(s_si), _sorted_values(s_full),
                       rtol=1e-10, atol=1e-12)


@_EQUAL_SIGNATURE_CASES
@pytest.mark.parametrize('refinement', ('per_sector_oversampling',
                                        'adaptive_spectrum',
                                        'sector_dimensions'))
def test_si_refinements_handle_equal_signature_halves(
        config_kwargs, sym, charges, dimensions, capacity, refinement):
    """Redistribution allocates within the fusion-rule capacity of every sector."""
    config = yastn.make_config(sym=sym, **config_kwargs)
    config.backend.random_seed(seed=44)
    r0, r1 = _equal_signature_halves(config, charges, dimensions)
    opts_svd = {'D_total': 3, 'tol': 0, 'fix_signs': True}
    opts_si = {'enabled': True, 'oversampling': 2, 'niter': 4, 'tol': 1e-12,
               'redistribute_sectors': True, 'refinement': refinement}

    _, _, X, Y, _ = si_proj_corners(r0, r1, opts_svd, opts_si)

    assert si_bases_compatible(_Half(r0), _Half(r1), X, Y)
    allocation = X.get_legs(1).tD
    assert all(dimension <= capacity[charge] for charge, dimension in allocation.items())
    assert sum(allocation.values()) == 5
    if refinement == 'per_sector_oversampling':
        # Probing every sector finds the globally dominant spectrum.
        _, s_si, _, _, _, _ = si_projector_svd(r0, r1, X, Y, opts_svd, opts_si)
        rho = yastn.tensordot(r0, r1, axes=(1, 1))
        _, s_full, _ = rho.svd_with_truncation(axes=(0, 1), sU=r0.s[1], **opts_svd)
        assert np.allclose(_sorted_values(s_si), _sorted_values(s_full),
                           rtol=1e-10, atol=1e-12)
