# Copyright 2026 The YASTN Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Manual comparison of SI and full-SVD charge-sector spectra.

These checks are not automated: the interesting output of
:func:`z2_si_sector_distribution` is a pair of singular-value plots that have
to be looked at. The module is therefore skipped under pytest and meant to be
run directly::

    python tests/peps/test_si_charge_sector_plots.py

which writes ``z2_si_singular_values.png`` into the working directory.
"""

import numpy as np
import pytest

import yastn
from yastn.tn.fpeps.envs._env_ctm_SI_projectors import (
    si_projector_svd,
    svd_charge_sector_dimensions,
)


pytestmark = pytest.mark.skip(
    reason="manual: results are singular-value plots requiring visual inspection")


def _random_matrix_for_sector_test(sym, sectors, seed):
    """Create a random block-diagonal matrix for executable tests below."""
    config = yastn.make_config(backend='np', sym=sym)
    config.backend.random_seed(seed)
    matrix = yastn.Tensor(config=config, s=(1, -1))
    for charge, dimension in sectors:
        kwargs = {} if charge is None else {'ts': (charge, charge)}
        matrix.set_block(Ds=(dimension, dimension), val='rand', **kwargs)
    return matrix


def _charge_counts_for_sector_test(matrix):
    _, singular_values, _ = matrix.svd(
        axes=(0, 1), sU=matrix.s[1], fix_signs=True)
    return svd_charge_sector_dimensions(singular_values)


class TestSvdChargeSectorDimensions:
    """Executable tests for :func:`svd_charge_sector_dimensions`."""

    @staticmethod
    def plot_z2_singular_values(s_ref, s_si, D_total, plot_path):
        """Plot full-SVD and SI singular values for both Z2 sectors."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, spectrum, title in zip(
                axes, (s_ref, s_si), ('Full SVD', 'SI')):
            sector_values = {}
            for charge in (0, 1):
                block = (charge, charge)
                if block not in spectrum.get_blocks_charge():
                    continue
                values = np.asarray(spectrum[block]).reshape(-1)
                values = np.sort(values)[::-1]
                sector_values[charge] = values
                ax.semilogy(range(1, len(values) + 1), values,
                            marker='.', label=f'charge {charge}')
            all_values = np.concatenate(tuple(sector_values.values()))
            if 0 < D_total <= len(all_values):
                cutoff = np.sort(all_values)[::-1][D_total - 1]
                ax.axhline(cutoff, color='black', linestyle='--', linewidth=1.5,
                           label=f'D_total cutoff ({cutoff:.3g})')
            ax.set_title(title)
            ax.set_xlabel('index within charge sector')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
        axes[0].set_ylabel('singular value')
        fig.suptitle(f'Z2 singular-value spectra (D_total={D_total})')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def test_dense(self):
        rho = _random_matrix_for_sector_test('none', ((None, 3),), seed=0)
        counts = _charge_counts_for_sector_test(rho)
        assert counts == {(): 3}, counts

    def test_u1(self):
        rho = _random_matrix_for_sector_test(
            'U1', ((-1, 2), (0, 3), (2, 1)), seed=1)
        counts = _charge_counts_for_sector_test(rho)
        assert counts == {(-1,): 2, (0,): 3, (2,): 1}, counts

    def test_z2(self):
        rho = _random_matrix_for_sector_test('Z2', ((0, 2), (1, 3)), seed=2)
        counts = _charge_counts_for_sector_test(rho)
        assert counts == {(0,): 2, (1,): 3}, counts

    @staticmethod
    def z2_si_sector_distribution(r0_sector_dims, r1_sector_dims,
                                  x_sector_dims, y_sector_dims,
                                  D_total=12, scale=1,
                                  distribution='random', plot_path=None):
        """Return SI/reference sector distributions and their projector error.

        ``scale`` can be a single number applied to both Z2 sectors or a
        ``{charge: factor}`` mapping used to bias their singular spectra.
        ``distribution`` controls the spectrum within each sector and accepts
        ``'random'``, ``'flat'``, ``'linear'``, ``'exponential'``,
        ``'powerlaw'``, a callable ``f(dimension, charge)``, or a
        ``{charge: distribution}`` mapping.
        """
        if r0_sector_dims != r1_sector_dims:
            raise ValueError("r0 and r1 must have matching Z2 sector dimensions.")
        if x_sector_dims != y_sector_dims:
            raise ValueError("X and Y must have matching SI sector dimensions.")

        sectors = tuple(sorted(r1_sector_dims.items()))
        config = _random_matrix_for_sector_test('Z2', sectors, seed=3).config
        rng = np.random.default_rng(3)
        r1 = yastn.Tensor(config=config, s=(1, -1))
        r0 = yastn.Tensor(config=config, s=(1, -1))
        for charge, dimension in sorted(r0_sector_dims.items()):
            block = (charge, charge)
            r0.set_block(ts=block, Ds=(dimension, dimension),
                         val=np.eye(dimension))
            factor = scale.get(charge, 1) if isinstance(scale, dict) else scale
            sector_distribution = (distribution[charge]
                                   if isinstance(distribution, dict)
                                   else distribution)
            if callable(sector_distribution):
                singular_values = np.asarray(
                    sector_distribution(dimension, charge))
            elif sector_distribution == 'flat':
                singular_values = np.ones(dimension)
            elif sector_distribution == 'linear':
                singular_values = np.linspace(1, 1e-2, dimension)
            elif sector_distribution == 'exponential':
                singular_values = np.geomspace(1, 1e-8, dimension)
            elif sector_distribution == 'powerlaw':
                singular_values = 1 / np.arange(1, dimension + 1)
            elif sector_distribution == 'random':
                singular_values = np.sort(rng.random(dimension))[::-1]
            else:
                raise ValueError(
                    f"Unknown singular-value distribution for charge "
                    f"{charge}: {sector_distribution!r}.")
            if singular_values.shape != (dimension,):
                raise ValueError(
                    "A custom distribution must return one value per dimension.")

            q_left, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
            q_right, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
            matrix = q_left @ np.diag(factor * singular_values) @ q_right.T
            r1.set_block(ts=block, Ds=matrix.shape, val=matrix)

        biased_rho = r0 @ r1
        x_leg = yastn.Leg(config, s=-1, t=tuple(sorted(x_sector_dims)),
                    D=tuple(x_sector_dims[q] for q in sorted(x_sector_dims)))
        y_leg = yastn.Leg(config, s=-1, t=tuple(sorted(y_sector_dims)),
                    D=tuple(y_sector_dims[q] for q in sorted(y_sector_dims)))

        def random_isometry(outer_leg, si_leg):
            basis = yastn.rand(config, legs=(outer_leg, si_leg))
            return yastn.qr(basis, axes=(0, 1), sQ=si_leg.s)[0]

        X = random_isometry(biased_rho.get_legs(1).conj(), x_leg)
        Yh = random_isometry(biased_rho.get_legs(0), y_leg)

        opts_svd = {'D_total': D_total, 'D_block': float('inf'), 'tol': 0}
        opts_si = {'niter': 100, 'tol': 1e-12}
        # ``si_projector_svd`` takes the second corner half in CTM order,
        # ``(outer, contracted)``, so that it forms tensordot(r0, r1, axes=(1, 1)).
        # ``r1`` is built here as ``(contracted, outer)`` for ``r0 @ r1``.
        u_si, s_si, v_si, _, _, s_si_all = si_projector_svd(
            r0, r1.T, X, Yh.H, opts_svd, opts_si, return_spectrum=True)
        _, s_ref_all, _ = biased_rho.svd(
            axes=(0, 1), sU=biased_rho.s[1], fix_signs=True)
        u_ref, s_ref, v_ref = biased_rho.svd_with_truncation(
            axes=(0, 1), sU=biased_rho.s[1], fix_signs=True, **opts_svd)

        si_counts = svd_charge_sector_dimensions(s_si)
        reference_counts = svd_charge_sector_dimensions(s_ref)

        left_error = (u_si @ u_si.H - u_ref @ u_ref.H).norm().item()
        right_error = (v_si.H @ v_si - v_ref.H @ v_ref).norm().item()
        error = max(left_error, right_error)

        if plot_path is not None:
            TestSvdChargeSectorDimensions.plot_z2_singular_values(
                s_ref_all, s_si_all, D_total, plot_path)

        return si_counts, reference_counts, error


    def test_rejects_nondiagonal(self):
        rho_u1 = _random_matrix_for_sector_test('U1', ((0, 2),), seed=4)
        try:
            svd_charge_sector_dimensions(rho_u1)
        except yastn.YastnError:
            pass
        else:
            raise AssertionError("A non-diagonal tensor should be rejected.")


def _run_manual_z2_comparison(plot_path='z2_si_singular_values.png'):
    """Compare SI against a full SVD on strongly biased Z2 sectors."""
    tests = TestSvdChargeSectorDimensions()
    tests.test_dense()
    tests.test_u1()
    tests.test_z2()
    tests.test_rejects_nondiagonal()

    r0_sector_dims = r1_sector_dims = {0: 240, 1: 320}
    x_sector_dims = y_sector_dims = {0: 12, 1: 12}
    si_counts, reference_counts, error = tests.z2_si_sector_distribution(
        r0_sector_dims, r1_sector_dims, x_sector_dims, y_sector_dims,
        D_total=12,
        scale={0: 10, 1: 1},
        distribution={0: 'powerlaw', 1: 'exponential'},
        plot_path=plot_path)
    print(f"si_counts={si_counts}, reference_counts={reference_counts}, "
          f"error={error}")
    print(f"wrote {plot_path}")
    return si_counts, reference_counts, error


if __name__ == '__main__':
    _run_manual_z2_comparison()
