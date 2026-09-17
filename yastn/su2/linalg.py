"""Linear-algebra helpers for reduced SU(2) tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..sym import sym_SU2

__all__ = ['MultipletMask', 'truncation_mask']


@dataclass(frozen=True)
class MultipletMask:
    """Selection masks for singular-value degeneracy spaces.

    ``masks[j][alpha]`` selects a *degeneracy* state.  Its dense contribution
    is always either zero or the full ``j + 1`` magnetic states, never a
    subset of them.
    """
    masks: dict[int, np.ndarray]
    kept_dimension: int
    discarded_weight: float


def truncation_mask(spectra: Mapping[int, np.ndarray], D_total=float('inf'),
                    D_block=float('inf'), tol=0.) -> MultipletMask:
    """Truncate SU(2) singular values without cutting a magnetic multiplet.

    Parameters
    ----------
    spectra
        Mapping ``two_j -> 1d singular values`` in the reduced degeneracy
        basis.  A selected value at ``two_j=j`` consumes ``j + 1`` states of
        the dense bond.
    D_total
        Maximum dense bond dimension.  Thus a spin-half multiplet costs two,
        a spin-one multiplet costs three, etc.
    D_block
        Maximum number of reduced degeneracy states in each irrep.
    tol
        Values no greater than ``tol * max(spectra)`` are discarded.
    """
    D_total = float(D_total)
    candidates, masks = [], {}
    maximum = max((float(np.max(np.abs(x))) for x in spectra.values() if len(x)), default=0.)
    for j, values in spectra.items():
        j = sym_SU2.validate_irrep(j)
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError('Each SU2 spectrum must be one-dimensional')
        masks[j] = np.zeros(len(values), dtype=bool)
        for alpha, value in enumerate(values):
            if abs(value) > tol * maximum:
                candidates.append((float(abs(value)), j, alpha))
    candidates.sort(reverse=True)
    used, n_per_block = 0, {j: 0 for j in masks}
    for _, j, alpha in candidates:
        cost = j + 1
        if used + cost <= D_total and n_per_block[j] < D_block:
            masks[j][alpha] = True
            used += cost
            n_per_block[j] += 1
    discarded = sum(float(np.sum(np.abs(np.asarray(values)[~masks[sym_SU2.validate_irrep(j)]]) ** 2))
                    for j, values in spectra.items())
    return MultipletMask(masks=masks, kept_dimension=used, discarded_weight=discarded)
