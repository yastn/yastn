# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""
Recycled subspace-iteration (SI) projectors for CTMRG.

An SI update replaces the dense SVD of the enlarged-corner product
``A = r0 @ r1.T`` by a randomized/recycled subspace iteration: a pair of
column-isometric bases ``X`` (spanning the row space of ``r1``) and ``Y``
(spanning the row space of ``r0``) is carried across CTM sweeps, refined by a
few applications of ``A`` and ``A.H``, and only the small ``rho = Y A X`` is
decomposed and truncated.

This module is a leaf: it depends on the tensor layer only, never on the CTM
environment classes that call into it.
"""
from __future__ import annotations
import sys
from typing import NamedTuple

from ....initialize import rand, zeros, eye, block
from ....sym import sym_none
from ....tensor import Tensor, YastnError, Leg, LegMeta, diag, tensordot, qr, truncation_mask
from ....tensor._auxiliary import _struct
from ....tensor._contractions import _match_legs_tensordot
from ...._profile import nsys_profile, nvtx_range


#: Smallest singular value, relative to the largest, that the SI convergence
#: criterion will weigh fully, in addition to  pseudo-inverse ``cutoff``.
#  Below it a direction is determined only to about ``eps * s_1 / s_i`` in angle, becoming a roundoff error.
#:
#: The floor is relative in ``s``, because that is the scale of the weights:
#: an SI iteration applies ``A.H A``, so the ``|R_ii|`` of its ``QR`` carry the
#: squared singular values and the weights take their square root; see
#: :func:`si_weights_from_triangular`.  ``A.H A`` resolves a direction only
#: while ``(s_i / s_1) ** 2`` stays above an ulp, which puts the floor at
#: ``sqrt(eps)``.
SI_WEIGHT_FLOOR = sys.float_info.epsilon ** 0.5


class _Half:
    r"""CTM corner half, with legs ``(external, contracted)``.

    Hides whether a half is given as a single tensor or as a pair of enlarged
    corners whose product it is. ``_Half(r)`` returns the matching
    specialization, and passes an already built half through.

    Specializations provide ``get_legs(axis)``, ``config``, ``contracted()``,
    and multiplication of the half with a matrix: ``mm`` for ``self @ M``,
    together with ``mm_T``, ``mm_H`` and ``mm_conj`` for the transposed,
    hermitian-conjugated and conjugated half.
    """

    def __new__(cls, r):
        if isinstance(r, _Half):
            return r
        if cls is _Half:
            cls = _HalfTensor if isinstance(r, Tensor) else _HalfPair
        return super().__new__(cls)


class _HalfTensor(_Half):
    """Corner half that is already contracted into a single tensor."""

    def __init__(self, r):
        if r is self:  # __new__ passed an existing half through
            return
        self.tensor = r

    @property
    def config(self):
        return self.tensor.config

    def get_legs(self, axis):
        return self.tensor.get_legs(axis)

    def contracted(self):
        return self.tensor

    def mm(self, M):  # self @ M
        return tensordot(self.tensor, M, axes=(1, 0))

    def mm_T(self, M):  # self.T @ M
        return tensordot(self.tensor, M, axes=(0, 0))

    def mm_H(self, M):  # self.H @ M
        return tensordot(self.tensor.conj(), M, axes=(0, 0))

    def mm_conj(self, M):  # self.conj() @ M
        return tensordot(self.tensor.conj(), M, axes=(1, 0))


class _HalfPair(_Half):
    """Corner half given by a pair of enlarged corners, ``r = f0 @ f1``.

    The pair is applied corner by corner, so the half is not formed, which
    would cost ``O(N^3)`` for two ``N x N`` corners. Only :meth:`contracted`
    builds it.
    """

    def __init__(self, r):
        if r is self:  # __new__ passed an existing half through
            return
        self.f0, self.f1 = r
        self._legs = None
        self._contracted = None

    @property
    def config(self):
        return self.f0.config

    def get_legs(self, axis):
        r"""Leg of the product of the two corners.

        Sectors annihilated by the contraction are absent from the product,
        so they are dropped here as well -- SI columns in such sectors would
        be annihilated on the first application of the half.
        """
        if self._legs is None:
            self._legs = self._product_legs()
        return self._legs[axis]

    def _product_legs(self):
        corner_legs = (self.f0.get_legs(0), self.f1.get_legs(1))
        if self.f0.ndim_n != 2 or self.f1.ndim_n != 2:
            return corner_legs  # meta-fused corners: keep their raw legs
        # Native legs, ordered as the effective ones, so that a corner
        # transposed lazily is described without touching its data.
        structs = tuple(_struct(legs=tuple(f.struct.legs[i] for i in f.trans),
                                n=f.struct.n, isdiag=False)
                        for f in (self.f0, self.f1))
        # The last output carries the block structure of the product.
        bl_c = _match_legs_tensordot(self.config.sym, *structs, [0], [1], [0], [1])[-1]
        return tuple(self._with_charges(leg, basic)
                     for leg, basic in zip(corner_legs, bl_c.struct.legs))

    @staticmethod
    def _with_charges(leg, basic):
        """Corner leg restricted to the charge sectors of the product."""
        if leg.tD == basic.tD:
            return leg
        return Leg(leg.sym, s=basic.s, t=basic.t, D=basic.D, hf=leg.hf)

    def contracted(self):
        if self._contracted is None:
            self._contracted = self.f0 @ self.f1
        return self._contracted

    def mm(self, M):  # self @ M
        return tensordot(self.f0, tensordot(self.f1, M, axes=(1, 0)), axes=(1, 0))

    def mm_T(self, M):  # self.T @ M
        return tensordot(self.f1, tensordot(self.f0, M, axes=(0, 0)), axes=(0, 0))

    def mm_H(self, M):  # self.H @ M
        return tensordot(self.f1.conj(), tensordot(self.f0.conj(), M, axes=(0, 0)), axes=(0, 0))

    def mm_conj(self, M):  # self.conj() @ M
        return tensordot(self.f0.conj(), tensordot(self.f1.conj(), M, axes=(1, 0)), axes=(1, 0))


class SI_state(NamedTuple):
    r"""Recycling state of one SI projector pair.

    ``age`` counts how many times the pair has been updated with SI; 
    used by redistribution schedule :func:`redistribute_due`.
    ``niter`` and ``error`` describe the last update: the number of power
    updates made and the subspace error between the last two X, Y iterates. 
    ``bases`` records where the last update got its bases from:
    ``'reused'`` when the incoming pair was used as it came, ``'rebased'`` when
    it was carried onto changed corner row legs, see :func:`si_rebase_bases`,
    and ``'reinitialized'`` when it was discarded for a fresh random start.
    """
    age: int = 0
    niter: int = 0
    error: float = float('inf')
    rank: int = 0
    bases: str = 'reused'


def _si_rank(opts_svd, opts_si):
    """Total size of an SI basis, including oversampling."""
    oversampling = opts_si.get('oversampling', 5)
    D_total = opts_svd.get('D_total')
    if isinstance(D_total, int):
        return D_total + oversampling
    D_block = opts_svd.get('D_block')
    if isinstance(D_block, int):
        return D_block + oversampling
    raise YastnError("SI projectors require an integer D_total or D_block in opts_svd.")


def _si_truncation_rank(opts_svd):
    """Number of directions the truncation keeps, or ``None`` if not a plain rank.

    This is the boundary :func:`si_weights_from_triangular` clips at: every
    direction that survives truncation carries the projectors, while the
    oversampled tail beyond it does not.
    """
    for key in ('D_total', 'D_block'):
        value = opts_svd.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _charge_order(charge):
    """Sorting key placing charges as 0, +1, -1, +2, -2, ... .

    The tuple fallback applies the same convention component by component.
    """
    return tuple((abs(q), q < 0) for q in charge)


def _distribute_si_rank_with_capacity(capacities, rank):
    r"""Distribute an SI rank without exceeding CTM-leg sector capacities.

    The returned mapping defines the auxiliary leg used by both SI bases, so
    ``X`` and ``Y`` always contain exactly the same charges and the same number
    of vectors in every charge sector. The allocation always has the requested
    oversampled rank; insufficient corner-leg capacity is an error.
    """
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise YastnError("SI rank must be a positive integer.")

    capacities = {charge: dimension
                  for charge, dimension in capacities.items()
                  if dimension > 0}
    if not capacities:
        raise YastnError("The CTM corner leg has no non-empty charge sectors.")

    total_capacity = sum(capacities.values())
    if rank > total_capacity:
        raise YastnError(
            f"Requested SI rank {rank} exceeds CTM corner-leg capacity "
            f"{total_capacity}; cannot construct an auxiliary leg of "
            f"dimension chi + oversampling.")

    ordered_charges = sorted(capacities, key=_charge_order)
    allocation = {charge: 0 for charge in ordered_charges}
    remaining = rank

    # Round-robin allocation is even whenever capacities permit it. Once a
    # small sector is full, its remaining share is assigned to larger sectors.
    while remaining:
        for charge in ordered_charges:
            if allocation[charge] < capacities[charge]:
                allocation[charge] += 1
                remaining -= 1
                if remaining == 0:
                    break

    return {charge: dimension for charge, dimension in allocation.items()
            if dimension > 0}


def _validate_ctm_corner_pair(r0, r1):
    """Validate the two closures of a pair of CTM corner halves.

    The contracted legs are closed on each other, so they have to be
    contractible. The external legs only carry the SI bases, where matching
    dimensions of shared charges suffice; their fusion histories may differ,
    which invalidates recycled bases but not the corners.
    """
    for axis in (0, 1):
        leg0 = r0.get_legs(axis)
        leg1 = r1.get_legs(axis)
        tD0, tD1 = leg0.tD, leg1.tD
        consistent = leg0.are_consistent(leg1) if axis else \
            all(tD0[charge] == tD1[charge] for charge in tD0.keys() & tD1.keys())
        if not consistent: raise YastnError(
                "CTM corner halves must have matching dimensions in every "
                "shared charge sector on both loop closures; "
                f"mismatch on axis {axis}.")


def _ctm_shared_sector_capacity(r0, r1):
    """Return capacities of sectors supported by both CTM corner halves."""
    capacity0 = r0.get_legs(0).tD
    capacity1 = r1.get_legs(0).tD
    return {charge: dimension for charge, dimension in capacity0.items()
            if charge in capacity1}


def initialize_si_bases(r0, r1, rank, charges=None):
    r"""Initialize compatible column-isometric SI bases from Gaussian noise.

    The auxiliary rank is spread as uniformly as possible over charge sectors
    of the matching external CTM legs. A sector cannot be assigned more
    columns than that sector has rows.
    """
    _validate_ctm_corner_pair(r0, r1)

    x_input = r1.get_legs(0).conj()  # right leg of r1
    y_input = r0.get_legs(0)  # left leg of r0
    sector_capacity = _ctm_shared_sector_capacity(r0, r1)

    if charges is None:
        charge_mapping = _distribute_si_rank_with_capacity(
            sector_capacity, rank)
    else:
        charge_mapping = dict(charges)
        unknown_charges = set(charge_mapping) - set(sector_capacity)
        if unknown_charges:
            raise YastnError(
                f"SI charge sectors are absent from the CTM corner leg: "
                f"{unknown_charges}.")
        if any(not isinstance(dimension, int) or isinstance(dimension, bool)
               or dimension <= 0
               for dimension in charge_mapping.values()):
            raise YastnError("SI charge-sector dimensions must be positive integers.")
        if not charge_mapping:
            raise YastnError("SI charge-sector mapping cannot be empty.")
        mapped_rank = sum(charge_mapping.values())
        if mapped_rank != rank:
            raise YastnError(
                f"Explicit SI charge-sector dimensions sum to {mapped_rank}, "
                f"but requested SI rank is {rank}.")
        for charge, dimension in charge_mapping.items():
            capacity = sector_capacity[charge]
            if dimension > capacity:
                raise YastnError(
                    f"SI dimension {dimension} exceeds capacity {capacity} "
                    f"in charge sector {charge}.")
    x_aux = Leg(
        r1.config,
        s=-x_input.s,
        t=tuple(charge_mapping.keys()),
        D=tuple(charge_mapping.values()),
    )

    X = rand(r1.config, legs=(x_input, x_aux), distribution='normal')
    Yh = rand(r0.config, legs=(y_input, x_aux), distribution='normal')

    X, _ = qr(X, axes=(0, 1), sQ=x_aux.s)
    Yh, _ = qr(Yh, axes=(0, 1), sQ=x_aux.s)
    return X, Yh.H

def si_bases_compatible(r0, r1, X, Y):
    """Whether recycled bases are compatible with the current corners.

    Beyond matching charge-sector dimensions, the bases must be contractible
    with the corners: a hard-fused corner leg can keep its aggregate sector
    dimensions while its sub-leg dimensions or fusion history change.
    """
    if X is None or Y is None:
        return False

    leg0_r0 = r0.get_legs(0)
    leg0_r1 = r1.get_legs(0)

    def is_compatible_subspace(basis_leg, corner_leg):
        """A refined basis may intentionally contain only selected sectors."""
        return (basis_leg.s == corner_leg.s
                and all(charge in corner_leg.tD
                        and corner_leg.tD[charge] == dimension
                        for charge, dimension in basis_leg.tD.items()))

    try:
        return (
            is_compatible_subspace(X.get_legs(0), leg0_r1.conj())
            and is_compatible_subspace(Y.get_legs(1), leg0_r0.conj())
            and X.get_legs(1) == Y.get_legs(0).conj()
            and X.get_legs(0).are_consistent(leg0_r1)
            and Y.get_legs(1).are_consistent(leg0_r0)
        )
    except (AttributeError, IndexError):
        return False

def si_weights_from_triangular(R, rank=None, cutoff=0):
    r"""Significance of each SI direction, from the ``R`` of its ``QR``.

    ``R`` is the triangular factor of the power-iterated ``Q R = A.H A Q_old``.
    An iteration applies the operator ``A.H A`` and hence ``|R_ii|`` grows as 
    *squared* singular value of direction ``i``. The returned weights are ``sqrt(|R_ii|)``, 
    i.e. proportional to the singular value itself.

    ``rank`` -- the number of directions the truncation keeps -- normalizes to
    the *smallest retained* weight and clips above, so that every retained
    direction weighs one and only the oversampled tail is suppressed. Without
    it the weights are normalized to a largest weight of one instead, which
    makes the criterion blind at the truncation boundary: the error is formed as
    ``1 - N/D`` with ``N/D`` approaching one, so a direction is resolved only
    while ``(s_i/s_1)^2`` stays above an ulp, i.e. ``s_i/s_1 > sqrt(eps)``.  On a
    CTM half spanning nine or more decades that hides the very directions the
    ``s^-1/2`` of the projectors amplifies most.  That argument reaches down to
    ``sqrt(eps)`` and no further: below it ``A.H A`` no longer resolves the
    direction at all, so what ``s^-1/2`` amplifies there is roundoff, and
    :data:`SI_WEIGHT_FLOOR` rather than ``rank`` sets the boundary.

    ``cutoff`` is the pseudo-inverse cutoff the projectors will be built with.
    ``rsqrt`` zeroes every ``s_i <= cutoff``, so such a direction contributes
    nothing to ``p0``/``p1`` and must not hold the iteration up either; it is
    the natural floor for the normalization. ``|R_ii|`` carries the squared
    singular values, so ``sqrt(|R_ii|)`` is on the scale of ``s`` and compares
    with ``cutoff`` directly.

    The normalization additionally never drops below :data:`SI_WEIGHT_FLOOR`
    times the largest weight, so that a ``chi`` reaching past what double
    precision resolves does not hand the iteration count back to roundoff even
    when no ``cutoff`` is set.  This is the binding term whenever ``rank`` --
    which comes from the *requested* ``D_total``, not from the half's numerical
    rank -- reaches into the roundoff tail, as it does on any CTM half whose
    spectrum decays faster than ``chi``.  Without it, ``values[rank-1]`` lands
    at roundoff, every direction above clips to a weight of one, and the
    criterion floors out at the level at which those directions re-randomize.

    Returns ``None`` when ``R`` vanishes identically -- every sampled direction
    is then annihilated and nothing distinguishes them -- which leaves
    :func:`si_subspace_error` on its unweighted mean.
    """
    w = abs(diag(R.detach())).sqrt()
    scale = w.norm(p='inf').item()
    if scale == 0:
        return None
    if rank is not None:
        values = sorted((value
                         for sector in svd_charge_sector_values(w).values()
                         for value in sector), reverse=True)
        retained = max(values[min(rank, len(values)) - 1],
                       cutoff, SI_WEIGHT_FLOOR * scale)
        if retained > 0:
            return (w / retained).clip(a_max=1.)
    return w / scale


def si_subspace_error(Q, Q_old, weights=None):
    r"""Weighted mean squared sine of the angles between two SI bases.

    Both tensors are expected to be column-isometric. Without ``weights`` this
    is ``1 - ||Q_old.H @ Q||_F^2 / rank``, the plain mean over all directions.

    ``weights`` -- a diagonal tensor of per-direction significance, see
    :func:`si_weights_from_triangular` -- instead gives
    ``1 - ||Q_old.H @ Q @ w||_F^2 / ||w||^2``, that is ``sum_i w_i^2 sin^2(t_i)
    / sum_i w_i^2``, where ``t_i`` is the angle between direction ``i`` of ``Q``
    and the span of ``Q_old``.

    Weighing matters because SI oversamples on purpose: once ``chi + p`` exceeds
    the numerical rank of the halves, the surplus directions are annihilated and
    the ``QR`` refills them with arbitrary completions that roundoff re-randomizes
    every iteration. Unweighted, each of them contributes up to ``1 / rank`` to
    the error forever, so the error floors out well above any useful tolerance
    and the iteration count ends up decided by roundoff. Weighted, they are
    suppressed by ``w_i^2`` and the error reports the directions that carry the
    spectrum.

    Either form vanishes exactly when ``Q`` and ``Q_old`` span the same space,
    for any gauge within it: ``Q_old.H @ Q`` is then unitary, and
    ``||U @ w||_F = ||w||`` for unitary ``U``.
    """
    if Q_old is None or Q.get_legs() != Q_old.get_legs():
        return float('inf')

    overlap = Q_old.detach().H @ Q.detach()
    if weights is None:
        error = 1.0 - overlap.norm() ** 2 / Q.get_shape(axes=1)
    else:
        error = 1.0 - (overlap @ weights).norm() ** 2 / weights.norm() ** 2
    # Roundoff can put the result just outside the mathematical interval.
    return max(0.0, min(1.0, error.item()))


def svd_charge_sector_dimensions(s):
    r"""Return the number of singular values in every charge sector.

    Parameters
    ----------
    s : Tensor
        Diagonal singular-value tensor returned by :meth:`Tensor.svd`.

    Returns
    -------
    dict
        Mapping ``charge -> amount``. Charges are always tuples, including
        ``()`` for tensors without symmetry and ``(q,)`` for U(1).
    """
    if not isinstance(s, Tensor) or not s.isdiag or s.ndim != 2:
        raise YastnError("Expected a diagonal rank-2 singular-value tensor.")
    return dict(s.get_legs(0).tD)


def svd_charge_sector_values(s):
    r"""Return singular values grouped by symmetry-charge sector.

    Parameters
    ----------
    s : Tensor
        Diagonal singular-value tensor returned by :meth:`Tensor.svd`.

    Returns
    -------
    dict
        Mapping ``charge -> list of singular values``. Charges are tuples,
        including ``()`` without symmetry and ``(q,)`` for U(1).
    """
    if not isinstance(s, Tensor) or not s.isdiag or s.ndim != 2:
        raise YastnError("Expected a diagonal rank-2 singular-value tensor.")

    return {
        charge: s[charge + charge].tolist()
        for charge in s.get_legs(0).t
    }


def _distribute_si_rank_proportionally(sector_weights, rank):
    """Distribute ``rank`` proportionally to nonnegative sector weights."""
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise YastnError("SI rank must be a positive integer.")
    if not sector_weights:
        raise YastnError("Cannot distribute SI rank without charge sectors.")
    if any(weight < 0 for weight in sector_weights.values()):
        raise YastnError("Charge-sector weights must be nonnegative.")

    total_weight = sum(sector_weights.values())
    if total_weight == 0:
        # Callers weight sectors by leg dimensions or by counts of dominant
        # singular values, both of which are positive in at least one sector.
        raise YastnError("Charge-sector weights cannot all be zero.")

    charge_mapping = {}
    fractional_numerators = {}
    for charge, weight in sector_weights.items():
        dimension, fractional_numerator = divmod(rank * weight,
                                                 total_weight)
        charge_mapping[charge] = dimension
        fractional_numerators[charge] = fractional_numerator

    remainder = rank - sum(charge_mapping.values())
    remainder_order = sorted(
        fractional_numerators,
        key=fractional_numerators.get,
        reverse=True)
    for charge in remainder_order[:remainder]:
        charge_mapping[charge] += 1

    return {charge: dimension for charge, dimension in charge_mapping.items()
            if dimension > 0}


def _si_refinement_adaptive_spectrum(r0, r1, X, Y, opts_svd, opts_si):
    """Return a stable SI charge mapping estimated from dominant spectra."""
    iterations = opts_si.get('adaptive_spectrum_iterations', opts_si.get('asvr_iterations', 5))
    chip = _si_rank(opts_svd, opts_si)
    sector_capacity = _ctm_shared_sector_capacity(r0, r1)
    charge_mapping = dict(X.get_legs(1).tD)

    # A symmetry-preserving subspace iteration cannot generate a charge sector
    # absent from its input bases. Seed every sector shared by both corners so
    # that adaptive-spectrum refinement can compare their spectra before refining the allocation.
    missing_charges = set(sector_capacity) - set(charge_mapping)
    if missing_charges:
        exploratory_mapping = _distribute_si_rank_with_capacity(
            sector_capacity, chip)
        unexplored_charges = set(sector_capacity) - set(exploratory_mapping)
        if unexplored_charges:
            raise YastnError(
                "Adaptive-spectrum refinement cannot probe every shared "
                "charge sector with SI rank "
                f"{chip}; increase D_total/D_block or oversampling. Missing "
                f"sectors: {unexplored_charges}.")
        charge_mapping = exploratory_mapping
        X, Y = _recycle_si_bases(
            r0, r1, X, Y, charge_mapping)

    for asvr_iteration in range(iterations):
        sall = _si_spectrum(r0, r1, X, Y, opts_si)
        sector_values = svd_charge_sector_values(sall)
        # Keep values above the largest per-sector floor so dominant sectors
        # receive more columns in the next allocation.
        largest_smallest_sector_value = max(values[-1]
                                            for values in sector_values.values())
        sector_dominant_values = {
            charge: sum(value >= largest_smallest_sector_value for value in values)
            for charge, values in sector_values.items()
        }
        refined_mapping = _distribute_si_rank_proportionally(
            sector_dominant_values, chip)
        if refined_mapping == charge_mapping:
            break
        charge_mapping = refined_mapping
        if asvr_iteration + 1 < iterations:
            X, Y = _recycle_si_bases(
                r0, r1, X, Y, charge_mapping)
    return charge_mapping


def _si_refinement_sector_dimensions(r0, r1, X, Y, opts_svd, opts_si):
    r"""Allocate SI rank from the relative sizes of CTM charge sectors.

    The auxiliary rank is distributed proportionally to the dimensions of the
    charge sectors shared by the external legs of ``r0`` and ``r1``. Integer
    dimensions are obtained by largest-remainder apportionment, and the result
    never exceeds a sector's capacity. ``X`` and ``Y`` are accepted to provide
    the same call signature as the other SI refinement methods.
    """
    sector_capacity = _ctm_shared_sector_capacity(r0, r1)
    rank = min(_si_rank(opts_svd, opts_si), sum(sector_capacity.values()))
    charge_mapping = _distribute_si_rank_proportionally(
        sector_capacity, rank)
    return charge_mapping


def _si_refinement_per_sector_oversampling(r0, r1, X, Y, opts_svd, opts_si):
    """Return an SI charge mapping estimated by per-sector oversampling."""
    chip = _si_rank(opts_svd, opts_si)
    oversampled_sector_values = {}
    for charge, capacity in _ctm_shared_sector_capacity(r0, r1).items():
        sector_rank = min(chip, capacity)
        X_charge, Y_charge = initialize_si_bases(
            r0, r1, sector_rank, charges={charge: sector_rank})
        sall = _si_spectrum(r0, r1, X_charge, Y_charge, opts_si)
        sector_values = list(
            svd_charge_sector_values(sall).get(charge, ()))
        # An exactly zero sector can be omitted from the block structure of
        # the reduced SVD even though its directions remain available on the
        # CTM legs. Preserve those structural null directions for allocation.
        sector_values.extend([0.] * (sector_rank - len(sector_values)))
        oversampled_sector_values[charge] = sector_values

    top_values = sorted(
        ((value, charge) for charge, values in oversampled_sector_values.items()
         for value in values),
        key=lambda item: item[0], reverse=True)[:chip]
    charge_mapping = {}
    for _, charge in top_values:
        charge_mapping[charge] = charge_mapping.get(charge, 0) + 1
    if not charge_mapping:
        raise YastnError(
            "Per-sector-oversampling refinement found no singular values.")
    return charge_mapping

@nsys_profile("si_refinement")
def si_refinement(r0, r1, X, Y, opts_svd, opts_si):
    r"""Refine and recycle SI bases with the selected allocation strategy.

    This is the single dispatch point for SI charge-sector refinement. Each
    strategy returns a charge mapping; basis resizing is centralized here so
    every method retains compatible columns in its public ``(X, Y)`` result.

    Each half is either a tensor or a pair of enlarged corners; see :class:`_Half`.
    """
    r0, r1 = _Half(r0), _Half(r1)
    _validate_ctm_corner_pair(r0, r1)
    refinement = opts_si.get('refinement', 'per_sector_oversampling')
    refinements = {
        'per_sector_oversampling': _si_refinement_per_sector_oversampling,
        'adaptive_spectrum': _si_refinement_adaptive_spectrum,
        'sector_dimensions': _si_refinement_sector_dimensions,
        # The former acronyms, kept so that a stored option still dispatches.
        'cwo': _si_refinement_per_sector_oversampling,
        'asvr': _si_refinement_adaptive_spectrum,
        'rds': _si_refinement_sector_dimensions,
    }
    try:
        refine = refinements[refinement]
    except KeyError:
        raise YastnError(
            "Unknown SI refinement method "
            f"{refinement!r}; expected 'per_sector_oversampling', "
            "'adaptive_spectrum', or 'sector_dimensions'.") from None

    sector_capacity = _ctm_shared_sector_capacity(r0, r1)
    target_rank = min(_si_rank(opts_svd, opts_si),
                      sum(sector_capacity.values()))
    reusable = X is not None and Y is not None
    if reusable:
        current_mapping_x = dict(X.get_legs(1).tD)
        current_mapping_y = dict(Y.get_legs(0).tD)

    # There is no allocation decision to make for a single charge sector.
    # Keeping an already correctly sized basis avoids both the refinement work
    # and an unnecessary random restart of a useful recycled subspace.
    if reusable and len(sector_capacity) == 1:
        charge = next(iter(sector_capacity))
        target_mapping = {charge: target_rank}
        if (current_mapping_x == target_mapping
                and current_mapping_y == target_mapping):
            return X, Y

    charge_mapping = refine(r0, r1, X, Y, opts_svd, opts_si)
    if (reusable and current_mapping_x == charge_mapping
            and current_mapping_y == charge_mapping):
        return X, Y
    return _recycle_si_bases(r0, r1, X, Y, charge_mapping)


def _validate_isometry(V):
    """Validate and return the two legs of a resizable isometry."""
    if not isinstance(V, Tensor) or V.ndim != 2 or V.ndim_n != 2 or V.isdiag:
        raise YastnError("Expected a non-diagonal rank-2 isometry.")
    legs = V.get_legs()
    if not all(isinstance(leg, Leg) for leg in legs):
        raise YastnError("Isometry resizing does not support meta-fused legs.")
    if legs[1].is_fused():
        raise YastnError(
            "The resized isometry leg must not have hard-fusion history.")
    return legs


def _validate_dense_isometry(V, dim):
    """Validate a dense, single-block isometry and a requested new dimension.

    Returns its two legs and its current number of columns.
    """
    left_leg, right_leg = _validate_isometry(V)
    if (V.config.sym.NSYM != 0 or len(right_leg.tD) != 1
            or V.nblocks != 1):
        raise YastnError(
            "Expected a dense isometry with one charge block; use "
            "symmetric_isometry_recycle for a symmetric isometry.")
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise YastnError("dim must be a positive integer.")
    return left_leg, right_leg, next(iter(right_leg.tD.values()))


def _validated_isometry_result(V, resized, dimension):
    """Validate and return a resized dense isometry."""
    expected_shape = (V.get_shape(axes=0), dimension)
    if resized.get_shape() != expected_shape:
        raise YastnError(
            f"Failed to construct isometry with shape {expected_shape}.")
    return resized


def _isometry_blocks_by_right_charge(V):
    """Map each right-leg charge to its tensor block charge and shape."""
    nsym = V.config.sym.NSYM
    return {
        tuple(block_charge[-nsym:]) if nsym else (): (block_charge, shape)
        for block_charge, shape in zip(V.get_blocks_charge(),
                                       V.get_blocks_shape())
    }


def _dense_isometry_sector(V, shape, block_charge=None):
    """Create a dense copy or random isometry for one symmetry sector."""
    config = V.config._replace(sym=sym_none)
    legs = tuple(Leg(config, s=leg.s, D=(dimension,))
                 for leg, dimension in zip(V.get_legs(), shape))
    if block_charge is None:
        sector = rand(
            config, legs=legs, n=(), distribution='normal',
            dtype=V.yastn_dtype, device=V.device)
        return qr(sector, axes=(0, 1), sQ=legs[1].s)[0]

    sector = zeros(
        config, legs=legs, n=(), dtype=V.yastn_dtype, device=V.device)
    sector[()] = V[block_charge]
    return sector


def _isometry_target_leg(V, charges):
    """Create the right leg specified by a charge-to-dimension mapping."""
    try:
        charge_dimensions = dict(charges)
    except (TypeError, ValueError):
        raise YastnError(
            "charges must be a mapping from charge to target dimension.") from None

    if any(not isinstance(dimension, int) or isinstance(dimension, bool)
           or dimension < 0 for dimension in charge_dimensions.values()):
        raise YastnError(
            "Isometry charge-sector dimensions must be nonnegative integers.")
    charge_dimensions = {charge: dimension
                         for charge, dimension in charge_dimensions.items()
                         if dimension > 0}
    if not charge_dimensions:
        raise YastnError("The resized isometry must have at least one column.")

    right = V.get_legs(1)
    try:
        leg = Leg(V.config, s=1,
                  t=tuple(charge_dimensions),
                  D=tuple(charge_dimensions.values()))
        return leg if right.s == 1 else leg.conj()
    except YastnError as error:
        raise YastnError(f"Invalid isometry charge distribution: {error}") from None


def isometry_expansion(V, dim, u=None):
    r"""Expand the right leg of a dense, single-block isometry.

    Existing columns are retained. New columns are obtained by projecting
    ``u`` (or Gaussian noise when ``u`` is omitted) away from the retained
    subspace and orthonormalizing the residual. ``u`` can contain one or
    several candidate columns, but it must contain exactly ``dim - V.shape[1]``
    columns.

    Parameters
    ----------
    V : Tensor
        Rank-2 column isometry to expand.
    dim : int
        Target number of columns.
    u : Tensor, optional
        Candidate matrix for the additional columns.
    """
    left_leg, right_leg, current_dimension = _validate_dense_isometry(V, dim)
    if dim <= current_dimension:
        raise YastnError(
            f"Cannot expand from {current_dimension} to dimension {dim}; "
            "the target dimension must be larger.")
    row_dimension = sum(left_leg.D)
    if dim > row_dimension:
        raise YastnError(
            f"Isometry dimension {dim} exceeds row-space capacity "
            f"{row_dimension}.")

    added_dimension = dim - current_dimension
    added_leg = Leg(V.config, s=right_leg.s, D=(added_dimension,))
    if u is None:
        addition = rand(
            V.config, legs=(left_leg, added_leg), n=V.n,
            distribution='normal', dtype=V.yastn_dtype, device=V.device)
    elif (not isinstance(u, Tensor) or u.ndim != 2 or u.ndim_n != 2
            or u.isdiag or u.n != V.n
            or u.get_legs(0) != V.get_legs(0)
            or u.get_legs(1).s != added_leg.s
            or u.get_legs(1).tD != added_leg.tD
            or u.dtype != V.dtype or u.device != V.device):
        raise YastnError(
            "u must be a matrix with the isometry's left leg and exactly "
            f"{added_dimension} additional columns.")
    else:
        addition = u

    addition = addition - V @ (V.H @ addition)
    # Reorthogonalize to limit roundoff when the candidate columns have a
    # large component in the span of V.
    addition = addition - V @ (V.H @ addition)
    addition, _ = qr(addition, axes=(0, 1), sQ=right_leg.s)
    if addition.get_shape(axes=1) != added_dimension:
        raise YastnError(
            "The requested expansion does not fit in the available row space.")

    expanded = block({(0,): V, (1,): addition},
                     common_legs=(0,)).drop_leg_history(axes=1)
    return _validated_isometry_result(V, expanded, dim)


def isometry_shrinkage(V, dim):
    r"""Shrink a dense, single-block isometry to its first ``dim`` columns.

    Parameters
    ----------
    V : Tensor
        Rank-2 column isometry to shrink.
    dim : int
        Target number of columns.
    """
    _, right_leg, current_dimension = _validate_dense_isometry(V, dim)
    if dim > current_dimension:
        raise YastnError(
            f"Cannot shrink from {current_dimension} to dimension {dim}; "
            "the target dimension cannot be larger.")
    if dim == current_dimension:
        return V

    shrunk_leg = Leg(V.config, s=right_leg.s, D=(dim,))
    selector = eye(
        V.config, legs=(right_leg.conj(), shrunk_leg), isdiag=False,
        dtype=V.yastn_dtype, device=V.device)
    return _validated_isometry_result(V, V @ selector, dim)


def _elementary_sublegs(leg):
    """Elementary legs a hard-fused leg is a product of, in order.

    Mirrors the recursion of :func:`yastn.eye` over fused legs, so the pairs it
    forms are exactly the ones the embedding is built from.
    """
    if not leg.is_fused():
        return (leg,)
    return tuple(elementary
                 for sub in leg.unfuse_leg()
                 for elementary in _elementary_sublegs(sub))


def _rows_embeddable(source, target):
    r"""Whether ``source`` embeds into ``target`` sub-leg by sub-leg.

    Only a hard-fused row leg qualifies. Its sub-legs are the CTM boundary bond
    and the double-layer PEPS bond, so the embedding pads the boundary factor
    and leaves the PEPS factor alone -- a correspondence that survives a change
    of ``chi``. A leg without that structure, such as the ``R``-factor leg the
    ``use_qr`` halves carry, only has an ordering induced by its own
    factorization, which does not relate two different dimensions.
    """
    if isinstance(source, LegMeta) or isinstance(target, LegMeta):
        return False  # eye() does not support meta-fused legs
    if not (source.is_fused() and target.is_fused()):
        return False
    if not (source.s == target.s
            and source.hf.tree == target.hf.tree
            and source.hf.op == target.hf.op
            and source.hf.s == target.hf.s):
        return False
    # The embedding can only produce charges the bases already carry, so a
    # sub-leg that gained one would leave the fused row leg short of the
    # corner's sector and the pair uncontractible.  Such a corner is left to a
    # fresh start, which explores the new sector from the outset.
    return all(set(tgt.tD) <= set(src.tD)
               for src, tgt in zip(_elementary_sublegs(source),
                                   _elementary_sublegs(target)))


def _rows_embedding_is_injective(source, target):
    """Whether every sector of ``source`` fits in ``target``.

    The embedding is a product of the elementary ones, so it is injective --
    plain zero-padding, which leaves columns orthonormal -- exactly when every
    factor is.
    """
    for src, tgt in zip(_elementary_sublegs(source), _elementary_sublegs(target)):
        tD = tgt.tD
        if any(charge not in tD or tD[charge] < dimension
               for charge, dimension in src.tD.items()):
            return False
    return True


def isometry_rebase_rows(V, target_leg):
    r"""Re-express a column isometry on a row leg of changed dimensions.

    Leading rows of every elementary charge sector are kept. Sectors that grow
    are zero-padded, which is exact: the columns stay orthonormal. Sectors that
    shrink or disappear cost the columns their norm, so the result is
    re-orthonormalized; the ``QR`` also completes directions the embedding
    annihilated, which is what a fresh random column would have provided.

    Returns ``None`` when the two legs are not related by such an embedding.
    """
    source = V.get_legs(0)
    if source == target_leg:
        return V
    if not _rows_embeddable(source, target_leg):
        return None
    embedding = eye(
        V.config, legs=(source.conj(), target_leg), isdiag=False,
        dtype=V.yastn_dtype, device=V.device)
    W = tensordot(embedding, V, axes=(0, 0))  # legs (target_leg, auxiliary)
    if not _rows_embedding_is_injective(source, target_leg):
        W, _ = qr(W, axes=(0, 1), sQ=V.get_legs(1).s)
    return W


def si_rebase_bases(r0, r1, X, Y):
    r"""Carry SI bases onto corner row legs of changed dimensions.

    The CTM row space grows over the first sweeps -- ``chi * D^2`` follows a
    ``chi`` that has not saturated yet -- and charge-sector redistribution moves
    it again later on. Both leave the recycled bases describing the right
    subspace in the wrong space. Embedding them costs one sparse contraction
    and keeps that description, where
    :func:`initialize_si_bases` would replace it with noise.

    The targets mirror :func:`initialize_si_bases`: ``X`` sits on the conjugate
    of the external leg of ``r1``, and ``Y.H`` on the external leg of ``r0``.
    Both have to rebase, since the two share an auxiliary leg; otherwise
    ``(None, None)`` is returned, and so it is for corners the embedding does
    not apply to, see :func:`_rows_embeddable`.
    """
    if X is None or Y is None:
        return None, None
    try:
        X_new = isometry_rebase_rows(X, r1.get_legs(0).conj())
        Yh_new = isometry_rebase_rows(Y.H, r0.get_legs(0))
    except YastnError:
        return None, None
    if X_new is None or Yh_new is None:
        return None, None
    return X_new, Yh_new.H


def symmetric_isometry_recycle(V, charges, left_leg=None):
    r"""Resize charge sectors on the right leg of an isometry.

    ``charges`` maps each right-leg charge to its *target* dimension. Sectors
    omitted from the mapping (or assigned zero) are removed. In each retained
    sector, leading columns of ``V`` are kept; sectors that grow are completed
    with orthonormal random columns. Row sectors paired with removed right-leg
    sectors are dropped by YASTN's canonical block-sparse representation.

    ``left_leg`` can supply a compatible expanded row space. This is useful
    when a previously removed charge sector has to be introduced again: the
    old basis no longer carries that row sector, but the current CTM corner
    does. Existing sectors must have the same row dimensions in both legs.

    """
    current_left_leg, right_leg = _validate_isometry(V)
    if left_leg is None:
        left_leg = current_left_leg
    elif not isinstance(left_leg, Leg):
        raise YastnError("left_leg must be a YASTN Leg.")

    common_charges = current_left_leg.tD.keys() & left_leg.tD.keys()
    if (current_left_leg.s != left_leg.s
            or any(current_left_leg.tD[charge] != left_leg.tD[charge]
                   for charge in common_charges)):
        raise YastnError(
            "left_leg must preserve dimensions of existing row sectors.")
    target_leg = _isometry_target_leg(V, charges)
    current_dimensions = right_leg.tD
    target_dimensions = target_leg.tD
    if current_dimensions == target_dimensions:
        return V

    current_blocks = _isometry_blocks_by_right_charge(V)
    probe = zeros(
        V.config, legs=(left_leg, target_leg), n=V.n,
        dtype=V.yastn_dtype, device=V.device)
    target_blocks = _isometry_blocks_by_right_charge(probe)

    for charge, dimension in target_dimensions.items():
        capacity = target_blocks.get(charge, (None, (0, 0)))[1][0]
        if dimension > capacity:
            raise YastnError(
                f"Isometry dimension {dimension} exceeds row-space capacity "
                f"{capacity} in charge sector {charge}.")

    resized = probe
    for charge, target_dimension in target_dimensions.items():
        target_block_charge, target_shape = target_blocks[charge]

        if charge not in current_blocks:
            # There is no zero-column tensor to pass to isometry_expansion,
            # so a newly introduced sector is initialized as an isometry.
            sector = _dense_isometry_sector(V, target_shape)
        else:
            current_block_charge, current_shape = current_blocks[charge]
            sector = _dense_isometry_sector(
                V, current_shape, current_block_charge)

            current_dimension = current_dimensions[charge]
            if target_dimension > current_dimension:
                sector = isometry_expansion(sector, target_dimension)
            elif target_dimension < current_dimension:
                sector = isometry_shrinkage(sector, target_dimension)

        resized[target_block_charge] = sector.to_raw_tensor()

    if resized.get_legs(1).tD != target_dimensions:
        raise YastnError("Failed to construct the requested charge distribution.")
    return resized


def _recycle_si_bases(r0, r1, X, Y, charge_mapping):
    """Resize both SI bases while preserving their compatible columns."""
    rank = sum(charge_mapping.values())
    if not si_bases_compatible(r0, r1, X, Y):
        return initialize_si_bases(
            r0, r1, rank, charges=charge_mapping)

    X = symmetric_isometry_recycle(
        X, charge_mapping, left_leg=r1.get_legs(0).conj())
    Yh = symmetric_isometry_recycle(
        Y.H, charge_mapping, left_leg=r0.get_legs(0))
    return X, Yh.H


@nsys_profile("_si_reduced_svd")
def _si_reduced_svd(r0, r1, X, Y, opts_si, spec_only=False, rank=None, cutoff=0):
    r"""Subspace-iterate the bases and  ``rho = Y A X``.

    Halves ``r0`` and ``r1`` are :class:`_Half` instances, built by the caller.

    Returns the converged bases together with the decomposition
    ``us, sall, vs`` of ``rho``, followed by an ``info`` dictionary holding the
    number of power updates performed (``niter``, at most the ``niter`` budget
    of ``opts_si``) and the subspace ``error`` they reached. 
    """
    _validate_ctm_corner_pair(r0, r1)
    niter = opts_si.get('niter', 5)
    tol = opts_si.get('tol', 1e-3)
    X_old, Yh_old = X, Y.H
    # With niter=0 the bases are used as they come in; no update is made and no
    # subspace error is available.
    n_iter, error = 0, float('inf')

    with nvtx_range("_si_reduced_svd SI"):
        for _ in range(niter):
            # A = r0 @ r1.T is applied as r0.mm(r1.mm_T(.)), and A.H as r1.mm_conj(r0.mm_H(.)).
            AX = r0.mm(r1.mm_T(X))
            X_next = r1.mm(r0.mm_T(AX.conj())).conj() # r1.mm_conj(r0.mm_H(AX))
            X, Rx = qr(X_next, axes=(0, 1), sQ=X.s[1])

            # Yh = Y.H
            AHY = r1.mm(r0.mm_T(Y.T)).conj() # r1.mm_conj(r0.mm_H(Yh))
            Yh_next = r0.mm(r1.mm_T(AHY))
            Yh, Ry = qr(Yh_next, axes=(0, 1), sQ=-Y.s[0])

            # The triangular factors weigh each direction by its significance,
            # so that oversampled directions at roundoff do not set the error.
            # ``rank`` and ``cutoff`` put that boundary where the projectors
            # put it -- at the truncation, and at the pseudo-inverse cutoff --
            # instead of at the largest singular value;
            # see :func:`si_weights_from_triangular`.
            error = max(si_subspace_error(X, X_old, si_weights_from_triangular(Rx, rank, cutoff)),
                        si_subspace_error(Yh, Yh_old, si_weights_from_triangular(Ry, rank, cutoff)))
            n_iter += 1

            Y = Yh.H
            if error < tol:
                break
            X_old, Yh_old = X, Yh

    rho = Y @ r0.mm(r1.mm_T(X))
    info = {'niter': n_iter}
    if n_iter>0: info.update({'error': error})
    if spec_only:
        sall= rho.svd(axes=(0, 1), sU=rho.s[1], fix_signs=True, compute_uv=False)
        return X, Y, None, sall, None, info
    us, sall, vs = rho.svd(axes=(0, 1), sU=rho.s[1], fix_signs=True)
    return X, Y, us, sall, vs, info


def _si_spectrum(r0, r1, X, Y, opts_si):
    r"""Reduced singular values alone, for charge-sector refinement.
    Refinement probes rank sectors against each other rather than building projectors.
    """
    return _si_reduced_svd(r0, r1, X, Y, opts_si, spec_only=True)[3]

@nsys_profile("si_projector_svd")
def si_projector_svd(r0, r1, X, Y, opts_svd, opts_si,
                     return_spectrum=False, cutoff=0):
    """Approximate the SVD of ``r0 @ r1.T`` using recycled subspaces.

    Each half is either a tensor or a pair of enlarged corners; see :class:`_Half`.

    Always returns the 6-tuple ``u, s, v, X_new, Y_new, info``, where ``info``
    reports the subspace iteration; see :func:`_si_reduced_svd`.

    With ``return_spectrum``, only the spectrum is computed: ``s`` is then the
    full, untruncated spectrum of the reduced ``rho``, and ``u``, ``v``,
    ``X_new`` and ``Y_new`` are all ``None``.  Without ``us`` and ``vs`` there
    are no projectors, and no rotation of the bases into the SVD gauge, so
    nothing recyclable is produced.
    """
    r0, r1 = _Half(r0), _Half(r1)
    rank = _si_truncation_rank(opts_svd)
    if return_spectrum:
        res = _si_reduced_svd(r0, r1, X, Y, opts_si, spec_only=True,
                              rank=rank, cutoff=cutoff)
        return None, res[3], None, None, None, res[5]

    X, Y, us, sall, vs, info = _si_reduced_svd(r0, r1, X, Y, opts_si,
                                               rank=rank, cutoff=cutoff)

    X_new = X @ vs.H
    Y_new = us.H @ Y
    u = Y_new.H #Y.H @ us
    v = X_new.H #vs @ X.H

    trunc_opts = {k: opts_svd[k] for k in (
        'tol', 'tol_block', 'D_block', 'D_total', 'largest_gap',
        'eps_multiplet', 'hermitian', 'mask_f') if k in opts_svd}
    mask = truncation_mask(sall, **trunc_opts)
    u, s, v = mask.apply_mask(u, sall, v, axes=(-1, 0, 0))

    return u, s, v, X_new, Y_new, info


def redistribute_due(age, opts_si):
    r"""Whether a projector pair of the given ``age`` is inside the schedule window.

    ``age`` counts how many times this projector pair has already been updated
    with SI. The window covers every update up to ``warmup``, and then every
    ``redistribute_frequency`` updates if that option is positive.

    Two callers read it. It is the condition under which
    ``redistribute_sectors`` actually redistributes; and its negation is one of
    the conditions for skipping an SI update under ``skip_SI_update``, so that
    a pair inside the window is never skipped.
    """
    warmup = opts_si.get('warmup', 5)
    frequency = opts_si.get('redistribute_frequency', 0)
    return (age <= warmup
            or (frequency > 0 and age > warmup
                and (age - warmup) % frequency == 0))


def si_proj_corners(r0, r1, opts_svd, opts_si, X=None, Y=None, cutoff=0):
    r"""Truncated SVD of ``r0 @ r1.T`` from recycled subspace-iteration bases.

    Returns the projector pair ``p0, p1`` of the truncated decomposition,
    the refreshed bases ``X_new, Y_new`` to be recycled by the next update,
    and the ``info`` of the subspace iteration; see :func:`_si_reduced_svd`.
    ``info['bases']`` reports where the bases came from, see :class:`SI_state`.

    Bases that no longer fit the corners are first carried onto the new row
    legs by :func:`si_rebase_bases`, and only redrawn from noise when that is
    not possible. ``opts_si['rebase'] = False`` skips the attempt.

    Each half is either a tensor or a pair of enlarged corners; see :class:`_Half`.
    """
    r0, r1 = _Half(r0), _Half(r1)
    _validate_ctm_corner_pair(r0, r1)
    # An eye-initialized CTM starts below its requested chi and grows over
    # the first updates.  During that growth the enlarged corners may not
    # yet accommodate chi + p rangefinder columns.  Use every currently
    # available shared direction; changed corner legs will invalidate and
    # enlarge the recycled bases on subsequent updates.
    capacity = _ctm_shared_sector_capacity(r0, r1)
    rank = min(_si_rank(opts_svd, opts_si), sum(capacity.values()))
    bases = 'reused'
    if not si_bases_compatible(r0, r1, X, Y):
        # Corner row legs that changed do not invalidate what the bases
        # describe, only the space they are written in; see si_rebase_bases.
        if opts_si.get('rebase', True):
            X_rebased, Y_rebased = si_rebase_bases(r0, r1, X, Y)
            if si_bases_compatible(r0, r1, X_rebased, Y_rebased):
                X, Y, bases = X_rebased, Y_rebased, 'rebased'
        if bases == 'reused':
            X, Y = initialize_si_bases(r0, r1, rank)
            bases = 'reinitialized'
        elif sum(X.get_legs(1).tD.values()) != rank or any(
                dimension > capacity.get(charge, 0)
                for charge, dimension in X.get_legs(1).tD.items()):
            # The auxiliary rank follows the capacity of the corner legs, which
            # moved along with them.
            X, Y = _recycle_si_bases(
                r0, r1, X, Y,
                _distribute_si_rank_with_capacity(capacity, rank))
    if opts_si.get('redistribute_sectors', False):
        X, Y = si_refinement(r0, r1, X, Y, opts_svd, opts_si)

    res= si_projector_svd(r0, r1, X, Y, opts_svd, opts_si, cutoff=cutoff)
    u, s, v, X_new, Y_new, info= res
    info["bases"] = bases

    rs = s.rsqrt(cutoff=cutoff)
    info["rank"] = (rs>0).trace().item()
    p0= r1.mm_T( (rs @ v).H ).unfuse_legs(axes=0)
    p1= r0.mm_T( (u @ rs).conj() ).unfuse_legs(axes=0)
    return p0, p1, X_new, Y_new, info
