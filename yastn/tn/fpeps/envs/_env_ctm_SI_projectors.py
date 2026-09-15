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

The corner halves ``r0`` and ``r1`` are passed as tuples of rank-2 factors,
typically pairs of enlarged corners, with ``r0 = r0[0] @ r0[1]`` and likewise
for ``r1``. The halves are only ever applied factor by factor, so a half of
two ``N x N`` corners is never formed at ``O(N^3)`` cost. A single tensor is
accepted as a one-factor half.

This module is a leaf: it depends on the tensor layer only, never on the CTM
environment classes that call into it.
"""
from __future__ import annotations

import logging

from ....initialize import rand, zeros, eye, block
from ....sym import sym_none
from ....tensor import Tensor, YastnError, Leg, tensordot, qr, truncation_mask
from ....tensor._auxiliary import get_blocks
from ...._profile import nsys_profile, nvtx_range

logger = logging.getLogger(__name__)


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


def si_enabled(opts_si):
    """Whether ``opts_si`` requests SI projectors."""
    return opts_si is not None and opts_si.get('enabled', False)


def _as_half(r):
    """Return a CTM corner half as the tuple of rank-2 factors whose product it is."""
    factors = (r,) if isinstance(r, Tensor) else r
    if (not isinstance(factors, (tuple, list)) or not factors
            or not all(isinstance(factor, Tensor) for factor in factors)):
        raise YastnError(
            "CTM corner halves must be YASTN tensors or tuples of YASTN tensors.")
    return tuple(factors)


def _half_leg(half, axis):
    """Leg of the product of ``half`` factors: 0 is external, 1 is contracted."""
    return half[0].get_legs(0) if axis == 0 else half[-1].get_legs(1)


def _validate_ctm_corner_pair(r0, r1):
    """Validate the two closures of a pair of CTM corner halves.

    Returns both halves as tuples of factors, see :func:`_as_half`.
    """
    r0, r1 = _as_half(r0), _as_half(r1)
    factors = r0 + r1
    if any(factor.ndim != 2 for factor in factors):
        raise YastnError("CTM corner halves must be rank-2 tensors.")
    if any(factor.config.sym.SYM_ID != r0[0].config.sym.SYM_ID
           for factor in factors):
        raise YastnError("CTM corner halves must use the same symmetry.")

    for axis in (0, 1):
        leg0 = _half_leg(r0, axis)
        leg1 = _half_leg(r1, axis)
        common_charges = leg0.tD.keys() & leg1.tD.keys()
        if any(leg0.tD[charge] != leg1.tD[charge]
               for charge in common_charges):
            raise YastnError(
                "CTM corner halves must have matching dimensions in every "
                "shared charge sector on both loop closures; "
                f"mismatch on axis {axis}.")
    return r0, r1


def _live_sectors(r0, r1):
    r"""Charges of the external legs connected by blocks through ``r0 @ r1.T``.

    Uses block charges only. Corner legs of a half given by its factors can
    carry sectors without blocks in the product; SI columns in such sectors
    are annihilated. Returns ``None`` for meta-fused factors.
    """
    if any(factor.ndim_n != 2 for factor in r0 + r1):
        return None
    nsym = r0[0].config.sym.NSYM

    def block_charges(half):  # (leg 0, leg 1) charges of blocks of each factor
        return [[(t[:nsym], t[nsym:]) for t in factor.get_blocks_charge()]
                for factor in half]

    def contracted(blocks):  # charges on contracted leg reached from external leg
        live = {t1 for _, t1 in blocks[0]}
        for factor in blocks[1:]:
            live = {t1 for t0, t1 in factor if t0 in live}
        return live

    def external(blocks, live):  # charges on external leg reached from live ones
        for factor in reversed(blocks):
            live = {t0 for t0, t1 in factor if t1 in live}
        return live

    blocks0, blocks1 = block_charges(r0), block_charges(r1)
    live = contracted(blocks0) & contracted(blocks1)
    return external(blocks0, live) & external(blocks1, live)


def _ctm_shared_sector_capacity(r0, r1):
    """Return capacities of sectors supported by both CTM corner halves."""
    capacity0 = _half_leg(r0, 0).tD
    capacity1 = _half_leg(r1, 0).tD
    live = _live_sectors(r0, r1)
    return {charge: dimension for charge, dimension in capacity0.items()
            if charge in capacity1 and (live is None or charge in live)}


def initialize_si_bases(r0, r1, rank, charges=None):
    r"""Initialize compatible column-isometric SI bases from Gaussian noise.

    The auxiliary rank is spread as uniformly as possible over charge sectors
    of the matching external CTM legs. A sector cannot be assigned more
    columns than that sector has rows.
    """
    r0, r1 = _validate_ctm_corner_pair(r0, r1)

    x_input = _half_leg(r1, 0).conj()  # right leg of r1
    y_input = _half_leg(r0, 0)  # left leg of r0
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
        r1[0].config,
        s=-x_input.s,
        t=tuple(charge_mapping.keys()),
        D=tuple(charge_mapping.values()),
    )

    X = rand(r1[0].config, legs=(x_input, x_aux), distribution='normal')
    Yh = rand(r0[0].config, legs=(y_input, x_aux), distribution='normal')

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
    r0, r1 = _as_half(r0), _as_half(r1)

    def is_compatible_subspace(basis_leg, corner_leg):
        """A refined basis may intentionally contain only selected sectors."""
        return (basis_leg.s == corner_leg.s
                and all(charge in corner_leg.tD
                        and corner_leg.tD[charge] == dimension
                        for charge, dimension in basis_leg.tD.items()))

    try:
        return (
            is_compatible_subspace(
                X.get_legs(0), _half_leg(r1, 0).conj())
            and is_compatible_subspace(
                Y.get_legs(1), _half_leg(r0, 0).conj())
            and X.get_legs(1) == Y.get_legs(0).conj()
            and all(X.dtype == factor.dtype for factor in r1)
            and all(Y.dtype == factor.dtype for factor in r0)
            and X.device == r1[0].device
            and Y.device == r0[0].device
            and X.get_legs(0).are_consistent(_half_leg(r1, 0))
            and Y.get_legs(1).are_consistent(_half_leg(r0, 0))
        )
    except (AttributeError, IndexError):
        return False

def si_subspace_error(Q, Q_old):
    r"""Mean squared sine of the principal angles between two SI bases.

    Both tensors are expected to be column-isometric.  The expression
    ``1 - ||Q_old.H @ Q||_F^2 / rank`` is invariant under rotations within
    either basis, unlike a direct tensor difference.
    """
    if Q_old is None or Q.get_legs() != Q_old.get_legs():
        return float('inf')

    rank = Q.get_shape(axes=1)
    overlap = Q_old.detach().H @ Q.detach()
    error = 1.0 - overlap.norm() ** 2 / rank
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


def _si_refinement_asvr(r0, r1, X, Y, opts_svd, opts_si):
    """Return a stable SI charge mapping estimated from dominant spectra."""
    iterations = opts_si.get('asvr_iterations', 5)
    chip = _si_rank(opts_svd, opts_si)
    sector_capacity = _ctm_shared_sector_capacity(r0, r1)
    charge_mapping = dict(X.get_legs(1).tD)

    # A symmetry-preserving subspace iteration cannot generate a charge sector
    # absent from its input bases. Seed every sector shared by both corners so
    # that ASVR can compare their spectra before refining the allocation.
    missing_charges = set(sector_capacity) - set(charge_mapping)
    if missing_charges:
        exploratory_mapping = _distribute_si_rank_with_capacity(
            sector_capacity, chip)
        unexplored_charges = set(sector_capacity) - set(exploratory_mapping)
        if unexplored_charges:
            raise YastnError(
                "ASVR cannot probe every shared charge sector with SI rank "
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


def _si_refinement_rds(r0, r1, X, Y, opts_svd, opts_si):
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


def _si_refinement_cwo(r0, r1, X, Y, opts_svd, opts_si):
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
        raise YastnError("CWO refinement found no singular values.")
    return charge_mapping

@nsys_profile("si_refinement")
def si_refinement(r0, r1, X, Y, opts_svd, opts_si):
    r"""Refine and recycle SI bases with the selected allocation strategy.

    This is the single dispatch point for SI charge-sector refinement. Each
    strategy returns a charge mapping; basis resizing is centralized here so
    every method retains compatible columns in its public ``(X, Y)`` result.
    """
    r0, r1 = _validate_ctm_corner_pair(r0, r1)
    refinement = opts_si.get('refinement', 'cwo')
    refinements = {
        'cwo': _si_refinement_cwo,
        'asvr': _si_refinement_asvr,
        'rds': _si_refinement_rds,
    }
    try:
        refine = refinements[refinement]
    except KeyError:
        raise YastnError(
            "Unknown SI refinement method "
            f"{refinement!r}; expected 'cwo', 'asvr', or 'rds'.") from None

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


def _apply_half(half, M):
    r"""Apply ``half = half[0] @ half[1] @ ...`` to the axis 0 of ``M``."""
    for factor in reversed(half):
        M = tensordot(factor, M, axes=(1, 0))
    return M


def _apply_half_T(half, M):
    r"""Apply ``half.T`` to the axis 0 of ``M``."""
    for factor in half:
        M = tensordot(factor, M, axes=(0, 0))
    return M


def _apply_half_H(half, M):
    r"""Apply ``half.H`` to the axis 0 of ``M``."""
    for factor in half:
        M = tensordot(factor.conj(), M, axes=(0, 0))
    return M


def _apply_half_conj(half, M):
    r"""Apply ``half.conj()`` to the axis 0 of ``M``."""
    for factor in reversed(half):
        M = tensordot(factor.conj(), M, axes=(1, 0))
    return M


def _apply_corner_product(r0, r1, X):
    r"""Apply A = tensordot(r0, r1, axes=(1, 1)) to X.

    r0 has indices (a, k), r1 has indices (b, k), and X has
    indices (b, p). The result has indices (a, p).
    """
    return _apply_half(r0, _apply_half_T(r1, X))


def _apply_corner_product_h(r0, r1, Z):
    r"""Apply A.H to Z without explicitly constructing A.

    Z has indices (a, p). The result has indices (b, p).
    """
    return _apply_half_conj(r1, _apply_half_H(r0, Z))


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
        X, charge_mapping, left_leg=_half_leg(r1, 0).conj())
    Yh = symmetric_isometry_recycle(
        Y.H, charge_mapping, left_leg=_half_leg(r0, 0))
    return X, Yh.H


@nsys_profile("_si_reduced_svd")
def _si_reduced_svd(r0, r1, X, Y, opts_si, spec_only=False):
    r"""Subspace-iterate the bases and decompose the reduced ``rho = Y A X``.

    Returns the converged bases together with the decomposition
    ``us, sall, vs`` of ``rho``. Everything here acts either on the small
    auxiliary legs or through ``_apply_corner_product``, so the full
    ``r0 @ r1.T`` is never formed.
    """
    r0, r1 = _validate_ctm_corner_pair(r0, r1)
    niter = opts_si.get('niter', 5)
    tol = opts_si.get('tol', 1e-3)
    X_old, Yh_old = X, Y.H

    with nvtx_range("_si_reduced_svd SI"):
        for _ in range(niter):
            AX = _apply_corner_product(r0, r1, X)
            X_next = _apply_corner_product_h(r0, r1, AX)
            X, _ = qr(X_next, axes=(0, 1), sQ=X.s[1])

            Yh = Y.H
            AHY = _apply_corner_product_h(r0, r1, Yh)
            Yh_next = _apply_corner_product(r0, r1, AHY)
            Yh, _ = qr(Yh_next, axes=(0, 1), sQ=Yh.s[1])

            error = max(si_subspace_error(X, X_old),
                        si_subspace_error(Yh, Yh_old))

            Y = Yh.H
            if error < tol:
                break
            X_old, Yh_old = X, Yh

    rho = Y @ _apply_corner_product(r0, r1, X)
    if spec_only:
        sall= rho.svd(axes=(0, 1), sU=rho.s[1], fix_signs=True, compute_uv=False)
        return X, Y, None, sall, None
    us, sall, vs = rho.svd(axes=(0, 1), sU=rho.s[1], fix_signs=True)
    return X, Y, us, sall, vs


def _si_spectrum(r0, r1, X, Y, opts_si):
    r"""Reduced singular values alone, for charge-sector refinement.

    Refinement strategies only read the spectrum, so this skips building the
    projectors and applying the truncation mask -- work proportional to the
    large CTM legs rather than to the auxiliary rank.
    """
    return _si_reduced_svd(r0, r1, X, Y, opts_si, spec_only=True)[3]


def _ritz_phases(u):
    r"""Phases fixing the gauge of Ritz vectors ``u = Y.H @ us`` as ``fix_signs`` in SVD.

    Returns a matrix, diagonal in values, with the phase of the largest
    element of each column of ``u``, with legs matching the leg 1 of ``u``.
    Returns ``None`` for meta-fused ``u``.
    """
    if u.ndim_n != 2:
        return None
    u = u.consume_transpose()
    leg = u.get_legs(1)
    phases = eye(u.config, legs=(leg.conj(), leg), isdiag=False,
                 dtype=u.yastn_dtype, device=u.device)
    blocks_u = get_blocks(u.config.sym, u.struct)
    blocks_p = get_blocks(phases.config.sym, phases.struct)
    index_p = {tuple(t[0].tolist()): i for i, t in enumerate(blocks_p.t)}
    meta = []
    for i, t in enumerate(blocks_u.t):
        j = index_p[tuple(t[1].tolist())]
        meta.append((None, None, tuple(blocks_u.slc[i]), tuple(blocks_u.D[i]),
                     None, tuple(blocks_p.slc[j]), tuple(blocks_p.D[j])))
    if len(meta) != len(index_p):
        return None
    backend = u.config.backend
    _, data = backend.fix_svd_signs(backend.clone(u.data), phases.data, meta)
    return phases._replace(data=data)


@nsys_profile("si_projector_svd")
def si_projector_svd(r0, r1, X, Y, opts_svd, opts_si,
                     return_spectrum=False):
    """Approximate the SVD of ``r0 @ r1.T`` using recycled subspaces."""
    X, Y, us, sall, vs = _si_reduced_svd(r0, r1, X, Y, opts_si)

    # Fixing signs of the small us, vs leaves the gauge of the projectors to
    # QR sign conventions within SI. Fix it on the external legs of r0 instead.
    # With corner pairs, those are the legs of neighbouring projector pairs,
    # whose recycled bases rely on a gauge that is stable between CTM steps.
    phases = _ritz_phases(Y.H @ us)
    if phases is not None:
        us = us @ phases.conj_blocks()
        vs = phases @ vs

    X_new = X @ vs.H
    Y_new = us.H @ Y
    u = Y.H @ us
    v = vs @ X.H

    trunc_opts = {k: opts_svd[k] for k in (
        'tol', 'tol_block', 'D_block', 'D_total', 'largest_gap',
        'eps_multiplet', 'hermitian', 'mask_f') if k in opts_svd}
    mask = truncation_mask(sall, **trunc_opts)
    u, s, v = mask.apply_mask(u, sall, v, axes=(-1, 0, 0))
    result = (u, s, v, X_new, Y_new)
    return result + (sall,) if return_spectrum else result


def si_correction_due(age, opts_si):
    r"""Whether a projector pair of the given ``age`` is due a sector redistribution.

    ``age`` counts how many times this projector pair has already been updated
    with SI. A correction fires once at ``warmup``, and then every
    ``correction_frequency`` updates if that option is positive.
    """
    warmup = opts_si.get('warmup', 5)
    frequency = opts_si.get('correction_frequency', 0)
    return (age == warmup
            or (frequency > 0 and age > warmup
                and (age - warmup) % frequency == 0))


def si_proj_corners(r0, r1, opts_svd, opts_si, X=None, Y=None, cutoff=0):
    r"""Projectors in between ``r0 @ r1.T`` from recycled subspace-iteration bases.

    SI counterpart of :func:`yastn.tn.fpeps.envs._env_ctm.proj_corners`.
    Corner halves ``r0`` and ``r1`` are tuples of rank-2 factors, e.g.,
    pairs of enlarged corners, with ``r0 = r0[0] @ r0[1]``; see the module
    docstring. Their products are never formed.

    Parameters
    ----------
    r0, r1: tuple[Tensor, ...] | Tensor
        Corner halves, each with legs (external, contracted).
    opts_svd: dict
        Truncation options; requires an integer ``D_total`` or ``D_block``.
    opts_si: dict
        SI options, see :meth:`yastn.tn.fpeps.EnvCTM.update_`.
    X, Y: Tensor | None
        Recycled bases from the previous update of this projector pair.
    cutoff: float
        Cutoff of the pseudo-inverse square root of the singular values.

    Returns
    -------
    p0, p1, X_new, Y_new
        Projectors built from ``r1`` and ``r0``, respectively, and
        refreshed bases to be recycled by the next update.
    """
    r0, r1 = _validate_ctm_corner_pair(r0, r1)
    # An eye-initialized CTM starts below its requested chi and grows over
    # the first updates.  During that growth the enlarged corners may not
    # yet accommodate chi + p rangefinder columns.  Use every currently
    # available shared direction; changed corner legs will invalidate and
    # enlarge the recycled bases on subsequent updates.
    rank = min(_si_rank(opts_svd, opts_si),
               sum(_ctm_shared_sector_capacity(r0, r1).values()))
    if not si_bases_compatible(r0, r1, X, Y):
        X, Y = initialize_si_bases(r0, r1, rank)
    if opts_si.get('correct', False):
        X, Y = si_refinement(r0, r1, X, Y, opts_svd, opts_si)
    u, s, v, X_new, Y_new = si_projector_svd(r0, r1, X, Y, opts_svd, opts_si)

    if opts_svd.get('verbosity', 0) > 2:
        logger.info(f"si_proj_corners S {s.get_legs(0)}")

    rs = s.rsqrt(cutoff=cutoff)
    p0 = _apply_half_T(r1, (rs @ v).conj().T).unfuse_legs(axes=0)
    p1 = _apply_half_T(r0, (u @ rs).conj()).unfuse_legs(axes=0)
    return p0, p1, X_new, Y_new
