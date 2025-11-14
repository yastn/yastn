# Copyright 2024 The YASTN Authors. All Rights Reserved.
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
""" Dynamical changing of lru_cache maxsize. """
from functools import lru_cache

from . import _merging, _contractions, _algebra

__all__ = ['clear_cache', 'get_cache_info', 'set_cache_maxsize']


def set_cache_maxsize(maxsize=0):
    """Change maxsize of lru_cache to reuses some metadata."""
    _contractions._meta_broadcast = lru_cache(maxsize)(_contractions._meta_broadcast.__wrapped__)
    _contractions._meta_tensordot_f2m = lru_cache(maxsize)(_contractions._meta_tensordot_f2m.__wrapped__)
    _contractions._meta_tensordot_fc = lru_cache(maxsize)(_contractions._meta_tensordot_fc.__wrapped__)
    _contractions._meta_tensordot_nf = lru_cache(maxsize)(_contractions._meta_tensordot_nf.__wrapped__)
    _contractions._meta_mask = lru_cache(maxsize)(_contractions._meta_mask.__wrapped__)
    _contractions._common_inds = lru_cache(maxsize)(_contractions._common_inds.__wrapped__)
    _contractions._meta_swap_gate = lru_cache(maxsize)(_contractions._meta_swap_gate.__wrapped__)
    _contractions._meta_swap_gate_charge = lru_cache(maxsize)(_contractions._meta_swap_gate_charge.__wrapped__)
    _contractions._meta_trace = lru_cache(maxsize)(_contractions._meta_trace.__wrapped__)
    _contractions._meta_vdot = lru_cache(maxsize)(_contractions._meta_vdot.__wrapped__)
    _contractions._meta_ncon = lru_cache(maxsize)(_contractions._meta_ncon.__wrapped__)
    _merging._meta_merge_to_matrix = lru_cache(maxsize)(_merging._meta_merge_to_matrix.__wrapped__)
    _merging._meta_unmerge_matrix = lru_cache(maxsize)(_merging._meta_unmerge_matrix.__wrapped__)
    _merging._masks_hfs_intersection = lru_cache(maxsize)(_merging._masks_hfs_intersection.__wrapped__)
    _merging._leg_structure_combine_charges_prod = lru_cache(maxsize)(_merging._leg_structure_combine_charges_prod.__wrapped__)
    _merging._meta_fuse_hard = lru_cache(maxsize)(_merging._meta_fuse_hard.__wrapped__)
    _merging._meta_unfuse_hard = lru_cache(maxsize)(_merging._meta_unfuse_hard.__wrapped__)
    _algebra._meta_addition = lru_cache(maxsize)(_algebra._meta_addition.__wrapped__)


def clear_cache():
    """Change maxsize of lru_cache to reuses some metadata."""
    _contractions._meta_broadcast.cache_clear()
    _contractions._meta_tensordot_f2m.cache_clear()
    _contractions._meta_tensordot_fc.cache_clear()
    _contractions._meta_tensordot_nf.cache_clear()
    _contractions._meta_mask.cache_clear()
    _contractions._common_inds.cache_clear()
    _contractions._meta_swap_gate.cache_clear()
    _contractions._meta_swap_gate_charge.cache_clear()
    _contractions._meta_trace.cache_clear()
    _contractions._meta_vdot.cache_clear()
    _contractions._meta_ncon.cache_clear()
    _merging._meta_merge_to_matrix.cache_clear()
    _merging._meta_unmerge_matrix.cache_clear()
    _merging._masks_hfs_intersection.cache_clear()
    _merging._leg_structure_combine_charges_prod.cache_clear()
    _merging._meta_fuse_hard.cache_clear()
    _merging._meta_unfuse_hard.cache_clear()
    _algebra._meta_addition.cache_clear()


def get_cache_info():
    """Return statistics of lru_caches used in yastn."""
    return {"merge_to_matrix": _merging._meta_merge_to_matrix.cache_info(),
            "unmerge_from_matrix": _merging._meta_unmerge_matrix.cache_info(),
            "fuse_hard": _merging._meta_fuse_hard.cache_info(),
            "unfuse_hard": _merging._meta_unfuse_hard.cache_info(),
            "intersect_hfs": _merging._masks_hfs_intersection.cache_info(),
            "combine_leg_structure": _merging._leg_structure_combine_charges_prod.cache_info(),
            "tensordot_f2m": _contractions._meta_tensordot_f2m.cache_info(),
            "tensordot_fc": _contractions._meta_tensordot_fc.cache_info(),
            "tensordot_nf": _contractions._meta_tensordot_nf.cache_info(),
            "tensordot_common_inds": _contractions._common_inds.cache_info(),
            "broadcast": _contractions._meta_broadcast.cache_info(),
            "mask": _contractions._meta_mask.cache_info(),
            "trace": _contractions._meta_trace.cache_info(),
            "vdot": _contractions._meta_vdot.cache_info(),
            "swap_gate": _contractions._meta_swap_gate.cache_info(),
            "swap_gate_charge": _contractions._meta_swap_gate_charge.cache_info(),
            "ncon": _contractions._meta_ncon.cache_info(),
            "addition": _algebra._meta_addition.cache_info()}
