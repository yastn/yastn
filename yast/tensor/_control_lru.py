# pylint: disable=protected-access
""" Dynamical changing of lru_cache maxsize. """
from functools import lru_cache
from . import _merging, _contractions


__all__ = ['set_cache_maxsize', 'get_cache_info', 'clear_cache']


def set_cache_maxsize(maxsize=0):
    """Change maxsize of lru_cache to reuses some metadata."""
    _contractions._meta_broadcast = lru_cache(maxsize)(_contractions._meta_broadcast.__wrapped__)
    _contractions._meta_tensordot_nomerge = lru_cache(maxsize)(_contractions._meta_tensordot_nomerge.__wrapped__)
    _contractions._swap_gate_meta = lru_cache(maxsize)(_contractions._swap_gate_meta.__wrapped__)
    _contractions._trace_meta = lru_cache(maxsize)(_contractions._trace_meta.__wrapped__)
    _contractions._trace_input = lru_cache(maxsize)(_contractions._trace_input.__wrapped__)
    _merging._meta_merge_to_matrix = lru_cache(maxsize)(_merging._meta_merge_to_matrix.__wrapped__)
    _merging._intersect_hfs = lru_cache(maxsize)(_merging._intersect_hfs.__wrapped__)
    _merging._leg_structure_combine_charges = lru_cache(maxsize)(_merging._leg_structure_combine_charges.__wrapped__)
    _merging._meta_fuse_hard = lru_cache(maxsize)(_merging._meta_fuse_hard.__wrapped__)
    _merging._meta_unfuse_hard = lru_cache(maxsize)(_merging._meta_unfuse_hard.__wrapped__)


def clear_cache():
    """Change maxsize of lru_cache to reuses some metadata."""
    _contractions._meta_broadcast.cache_clear()
    _contractions._meta_tensordot_nomerge.cache_clear()
    _contractions._swap_gate_meta.cache_clear()
    _contractions._trace_meta.cache_clear()
    _contractions._trace_input.cache_clear()
    _merging._meta_merge_to_matrix.cache_clear()
    _merging._intersect_hfs.cache_clear()
    _merging._leg_structure_combine_charges.cache_clear()
    _merging._meta_fuse_hard.cache_clear()
    _merging._meta_unfuse_hard.cache_clear()


def get_cache_info():
    """Return statistics of lru_caches used in yast."""
    return {"merge_to_matrix": _merging._meta_merge_to_matrix.cache_info(),
            "tensordot_nomerge": _contractions._meta_tensordot_nomerge.cache_info(),
            "broadcast": _contractions._meta_broadcast.cache_info(),
            "swap_gate": _contractions._swap_gate_meta.cache_info(),
            "trace_meta": _contractions._trace_meta.cache_info(),
            "trace_input": _contractions._trace_input.cache_info(),
            "fuse_hard": _merging._meta_fuse_hard.cache_info(),
            "unfuse_hard": _merging._meta_unfuse_hard.cache_info(),
            "intersect_hfs": _merging._intersect_hfs.cache_info(),
            "combine_leg_structure": _merging._leg_structure_combine_charges.cache_info()}

