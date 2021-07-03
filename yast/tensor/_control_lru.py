""" Dynamical changing of lru_cache maxsize. """
from . import _merging
from functools import lru_cache


__all__ = ['set_cache_maxsize', 'get_cache_info']


def set_cache_maxsize(maxsize=0):
    """Change maxsize of lru_cache to reuses some metadata."""
    _merging._meta_merge_to_matrix = lru_cache(maxsize)(_merging._meta_merge_to_matrix.__wrapped__)
    _merging._intersect_hfs = lru_cache(maxsize)(_merging._intersect_hfs.__wrapped__)
    _merging._leg_structure_combine_charges = lru_cache(maxsize)(_merging._leg_structure_combine_charges.__wrapped__)
    _merging._meta_fuse_hard = lru_cache(maxsize)(_merging._meta_fuse_hard.__wrapped__)
    _merging._meta_unfuse_hard = lru_cache(maxsize)(_merging._meta_unfuse_hard.__wrapped__)


def get_cache_info():
    return {"meta_merge_to_matrix": _merging._meta_merge_to_matrix.cache_info(),
            "meta_fuse_hard": _merging._meta_fuse_hard.cache_info(),
            "meta_unfuse_hard": _merging._meta_unfuse_hard.cache_info(),
            "_intersect_hfs": _merging._intersect_hfs.cache_info(),
            "leg_structure_combine_charges": _merging._leg_structure_combine_charges.cache_info()}
