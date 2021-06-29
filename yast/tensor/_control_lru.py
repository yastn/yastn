""" Dynamical changing of lru_cache maxsize. """
from . import _merging
from functools import lru_cache


__all__ = ['set_cache_maxsize', 'get_cache_info']


def set_cache_maxsize(maxsize=0):
    """Change maxsize of lru_cache to reuses some metadata."""
    _merging._meta_merge_to_matrix = lru_cache(maxsize)(_merging._meta_merge_to_matrix.__wrapped__)


def get_cache_info():
    return _merging._meta_merge_to_matrix.cache_info()
