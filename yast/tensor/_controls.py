""" Testing and controls. """
import numpy as np
from functools import lru_cache
from . import _merging
from ._auxliary import _flatten, _tarray

__all__ = ['check_signatures_match', 'check_consistency', 'set_cache_maxsize', 'get_cache_info',
           'are_independent', 'is_consistent']

_check = {"signatures_match": True, "consistency": True}

class YastError(Exception):
    """Errors cought by checks in yast."""


def check_signatures_match(value=True):
    """Set the value of the flag check_signatures_match."""
    _check["signatures_match"] = bool(value)


def check_consistency(value=True):
    """Set the value of the flag check_consistency."""
    _check["consistency"] = bool(value)


def set_cache_maxsize(maxsize=0):
    """Change maxsize of lru_cache to reuses some metadata."""
    _merging._meta_merge_to_matrix = lru_cache(maxsize)(_merging._meta_merge_to_matrix.__wrapped__)


def get_cache_info():
    return _merging._meta_merge_to_matrix.cache_info()


def _test_configs_match(a, b):
    """Check if config's of two tensors allow for performing operations mixing them. """
    if a.config.device != b.config.device:
        raise YastError('Devices of the two tensors do not match.')
    if a.config.sym.SYM_ID != b.config.sym.SYM_ID:
        raise YastError('Two tensors have different symmetries.')
    if a.config.backend.BACKEND_ID != b.config.backend.BACKEND_ID:
        raise YastError('Two tensors have different backends.')


def _test_tensors_match(a, b):
    if _check["signatures_match"] and (a.struct.s != b.struct.s or a.struct.n != b.struct.n):
        raise YastError('Tensor signatures do not match')
    if _check["consistency"] and a.meta_fusion != b.meta_fusion:
        raise YastError('Fusion trees do not match')
    if a.hard_fusion != b.hard_fusion:
        raise YastError('Hard fusions of the two tensors do not match')


def _test_fusions_match(a, b):
    if _check["consistency"] and a.meta_fusion != b.meta_fusion:
        raise YastError('Fusion trees do not match')


def _test_all_axes(a, axes):
    if _check["consistency"]:
        axes = tuple(_flatten(axes))
        if a.nlegs != len(axes):
            raise YastError('Two few indices in axes')
        if sorted(set(axes)) != list(range(a.nlegs)):
            raise YastError('Repeated axis')


def _test_hard_fusion_match(hf1, hf2, mconj):
    if hf1.t != hf2.t or hf1.D != hf2.D or hf1.tree != hf2.tree:
        raise YastError('Hard fusions do not match')
    if (mconj == 1 and hf1.s != hf2.s) or (mconj == -1 and hf1.s != hf2.ms):
        raise YastError('Hard fusions do not match. Singnature problem.')


def are_independent(a, b):
    """
    Test if all elements of two yast tensors are independent objects in memory.
    """
    test = []
    test.append(a is b)
    test.append(a.A is b.A)
    for key in a.A.keys():
        if key in b.A:
            test.append(a.config.backend.is_independent(a.A[key], b.A[key]))
    return not any(test)


def is_consistent(a):
    """
    Test is yast tensor is does not contain inconsistent structures

    Check that:
    1) tset and Dset correspond to A
    2) tset follow symmetry rule f(s@t)==n
    3) block dimensions are consistent (this requires config.test=True)
    """
    for ind, D in zip(a.struct.t, a.struct.D):
        assert ind in a.A, 'index in struct.t not in dict A'
        assert a.config.backend.get_shape(a.A[ind]) == D, 'block dimensions do not match struct.D'
    assert len(a.struct.t) == len(a.A), 'length of struct.t do not match dict A'
    assert len(a.struct.D) == len(a.A), 'length of struct.D do not match dict A'

    tset = _tarray(a)
    sa = np.array(a.struct.s, dtype=int)
    na = np.array(a.struct.n, dtype=int)
    assert np.all(a.config.sym.fuse(tset, sa, 1) == na), 'charges of some block do not satisfy symmetry condition'
    for n in range(a.nlegs):
        a.get_leg_structure(n, native=True)
    device = {a.config.backend.get_device(x) for x in a.A.values()}
    assert len(device) <= 1, 'not all blocks reside on the same device'
    if len(device) == 1:
        assert device.pop() == a.config.device, 'device of blocks inconsistent with config.device'
    return True
