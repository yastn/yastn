""" Error handling for yast tensor. """
import numpy as np
from ._auxliary import _tarray

__all__ = ['check_signatures_match', 'check_consistency', 'allow_cache_meta', 'YastError', 'are_independent', 'is_consistent']


_check = {"signatures_match": True, "consistency": True, "cache_meta": True}


class YastError(Exception):
    """Errors cought by checks in yast."""


def check_signatures_match(value=True):
    """Set the value of the flag check_signatures_match."""
    _check["signatures_match"] = bool(value)


def check_consistency(value=True):
    """Set the value of the flag check_consistency."""
    _check["consistency"] = bool(value)


def allow_cache_meta(value=True):
    """Set the value of the flag that permits to reuses some metadata."""
    _check["cache_meta"] = bool(value)


def _test_configs_match(a, b):
    # if a.config != b.config:
    if not (a.config.dtype == b.config.dtype and
            a.config.dtype == b.config.dtype and
            a.config.sym.name == b.config.sym.name and
            a.config.backend._backend_id == b.config.backend._backend_id):
        raise YastError('configs do not match')


def _test_tensors_match(a, b):
    if _check["signatures_match"] and (not np.all(a.s == b.s) or not np.all(a.n == b.n)):
        raise YastError('Tensor signatures do not match')
    if _check["consistency"] and not a.meta_fusion == b.meta_fusion:
        raise YastError('Fusion trees do not match')


def _test_fusions_match(a, b):
    if _check["consistency"] and not a.meta_fusion == b.meta_fusion:
        raise YastError('Fusion trees do not match')


def _test_axes_split(a, out_l, out_r):
    if _check["consistency"]:
        if not a.nlegs == len(out_l) + len(out_r):
            raise YastError('Two few indices in axes')
        if not sorted(set(out_l + out_r)) == list(range(a.nlegs)):
            raise YastError('Repeated axis')


def are_independent(a, b):
    """
    Test if all elements of two yast tensors are independent objects in memory.
    """
    test = []
    test.append(a is b)
    test.append(a.A is b.A)
    test.append(a.n is b.n)
    test.append(a.s is b.s)
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
    test = []
    for ind, D in zip(a.struct.t, a.struct.D):
        test.append(ind in a.A)
        test.append(a.config.backend.get_shape(a.A[ind]) == D)
    test.append(len(a.struct.t) == len(a.A))
    test.append(len(a.struct.D) == len(a.A))

    tset = _tarray(a)
    test.append(np.all(a.config.sym.fuse(tset, a.s, 1) == a.n))
    for n in range(a.nlegs):
        a.get_leg_structure(n, native=True)

    return all(test)