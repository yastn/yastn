""" Testing and auxliary functions. """

from functools import lru_cache
from typing import NamedTuple
from itertools import accumulate, chain
from . import _merging
import numpy as np
from ..sym import sym_none

__all__ = ['check_signatures_match', 'check_consistency', 'set_cache_maxsize', 'get_cache_info',
           'are_independent', 'is_consistent']

_check = {"signatures_match": True, "consistency": True}


class _struct(NamedTuple):
    t: tuple
    D: tuple
    s: tuple
    n: tuple


class _config(NamedTuple):
    backend: any = None
    sym: any = sym_none
    device: str = 'cpu'


def _flatten(nested_iterator):
    for item in nested_iterator:
        try:
            yield from _flatten(item)
        except TypeError:
            yield item


def _unpack_axes(a, *args):
    """Unpack meta axes into native axes based on a.meta_fusion"""
    clegs = tuple(accumulate(x[0] for x in a.meta_fusion))
    return tuple(tuple(chain(*(range(clegs[ii] - a.meta_fusion[ii][0], clegs[ii]) for ii in axes))) for axes in args)


def _clear_axes(*args):
    return ((axis,) if isinstance(axis, int) else tuple(axis) for axis in args)


def _common_keys(d1, d2):
    """ Divide keys into: common, only in d1 and only in d2. Returns: keys12, keys1, keys2. """
    s1 = set(d1)
    s2 = set(d2)
    return tuple(s1 & s2), tuple(s1 - s2), tuple(s2 - s1)


def _common_rows(a, b):
    """ Return indices (as tuple) of nparray a rows that are in b, and vice versa. """
    la = [tuple(x.flat) for x in a]
    lb = [tuple(x.flat) for x in b]
    sa = set(la)
    sb = set(lb)
    ia = tuple(ii for ii, el in enumerate(la) if el in sb)
    ib = tuple(ii for ii, el in enumerate(lb) if el in sa)
    return ia, ib


def _tarray(a):
    return np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), a.nlegs, a.config.sym.NSYM))


def _Darray(a):
    return np.array(a.struct.D, dtype=int).reshape(len(a.struct.D), a.nlegs)


def update_struct(a):
    """Updates meta-information about charges and dimensions of all blocks."""
    d = a.A
    a.A = {k: d[k] for k in sorted(d)}
    t = tuple(a.A.keys())
    D = tuple(a.config.backend.get_shape(x) for x in a.A.values())
    a.struct = _struct(t, D, tuple(a.s), tuple(a.n))


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
    # if a.config != b.config:
    if not (a.config.device == b.config.device
            and a.config.sym.SYM_ID == b.config.sym.SYM_ID
            and a.config.backend.BACKEND_ID == b.config.backend.BACKEND_ID):
        raise YastError('configs do not match')


def _test_tensors_match(a, b):
    if _check["signatures_match"] and (not np.all(a.s == b.s) or not np.all(a.n == b.n)):
        raise YastError('Tensor signatures do not match')
    if _check["consistency"] and not a.meta_fusion == b.meta_fusion:
        raise YastError('Fusion trees do not match')


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
