""" Auxliary functions for syntax """
from itertools import accumulate, chain
import numpy as np


def _unpack_axes(a, *args):
    """Unpack meta axes into native axes based on self.meta_fusion"""
    clegs = tuple(accumulate(x[0] for x in a.meta_fusion))
    return (tuple(chain(*(range(clegs[ii] - a.meta_fusion[ii][0], clegs[ii]) for ii in axes))) for axes in args)


def _clear_axes(*args):
    return ((axis,) if isinstance(axis, int) else tuple(axis) for axis in args)


def _common_keys(d1, d2):
    """
    Divide keys into: common, only in d1 and only in d2.

    Returns: keys12, keys1, keys2
    """
    s1 = set(d1)
    s2 = set(d2)
    return tuple(s1 & s2), tuple(s1 - s2), tuple(s2 - s1)


def _indices_common_rows(a, b):
    """
    Return indices of a that are in b, and indices of b that are in a.
    """
    la = [tuple(x.flat) for x in a]
    lb = [tuple(x.flat) for x in b]
    sa = set(la)
    sb = set(lb)
    ia = np.array([ii for ii, el in enumerate(la) if el in sb], dtype=np.intp)
    ib = np.array([ii for ii, el in enumerate(lb) if el in sa], dtype=np.intp)
    return ia, ib
