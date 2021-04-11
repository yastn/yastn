""" Auxliary functions for syntax """
from collections import namedtuple
from itertools import accumulate, chain
from ..sym import sym_none
import numpy as np


_struct = namedtuple('_struct', ('t', 'D', 's', 'n'))


_config = namedtuple('_config', ('backend', 'sym', 'dtype', 'device'),
                     defaults = (None, sym_none, 'float64', 'cpu'))


def _flatten(nested_iterator):
    for item in nested_iterator:
        try:
            yield from _flatten(item)
        except TypeError:
            yield item


def _unpack_axes(a, *args):
    """Unpack meta axes into native axes based on a.meta_fusion"""
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
    ia = tuple(ii for ii, el in enumerate(la) if el in sb)
    ib = tuple(ii for ii, el in enumerate(lb) if el in sa)
    # ia = np.array([ii for ii, el in enumerate(la) if el in sb], dtype=np.intp)
    # ib = np.array([ii for ii, el in enumerate(lb) if el in sa], dtype=np.intp)
    return ia, ib


def _tarray(a):
    return np.array(a.struct.t, dtype=int).reshape(len(a.struct.t), a.nlegs, a.config.sym.nsym)


def _Darray(a):
    return np.array(a.struct.D, dtype=int).reshape(len(a.struct.D), a.nlegs)


def _tDarrays(a):
    return np.array(a.struct.t, dtype=int).reshape(len(a.struct.t), a.nlegs, a.config.sym.nsym), np.array(a.struct.D, dtype=int).reshape(len(a.struct.D), a.nlegs)


def update_struct(a):
    """Updates meta-information about charges and dimensions of all blocks."""
    d = a.A
    a.A = {k: d[k] for k in sorted(d)}
    t = tuple(a.A.keys())
    D = tuple(a.config.backend.get_shape(x) for x in a.A.values())
    a.struct = _struct(t, D, tuple(a.s), tuple(a.n))
