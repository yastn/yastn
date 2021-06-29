""" Auxliary functions. """

from typing import NamedTuple
from itertools import accumulate, chain
import numpy as np
from ..sym import sym_none


class _struct(NamedTuple):
    t: tuple = ()
    D: tuple = ()
    s: tuple = ()
    n: tuple = ()


class _config(NamedTuple):
    backend: any = None
    sym: any = sym_none
    device: str = 'cpu'
    default_device: str = 'cpu'
    default_dtype: str = 'float64'


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
    a.struct = _struct(t, D, a.struct.s, a.struct.n)
