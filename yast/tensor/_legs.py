import numpy as np
from typing import NamedTuple
from ._auxliary import _flatten
from ._tests import YastError
from ..sym import sym_none

__all__ = ['Leg']


class _Leg(NamedTuple):
    sym: any = sym_none
    s: int = 1  # leg signature in (1, -1)
    t: tuple = ()  # leg charges
    D: tuple = ()  # and their dimensions
    fused: tuple = ()  # subspaces

    def conj(self):
        """ switch leg signature """
        return self._replace(s=-self.s)


class _metaLeg(NamedTuple):
    legs: tuple = ()
    mf: tuple = (1,)  # order of (meta) fusions


def Leg(config, s=1, t=(), D=(), **kwargs):
    """ Check input to create a new _Leg. """

    sym = config if hasattr(config, 'SYM_ID') else config.sym
    if s not in (-1, 1):
        raise YastError('Signature of Leg should be 1 or -1')

    D = tuple(_flatten(D))
    t = tuple(_flatten(t))
    if not all(int(x) == x and x > 0 for x in D):
        raise YastError('D should be a tuple of positive ints')
    if not all(int(x) == x for x in t):
        raise YastError('Charges should be ints')
    if len(D) * sym.NSYM != len(t) or (sym.NSYM == 0 and len(D) != 1):
        raise YastError('Number of provided charges and bond dimensions do not match sym.NSYM')
    newt = tuple(tuple(x.flat) for x in sym.fuse(np.array(t).reshape((len(D), 1, sym.NSYM)), (s,), s))
    oldt = tuple(tuple(x.flat) for x in np.array(t).reshape(len(D), sym.NSYM))
    if oldt != newt:
        raise YastError('Provided charges are outside of the natural range for specified symmetry.')
    if len(set(newt)) != len(newt):
        raise YastError('Repeated charge index.')
    tD = {x: d for x, d in zip(newt, D)}
    t = tuple(sorted(newt))
    D = tuple(tD[x] for x in t)
    return _Leg(sym=sym, s=s, t=t, D=D)
