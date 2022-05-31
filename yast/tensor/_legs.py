import numpy as np
from typing import NamedTuple
from ._auxliary import _flatten
from ._tests import YastError
from ..sym import sym_none
from ._merging import _Fusion

__all__ = ['Leg', 'leg_union']


class _Leg(NamedTuple):
    sym: any = sym_none
    s: int = 1  # leg signature in (1, -1)
    t: tuple = ()  # leg charges
    D: tuple = ()  # and their dimensions
    hf: tuple = ()  # fused subspaces

    def conj(self):
        """ switch leg signature """
        return self._replace(s=-self.s, hf=self.hf.conj())


class _metaLeg(NamedTuple):
    legs: tuple = ()
    mf: tuple = (1,)  # order of (meta) fusions
    t: tuple = ()  # meta-leg charges combinations
    D: tuple = ()  # and their dimensions

    def conj(self):
        """ switch leg signature """
        return self._replace(legs=tuple(leg.conj() for leg in self.legs))


def Leg(config, s=1, t=(), D=(), hf=None):
    """ 
    Create a new _Leg.

    Verifies if the input is consistent.
    """

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

    if hf is None:
        hf = _Fusion(s=(s,))
    if hf.s[0] != s:
        raise YastError('Provided hard_fusion and signature do not match')
    return _Leg(sym=sym, s=s, t=t, D=D, hf=hf)



def _combine_tD(*legs):
    tD = {}
    for leg in legs:
        for t, D in zip(leg.t, leg.D):
            if t in tD and tD[t] != D:
                raise YastError('Legs have inconsistent dimensions')
            tD[t] = D
    t = tuple(sorted(tD.keys()))
    D = tuple(tD[x] for x in t)
    return t, D


def _leg_union(*legs):
    """
    Output _Leg that represent space being an union of spaces of a list of legs.
    """
    legs = list(legs)
    if any(leg.sym.SYM_ID != legs[0].sym.SYM_ID for leg in legs):
        raise YastError('Legs have different symmetries')
    if any(leg.s != legs[0].s for leg in legs):
        raise YastError('Legs have different signatures')
    if any(leg.hf != legs[0].hf for leg in legs):
        raise YastError('Leg union does not support union of fused spaces - TODO')
    t, D = _combine_tD(*legs)
    return _Leg(sym=legs[0].sym, s=legs[0].s, t=t, D=D, hf=legs[0].hf)


def leg_union(*legs):
    """
    Output _Leg that represent space being an union of spaces of a list of legs.
    """
    legs = list(legs)
    if all(isinstance(leg, _Leg) for leg in legs):
        return _leg_union(*legs)
    if all(isinstance(leg, _metaLeg) for leg in legs):
        mf = legs[0].mf
        if any(leg.mf != mf for leg in legs):
            raise YastError('Meta-fusions do not match')
        new_nlegs = tuple(_leg_union(*(mleg.legs[n] for mleg in legs)) for n in range(mf[0]))
        t, D = _combine_tD(*legs)
        return _metaLeg(legs=new_nlegs, mf=legs[0].mf, t=t, D=D)
    raise YastError('All arguments of leg_union should be Legs or meta-fused Legs.')
