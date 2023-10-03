""" Auxliary functions. """
from typing import NamedTuple
from itertools import accumulate, chain
from ..sym import sym_none


class _struct(NamedTuple):
    s: tuple = ()  # leg signatures
    n: tuple = ()  # tensor charge
    diag: bool = False  # isdiag
    t: tuple = ()  # list of block charges
    D: tuple = ()  # list of block shapes
    size: int = 0  # total data size


class _slc(NamedTuple):
    slcs: tuple = ()  # slice
    D: tuple = ()  # reshape
    Dp: int = 0  # product of D


class _config(NamedTuple):
    backend: any = None
    sym: any = sym_none
    fermionic: tuple = False
    default_device: str = 'cpu'
    default_dtype: str = 'float64'
    default_fusion: str = 'hard'
    force_fusion: str = None


def _flatten(nested_iterator):
    for item in nested_iterator:
        try:
            yield from _flatten(item)
        except TypeError:
            yield item


def _unpack_axes(mfs, *args):
    """ Unpack meta axes into native axes based on a.mfs """
    clegs = tuple(accumulate(x[0] for x in mfs))
    return tuple(tuple(chain(*(range(clegs[ii] - mfs[ii][0], clegs[ii]) for ii in axes))) for axes in args)


def _clear_axes(*args):
    return ((axis,) if isinstance(axis, int) else tuple(axis) for axis in args)


def _ntree_to_mf(ntree):
    """ Change nested lists into linear fusion tree. """
    mf = ()
    for subtree in ntree:
        mf = mf + _ntree_to_mf(subtree)
    nlegs = max(1, sum(x == 1 for x in mf))
    return (nlegs,) + mf


def _mf_to_ntree(mf):
    """ Change linear fusion tree into nested lists. """
    ntree = []
    if mf[0] > 1:
        pos_init, cum = 1, 0
        for pos, nlegs in enumerate(mf[1:]):
            if cum == 0:
                cum = nlegs
            if nlegs == 1:
                cum = cum - 1
                if cum == 0:
                    ntree.append(_mf_to_ntree(mf[pos_init:pos + 2]))
                    pos_init = pos + 2
    return ntree


def _unpack_legs(legs):
    """ Return native legs and mfs. """
    ulegs, mfs = [], []
    for leg in legs:
        if isinstance(leg.fusion, tuple):  # meta-fused
            ulegs.extend(leg.legs)
            mfs.append(leg.fusion)
        else:  #_Leg
            ulegs.append(leg)
            mfs.append((1,))
    return tuple(ulegs), tuple(mfs)
