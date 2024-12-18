# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Auxliary functions used by yastn.Tensor. """
from typing import NamedTuple
from itertools import accumulate, chain
from ..sym import sym_none
from dataclasses import dataclass


@dataclass
class Method():
    """
    Auxiliary mutable method class.
    It introduces the mechanism to change the method used in :meth:`yastn.tn.mps.dmrg_`, :meth:`yastn.tn.mps.tdvp_`,
    and other generator functions in between consecutive sweeps.
    Updating the value in place will inject the new value back into the generator.
    """
    string: str = ''

    def __eq__(self, string):
        return string == self.string

    def __str__(self):
        return self.string

    def update_(self, string):
        """ Update the method name in place. """
        self.string = str(string)


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
    """ Unpack meta axes into native axes based on ``a.mfs``. """
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
