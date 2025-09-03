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
import abc
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import NamedTuple
from ..sym import sym_none


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


class SpecialTensor(metaclass=abc.ABCMeta):
    """
    A parent class to create a special tensor-like object.

    ``yastn.tensordot(a, b, axes)`` check if ``a`` or ``b`` is an instance of SpecialTensor
    and calls ``a.tensordo(b, axes)`` or ``b.tensordo(a, axes, reverse=True)``
    """

    @abc.abstractmethod
    def tensordot(self, b, axes, reverse=False):
        pass  # pragma: no cover


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
    tensordot_policy: str = 'fuse_contracted'


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


def _unpack_legs(legs):
    """ Return native legs and mfs. """
    ulegs, mfs = [], []
    for leg in legs:
        if hasattr(leg, 'mf'):  # meta-fused
            mfs.append(leg.mf)
            ulegs.extend(leg.legs)
        else:  # _Leg
            mfs.append((1,))
            ulegs.append(leg)
    return tuple(ulegs), tuple(mfs)


def _join_contiguous_slices(slcs_a, slcs_b):
    if not slcs_a:
        return ()
    meta = []
    tmp_a = slcs_a[0]
    tmp_b = slcs_b[0]
    for sl_a, sl_b in zip(slcs_a[1:], slcs_b[1:]):
        if tmp_a[1] == sl_a[0] and tmp_b[1] == sl_b[0]:
            tmp_a = (tmp_a[0], sl_a[1])
            tmp_b = (tmp_b[0], sl_b[1])
        else:
            meta.append((tmp_a, tmp_b))
            tmp_a = sl_a
            tmp_b = sl_b
    meta.append((tmp_a, tmp_b))
    return tuple(meta)
