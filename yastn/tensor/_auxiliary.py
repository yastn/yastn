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
""" Auxiliary functions used by yastn.Tensor. """
from __future__ import annotations

from itertools import accumulate, chain
from typing import NamedTuple

import numpy as np

from ..sym import sym_none

__all__ = ['_config', '_struct', 'get_blocks', 'sign_canonical_order', 'swap_charges']


class _blocks(NamedTuple):
    t: np.array = None  # leg signatures
    D: np.array = None  # tensor charge
    slc: np.array = None  # isdiag
    size: int = 0  # list of block charges
    nblocks: int = 0  # list of block shapes
    legs: tuple = ()  # total data size
    n: tuple = ()  # tensor charge
    isdiag: bool = False  # isdiag


class _struct(NamedTuple):
    legs: tuple = ()  # tuple[LegBasic]
    n: tuple = ()  # tensor charge
    isdiag: bool = False  # isdiag


class _config(NamedTuple):
    backend: any = None
    sym: any = sym_none
    fermionic: tuple = False
    default_device: str = 'cpu'
    default_dtype: str = 'float64'
    default_fusion: str = 'hard'
    force_fusion: str = None
    tensordot_policy: str = 'fuse_contracted'
    profile: bool = False


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


def swap_charges(charges_0, charges_1, fss) -> int:
    """ Calculates a sign accumulated while swapping lists of charges. """
    if not fss:
        return 1
    t0 = np.array(charges_0, dtype=np.int64)
    t1 = np.array(charges_1, dtype=np.int64)
    if fss is True:
        return 1 - 2 * (np.sum(t0 * t1, dtype=np.int64).item() % 2)
    return 1 - 2 * (np.sum((t0 * t1)[:, fss], dtype=np.int64).item() % 2)


def sign_canonical_order(*operators, sites=None, f_ordered=None) -> int:
    """
    Calculates a sign corresponding to the commutation of operators into canonical order,
    where the corresponding sites get ordered according to fermionic order.
    We assume that in canonical ordering, the operators at sites appearing
    later in the fermionic order are applied first.

    For instance, consider operators O, P at sites=(s0, s1),
    which corresponds to a product operator Q = O_s0 P_s1.
    If s0 <= s1 in fermionic order, then the sign is 1.
    If s1 < s0 in fermionic order, Q = sign * P_s1 O_s0,
    where the sign follows from swapping the charges carried by O and P.
    Operators at the same site are not swapped.

    Parameters
    ----------
    operators: Sequence[yastn.Tensor]
        List of local operators to calculate <O_s0 Ps_s1 ...>.

    sites: Sequence[Sites]
        A list of sites [s0, s1, ...] matching the operators.

    f_order: Callable
        Function with 2 arguments, telling whether 2 sites are fermionically ordered.
    """

    if not operators or not operators[0].config.fermionic:
        return 1

    sites = list(sites)
    # operators = list(operators)
    charges = [op.n for op in operators]

    charges_0, charges_1 = [], []

    while sites:
        first_ind, first_site = 0, sites[0]
        for ind, site in enumerate(sites[1:], start=1):
            if not f_ordered(first_site, site):
                first_ind, first_site = ind, site
        sites.pop(first_ind)
        c1 = charges.pop(first_ind)
        for c0 in charges[:first_ind]:
            charges_0.append(c0)
            charges_1.append(c1)

    if len(charges_0) == 0:
        return 1
    return swap_charges(charges_0, charges_1, operators[0].config.fermionic)


def get_blocks(sym, legs, n, isdiag=False):
    """
    Generate all allowed block charges, their dimensions, slices, total size and trimed legs.
    Assume that legs have sorted charges
    """
    s = tuple(leg.s for leg in legs)
    taxes = tuple(leg.t for leg in legs)
    tblocks, iblocks, icharges = get_blocks_charges(sym, taxes, s, n)

    Dblocks = np.empty(iblocks.shape, dtype=np.int64)
    for i, leg in enumerate(legs):
        Dax = np.array(leg.D, dtype=np.int64)
        Dblocks[:, i] = Dax[iblocks[:, i]]

    nblocks = len(iblocks)
    #
    slices = np.zeros((nblocks, 2), dtype=np.int64)
    Dp = Dblocks[:, 0] if isdiag else np.prod(Dblocks, axis=1, dtype=np.int64)
    np.cumsum(Dp, out=slices[:, 1])
    slices[1:, 0] = slices[:-1, 1]
    size = np.sum(Dp, dtype=np.int64).item()
    #
    # recalculate legs, in case some leg charges do not appear in any block
    new_legs = []
    for leg, inds in zip(legs, icharges):
        tl = tuple(leg.t[i] for i in inds)
        Dl = tuple(leg.D[i] for i in inds)
        leg = type(leg)(s=leg.s, t=tl, D=Dl)
        new_legs.append(leg)
    #
    return _blocks(t=tblocks, D=Dblocks, slc=slices, size=size, nblocks=nblocks, legs=tuple(new_legs), n=n, isdiag=isdiag)


def get_blocks_charges(sym, taxes, s, n):
    nsym = sym.NSYM
    ndim = len(taxes)
    if ndim > 0:
        indices = np.indices([len(tax) for tax in taxes]).reshape(ndim, -1).T
    else:
        indices = np.zeros((1, ndim), dtype=np.int64)

    comb_t = np.empty((len(indices), ndim, nsym), dtype=np.int64)
    for i, tax in enumerate(taxes):
        tax = np.array(tax, dtype=np.int64).reshape(len(tax), nsym)
        comb_t[:, i, :] = tax[indices[:, i], :]

    s = np.array(s, dtype=np.int64)
    ind = np.all(sym.fuse(comb_t, s, 1) == n, axis=1)

    tblocks = comb_t[ind]
    iblocks = indices[ind]
    icharges = [sorted(np.unique(iblocks[:, i]).tolist()) for i in range(ndim)]
    return tblocks, iblocks, icharges


def get_sub_slices(st, st_full):
    inds = np.zeros(st.nblocks, dtype=np.int64)
    ic = 0
    for it, tt in enumerate(st.t):
        while not np.array_equal(tt, st_full.t[ic]):
            ic += 1
        inds[it] = ic
    return st_full.slc[inds]


def update_old_struct(st_new):
    struct_new = _struct(legs=st_new.legs, n=st_new.n, isdiag=st_new.isdiag)
    return struct_new


def find_index(tset, tt, sorted=True):
    rs, *cs = tset.shape
    cp = np.prod(cs, dtype=np.int64)
    if not isinstance(tt, np.ndarray):
        tt = np.array(tt, dtype=np.int64)
    if cp == 0 and rs == 1 and tt.size == 0:
        return 0
    elif cp > 0 and rs > 0:
        tset = tset.reshape(rs, cp)
        tt = tt.reshape(cp)
        struct_dt = np.dtype([('', tset.dtype)] * cp)
        tset_view = np.ascontiguousarray(tset).view(struct_dt).ravel()
        tt_view = np.ascontiguousarray(tt).view(struct_dt).ravel()
        if not sorted:
            ind = np.where(tset_view == tt_view)[0]
            if len(ind) == 1:
                return ind[0]
        else:
            ind = np.searchsorted(tset_view, tt_view)[0]
            if ind < len(tset) and np.array_equal(tset[ind], tt):
                return ind
    raise ValueError()


def argsort_t(tset):
    rs, *cs = tset.shape
    cp = np.prod(cs, dtype=np.int64)
    if rs == 0:
        return np.array([], dtype=np.int64)
    if cp == 0 and rs == 1:
        return np.array([0], dtype=np.int64)
    tset = tset.reshape(rs, cp)
    struct_dt = np.dtype([('', tset.dtype)] * cp)
    tset_view = np.ascontiguousarray(tset).view(struct_dt).ravel()
    return np.argsort(tset_view)


def find_matching_indices(tset1, tset2, both=True):
    rs1, *cs1 = tset1.shape
    rs2, *cs2 = tset2.shape
    assert cs1 == cs2, "Sanity check."
    cp = np.prod(cs1, dtype=np.int64)
    if cp == 0 and rs1 == 1 and rs2 == 1:
        ind1 = ind2 = np.array([0], dtype=np.int64)
    elif cp > 0 and rs1 > 0 and rs2 > 0:
        tset1 = tset1.reshape(rs1, cp)
        tset2 = tset2.reshape(rs2, cp)
        struct_dt = np.dtype([('', tset1.dtype)] * cp)
        tset1_view = np.ascontiguousarray(tset1).view(struct_dt).ravel()
        tset2_view = np.ascontiguousarray(tset2).view(struct_dt).ravel()

        ind1 = np.searchsorted(tset1_view, tset2_view)
        mask = ind1 < rs1
        safe_ind = np.where(mask, ind1, 0)
        mask = mask & (tset1_view[safe_ind] == tset2_view)
        ind1 = ind1[mask]
        if both:
            ind2 = np.flatnonzero(mask)
    else:
        ind1 = ind2 = np.array([], dtype=np.int64)
    return (ind1, ind2) if both else ind1
