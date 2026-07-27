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

import hashlib
from functools import lru_cache
from itertools import accumulate, chain
from math import prod
from typing import NamedTuple, Sequence

import numpy as np

from .._profile import nsys_profile
from ..sym import sym_none

__all__ = ['_config', '_struct', 'get_blocks', 'hash_blocks', 'sign_canonical_order', 'swap_charges', 'find_matching_indices', 'HashedMask']


class _config(NamedTuple):
    backend: any = None
    sym: any = sym_none
    fermionic: tuple = False
    default_device: str = 'cpu'
    default_dtype: str = 'float64'
    default_fusion: str = 'hard'
    force_fusion: str = None
    tensordot_policy: str = 'fuse_contracted'


class HashedMask:
    __slots__ = ('_arr', '_hash')
    _NONE_HASH = hash(None)

    def __init__(self, arr):
        if arr is None or sum(arr) == len(arr):  # no mask, or all True
            self._arr = None
            self._hash = self._NONE_HASH
        else:
            if isinstance(arr, (tuple, list)):
                arr = np.array(arr, dtype=bool)
            arr.flags.writeable = False
            self._arr = arr
            self._hash = hash(arr.data.tobytes())

    @property
    def array(self) -> np.ndarray | None:
        return self._arr

    def tolist(self) -> np.ndarray | None:
        return self._arr if self._arr is None else self._arr.tolist()

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if not isinstance(other, HashedMask):
            return False
        if self is other:
            return True
        if self._hash != other._hash:
            return False
        if self._arr is None or other._arr is None:
            return self._arr is other._arr
        return np.array_equal(self._arr, other._arr)


class _struct(NamedTuple):
    legs: tuple = ()  # tuple[LegBasic]
    n: tuple = ()  # tensor charge
    isdiag: bool = False  # isdiag
    mask: HashedMask = HashedMask(None)

    def replace(self, **kwargs):
        if 'mask' in kwargs and not isinstance(kwargs['mask'], HashedMask):
            kwargs['mask'] = HashedMask(kwargs['mask'])
        if 'legs' in kwargs and not isinstance(kwargs['legs'], tuple):
            kwargs['legs'] = tuple(kwargs['legs'])
        return self._replace(**kwargs)

    def mask_from_ind(self, nblocks, ind):
        mask = np.zeros(nblocks, dtype=bool)
        mask[ind] = True
        return self.replace(mask=mask)


class _blocks(NamedTuple):
    t: np.array = None  # list of block charges nblocks x ndim_n x nsym
    D: np.array = None  # list of block shapes  nblocks x ndim_n
    slc: np.array = None  # list of block slices nblocks x 2
    size: int = 0  # data size
    nblocks: int = 0  # number of blocks
    coords: np.array = None  # list of block coordinates


def hash_blocks(bl, out=str) -> str | bytes:
    """
    Persistent, cross-process hash of a NamedTuple whose fields are either
    hashable Python objects or numpy arrays (e.g. :class:`_blocks`).

    The digest is stable across interpreter runs. Array fields are hashed by
    content together with their shape and dtype; contiguity is canonicalized so
    that equivalent C-/F-ordered arrays yield the same digest.
    """
    h = hashlib.blake2b()
    for v in bl:
        if isinstance(v, np.ndarray):
            v = np.ascontiguousarray(v)
            h.update(str(v.shape).encode())
            h.update(v.dtype.str.encode())
            h.update(v.tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest() if out is str else h.digest()


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


def _compress_slices(meta):
    if len(meta) == 0:
        return meta
    s1, s2 = meta.shape
    c2 = s2 // 2
    slcs = meta.reshape(s1, c2, 2).transpose((1, 0, 2))
    inds = np.any(slcs[:, 1:, 0] != slcs[:, :-1, 1], axis=0)
    mask = np.ones((s1, 2), dtype=bool)
    mask[1:, 0] = inds
    mask[:-1, 1] = inds
    mask = mask.reshape(2 * s1)
    n1 = np.sum(inds) + 1
    slcs = slcs.reshape(c2, 2 * s1)
    joined_slcs = slcs[:, mask].reshape(c2, n1, 2).transpose((1, 0, 2))
    return np.ascontiguousarray(joined_slcs.reshape(n1, s2))


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


@lru_cache(maxsize=1024)
@nsys_profile
def get_blocks(sym, struct) -> _blocks:
    """
    Generate all allowed block charges, their dimensions, slices, coordinates and total size.
    """
    saxes = tuple(leg.s for leg in struct.legs)
    taxes = tuple(leg.t for leg in struct.legs)
    tblocks, iblocks = get_blocks_charges_all(sym, taxes, saxes, struct.n)

    if struct.mask.array is not None:
        tblocks = tblocks[struct.mask.array]
        iblocks = iblocks[struct.mask.array]

    Dblocks = np.empty(iblocks.shape, dtype=np.int64)
    for i, leg in enumerate(struct.legs):
        Dax = np.array(leg.D, dtype=np.int64)
        Dblocks[:, i] = Dax[iblocks[:, i]]

    nblocks = len(iblocks)
    #
    slices = np.zeros((nblocks, 2), dtype=np.int64)
    Dp = Dblocks[:, 0] if struct.isdiag else np.prod(Dblocks, axis=1, dtype=np.int64)
    np.cumsum(Dp, out=slices[:, 1])
    slices[1:, 0] = slices[:-1, 1]
    size = np.sum(Dp, dtype=np.int64).item()
    #
    return _blocks(t=tblocks, D=Dblocks, slc=slices, size=size, nblocks=nblocks, coords=iblocks)


@nsys_profile
def get_trimmed_struct(sym, struct, sub_legs=None):
    saxes = tuple(int(leg.s) for leg in struct.legs)
    # taxes_full = tuple(leg.t for leg in struct.legs)
    # taxes_full = tuple(tuple(tt for tt, d in zip(leg.t, leg.D) if d > 0) for leg in struct.legs)
    taxes_full = tuple(tuple(tuple(map(int, tt)) for tt, d in zip(leg.t, leg.D) if d > 0) for leg in struct.legs)
    if sub_legs is None:
        taxes_sub = taxes_full
    else:
        # taxes_sub = tuple(leg.t for leg in sub_legs)
        taxes_sub = tuple(tuple(tuple(map(int, tt)) for tt, d in zip(leg.t, leg.D) if d > 0) for leg in sub_legs)
    taxes_new, mask_sub = get_trimmed_struct_engine(sym, taxes_full, saxes, struct.n, struct.mask, taxes_sub)
    legs_old = struct.legs if sub_legs is None else sub_legs  # use sub_legs to update bond dims e.g. in mask
    legs_new = tuple(leg.trim(tax) for leg, tax in zip(legs_old, taxes_new))
    return _struct(legs=tuple(legs_new), n=struct.n, isdiag=struct.isdiag, mask=mask_sub)


@lru_cache(maxsize=1024)
@nsys_profile
def get_trimmed_struct_engine(sym, taxes_full, saxes, n, mask, taxes_sub):
    #
    tblocks_full, _ = get_blocks_charges_all(sym, taxes_full, saxes, n)
    if mask.array is not None:
        tblocks_full = tblocks_full[mask.array]
    #
    tblocks_sub, iblocks_sub = get_blocks_charges_all(sym, taxes_sub, saxes, n)
    inds_sub = find_matching_indices(tblocks_sub, tblocks_full, both=False)
    iblocks_sub = iblocks_sub[inds_sub]
    icharges_sub = [np.unique(iblocks_sub[:, i]).tolist() for i in range(len(taxes_sub))]
    taxes_new = tuple(tuple(tax[i] for i in inds) for tax, inds in zip(taxes_sub, icharges_sub))

    if taxes_new != taxes_sub:
        return get_trimmed_struct_engine(sym, taxes_full, saxes, n, mask, taxes_new)

    if mask.array is not None:
        mask_sub = np.zeros(len(tblocks_sub), dtype=bool)
        mask_sub[inds_sub] = True
        mask_sub = HashedMask(mask_sub)
    else:
        mask_sub = mask

    return taxes_new, mask_sub


def _product_indices(sizes):
    """ Indices of the cartesian product of ``range(n)``, in lexicographic (C) order. """
    if len(sizes) == 0:
        return np.zeros((1, 0), dtype=np.int64)
    return np.indices(sizes, dtype=np.int64).reshape(len(sizes), -1).T


def _row_view(a):
    """ 1d structured view of the rows of a 2d array, so that rows can be sorted and searched. """
    a = np.ascontiguousarray(a)
    return a.view([(f'f{i}', a.dtype) for i in range(a.shape[1])]).ravel()


def _group_charges(sym, tarr, s, axes):
    """ Cartesian product of charges on ``axes``: indices, charges, and what they fuse to. """
    indices = _product_indices([len(tarr[i]) for i in axes])
    t = np.empty((len(indices), len(axes), sym.NSYM), dtype=np.int64)
    for i, ax in enumerate(axes):
        t[:, i, :] = tarr[ax][indices[:, i], :]
    return indices, t, sym.fuse(t, np.array([s[ax] for ax in axes], dtype=np.int64), 1)


@lru_cache(maxsize=1024)
@nsys_profile
def get_blocks_charges_all(sym, taxes: Sequence[Sequence[int]], s: Sequence[int], n: Sequence[int]):
    """
    Params
    ------
    taxes: List of lists of charges for each leg
    """
    _SPLIT_MIN = 4096
    nsym = sym.NSYM
    ndim = len(taxes)
    sizes = [len(tax) for tax in taxes]

    if 0 in sizes:  # a leg carrying no charges admits no blocks
        return (np.zeros((0, ndim, nsym), dtype=np.int64),
                np.zeros((0, ndim), dtype=np.int64))

    tarr = tuple(np.array(tax, dtype=np.int64).reshape(len(tax), nsym) for tax in taxes)

    # Enumerating every combination of charges costs prod(sizes), i.e. exponentially many in the
    # rank, while only a small fraction of them fuses to n. Instead, split the legs into two
    # groups of comparable size, enumerate each group on its own -- about sqrt(prod(sizes))
    # combinations each -- and join the two groups on the charge they have to share.
    h = ndim if nsym == 0 or prod(sizes) <= _SPLIT_MIN else \
        min((max(prod(sizes[:c]), prod(sizes[c:])), c) for c in range(ndim + 1))[1]

    lind, lt, lfused = _group_charges(sym, tarr, s, range(h))

    if h == ndim:  # a single group, so filter it directly
        ind = np.all(lfused == np.array(n, dtype=np.int64), axis=1)
        tblocks, iblocks = lt[ind], lind[ind]
    elif sym.add_charges(n) != tuple(n):
        # fused charges are canonical representatives of their sector, so a non-canonical n
        # is matched by none of them
        tblocks = np.zeros((0, ndim, nsym), dtype=np.int64)
        iblocks = np.zeros((0, ndim), dtype=np.int64)
    else:
        rind, rt, rfused = _group_charges(sym, tarr, s, range(h, ndim))

        # charge the first group has to carry to complete each combination of the second, n - r
        nt = np.broadcast_to(np.array(n, dtype=np.int64), (len(rfused), nsym))
        rneed = sym.fuse(np.stack([nt, rfused], axis=1), np.array([1, -1], dtype=np.int64), 1)

        # join the groups on that charge, keeping the lexicographic order of the indices
        order = np.argsort(_row_view(rneed), kind='stable')
        rkeys = _row_view(rneed[order])
        lkeys = _row_view(lfused)
        lo = np.searchsorted(rkeys, lkeys, side='left')
        counts = np.searchsorted(rkeys, lkeys, side='right') - lo

        nblocks = int(counts.sum())
        starts = np.zeros(len(counts), dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        # ragged gather: for each row of the first group, its matches in their original order
        lsel = np.repeat(np.arange(len(lind), dtype=np.int64), counts)
        rsel = order[np.repeat(lo - starts, counts) + np.arange(nblocks, dtype=np.int64)]

        iblocks = np.empty((nblocks, ndim), dtype=np.int64)
        iblocks[:, :h], iblocks[:, h:] = lind[lsel], rind[rsel]
        tblocks = np.empty((nblocks, ndim, nsym), dtype=np.int64)
        tblocks[:, :h], tblocks[:, h:] = lt[lsel], rt[rsel]

    return tblocks, iblocks


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


@nsys_profile
def find_matching_indices(tset1, tset2, both=True):
    rs1, *cs1 = tset1.shape
    rs2, *cs2 = tset2.shape
    assert cs1 == cs2, "Sanity check. Contact developers."
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


@nsys_profile
def locate_rows(tset, query):
    """
    Index of each row of ``query`` within ``tset``, or ``len(tset)`` when the row is absent.

    ``tset`` must have unique rows sorted in the same (per-column, lexicographic) order
    produced by ``np.unique(..., axis=0)``; ``query`` may contain arbitrary/repeated rows.
    Unlike :func:`find_matching_indices`, the result has one entry per row of ``query``
    (misses flagged with the sentinel ``len(tset)``), so it maps rows to their class id.
    """
    rs, *cs = tset.shape
    rq, *cq = query.shape
    assert cs == cq, "Sanity check. Contact developers."
    cp = np.prod(cs, dtype=np.int64)
    if rs == 0:  # empty tset: every query row is absent
        return np.full(rq, rs, dtype=np.int64)
    if cp == 0:  # zero-width rows collapse to a single (empty) class present in tset
        return np.zeros(rq, dtype=np.int64)
    tset = tset.reshape(rs, cp)
    query = query.reshape(rq, cp)
    struct_dt = np.dtype([('', tset.dtype)] * cp)
    tset_view = np.ascontiguousarray(tset).view(struct_dt).ravel()
    query_view = np.ascontiguousarray(query).view(struct_dt).ravel()

    ids = np.searchsorted(tset_view, query_view)
    hit = ids < rs
    safe_ind = np.where(hit, ids, 0)
    hit &= tset_view[safe_ind] == query_view
    ids[~hit] = rs  # sentinel for rows absent from tset
    return ids
