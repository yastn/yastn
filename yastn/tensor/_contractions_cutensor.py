# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
"""
Optimized (v2) builder of the cutensor tensordot meta.

This module isolates the accelerated variant of ``_meta_tensordot_cutensor`` and the row
helpers it relies on. The original reference variant (v1) lives in :mod:`._contractions`,
together with the pieces both variants share (leg matching ``_match_legs_tensordot`` and the
pair-index alignment ``_matched_pair_indices``); the A/B toggle ``_META_CUTENSOR_VERSION``
there selects between them. The int64 row encoder (:func:`_encode_rows_shared` /
:func:`_row_keys_pair`) stays in :mod:`._auxiliary`, where it also backs
:func:`find_matching_indices`.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .._profile import nsys_profile, nvtx_range
from ._auxiliary import (_encode_rows_shared, _row_keys_pair, get_blocks, hash_blocks,
                         _struct, HashedMask, get_blocks_charges_all, find_matching_indices)
from ._contractions import (_convert_bl_for_cutensor, _cutensor_meta, _match_legs_tensordot,
                            get_blocks_and_subslices, _matched_pair_indices,)
from ._tests import YastnError

#: Below this many rows the GPU (v3) matcher / mask fall back to numpy (launch+transfer
#: overhead dominates). Tunable via env ``YASTN_META_CUTENSOR_GPU_MIN``.
_GPU_MIN = int(os.environ.get("YASTN_META_CUTENSOR_GPU_MIN", "4096"))


@nsys_profile
def locate_rows(tset, query):
    """
    Index of each row of ``query`` within ``tset``, or ``len(tset)`` when the row is absent.

    ``tset`` must have unique rows sorted in the same (per-column, lexicographic) order
    produced by ``np.unique(..., axis=0)``; ``query`` may contain arbitrary/repeated rows.
    Unlike :func:`yastn.tensor._auxiliary.find_matching_indices`, the result has one entry
    per row of ``query`` (misses flagged with the sentinel ``len(tset)``), so it maps rows
    to their class id.
    """
    rs, *cs = tset.shape
    rq, *cq = query.shape
    assert cs == cq, "Sanity check. Contact developers."
    cp = np.prod(cs, dtype=np.int64)
    if rs == 0:  # empty tset: every query row is absent
        return np.full(rq, rs, dtype=np.int64)
    if cp == 0:  # zero-width rows collapse to a single (empty) class present in tset
        return np.zeros(rq, dtype=np.int64)
    v_t, v_q = _row_keys_pair(tset.reshape(rs, cp), query.reshape(rq, cp))
    ids = np.searchsorted(v_t, v_q)
    hit = ids < rs
    safe_ind = np.where(hit, ids, 0)
    hit &= v_t[safe_ind] == v_q
    ids[~hit] = rs  # sentinel for rows absent from tset
    return ids


def unique_rows(t, return_inverse=False, return_counts=False):
    """
    Drop-in accelerator for ``np.unique(t, axis=0, ...)`` on integer charge blocks.

    Encodes rows as order-preserving int64 keys and uniques those in 1-D, which is markedly
    faster than numpy's structured/void ``axis=0`` sort; falls back to ``np.unique(axis=0)``
    when the key space would overflow. Returns rows in the same order as ``np.unique(axis=0)``
    and preserves the input's row shape ``t.shape[1:]``.
    """
    rs = t.shape[0]
    cp = int(np.prod(t.shape[1:], dtype=np.int64))
    enc = _encode_rows_shared([t.reshape(rs, cp)]) if rs > 0 else None
    if enc is None:
        return np.unique(t, axis=0, return_inverse=return_inverse, return_counts=return_counts)
    uk, first, inv, cnt = np.unique(enc[0], return_index=True, return_inverse=True, return_counts=True)
    res = [t[first]]  # representative row per class, in key-sorted (== structured-sorted) order
    if return_inverse:
        res.append(inv.reshape(-1))
    if return_counts:
        res.append(cnt)
    return res[0] if len(res) == 1 else tuple(res)


@lru_cache(maxsize=1024)
@nsys_profile
def _meta_tensordot_cutensor_v2(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b):
    struct_a_sub, struct_b_sub, struct_c = _match_legs_tensordot(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    bl_a, slc_a = get_blocks_and_subslices(sym, struct_a_sub, struct_a)
    bl_b, slc_b = get_blocks_and_subslices(sym, struct_b_sub, struct_b)
    bl_c = get_blocks(sym, struct_c)
    
    with nvtx_range("unique in blocks"):
        unique_a, inv_a, count_a = unique_rows(bl_a.t[:, nin_a, :], return_inverse=True, return_counts=True) # if more blocks of a contribute to given contracted sector (in b)
        arg_a = np.argsort(inv_a)
        #
        unique_b, inv_b, count_b = unique_rows(bl_b.t[:, nin_b, :], return_inverse=True, return_counts=True)
        arg_b = np.argsort(inv_b)
    #
    ind_a, ind_b = _matched_pair_indices(unique_a, count_a, arg_a, unique_b, count_b, arg_b)
    #
    with nvtx_range("unique out blocks"):
        # A candidate output block of bl_c is produced iff its (out_a, out_b) charge tuple
        # arises from some matched (a, b) pair. Encode each out_a / out_b tuple as a small
        # integer id (one class per distinct tuple among the blocks) and reduce the per-pair
        # test to 1-D int64 operations. This avoids building and sorting the ~nn wide rows
        # that column_stack + np.unique(axis=0) would otherwise require (nn = matched pairs).
        na, nb, nsym = len(nout_a), len(nout_b), bl_a.t.shape[2]  # explicit widths: bl_*.nblocks may be 0
        ua, id_a = unique_rows(bl_a.t[:, nout_a, :].reshape(bl_a.nblocks, na * nsym), return_inverse=True)
        ub, id_b = unique_rows(bl_b.t[:, nout_b, :].reshape(bl_b.nblocks, nb * nsym), return_inverse=True)
        id_a, id_b, n_b = id_a.reshape(-1), id_b.reshape(-1), len(ub)
        #
        keys = np.unique(id_a[ind_a] * n_b + id_b[ind_b])  # produced (out_a, out_b) classes
        #
        # struct_c legs are ordered (out_a from nout_a, then out_b from nout_b), so bl_c.t
        # splits into its out_a (:na) and out_b (na:) columns aligned with ua / ub above
        cid_a = locate_rows(ua, bl_c.t[:, :na, :].reshape(bl_c.nblocks, na * nsym))
        cid_b = locate_rows(ub, bl_c.t[:, na:, :].reshape(bl_c.nblocks, nb * nsym))
        c_keys = np.where((cid_a < len(ua)) & (cid_b < n_b), cid_a * n_b + cid_b, -1)
        #
        mask = np.isin(c_keys, keys)  # -1 sentinel (tuple absent from a/b blocks) never matches
        struct_c = struct_c.replace(mask=mask)
        bl_c = get_blocks(sym, struct_c)
    
    # dot product, which cannot be simply dispatched to vdot
    #              and either one of operands is (effectively) zero
    if not (len(slc_a) > 0 and len(slc_b) > 0):
        return struct_c, 0, *([None] * 18)
    else:
        dot_product = len(nout_a) + len(nout_b) == 0
        a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets = \
            _convert_bl_for_cutensor(struct_a_sub, bl_a, slc_a, dot_product=(dot_product and len(slc_a) < len(slc_b)))
        b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets = \
            _convert_bl_for_cutensor(struct_b_sub, bl_b, slc_b, dot_product=(dot_product and len(slc_a) >= len(slc_b)))
        c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets = \
            _convert_bl_for_cutensor(struct_c, bl_c, dot_product=dot_product)

        h_a, h_b, h_c = hash_blocks(_cutensor_meta(a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides), out=bytes),\
                    hash_blocks(_cutensor_meta(b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides), out=bytes),\
                    hash_blocks(_cutensor_meta(c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides), out=bytes)

    return (struct_c, bl_c.size, h_a, h_b, h_c,
            a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets,
            b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets,
            c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets)


# =====================================================================================
# GPU (v3) path. torch is imported lazily *inside* functions so that ``import yastn`` and
# ``_control_lru`` (which imports this module) never pull torch for numpy-only users.
# The shared find_matching_indices and trimming stay untouched; v3 uses its own GPU copies.
# =====================================================================================


def _enc_torch(tsets, torch):
    """
    torch twin of :func:`_encode_rows_shared`: order-preserving int64 mixed-radix keys per row
    of each 2-D tensor in ``tsets`` (shared per-column value set), or ``None`` on int64 overflow.
    """
    cp = tsets[0].shape[1]
    sizes = [t.shape[0] for t in tsets]
    dev = tsets[0].device
    if cp == 0:
        keys = torch.zeros(sum(sizes), dtype=torch.int64, device=dev)
    else:
        stacked = torch.cat(tsets, 0) if len(tsets) > 1 else tsets[0]
        invs, radices, prod = [], [], 1
        for j in range(cp):
            u, inv = torch.unique(stacked[:, j], return_inverse=True)
            invs.append(inv.reshape(-1)); radices.append(u.numel()); prod *= u.numel()
            if prod >= (1 << 62):
                return None
        keys = torch.zeros(sum(sizes), dtype=torch.int64, device=dev)
        w = 1
        for j in range(cp - 1, -1, -1):  # column 0 most significant (matches np.unique(axis=0))
            keys += invs[j].to(torch.int64) * w
            w *= radices[j]
    out, off = [], 0
    for s in sizes:
        out.append(keys[off: off + s]); off += s
    return out


def _locate_torch(sorted_u, q, torch):
    """ torch twin of :func:`locate_rows` on 1-D keys: index of each ``q`` in ``sorted_u`` or
    the sentinel ``len(sorted_u)`` when absent. """
    n = sorted_u.numel()
    if n == 0:
        return torch.zeros(q.numel(), dtype=torch.int64, device=q.device)  # sentinel 0 == n
    ids = torch.searchsorted(sorted_u, q)
    hit = ids < n
    safe = torch.where(hit, ids, torch.zeros_like(ids))
    hit = hit & (sorted_u[safe] == q)
    ids = ids.clone()
    ids[~hit] = n
    return ids


def _matched_pairs_torch(sa, sb, torch):
    """
    Block-index lists ``(ind_a, ind_b)`` of every matching (a, b) pair, given the shared-space
    contracted-sector key of each a-/b-block. torch twin of ``_indices_from_counts`` + the
    ``arg_*`` gather: sort by sector, count per sector, union-align, then a segmented cartesian
    product. Pair order is irrelevant downstream (only the set of produced keys is used).
    """
    dev = sa.device
    ord_a = torch.argsort(sa); ord_b = torch.argsort(sb)
    ua, ca = torch.unique_consecutive(sa[ord_a], return_counts=True)
    ub, cb = torch.unique_consecutive(sb[ord_b], return_counts=True)
    starts_a = torch.cumsum(ca, 0) - ca
    starts_b = torch.cumsum(cb, 0) - cb
    uni = torch.unique(torch.cat([ua, ub]))
    pa = torch.searchsorted(uni, ua); pb = torch.searchsorted(uni, ub)
    ca2 = torch.zeros(uni.numel(), dtype=torch.int64, device=dev); ca2[pa] = ca
    cb2 = torch.zeros_like(ca2); cb2[pb] = cb
    sa2 = torch.zeros_like(ca2); sa2[pa] = starts_a
    sb2 = torch.zeros_like(ca2); sb2[pb] = starts_b
    prod = ca2 * cb2
    nn = int(prod.sum().item())
    if nn == 0:
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        return empty, empty
    seg = torch.repeat_interleave(torch.arange(uni.numel(), device=dev), prod)
    within = torch.arange(nn, device=dev) - (torch.cumsum(prod, 0) - prod)[seg]
    cbseg = cb2[seg]
    ind_a = ord_a[sa2[seg] + within // cbseg]
    ind_b = ord_b[sb2[seg] + within % cbseg]
    return ind_a, ind_b


def _output_mask_gpu(bl_a, bl_b, bl_c, nout_a, nin_a, nin_b, nout_b, device):
    """
    GPU twin of v2's mask-c step: boolean mask (numpy) over ``bl_c`` blocks that are actually
    produced. Only ``bl_*.t`` cross to the device (nn stays on GPU); returns ``None`` on int64
    overflow so the caller can fall back to numpy.
    """
    import torch
    nba, nbb, nbc = bl_a.nblocks, bl_b.nblocks, bl_c.nblocks
    na, nb = len(nout_a), len(nout_b)
    nsym = bl_a.t.shape[2]
    ta = torch.as_tensor(bl_a.t, device=device)
    tb = torch.as_tensor(bl_b.t, device=device)
    tc = torch.as_tensor(bl_c.t, device=device)
    # contracted-sector keys (shared a/b) -> matched (a, b) pairs
    enc = _enc_torch([ta[:, nin_a, :].reshape(nba, len(nin_a) * nsym),
                      tb[:, nin_b, :].reshape(nbb, len(nin_b) * nsym)], torch)
    if enc is None:
        return None
    ind_a, ind_b = _matched_pairs_torch(enc[0], enc[1], torch)
    # out-charge ids: a-out shared with bl_c[:na], b-out shared with bl_c[na:]
    enca = _enc_torch([ta[:, nout_a, :].reshape(nba, na * nsym),
                       tc[:, :na, :].reshape(nbc, na * nsym)], torch)
    encb = _enc_torch([tb[:, nout_b, :].reshape(nbb, nb * nsym),
                       tc[:, na:, :].reshape(nbc, nb * nsym)], torch)
    if enca is None or encb is None:
        return None
    ua_keys, id_a = torch.unique(enca[0], return_inverse=True)
    ub_keys, id_b = torch.unique(encb[0], return_inverse=True)
    cid_a = _locate_torch(ua_keys, enca[1], torch)
    cid_b = _locate_torch(ub_keys, encb[1], torch)
    n_b = ub_keys.numel()
    keys = torch.unique(id_a.reshape(-1)[ind_a] * n_b + id_b.reshape(-1)[ind_b])
    valid = (cid_a < ua_keys.numel()) & (cid_b < n_b)
    c_keys = torch.where(valid, cid_a * n_b + cid_b, torch.full_like(cid_a, -1))
    return torch.isin(c_keys, keys).cpu().numpy()


def _find_matching_indices_gpu(tset1, tset2, both=True, device=None):
    """
    GPU-accelerated drop-in for :func:`find_matching_indices` (numpy in / numpy out), used only
    by the v3-local trimming copies. Falls back to the untouched numpy ``find_matching_indices``
    for cpu device, small inputs, or int64 overflow. ``tset1`` must be sorted-unique (as the
    numpy version already requires).
    """
    rs1, *cs1 = tset1.shape
    rs2, *cs2 = tset2.shape
    cp = int(np.prod(cs1, dtype=np.int64))
    if (device is None or getattr(device, 'type', None) != 'cuda'
            or cp == 0 or rs1 == 0 or rs2 == 0 or max(rs1, rs2) < _GPU_MIN):
        return find_matching_indices(tset1, tset2, both=both)
    import torch
    t1 = torch.as_tensor(np.ascontiguousarray(tset1.reshape(rs1, cp)), device=device)
    t2 = torch.as_tensor(np.ascontiguousarray(tset2.reshape(rs2, cp)), device=device)
    enc = _enc_torch([t1, t2], torch)
    if enc is None:
        return find_matching_indices(tset1, tset2, both=both)
    v1, v2 = enc  # v1 ascending: tset1 sorted + order-preserving encoding
    ind1 = torch.searchsorted(v1, v2)
    m = ind1 < rs1
    safe = torch.where(m, ind1, torch.zeros_like(ind1))
    m = m & (v1[safe] == v2)
    out1 = ind1[m].cpu().numpy()
    if both:
        return out1, torch.nonzero(m, as_tuple=True)[0].cpu().numpy()
    return out1


# --- v3-local copies of the trimming path (shared versions in _contractions / _auxiliary are
# --- left untouched); the only change is find_matching_indices -> _find_matching_indices_gpu.

@lru_cache(maxsize=1024)
@nsys_profile
def _get_trimmed_struct_engine_gpu(sym, taxes_full, saxes, n, mask, taxes_sub, device):
    tblocks_full, _ = get_blocks_charges_all(sym, taxes_full, saxes, n)
    if mask.array is not None:
        tblocks_full = tblocks_full[mask.array]
    tblocks_sub, iblocks_sub = get_blocks_charges_all(sym, taxes_sub, saxes, n)
    inds_sub = _find_matching_indices_gpu(tblocks_sub, tblocks_full, both=False, device=device)
    iblocks_sub = iblocks_sub[inds_sub]
    icharges_sub = [np.unique(iblocks_sub[:, i]).tolist() for i in range(len(taxes_sub))]
    taxes_new = tuple(tuple(tax[i] for i in inds) for tax, inds in zip(taxes_sub, icharges_sub))

    if taxes_new != taxes_sub:
        return _get_trimmed_struct_engine_gpu(sym, taxes_full, saxes, n, mask, taxes_new, device)

    if mask.array is not None:
        mask_sub = np.zeros(len(tblocks_sub), dtype=bool)
        mask_sub[inds_sub] = True
        mask_sub = HashedMask(mask_sub)
    else:
        mask_sub = mask
    return taxes_new, mask_sub


@nsys_profile
def _get_trimmed_struct_gpu(sym, struct, sub_legs, device):
    saxes = tuple(int(leg.s) for leg in struct.legs)
    taxes_full = tuple(tuple(tuple(map(int, tt)) for tt, d in zip(leg.t, leg.D) if d > 0) for leg in struct.legs)
    if sub_legs is None:
        taxes_sub = taxes_full
    else:
        taxes_sub = tuple(tuple(tuple(map(int, tt)) for tt, d in zip(leg.t, leg.D) if d > 0) for leg in sub_legs)
    taxes_new, mask_sub = _get_trimmed_struct_engine_gpu(sym, taxes_full, saxes, struct.n, struct.mask, taxes_sub, device)
    legs_new = tuple(leg.trim(tax) for leg, tax in zip(struct.legs, taxes_new))
    return _struct(legs=tuple(legs_new), n=struct.n, isdiag=struct.isdiag, mask=mask_sub)


@nsys_profile
def _match_legs_tensordot_gpu(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b, device):
    assert not struct_a.isdiag, "Sanity check. Contact developers."
    assert not struct_b.isdiag, "Sanity check. Contact developers."
    n_c = sym.add_charges(struct_a.n, struct_b.n)
    #
    struct_a_0, struct_b_0 = struct_a, struct_b
    legs_a_new, legs_b_new = list(struct_a.legs), list(struct_b.legs)
    while True:
        for ia, ib in zip(nin_a, nin_b):
            try:
                leg = struct_a_0.legs[ia].intersection(struct_b_0.legs[ib].conj())
            except ValueError:
                raise YastnError('Bond dimensions of some charges do not match.')
            legs_a_new[ia] = leg
            legs_b_new[ib] = leg.conj()

        struct_a_1 = _get_trimmed_struct_gpu(sym, struct_a_0, legs_a_new, device)
        struct_b_1 = _get_trimmed_struct_gpu(sym, struct_b_0, legs_b_new, device)

        legs_c = (*(struct_a_1.legs[ax] for ax in nout_a), *(struct_b_1.legs[ax] for ax in nout_b))
        struct_c_0 = _struct(legs=legs_c, n=n_c, isdiag=False)
        struct_c_1 = _get_trimmed_struct_gpu(sym, struct_c_0, None, device)

        if struct_a_0 == struct_a_1 and struct_b_0 == struct_b_1 and struct_c_0 == struct_c_1:
            break

        struct_a_0 = struct_a_1
        struct_b_0 = struct_b_1
        for ii, ax in enumerate(nout_a):
            legs_a_new[ax] = struct_c_1.legs[ii].intersection(struct_a_1.legs[ax])
        for ii, ax in enumerate(nout_b, start=len(nout_a)):
            legs_b_new[ax] = struct_c_1.legs[ii].intersection(struct_b_1.legs[ax])

    return struct_a_1, struct_b_1, struct_c_1


def _get_blocks_and_subslices_gpu(sym, struct_sub, struct_full, device):
    bl = get_blocks(sym, struct_sub)
    if struct_sub == struct_full:
        slc = bl.slc
    else:
        bl_full = get_blocks(sym, struct_full)
        slc = bl_full.slc[_find_matching_indices_gpu(bl_full.t, bl.t, both=False, device=device)]
    return bl, slc


@lru_cache(maxsize=1024)
@nsys_profile
def _meta_tensordot_cutensor_v3(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b, device):
    if device is None or getattr(device, 'type', None) != 'cuda':  # no GPU -> optimized CPU path
        return _meta_tensordot_cutensor_v2(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    struct_a_sub, struct_b_sub, struct_c = \
        _match_legs_tensordot_gpu(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b, device)
    bl_a, slc_a = _get_blocks_and_subslices_gpu(sym, struct_a_sub, struct_a, device)
    bl_b, slc_b = _get_blocks_and_subslices_gpu(sym, struct_b_sub, struct_b, device)
    bl_c = get_blocks(sym, struct_c)

    with nvtx_range("unique out blocks"):
        mask = _output_mask_gpu(bl_a, bl_b, bl_c, nout_a, nin_a, nin_b, nout_b, device)
    if mask is None:  # int64 overflow -> optimized CPU path (rare)
        return _meta_tensordot_cutensor_v2(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b)
    struct_c = struct_c.replace(mask=mask)
    bl_c = get_blocks(sym, struct_c)

    # dot product, which cannot be simply dispatched to vdot
    #              and either one of operands is (effectively) zero
    if not (len(slc_a) > 0 and len(slc_b) > 0):
        return struct_c, 0, *([None] * 18)
    else:
        dot_product = len(nout_a) + len(nout_b) == 0
        a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets = \
            _convert_bl_for_cutensor(struct_a_sub, bl_a, slc_a, dot_product=(dot_product and len(slc_a) < len(slc_b)))
        b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets = \
            _convert_bl_for_cutensor(struct_b_sub, bl_b, slc_b, dot_product=(dot_product and len(slc_a) >= len(slc_b)))
        c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets = \
            _convert_bl_for_cutensor(struct_c, bl_c, dot_product=dot_product)

        h_a, h_b, h_c = hash_blocks(_cutensor_meta(a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides), out=bytes),\
                    hash_blocks(_cutensor_meta(b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides), out=bytes),\
                    hash_blocks(_cutensor_meta(c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides), out=bytes)

    return (struct_c, bl_c.size, h_a, h_b, h_c,
            a_numSectionsPerMode, a_sectionExtents, a_coords, a_strides, a_offsets,
            b_numSectionsPerMode, b_sectionExtents, b_coords, b_strides, b_offsets,
            c_numSectionsPerMode, c_sectionExtents, c_coords, c_strides, c_offsets)
