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

from functools import lru_cache

import numpy as np

from .._profile import nsys_profile, nvtx_range
from ._auxiliary import _encode_rows_shared, _row_keys_pair, get_blocks, hash_blocks
from ._contractions import (_convert_bl_for_cutensor, _cutensor_meta, _match_legs_tensordot, 
                            get_blocks_and_subslices, _matched_pair_indices,)


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
