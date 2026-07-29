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
from ..backend import import_backend

#: Below this many rows the GPU matcher / mask fall back to numpy (launch+transfer
#: overhead dominates). Tunable via env ``YASTN_META_CUTENSOR_GPU_MIN``.
_GPU_MIN = int(os.environ.get("YASTN_META_CUTENSOR_GPU_MIN", "4096"))


# =====================================================================================
# GPU path. The concrete array backend (torch now; cupy/jax in future) is loaded from
# ``backend_id`` via :func:`import_backend` *inside* the entrypoint and injected into the
# helpers below, so ``import yastn`` and ``_control_lru`` (which imports this module) never
# pull a framework for numpy-only users. The helpers use only backend-agnostic primitives
# (numpy-aligned names on the backend module) plus universal array methods/operators.
# The shared find_matching_indices and trimming stay untouched; v3 uses its own GPU copies.
# =====================================================================================


def _enc_rows(tsets, backend, device):
    """
    Backend twin of :func:`_encode_rows_shared`: order-preserving int64 mixed-radix keys per row
    of each 2-D tensor in ``tsets`` (shared per-column value set), or ``None`` on int64 overflow.
    """
    cp = tsets[0].shape[1]
    sizes = [t.shape[0] for t in tsets]
    if cp == 0:
        keys = backend.zeros(sum(sizes), dtype='int64', device=device)
    else:
        stacked = backend.concatenate(tsets, axis=0) if len(tsets) > 1 else tsets[0]
        invs, radices, prod = [], [], 1
        for j in range(cp):
            u, inv = backend.unique(stacked[:, j], return_inverse=True)
            invs.append(inv.reshape(-1)); radices.append(backend.get_size(u)); prod *= backend.get_size(u)
            if prod >= (1 << 62):
                return None
        keys = backend.zeros(sum(sizes), dtype='int64', device=device)
        w = 1
        for j in range(cp - 1, -1, -1):  # column 0 most significant (matches np.unique(axis=0))
            keys += backend.move_to(invs[j], dtype='int64') * w
            w *= radices[j]
    out, off = [], 0
    for s in sizes:
        out.append(keys[off: off + s]); off += s
    return out


def _locate(sorted_u, q, backend, device):
    """ Backend twin of :func:`locate_rows` on 1-D keys: index of each ``q`` in ``sorted_u`` or
    the sentinel ``len(sorted_u)`` when absent. """
    n = backend.get_size(sorted_u)
    if n == 0:
        return backend.zeros(backend.get_size(q), dtype='int64', device=device)  # sentinel 0 == n
    ids = backend.searchsorted(sorted_u, q)
    hit = ids < n
    safe = backend.where(hit, ids, backend.zeros_like(ids))
    hit = hit & (sorted_u[safe] == q)
    ids = backend.clone(ids)
    ids[~hit] = n
    return ids


def _matched_pairs(sa, sb, backend, device):
    """
    Block-index lists ``(ind_a, ind_b)`` of every matching (a, b) pair, given the shared-space
    contracted-sector key of each a-/b-block. Backend twin of ``_indices_from_counts`` + the
    ``arg_*`` gather: sort by sector, count per sector, union-align, then a segmented cartesian
    product. Pair order is irrelevant downstream (only the set of produced keys is used).
    """
    ord_a = backend.argsort(sa); ord_b = backend.argsort(sb)
    ua, ca = backend.unique(sa[ord_a], return_counts=True)  # sorted -> unique == unique_consecutive
    ub, cb = backend.unique(sb[ord_b], return_counts=True)
    starts_a = backend.cumsum(ca, axis=0) - ca
    starts_b = backend.cumsum(cb, axis=0) - cb
    uni = backend.unique(backend.concatenate([ua, ub]))
    pa = backend.searchsorted(uni, ua); pb = backend.searchsorted(uni, ub)
    ca2 = backend.zeros(backend.get_size(uni), dtype='int64', device=device); ca2[pa] = ca
    cb2 = backend.zeros_like(ca2); cb2[pb] = cb
    sa2 = backend.zeros_like(ca2); sa2[pa] = starts_a
    sb2 = backend.zeros_like(ca2); sb2[pb] = starts_b
    prod = ca2 * cb2
    nn = int(backend.item(prod.sum()))
    if nn == 0:
        empty = backend.zeros(0, dtype='int64', device=device)
        return empty, empty
    seg = backend.repeat(backend.arange(backend.get_size(uni), device=device), prod)
    within = backend.arange(nn, device=device) - (backend.cumsum(prod, axis=0) - prod)[seg]
    cbseg = cb2[seg]
    ind_a = ord_a[sa2[seg] + within // cbseg]
    ind_b = ord_b[sb2[seg] + within % cbseg]
    return ind_a, ind_b


def _output_mask_gpu(bl_a, bl_b, bl_c, nout_a, nin_a, nin_b, nout_b, device, backend):
    """
    GPU twin of v2's mask-c step: boolean mask (numpy) over ``bl_c`` blocks that are actually
    produced. Only ``bl_*.t`` cross to the device (nn stays on GPU); returns ``None`` on int64
    overflow so the caller can fall back to numpy.
    """
    nba, nbb, nbc = bl_a.nblocks, bl_b.nblocks, bl_c.nblocks
    na, nb = len(nout_a), len(nout_b)
    nsym = bl_a.t.shape[2]
    ta = backend.to_tensor(bl_a.t, dtype='int64', device=device)
    tb = backend.to_tensor(bl_b.t, dtype='int64', device=device)
    tc = backend.to_tensor(bl_c.t, dtype='int64', device=device)
    # contracted-sector keys (shared a/b) -> matched (a, b) pairs
    enc = _enc_rows([ta[:, nin_a, :].reshape(nba, len(nin_a) * nsym),
                     tb[:, nin_b, :].reshape(nbb, len(nin_b) * nsym)], backend, device)
    if enc is None:
        return None
    ind_a, ind_b = _matched_pairs(enc[0], enc[1], backend, device)
    # out-charge ids: a-out shared with bl_c[:na], b-out shared with bl_c[na:]
    enca = _enc_rows([ta[:, nout_a, :].reshape(nba, na * nsym),
                      tc[:, :na, :].reshape(nbc, na * nsym)], backend, device)
    encb = _enc_rows([tb[:, nout_b, :].reshape(nbb, nb * nsym),
                      tc[:, na:, :].reshape(nbc, nb * nsym)], backend, device)
    if enca is None or encb is None:
        return None
    ua_keys, id_a = backend.unique(enca[0], return_inverse=True)
    ub_keys, id_b = backend.unique(encb[0], return_inverse=True)
    cid_a = _locate(ua_keys, enca[1], backend, device)
    cid_b = _locate(ub_keys, encb[1], backend, device)
    n_b = backend.get_size(ub_keys)
    keys = backend.unique(id_a.reshape(-1)[ind_a] * n_b + id_b.reshape(-1)[ind_b])
    valid = (cid_a < backend.get_size(ua_keys)) & (cid_b < n_b)
    c_keys = backend.where(valid, cid_a * n_b + cid_b, backend.full_like(cid_a, -1))
    return backend.to_numpy(backend.isin(c_keys, keys))


def _find_matching_indices_gpu(tset1, tset2, both=True, device=None, backend=None):
    """
    GPU-accelerated drop-in for :func:`find_matching_indices` (numpy in / numpy out), used only
    by the v3-local trimming copies. The device is validated once at the entrypoint, so this
    assumes a valid GPU ``device``; it still falls back to the untouched numpy
    ``find_matching_indices`` for small inputs or int64 overflow (perf/correctness, not device
    validation). ``tset1`` must be sorted-unique (as the numpy version already requires).
    """
    rs1, *cs1 = tset1.shape
    rs2, *cs2 = tset2.shape
    cp = int(np.prod(cs1, dtype=np.int64))
    if cp == 0 or rs1 == 0 or rs2 == 0 or max(rs1, rs2) < _GPU_MIN:
        return find_matching_indices(tset1, tset2, both=both)
    t1 = backend.to_tensor(np.ascontiguousarray(tset1.reshape(rs1, cp)), dtype='int64', device=device)
    t2 = backend.to_tensor(np.ascontiguousarray(tset2.reshape(rs2, cp)), dtype='int64', device=device)
    enc = _enc_rows([t1, t2], backend, device)
    if enc is None:
        return find_matching_indices(tset1, tset2, both=both)
    v1, v2 = enc  # v1 ascending: tset1 sorted + order-preserving encoding
    ind1 = backend.searchsorted(v1, v2)
    m = ind1 < rs1
    safe = backend.where(m, ind1, backend.zeros_like(ind1))
    m = m & (v1[safe] == v2)
    out1 = backend.to_numpy(ind1[m])
    if both:
        return out1, backend.to_numpy(backend.nonzero(m)[0])
    return out1


# --- v3-local copies of the trimming path (shared versions in _contractions / _auxiliary are
# --- left untouched); the only change is find_matching_indices -> _find_matching_indices_gpu.

@lru_cache(maxsize=1024)
@nsys_profile
def _get_trimmed_struct_engine_gpu(sym, taxes_full, saxes, n, mask, taxes_sub, device, backend_id):
    backend = import_backend(backend_id)
    tblocks_full, _ = get_blocks_charges_all(sym, taxes_full, saxes, n)
    if mask.array is not None:
        tblocks_full = tblocks_full[mask.array]
    tblocks_sub, iblocks_sub = get_blocks_charges_all(sym, taxes_sub, saxes, n)
    inds_sub = _find_matching_indices_gpu(tblocks_sub, tblocks_full, both=False, device=device, backend=backend)
    iblocks_sub = iblocks_sub[inds_sub]
    icharges_sub = [np.unique(iblocks_sub[:, i]).tolist() for i in range(len(taxes_sub))]
    taxes_new = tuple(tuple(tax[i] for i in inds) for tax, inds in zip(taxes_sub, icharges_sub))

    if taxes_new != taxes_sub:
        return _get_trimmed_struct_engine_gpu(sym, taxes_full, saxes, n, mask, taxes_new, device, backend_id)

    if mask.array is not None:
        mask_sub = np.zeros(len(tblocks_sub), dtype=bool)
        mask_sub[inds_sub] = True
        mask_sub = HashedMask(mask_sub)
    else:
        mask_sub = mask
    return taxes_new, mask_sub


@nsys_profile
def _get_trimmed_struct_gpu(sym, struct, sub_legs, device, backend_id):
    saxes = tuple(int(leg.s) for leg in struct.legs)
    taxes_full = tuple(tuple(tuple(map(int, tt)) for tt, d in zip(leg.t, leg.D) if d > 0) for leg in struct.legs)
    if sub_legs is None:
        taxes_sub = taxes_full
    else:
        taxes_sub = tuple(tuple(tuple(map(int, tt)) for tt, d in zip(leg.t, leg.D) if d > 0) for leg in sub_legs)
    taxes_new, mask_sub = _get_trimmed_struct_engine_gpu(sym, taxes_full, saxes, struct.n, struct.mask, taxes_sub, device, backend_id)
    legs_new = tuple(leg.trim(tax) for leg, tax in zip(struct.legs, taxes_new))
    return _struct(legs=tuple(legs_new), n=struct.n, isdiag=struct.isdiag, mask=mask_sub)


@nsys_profile
def _match_legs_tensordot_gpu(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b, device, backend_id):
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

        struct_a_1 = _get_trimmed_struct_gpu(sym, struct_a_0, legs_a_new, device, backend_id)
        struct_b_1 = _get_trimmed_struct_gpu(sym, struct_b_0, legs_b_new, device, backend_id)

        legs_c = (*(struct_a_1.legs[ax] for ax in nout_a), *(struct_b_1.legs[ax] for ax in nout_b))
        struct_c_0 = _struct(legs=legs_c, n=n_c, isdiag=False)
        struct_c_1 = _get_trimmed_struct_gpu(sym, struct_c_0, None, device, backend_id)

        if struct_a_0 == struct_a_1 and struct_b_0 == struct_b_1 and struct_c_0 == struct_c_1:
            break

        struct_a_0 = struct_a_1
        struct_b_0 = struct_b_1
        for ii, ax in enumerate(nout_a):
            legs_a_new[ax] = struct_c_1.legs[ii].intersection(struct_a_1.legs[ax])
        for ii, ax in enumerate(nout_b, start=len(nout_a)):
            legs_b_new[ax] = struct_c_1.legs[ii].intersection(struct_b_1.legs[ax])

    return struct_a_1, struct_b_1, struct_c_1


def _get_blocks_and_subslices_gpu(sym, struct_sub, struct_full, device, backend):
    bl = get_blocks(sym, struct_sub)
    if struct_sub == struct_full:
        slc = bl.slc
    else:
        bl_full = get_blocks(sym, struct_full)
        slc = bl_full.slc[_find_matching_indices_gpu(bl_full.t, bl.t, both=False, device=device, backend=backend)]
    return bl, slc


@lru_cache(maxsize=1024)
@nsys_profile
def _meta_tensordot_cutensor_gpu(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b, 
                                 lazy_threshold:float=None, device:str=None, backend_id:str=None):
    backend = import_backend(backend_id)
    if backend.is_cpu_device(device):
        raise ValueError("GPU device is required for _meta_tensordot_cutensor_gpu.")
    struct_a_sub, struct_b_sub, struct_c = \
        _match_legs_tensordot_gpu(sym, struct_a, struct_b, nout_a, nin_a, nin_b, nout_b, device, backend_id)
    bl_a, slc_a = _get_blocks_and_subslices_gpu(sym, struct_a_sub, struct_a, device, backend)
    bl_b, slc_b = _get_blocks_and_subslices_gpu(sym, struct_b_sub, struct_b, device, backend)
    bl_c = get_blocks(sym, struct_c)

    if lazy_threshold and bl_c.nblocks:
        with nvtx_range("unique out blocks"):
            mask = _output_mask_gpu(bl_a, bl_b, bl_c, nout_a, nin_a, nin_b, nout_b, device, backend)
        if mask is None:  # int64 overflow -> optimized CPU path (rare)
            return None
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
