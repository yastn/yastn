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
import logging
from functools import lru_cache
from contextlib import nullcontext
from itertools import product, accumulate
import gc, subprocess
import time
from typing import Hashable, Mapping, Sequence, Union

import numpy as np
import opt_einsum as oe  # type: ignore
from opt_einsum.contract import PathInfo  # type: ignore
try:
    from opt_einsum.contract import _VALID_CONTRACT_KWARGS
except:
    _VALID_CONTRACT_KWARGS = {'optimize', 'memory_limit', 'einsum_call', 'use_blas', 'shapes'}
from . import Tensor, ncon, split_data_and_meta, combine_data_and_meta
from ..initialize import block as yastn_block
from ._legs import Leg
from ._einsum import ncon_prefilter
from ._auxiliary import _clear_axes, _slc
from ._merging import _meta_mask
from ._tests import YastnError

log = logging.getLogger(__name__)


class SlicedLeg:
    r"""
    Describes a subset of a YASTN tensor leg: a selection of charge sectors,
    each optionally restricted to a contiguous slice within the sector's full
    block dimension.

    A collection of non-overlapping ``SlicedLeg`` objects whose union covers
    all charge sectors of a leg forms a valid partition.  Iterating over such
    a partition and summing the partial contractions reproduces the full result,
    while each individual contraction operates on a strictly smaller tensor.

    Parameters
    ----------
    t : Sequence[tuple | int]
        Charge sectors included in this slice.  Each element must be a tuple
        of ints (one per symmetry component, length ``NSYM``), or a plain int
        for single-component symmetries (normalised to a 1-tuple internally).
    D : Sequence[int]
        Dimension of each charge sector *within this slice*.  Must equal
        ``len(t)``.  Use ``full_D`` when the entire block is selected.
    slices : dict[tuple, slice], optional
        Maps each charge tuple to a ``slice`` object into the full block
        dimension of that sector.  If omitted, every sector uses
        ``slice(None)`` (i.e. the full block is selected).
    """

    def __init__(self, t, D, slices=None):
        # normalise each charge to a tuple of ints
        self.t = tuple(
            tuple(ti) if hasattr(ti, '__iter__') else (int(ti),) for ti in t
        )
        self.D = tuple(int(d) for d in D)
        if len(self.t) != len(self.D):
            raise ValueError("SlicedLeg: len(t) must equal len(D)")
        if slices is None:
            self.slices = {ti: slice(None) for ti in self.t}
        else:
            # normalise keys the same way
            self.slices = {
                (tuple(k) if hasattr(k, '__iter__') else (int(k),)): v
                for k, v in slices.items()
            }

    @property
    def tD(self):
        """Dict mapping charge tuple → dimension in this slice."""
        return dict(zip(self.t, self.D))

    def __repr__(self):
        return f"SlicedLeg(t={self.t}, D={self.D}, slices={self.slices})"


def make_sliced_legs(leg):
    r"""
    Split a YASTN :class:`yastn.Leg` into one :class:`SlicedLeg` per charge
    sector (the simplest non-overlapping partition).

    Each returned :class:`SlicedLeg` covers exactly one charge sector and
    includes the full block dimension of that sector (``slice(None)``).

    Parameters
    ----------
    leg : yastn.Leg

    Returns
    -------
    list[SlicedLeg]
    """
    return [SlicedLeg(t=(ti,), D=(Di,)) for ti, Di in zip(leg.t, leg.D)]


def slice_leg_uniform(leg: Leg, size: int):
    r"""
    Uniformly slice YASTN :class:`yastn.Leg` into segments of at most ``size``.
    Resulting slices can span multiple charge sectors.

    The returned list of :class:`SlicedLeg` objects forms a valid partition of
    the leg, with each slice selecting a contiguous subset of the block
    dimension within each charge sector.  The last slice  may be
    smaller than the specified size.

    Parameters
    ----------
    leg : yastn.Leg
    size : int

    Returns
    -------
    list[SlicedLeg]
    """
    sliced_legs = []
    ts, Ds, slices = [], [], {}
    remaining = size
    for t, D in zip(leg.t, leg.D):
        sector_offset = 0
        while sector_offset < D:
            take = min(D - sector_offset, remaining)
            ts.append(t)
            Ds.append(take)
            slices[t] = slice(sector_offset, sector_offset + take)
            sector_offset += take
            remaining -= take
            if remaining == 0:
                sliced_legs.append(SlicedLeg(t=ts, D=Ds, slices=slices))
                ts, Ds, slices = [], [], {}
                remaining = size
    if ts:
        sliced_legs.append(SlicedLeg(t=ts, D=Ds, slices=slices))
    return sliced_legs


def _build_mask_tensor(sliced_leg, full_leg, config, device=None):
    r"""
    Build a YASTN diagonal mask tensor for ``apply_mask``.

    For each charge sector in ``sliced_leg``, the diagonal block has 1.0 at
    positions selected by the corresponding slice and 0.0 elsewhere.
    Charge sectors absent from ``sliced_leg`` are simply omitted (the block is
    empty / all-zero and therefore not stored).

    Parameters
    ----------
    sliced_leg : SlicedLeg
    full_leg : yastn.Leg
        The full leg of the target tensor (provides total block dimensions).
    config : yastn config
    device : str, optional
        Target device for the mask tensor.  When provided and different from
        ``config.default_device``, the finished mask is moved with ``.to()``.

    Returns
    -------
    yastn.Tensor  (diagonal)
    """
    nsym = config.sym.NSYM
    n = (0,) * nsym
    mask_tensor = Tensor(config=config, s=(full_leg.s, -full_leg.s), n=n, isdiag=True)
    sl_tD = sliced_leg.tD
    for ti, full_D in zip(full_leg.t, full_leg.D):
        if ti not in sl_tD:
            continue
        sl = sliced_leg.slices.get(ti, slice(None))
        data = np.zeros(full_D)
        data[sl] = 1.0
        mask_tensor.set_block(ts=ti, Ds=full_D, val=data)
    if device is not None and device != mask_tensor.device:
        mask_tensor = mask_tensor.to(device)
    return mask_tensor


def _ncon_checkpointed(tensors, inds, conjs, order, swap):
    r"""Run a single ``ncon`` call under ``torch.utils.checkpoint``.

    Used when ``checkpoint_loop=True`` but ``unroll is None``, so there is
    no combo loop — just one full contraction to checkpoint.
    """
    input_datas, input_metas = zip(
        *(split_data_and_meta(t.to_dict(level=0), squeeze=True) for t in tensors)
    )
    _out_meta = [None]

    def _fn(*datas):
        reconst = [Tensor.from_dict(combine_data_and_meta(d, m))
                   for d, m in zip(datas, input_metas)]
        result = ncon(reconst, inds, conjs=conjs, order=order, swap=swap)
        r_data, r_meta = split_data_and_meta(result.to_dict(level=0), squeeze=True)
        _out_meta[0] = r_meta
        return r_data

    checkpoint = tensors[0].config.backend.checkpoint
    result_data = checkpoint(_fn, *input_datas, use_reentrant=True)
    return Tensor.from_dict(combine_data_and_meta(result_data, _out_meta[0]))


def _iteration_checkpointed(tensors, sl_map, tensor_unroll_info, mask_cache, index_groups,
                            out_ig, optimize, unroll_labels, pf_trim=None, swap=None,
                            release_cuda_cache=False) -> Tensor:
    r"""
    Run one unroll-loop iteration under ``torch.utils.checkpoint``.

    Uses :func:`yastn.split_data_and_meta` / :func:`yastn.combine_data_and_meta`
    to separate raw tensor data (tracked by autograd) from YASTN structural
    metadata (Python objects, not tracked).  ``ncon`` is executed inside the
    checkpointed region so its intermediate activations are not stored during
    the forward pass.

    The output YASTN tensor structure is captured in a mutable closure cell
    that is set during the checkpointed forward call and read back immediately
    after :func:`torch.utils.checkpoint.checkpoint` returns — no separate
    no-grad structural pre-run is needed.

    Parameters
    ----------
    tensors : sequence of yastn.Tensor
        Unmasked input tensors for this iteration.
    index_groups : list[list]
        Index groups for each tensor.
    out_ig : tuple | list
        Output index group.
    optimize : list[tuple[int, int]]
        Pairwise contraction path.

    Returns
    -------
    yastn.Tensor
    """
    # Split each input tensor into raw data + structural metadata.
    # Metadata (struct, slices, config, …) are pure Python objects — no memory
    # overhead from storing them.  Only the _data tensors participate in autograd.
    input_datas, input_metas = zip(
        *(split_data_and_meta(t.to_dict(level=0), squeeze=True) for t in tensors)
    )

    # Mutable cell: _fn sets this during its (first) forward call so the
    # output YASTN structural metadata is available after checkpoint() returns.
    # On the backward re-run _fn sets the same value again — harmlessly.
    _out_meta = [None]

    def _fn(*datas):
        # Reconstruct YASTN tensors from structural metadata + current data.
        masked = [Tensor.from_dict(combine_data_and_meta(d, m))
                  for d, m in zip(datas, input_metas)]

        for k in range(len(masked)):
            for u in unroll_labels:
                if (k, u) not in tensor_unroll_info:
                    continue
                user_ax = tensor_unroll_info[(k, u)][0]
                mask_t = mask_cache[(k, u, id(sl_map[u]), masked[k].device)]
                masked[k] = mask_t.apply_mask(masked[k], axes=user_ax)

        if pf_trim is not None:
            for k in range(len(masked)):
                trim_k = pf_trim.get(k)
                if trim_k is not None and len(trim_k) < len(masked[k].struct.t):
                    masked[k] = _filter_tensor_blocks(masked[k], trim_k)

        interleaved = []
        for T, ig in zip(masked, index_groups):
            interleaved.append(T)
            interleaved.append(ig)
        interleaved.append(out_ig)
        ts, inds, conjs, order, ncon_swap = _convert_path_to_ncon_args(
            *interleaved, optimize=optimize, swap=swap
        )
        partial = ncon(ts, inds, conjs=conjs, order=order, swap=ncon_swap,
                       release_cuda_cache=release_cuda_cache)

        # Capture output structure so it can be read back outside checkpoint().
        p_data, p_meta = split_data_and_meta(partial.to_dict(level=0), squeeze=True)
        _out_meta[0] = p_meta
        return p_data

    assert hasattr(tensors[0].config.backend, "checkpoint"), "Backend does not support checkpointing"
    checkpoint = tensors[0].config.backend.checkpoint
    result_data = checkpoint(_fn, *input_datas, use_reentrant=True)
    return Tensor.from_dict(combine_data_and_meta(result_data, _out_meta[0]))



def _filter_tensor_blocks(tensor, block_indices):
    r"""
    Return a compact tensor retaining only the specified blocks.

    The retained blocks are packed into a new contiguous data buffer so the
    returned tensor preserves the invariant ``len(_data) == struct.size``.

    Parameters
    ----------
    tensor : yastn.Tensor
    block_indices : frozenset[int] | None
        Indices into ``tensor.struct.t`` to retain.
        ``None`` means keep all blocks.

    Returns
    -------
    yastn.Tensor
    """
    if block_indices is None or len(block_indices) == len(tensor.struct.t):
        return tensor

    indices = sorted(block_indices)
    if not indices:
        return tensor._replace(struct=tensor.struct._replace(t=(), D=(), size=0),
                               slices=(), data=tensor._data[:0])

    new_t = tuple(tensor.struct.t[i] for i in indices)
    new_D = tuple(tensor.struct.D[i] for i in indices)
    new_Dp = tuple(tensor.slices[i].Dp for i in indices)
    new_slices = tuple(
        _slc(((stop - dp, stop),), ds, dp)
        for stop, dp, ds in zip(accumulate(new_Dp), new_Dp, new_D)
    )
    new_size = sum(new_Dp)
    new_struct = tensor.struct._replace(t=new_t, D=new_D, size=new_size)
    new_data = tensor.config.backend.zeros(new_size, dtype=tensor.yastn_dtype, device=tensor.device)
    for new_slc, old_idx in zip(new_slices, indices):
        old_slc = tensor.slices[old_idx].slcs[0]
        new_data[slice(*new_slc.slcs[0])] = tensor._data[slice(*old_slc)]

    return tensor._replace(struct=new_struct, slices=new_slices, data=new_data)


def _apply_meta_mask(tensor, mask_t, mask_D, axis):
    r"""Metadata-only version of apply_mask for a single axis."""
    ax = axis % len(tensor.mfs)
    if tensor.mfs[ax] != (1,):
        raise ValueError('Second tensor`s leg specified by axis cannot be fused.')
    ax = sum(tensor.mfs[ii][0] for ii in range(ax))
    ax = tensor.trans[ax]
    if tensor.hfs[ax].tree != (1,):
        raise ValueError('Second tensor`s leg specified by axes cannot be fused.')
    _, struct, slices, _, _ = _meta_mask(tensor.struct, tensor.slices, tensor.isdiag, mask_t, mask_D, ax)
    return tensor._replace(struct=struct, slices=slices)


def _contract_with_sliced_unroll(*args, unroll, optimize, checkpoint_loop=False, swap=None, devices=None, **kwargs):
    r"""
    Contract a tensor network with block-sparse index unrolling.

    For each combination of :class:`SlicedLeg` objects (one per unroll index),
    mask the relevant tensors with :meth:`~yastn.Tensor.apply_mask`, run the
    full ``ncon`` contraction on the reduced tensors, then accumulate the
    partial results.

    Both contracted and output indices may be unrolled.  Partials are grouped
    by their position along output-unrolled axes; contracted-only unroll combos
    are summed with ``+`` within each group.  The groups are then assembled into
    the final tensor via :func:`yastn.block`.

    Parameters
    ----------
    *args : interleaved ``(T1, ig1, T2, ig2, ..., out_ig)``
        Tensor network in opt_einsum's interleaved format.  An explicit output
        index group is required (odd number of elements in ``args``).
    unroll : dict[label, list[SlicedLeg]]
        Keys are index labels to unroll; values are lists of non-overlapping
        :class:`SlicedLeg` objects whose union spans all charge sectors of
        that index.
    optimize : list[tuple[int, int]]
        Pairwise contraction path (as returned by :func:`get_contraction_path`).
    checkpoint_loop : bool
        If ``True`` and PyTorch is available, each loop-body iteration is
        wrapped in :func:`torch.utils.checkpoint.checkpoint`.  Masking and
        ``ncon`` intermediates are recomputed during backward instead of
        stored, trading extra forward computation for lower peak memory.
        Default: ``False``.
    devices : list[str], optional
        List of device strings (e.g. ``['cuda:0', 'cuda:1']``) for
        dispatching loop iterations round-robin across multiple devices.
        Input tensors are pre-moved to each device once (autograd-preserving);
        partials are accumulated per-device and gathered back to the original
        device after the loop.  ``None`` (default) runs everything on the
        original device.
    **kwargs :
        Forwarded to ``ncon``.

    Returns
    -------
    yastn.Tensor
    """
    assert len(args) % 2 == 1, \
        "_contract_with_sliced_unroll requires explicit output index group"

    tensors = args[0 : 2 * (len(args) // 2) : 2]
    index_groups = list(args[1 : 2 * (len(args) // 2) : 2])
    out_ig = args[-1]

    unroll_labels = list(unroll.keys())

    # For each (tensor_index, unroll_label) pair: record the user-facing axis
    # and the full Leg object at that position.
    # Build per-ig index lookup to avoid repeated list(ig).index(u) calls.
    tensor_unroll_info = {}  # (k, u) -> (user_axis, full_leg)
    for k, (T, ig) in enumerate(zip(tensors, index_groups)):
        ig_list = list(ig)
        ig_index = {label: idx for idx, label in enumerate(ig_list)}
        for u in unroll_labels:
            if u in ig_index:
                user_ax = ig_index[u]
                full_leg = T.get_legs(user_ax)
                tensor_unroll_info[(k, u)] = (user_ax, full_leg)

    # For output indices that are unrolled, map output axis -> (label, full_leg).
    # The full leg is taken from the input tensor that carries that output index.
    out_ig_list = list(out_ig)
    output_unroll_info = {}  # out_ax -> (label, full_leg)
    for u in unroll_labels:
        if u in out_ig_list:
            out_ax = out_ig_list.index(u)
            for k in range(len(tensors)):
                if (k, u) in tensor_unroll_info:
                    _, full_leg = tensor_unroll_info[(k, u)]
                    output_unroll_info[out_ax] = (u, full_leg)
                    break

    # Output-unroll labels ordered by their axis index in the output tensor.
    # Contracted-only unroll labels are not in this list.
    output_unroll_labels = [u for _, (u, _) in sorted(output_unroll_info.items())]

    # --- Multi-device setup ---
    # Normalize devices; None means single-device (original behavior).
    original_device = tensors[0].device
    if devices is not None:
        if not isinstance(devices, (list, tuple)):
            devices = [devices]
        devices = list(dict.fromkeys(str(d) for d in devices))  # deduplicate, preserve order
        if len(devices) == 0 or (len(devices) == 1 and devices[0] == original_device):
            devices = None

    multi_device = devices is not None

    # Detect whether a non-PyTorch allocator (e.g. cuTENSOR) is in use.
    _uses_external_allocator = getattr(tensors[0].config.backend, 'BACKEND_ID', '') == 'torch_cpp'

    # On any CUDA device, periodically release PyTorch's cached GPU memory
    # to prevent allocator fragmentation that can cause OOM.
    _needs_cache_release = _uses_external_allocator or 'cuda' in str(original_device)

    def _release_cuda_cache(devs):
        r"""Release PyTorch's cached-but-unused GPU memory on *devs*."""
        import torch as _torch
        for d in devs:
            if 'cuda' in str(d):
                with _torch.cuda.device(d):
                    _torch.cuda.empty_cache()

    if multi_device:
        import torch as _torch

        # Pre-move input tensors to each target device once.
        # .to(device) is autograd-tracked: backward sends gradients back.
        tensors_by_device = {}
        for dev in devices:
            if dev == original_device:
                tensors_by_device[dev] = tensors
            else:
                tensors_by_device[dev] = tuple(t.to(dev) for t in tensors)

        # Release PyTorch's cached (but unused) GPU memory so that it is
        # available to non-PyTorch allocators (e.g. cudaMalloc used by cuTENSOR).
        if _uses_external_allocator:
            _release_cuda_cache(devices)

        # Build worker list: one (device, stream_context) per device.
        _workers = []  # list of (device, stream_context)
        _all_streams = []
        if _torch.cuda.is_available():
            cuda_devices = [d for d in devices if d.startswith('cuda')]
            for dev in cuda_devices:
                s = _torch.cuda.Stream(device=dev)
                _all_streams.append(s)
                _workers.append((dev, _torch.cuda.stream(s)))
            # Ensure pre-moved tensors are visible to non-default streams:
            # record an event on each device's default stream and have
            # every worker stream on that device wait on it.
            for dev in cuda_devices:
                with _torch.cuda.device(dev):
                    ev = _torch.cuda.current_stream().record_event()
                for s in _all_streams:
                    if str(s.device) == dev:
                        s.wait_event(ev)
            # CPU devices get no CUDA stream.
            for dev in devices:
                if not dev.startswith('cuda'):
                    _workers.append((dev, nullcontext()))
        if not _workers:
            # Fallback when torch/CUDA is unavailable.
            for dev in devices:
                _workers.append((dev, nullcontext()))

    # --- Pre-compute mask tensors for all (tensor, label, sliced_leg) triples ---
    # Avoids rebuilding identical masks on every loop iteration.
    # Key: (tensor_index, label, id(sliced_leg)) -> mask Tensor
    # For multi-device: also cache per-device copies.
    mask_cache = {}
    for (k, u), (user_ax, full_leg) in tensor_unroll_info.items():
        cfg = tensors[k].config
        dev0 = tensors[k].device
        for sl in unroll[u]:
            mask_t = _build_mask_tensor(sl, full_leg, cfg, device=dev0)
            mask_cache[(k, u, id(sl), dev0)] = mask_t
            if multi_device:
                for dev in devices:
                    if dev != dev0:
                        mask_cache[(k, u, id(sl), dev)] = mask_t.to(dev)

    # --- Pre-compute ncon structural args (index-only, independent of tensor data) ---
    # Build a template interleaved list to derive ncon indices, conjs, order, swap.
    _template_interleaved = []
    for T, ig in zip(tensors, index_groups):
        _template_interleaved.append(T)
        _template_interleaved.append(ig)
    _template_interleaved.append(out_ig)
    _, ncon_igs, ncon_conjs, ncon_order, ncon_swap = _convert_path_to_ncon_args(
        *_template_interleaved, optimize=optimize, swap=swap
    )

    # --- Pre-compute metadata prefilter inputs ---
    nsym = tensors[0].config.sym.NSYM
    _pf_inds = tuple(_clear_axes(*ncon_igs)) if nsym > 0 else None

    # --- Pre-compute output_pos_key lookup ---
    # Maps id(sliced_leg) -> index in unroll[u] for each output-unrolled label.
    sl_to_idx = {u: {id(sl): i for i, sl in enumerate(unroll[u])}
                 for u in output_unroll_labels}

    def _apply_masks_for_combo(base_tensors, sl_map, target_device):
        masked = list(base_tensors)
        for k in range(len(base_tensors)):
            for u in unroll_labels:
                if (k, u) not in tensor_unroll_info:
                    continue
                user_ax = tensor_unroll_info[(k, u)][0]
                mask_t = mask_cache[(k, u, id(sl_map[u]), target_device)]
                masked[k] = mask_t.apply_mask(masked[k], axes=user_ax)
        return masked

    def _apply_masks_for_combo_meta(base_tensors, sl_map):
        masked = list(base_tensors)
        for k in range(len(base_tensors)):
            for u in unroll_labels:
                if (k, u) not in tensor_unroll_info:
                    continue
                user_ax = tensor_unroll_info[(k, u)][0]
                masked[k] = _apply_meta_mask(masked[k], sl_map[u].t, sl_map[u].D, user_ax)
        return masked

    # Cartesian product over SlicedLeg choices, one per unroll label.
    # Partials are grouped by their position along output-unrolled axes.
    # Contracted-only unroll combos accumulate into the same key via +.
    #
    # With multi-device: accumulate per-device first (no cross-device sync
    # in the loop), then gather to original_device after the loop.
    combos = list(product(*(unroll[u] for u in unroll_labels)))
    combo_entries = []
    for combo in combos:
        sl_map = {u: sl for u, sl in zip(unroll_labels, combo)}
        output_pos_key = tuple(sl_to_idx[u][id(sl_map[u])] for u in output_unroll_labels)
        combo_entries.append((sl_map, output_pos_key))

    def _contract_single_combo(base_tensors, sl_map, pf_trim=None, use_checkpoint=False):
        if use_checkpoint:
            return _iteration_checkpointed(
                base_tensors, sl_map, tensor_unroll_info, mask_cache, index_groups,
                out_ig, optimize, unroll_labels, pf_trim=pf_trim, swap=swap,
                release_cuda_cache=_needs_cache_release,
            )

        target_device = base_tensors[0].device if multi_device and base_tensors is not tensors else original_device
        masked_tensors = _apply_masks_for_combo(base_tensors, sl_map, target_device)
        if pf_trim is not None:
            for k in range(len(masked_tensors)):
                trim_k = pf_trim.get(k)
                if trim_k is not None and len(trim_k) < len(masked_tensors[k].struct.t):
                    masked_tensors[k] = _filter_tensor_blocks(masked_tensors[k], trim_k)
        return ncon(masked_tensors, ncon_igs, conjs=ncon_conjs, order=ncon_order, swap=ncon_swap,
                    release_cuda_cache=_needs_cache_release)

    # How often to release PyTorch's cache for external allocators (cuTENSOR).
    # Every _CACHE_RELEASE_INTERVAL combos that actually produce a contraction,
    # we call empty_cache() so cudaMalloc can reclaim freed intermediates.
    # Interval of 1 prevents cache fragmentation that can cause OOM with
    # large intermediates even when total free memory is sufficient.
    _CACHE_RELEASE_INTERVAL = 1

    def _process_combos(assigned, iter_tensors, dev, stream_ctx):
        r"""Process a list of (n, sl_map, output_pos_key) entries on one device.

        Returns a dict {output_pos_key: accumulated_partial}.
        Thread-safe: reads only from shared immutable state; writes only to
        the returned local dict.
        """
        local_partials = {}
        _cfg = iter_tensors[0].config
        contractions_since_release = 0
        for n, sl_map, output_pos_key in assigned:
            with stream_ctx:
                if _cfg.profile:
                    tag = f"_contract_with_sliced_unroll {n}"
                    if dev is not None:
                        tag += f" [device={dev}]"
                    _cfg.backend.nvtx.range_push(tag)

                # --- Metadata-only masks for skip / prefilter decisions ---
                masked_meta_tensors = _apply_masks_for_combo_meta(iter_tensors, sl_map)
                skip = False
                for k in range(len(iter_tensors)):
                    if any((k, u) in tensor_unroll_info for u in unroll_labels) and not masked_meta_tensors[k].struct.t:
                        skip = True
                        break
                if skip:
                    if _cfg.profile: _cfg.backend.nvtx.range_pop()
                    continue

                # --- Prefilter: skip cross-tensor zeros ---
                pf_trim = None
                if _pf_inds is not None:
                    ts_meta = {k: (masked_meta_tensors[k].struct.t, masked_meta_tensors[k].ndim_n,
                                   masked_meta_tensors[k].trans, masked_meta_tensors[k].mfs)
                               for k in range(len(masked_meta_tensors))}
                    pf_trim = ncon_prefilter(ts_meta, _pf_inds, nsym)
                    if pf_trim is None:
                        if _cfg.profile: _cfg.backend.nvtx.range_pop()
                        continue

                partial = _contract_single_combo(iter_tensors, sl_map, pf_trim=pf_trim, use_checkpoint=checkpoint_loop)

                prev = local_partials.get(output_pos_key)
                local_partials[output_pos_key] = partial if prev is None else prev + partial

                if _cfg.profile: _cfg.backend.nvtx.range_pop()

                # Periodically release PyTorch's cached-but-unused GPU memory
                # to prevent allocator fragmentation.
                if _needs_cache_release:
                    contractions_since_release += 1
                    if contractions_since_release >= _CACHE_RELEASE_INTERVAL:
                        _release_cuda_cache([dev] if dev is not None else [original_device])
                        contractions_since_release = 0
        return local_partials

    # Release PyTorch's cache before the combo loop.
    if _needs_cache_release:
        _release_cuda_cache(devices if multi_device else [original_device])

    output_pos_partials = {}

    if multi_device:
        # --- Threaded multi-device dispatch ---
        # Round-robin combo entries across all workers (not just devices).
        # Each worker thread processes its assigned combos sequentially on
        # its own CUDA stream.  CUDA kernels release the GIL, so while one
        # worker's kernel runs, another worker can do Python metadata prep —
        # pipelining Python overhead with GPU execution.
        from concurrent.futures import ThreadPoolExecutor

        n_workers = len(_workers)
        worker_assigned = [[] for _ in range(n_workers)]
        for n, (sl_map, output_pos_key) in enumerate(combo_entries):
            worker_assigned[n % n_workers].append((n, sl_map, output_pos_key))

        # Build per-worker args: (assigned_combos, tensors, device, stream_ctx).
        worker_args = []
        for w_idx, (dev, stream_ctx) in enumerate(_workers):
            if not worker_assigned[w_idx]:
                continue
            worker_args.append((worker_assigned[w_idx], tensors_by_device[dev], dev, stream_ctx))

        with ThreadPoolExecutor(max_workers=len(worker_args)) as pool:
            futures = [pool.submit(_process_combos, *wa) for wa in worker_args]
            worker_results = [f.result() for f in futures]

        # Synchronize all streams before cross-device gather.
        for s in _all_streams:
            s.synchronize()

        # Gather worker partials back to the original device.
        for wa, local_partials in zip(worker_args, worker_results):
            dev = wa[2]
            for key, partial in local_partials.items():
                if dev != original_device:
                    partial = partial.to(original_device)
                prev = output_pos_partials.get(key)
                output_pos_partials[key] = partial if prev is None else prev + partial
    else:
        # --- Single-device sequential path ---
        assigned = [(n, sl_map, opk) for n, (sl_map, opk) in enumerate(combo_entries)]
        output_pos_partials = _process_combos(assigned, tensors, None, nullcontext())

    if not output_unroll_info:
        result = output_pos_partials.get((), None)
        if result is None and combo_entries:
            raise YastnError("No valid charge sectors found for contraction.")
        return result

    if not output_pos_partials and combo_entries:
        raise YastnError("No valid charge sectors found for contraction.")

    # Assemble partial results at different output positions using block().
    # The output-unrolled axes are the blocked axes; all others are common_legs.
    blocked_axes = sorted(output_unroll_info.keys())
    first_partial = next(iter(output_pos_partials.values()))
    ndim_out = first_partial.ndim_n
    common_legs_axes = [ax for ax in range(ndim_out) if ax not in blocked_axes]
    result = yastn_block(output_pos_partials, common_legs=common_legs_axes)
    # block() records fusion history in hfs, making the result incompatible
    # with plain ncon outputs.  Drop it so the caller gets an ordinary tensor.
    return result.drop_leg_history()


def _validate_and_resolve_unroll(*args,
        unroll=None) \
            -> Mapping[Hashable,Union[Sequence[SlicedLeg],int]]:
    r"""
    Validates the ``unroll`` argument and resolves any integer values
    to lists of :class:`SlicedLeg` objects via uniform slicing of the corresponding leg in the input tensors.

    :param unroll: Mapping[Hashable,Union[Sequence[SlicedLeg],int]] or None

    Returns
    -------
    Mapping[Hashable,Sequence[SlicedLeg]] or None
    """
    if unroll is None:
        return None
    assert isinstance(unroll, dict), "unroll must be a dict or None"
    for k, v in unroll.items():
        assert isinstance(v, (list, int)), "unroll values must be either list of SlicedLeg or integer"
        if isinstance(v, list):
            assert all(isinstance(sl, SlicedLeg) for sl in v), "unroll list values must be of type SlicedLeg"
        else:
            assert v > 0, "unroll integer value must be positive"
            # Find the leg with label k in the network and slice it uniformly
            found = False
            for t, ig in zip(args[0 : 2 * (len(args) // 2) : 2],
                             args[1 : 2 * (len(args) // 2) : 2]):
                if k in ig:
                    user_ax = list(ig).index(k)
                    full_leg = t.get_legs(user_ax)
                    unroll[k] = slice_leg_uniform(full_leg, v)
                    found = True
                    break
            if not found:
                raise ValueError(f"Index label {k} not found in any tensor index group.")
    return unroll


def _convert_path_to_ncon_args(*args, **kwargs):
    path= kwargs.pop("optimize", None)
    assert path is not None, "optimize (contraction path) has to be provided in kwargs"
    swap= kwargs.pop("swap", None)
    # TODO assert on format of path Sequence[Tuple[int]]

    # args is either, case I, an interleaved sequence of tensors and index groups (Sequence[Hashable])
    # or, case II, there is a single extra index group at the end specifying output indices
    # case II
    tensors= args[0 : 2 * (len(args) // 2) : 2]
    in_igs= list(list(ig) for ig in args[1 : 2 * (len(args) // 2) : 2] )
    if len(args) % 2 == 1:
        out_ig= args[-1]
    else:
        raise NotImplementedError("ncon conversion for interleaved format without explicit output indices is not implemented")

    # path is a sequence of tuple[int] specifying which tensors to contract at each step, starting from first pair
    # path uses positions in current list of tensors, which is shrinking at each step
    if all([len(e)==2 for e in path]):
        # pairwise contractions only - straightforward conversion
        conjs= [False]*len(tensors)
        order= None

        # from original labeling to ncon labeling
        igs_to_ncon_igs= {i: None for i in list(set(sum(in_igs,[])))}

        # convert indexing to ncon format (with implied order given by path)
        i=1
        im=len(in_igs)-1 # index of last original tensor in_igs
        orig_ig_inds= list(range(len(in_igs)))
        for n,p in enumerate(path):
            ig2= in_igs.pop(p[1])
            ig1= in_igs.pop(p[0])
            i_ig2= orig_ig_inds.pop(p[1])
            i_ig1= orig_ig_inds.pop(p[0])
            orig_ig_inds.append(-n-1)

            # find common indices - to be contracted over
            common_inds= set(ig1).intersection(set(ig2))
            out_inds= list([ind for ind in ig1+ig2 if not (ind in common_inds)])
            # print(f"{n} {p}->{(i_ig1,i_ig2)} {ig1} {ig2} common {common_inds} -> {out_inds} boundary im {im}")

            # assign ascending positive labels to common indices
            #
            for ci in common_inds:
                if igs_to_ncon_igs[ci] is None:
                    igs_to_ncon_igs[ci]= i
                    i+= 1
                else:
                    assert igs_to_ncon_igs[ci]== i, "Inconsistent mapping to ncon indices"

            # Append index group for the resulting tensor at the end, move last original index
            # NOTE we are not handling order here
            if (p[0] <= im or p[1] <= im):
                im= im- (p[0] <= im) - (p[1] <= im)
            in_igs.append(out_inds)

        # assign outgoing indices
        for i,oi in enumerate(out_ig):
            if igs_to_ncon_igs[oi] is None:
                igs_to_ncon_igs[oi]= -i-1

    else:
        raise NotImplementedError(
            "contract_with_unroll only supports pairwise contraction paths. "
            "Got a path step contracting more than 2 tensors simultaneously."
        )

    ncon_igs= [ [igs_to_ncon_igs[i] for i in ig] for ig in args[1 : 2 * (len(args) // 2) : 2] ]

    # convert swap labels to ncon indices
    ncon_swap = None
    if swap is not None:
        ncon_swap = [tuple(igs_to_ncon_igs[v] for v in ss) for ss in swap]

    return tensors, ncon_igs, conjs, order, ncon_swap


def _model_shape_as_dense(t: Tensor):
    return tuple(sum(s.D) for s in t.get_legs())

def _preprocess_interleaved_to_expr_and_shapes(*args, unroll=[]):
    r"""Casts interleaved einsum input into default format, stripping
    away unrolled indices if any.
    Collects shapes of the input and output tensors, labeling shapes
    of unrolled indices as negative values.
    Collects shapes of unrolled indices.

    This functions preprocesses the input for _get_contraction_path_cached
    allowing for caching.

    :param args: input to einsum in interleaved format
    :param unroll: indices to unroll
    """
    # assert that unroll indices are contracted over, i.e. appear at least twice for
    # at least two different tensors
    if len(unroll) > 0:
        assert not any(
            [sum([u_i in x for x in (args[1::2] + (args[-1],))]) < 2 for u_i in unroll]
        ), "Invalid choice of unrolled index"

    # cast interleaved format to default einsum while dropping unrolled indices
    #
    # the interleaved format has a) even number of elements, if the (i) the result is a scalar
    #                               or (ii) tensor sorted in default index order
    #                            b) odd number of elements if the result is a tensor and order of output indices
    #                               is explicitly specified
    to_ints = set([i for ig in args[1::2] for i in ig])
    to_ints = {i: idx for idx, i in enumerate(to_ints)}

    expr = ",".join(
        [
            "".join(["" if y in unroll else oe.get_symbol(to_ints[y]) for y in x])
            for x in args[1::2]
        ]
    )
    expr += "->" + "".join(
        ["" if y in unroll else oe.get_symbol(to_ints[y]) for y in args[-1]]
    )

    # NOTE shapes are used in performance model for contraction path search
    #      Here, we use shapes of t.to_dense() as proxy for block-sparse tensor shape
    #
    # assign shape to each index label
    i_to_s = {
        i: s
        for ig, t in zip(args[1::2], args[0 : 2 * (len(args) // 2) : 2])
        for i, s in zip(ig, _model_shape_as_dense(t))
    }

    # create shapes information, labeling shapes on unrolled dimensions as negative
    shapes = tuple(
        tuple(i_to_s[i] if not (i in unroll) else -i_to_s[i] for i in ig)
        for ig in args[1::2] + (args[-1],)
    )
    unrolled_shapes = tuple(i_to_s[i] for i in unroll)

    return expr, shapes, unrolled_shapes

@lru_cache(maxsize=128)
def _log_input_mem_size(shapes : tuple[tuple[int]],names=None,who=None,**kwargs):
    # log the total size (memory footprint) of tensors entering contraction
    if kwargs.get("verbosity",1):
        if names:
            assert len(names) == len(shapes),"Number of names has to match number of operands"
            in_mem_list=", ".join(f"{n} {t} {np.asarray(t).prod()}" for n, t in zip(names, shapes))
        else:
            in_mem_list=", ".join(f"{n} {t} {np.asarray(t).prod()}" for n, t in enumerate(shapes))
        log.info(f"{who} input sizes "+in_mem_list)
        log.info(f"{who} total size {sum(np.asarray(t).prod() for t in shapes)}")


def get_contraction_path(*tn_to_contract, unroll=None,
                         names:Sequence[str]=None, who:str=None, **kwargs)-> tuple[Sequence[tuple[int]], PathInfo]:
    r"""Returns optimal contraction path for tensor network contraction specified in interleaved
    format. Takes into account unrolled indices if any.

    :param tn_to_contract: input to einsum in interleaved format. Explicit index labeling
                           of output is required
    :param unroll: Mapping[Hashable,Union[Sequence[SlicedLeg],int]]
        indices to unroll
    :param names: string labels for tensors used for more readable logging. The order of
                  names has to follow order of tensors as they appear in ``tn_to_contract``
    :param who: string id for logging identifying this optimal contraction path search

    Returns
    -------
    path : Sequence[tuple[int]]
        Optimal contraction path as a sequence of tuples specifying which pair of tensors to contract at each step.
        The path is in terms of positions in current list of tensors, which is shrinking at each step.
    path_info : opt_einsum.contract.PathInfo
         Detailed information about the contraction path, including shapes and memory usage of intermediate tensors.
    """
    # require explicit specification of output index labels
    assert (
        len(tn_to_contract) % 2 == 1
    ), "Explicit specification of output index labels is required"
    unroll = _validate_and_resolve_unroll(*tn_to_contract, unroll=unroll)

    # TODO how to report block-sparse memory footprint & shape
    #      Here, we pass shape of the underlying 1D data array
    _log_input_mem_size(tuple(tuple(t._data.shape) for t in tn_to_contract[:-1][0::2]),names=names,who=who,**kwargs)

    if isinstance(unroll, list):
        raise ValueError(
            "unroll must be a dict or None, got a list. "
            "Pass unroll=None (or omit) for no unrolling, or a dict mapping labels to SlicedLegs."
        )

    if isinstance(unroll, dict) and unroll:
        tensors_orig = tn_to_contract[0 : 2 * (len(tn_to_contract) // 2) : 2]
        igs = list(tn_to_contract[1 : 2 * (len(tn_to_contract) // 2) : 2])
        out_ig = tn_to_contract[-1]

        rep_tensors = list(tensors_orig)
        for k, (t, ig) in enumerate(zip(tensors_orig, igs)):
            for label, sliced_legs in unroll.items():
                if label in ig:
                    user_ax = list(ig).index(label)
                    full_leg = rep_tensors[k].get_legs(user_ax)
                    rep_sl = max(sliced_legs, key=lambda sl: sum(sl.D))
                    mask_t = _build_mask_tensor(rep_sl, full_leg, rep_tensors[k].config)
                    candidate = mask_t.apply_mask(rep_tensors[k], axes=user_ax)
                    if candidate.struct.t:          # non-empty after masking
                        rep_tensors[k] = candidate  # else keep current (fallback)

        rep_args = []
        for t, ig in zip(rep_tensors, igs):
            rep_args.extend([t, ig])
        rep_args.append(out_ig)
        tn_for_path = tuple(rep_args)
        unroll_for_path = list(unroll.keys())
    else:
        tn_for_path = tn_to_contract
        unroll_for_path = []

    expr, shapes, unrolled_shapes = _preprocess_interleaved_to_expr_and_shapes(
        *tn_for_path, unroll=unroll_for_path
    )

    # TODO only shapes are used in performance model for contraction path search
    #      This is works straightforwardly for block-sparse, as shapes of t.to_dense()
    #      Better alternatives ?
    t0= time.perf_counter()
    res= _get_contraction_path_cached(
        expr, shapes, unrolled=unrolled_shapes, names=names, who=who, **kwargs
    )
    t1= time.perf_counter()
    log.info(f"{who} contraction path search took {t1-t0} [s]")
    return res


# Time budget for the DynamicProgramming path search. If DP does not finish
# within this many seconds, fall back to opt_einsum's "random-greedy".
_DP_SEARCH_TIMEOUT_S = 120


def _run_with_timeout(func, timeout):
    r"""Run ``func()`` in a daemon thread, returning ``(result, timed_out)``.

    If ``func`` does not return within ``timeout`` seconds, returns
    ``(None, True)`` and lets the background thread finish on its own (it
    cannot be safely interrupted in Python, but it is daemonic so it will
    not block interpreter shutdown). Exceptions raised by ``func`` are
    re-raised in the caller.
    """
    import threading
    result = [None]
    exc = [None]
    done = threading.Event()

    def _worker():
        try:
            result[0] = func()
        except BaseException as e:  # noqa: BLE001 — re-raised below
            exc[0] = e
        finally:
            done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    finished = done.wait(timeout)
    if not finished:
        return None, True
    if exc[0] is not None:
        raise exc[0]
    return result[0], False


@lru_cache(maxsize=128)
def _get_contraction_path_cached(
    expr, shapes, unrolled=(), names=None, who=None, **kwargs
):
    r"""Cachable function finding optimal contraction path for tensor network contraction
    specified in default einsum format with shapes only.

    :param expr: input to einsum in default format
    :param shapes: shapes of tensors to be contracted
    :param unrolled: shapes of unrolled indices
    :param names: string labels for tensors used for more readable logging. The order of
                  names has to follow order of tensors as they appear in ``tn_to_contract``
    :param who: string id for logging identifying this optimal contraction path search
    """
    optimizer = kwargs.pop("optimizer", None)
    # Default is DynamicProgramming (better peak memory on typical PEPS networks);
    # if the search exceeds the budget we fall back to opt_einsum's "random-greedy".
    _dp_with_timeout = False
    if optimizer in [None, "default", "dp", "dynamic-programming"]:
        optimizer = oe.DynamicProgramming(
            minimize="flops",  # 'size' optimize for largest intermediate tensor size, 'flops' for computation complexity
            search_outer=False,  # search through outer products as well
            cost_cap=True,  # don't use cost-capping strategy
        )
        _dp_with_timeout = True

    # pre-process shapes, by dropping negative values (unrolled index) and last tuple,
    # which holds shapes of output tensor
    shapes_unrolled = tuple(tuple(x for x in s if x > 0) for s in shapes[:-1])
    path = kwargs.pop("path", None)
    kwargs.pop("shapes", False)
    if not path:
        if _dp_with_timeout:
            def _search():
                return oe.contract_path(
                    expr, *shapes_unrolled, optimize=optimizer, shapes=True, **kwargs
                )
            res, timed_out = _run_with_timeout(_search, _DP_SEARCH_TIMEOUT_S)
            if timed_out:
                log.warning(
                    f"{who} DynamicProgramming path search exceeded "
                    f"{_DP_SEARCH_TIMEOUT_S}s; falling back to optimizer='random-greedy'."
                )
                optimizer = "random-greedy"
                path, path_info = oe.contract_path(
                    expr, *shapes_unrolled, optimize=optimizer, shapes=True, **kwargs
                )
            else:
                path, path_info = res
        else:
            path, path_info = oe.contract_path(
                expr, *shapes_unrolled, optimize=optimizer, shapes=True, **kwargs
            )  # ,use_blas=)

    path_info, mem_list = _get_contraction_path_info(
        path, expr, *shapes_unrolled, unrolled=unrolled, names=names, shapes=True
    )
    log.info(
        f"{who} optimizer {optimizer}"
        + (f" unrolled {unrolled}" if len(unrolled) > 0 else "")
        + f"\n{path}\n{path_info}\npeak-mem {max(mem_list):4.3e} mem {[f'{x:4.3e}' for x in mem_list]}"
    )
    return path, path_info


def _get_contraction_path_info(path, *operands, **kwargs):
    r"""opt_einsum contraction path reporting function extended
    to use user-supplied tensor labels ``names`` for description of individual operations.

    :param names: string labels for tensors used for more readable logging. The order of
                  names has to follow the order of tensors as they appear in ``operands``
    """
    names = kwargs.pop("names", None)
    unrolled = kwargs.pop("unrolled", ())

    unknown_kwargs = set(kwargs) - _VALID_CONTRACT_KWARGS
    if len(unknown_kwargs):
        raise TypeError(
            "einsum_path: Did not understand the following kwargs: {}".format(
                unknown_kwargs
            )
        )

    shapes = kwargs.pop("shapes", False)
    use_blas = kwargs.pop("use_blas", True)

    # Python side parsing
    input_subscripts, output_subscript, operands = oe.parser.parse_einsum_input(
        operands
    )

    # Build a few useful list and sets
    input_list = input_subscripts.split(",")
    if names:
        inputs_to_names = list(names)
    input_sets = [set(x) for x in input_list]
    if shapes:
        input_shps = operands
    else:
        input_shps = [x.shape for x in operands]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(",", ""))

    # Get length of each unique dimension and ensure all dimensions are correct
    size_dict = {}
    for tnum, term in enumerate(input_list):
        sh = input_shps[tnum]

        if len(sh) != len(term):
            raise ValueError(
                "Einstein sum subscript '{}' does not contain the "
                "correct number of indices for operand {}.".format(
                    input_list[tnum], tnum
                )
            )
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])

            if char in size_dict:
                # For broadcasting cases we always want the largest dim size
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError(
                        "Size of label '{}' for operand {} ({}) does not match previous "
                        "terms ({}).".format(char, tnum, size_dict[char], dim)
                    )
            else:
                size_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = [
        oe.helpers.compute_size_by_dict(term, size_dict)
        for term in input_list + [output_subscript]
    ]

    num_ops = len(input_list)

    # Compute naive cost
    # This isnt quite right, need to look into exactly how einsum does this
    # indices_in_input = input_subscripts.replace(',', '')

    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = oe.helpers.flop_count(indices, inner_product, num_ops, size_dict)

    cost_list = []
    scale_list = []
    size_list = [] # sizes of outputs
    contraction_list = []
    mem_list= []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))

        try:
            contract_tuple = oe.helpers.find_contraction(
                contract_inds, input_sets, output_set
            )
        except TypeError as e:
            # inputt_sets have to be frozenset sets for v3.4.0
            contract_tuple = oe.helpers.find_contraction(
                contract_inds, [frozenset(s) for s in input_sets], output_set
            )
        out_inds, input_sets, idx_removed, idx_contract = contract_tuple

        # Compute cost, scale, and size
        cost = oe.helpers.flop_count(
            idx_contract, idx_removed, len(contract_inds), size_dict
        )
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(oe.helpers.compute_size_by_dict(out_inds, size_dict))

        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        if names:
            tmp_inds_to_names = [inputs_to_names.pop(x) for x in contract_inds]
        # make list
        input_shps= list(input_shps) if type(input_shps)==tuple else input_shps
        tmp_shapes = [input_shps.pop(x) for x in contract_inds]

        if use_blas:
            do_blas = oe.blas.can_blas(tmp_inputs, out_inds, idx_removed, tmp_shapes)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        shp_result = oe.parser.find_output_shape(tmp_inputs, tmp_shapes, idx_result)

        input_list.append(idx_result)
        if names:
            inputs_to_names.append(f"_TMP_{cnum}")
        input_shps.append(np.asarray(shp_result))

        # sum the currently contracted ops and remaining ops
        # Here, all shapes have to by np.ndarray
        mem_list.append(sum([x.prod() if hasattr(x,'prod') else np.asarray(x).prod() for x in tmp_shapes+input_shps]))

        einsum_str= ",".join(tmp_inputs) + "->" + idx_result
        if names:
            einsum_str = ",".join(tmp_inds_to_names) + "->" + inputs_to_names[-1]

        # for large expressions saving the remaining terms at each step can
        # incur a large memory footprint - and also be messy to print
        if len(input_list) <= 20:
            remaining = tuple(input_list)
        else:
            remaining = None

        contraction = (contract_inds, idx_removed, einsum_str, remaining, do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    # TODO associate input_subscripts with names
    path_print = PathInfo(
        contraction_list,
        input_subscripts,
        output_subscript,
        indices,
        path,
        scale_list,
        naive_cost,
        opt_cost,
        size_list,
        size_dict,
    )

    return path_print, mem_list


def contract_with_unroll_compute_constants(*args, **kwargs):
    r"""Like :func:`contract_with_unroll`, but tensors that carry no unrolled
    index (constants) are contracted together once before the loop starts.
    Only the smaller variable sub-network is re-evaluated on every iteration,
    avoiding repeated work.

    Constant tensors that are disconnected in the network (share no index) are
    pre-contracted independently, keeping each intermediate small.

    :param args: interleaved format ``(T1, ig1, T2, ig2, ..., out_ig)``
    :param unroll: Mapping[Hashable,Union[Sequence[SlicedLeg],int]] or None
        specifying indices to unroll and how to slice them.
    :param optimize: contraction path for the full network
    :param checkpoint_loop: if True, each unrolled loop iteration is wrapped in
        :func:`torch.utils.checkpoint.checkpoint`.
    """
    checkpoint_loop = kwargs.pop("checkpoint_loop", False)
    who = kwargs.pop("who", None)
    kwargs.pop("verbosity", None)
    path_search_kwargs = {k: kwargs[k] for k in ("optimizer", "memory_limit")
                          if k in kwargs}
    if who is not None:
        path_search_kwargs["who"] = who
    unroll = kwargs.pop("unroll", None)
    swap = kwargs.pop("swap", None)
    unroll = _validate_and_resolve_unroll(*args, unroll=unroll)

    if unroll is None:
        # convert to ncon call
        ts, inds, conjs, order, ncon_swap = _convert_path_to_ncon_args(*args, swap=swap, **kwargs)
        if checkpoint_loop:
            return _ncon_checkpointed(ts, inds, conjs, order, ncon_swap)
        return ncon(ts, inds, conjs=conjs, order=order, swap=ncon_swap)

    path = kwargs.pop("optimize", None)
    assert path is not None, "optimize (contraction path) must be provided"

    tensors = args[0 : 2 * (len(args) // 2) : 2]
    index_groups = list(args[1 : 2 * (len(args) // 2) : 2])
    out_ig = args[-1]
    unroll_labels = set(unroll.keys())

    # Partition tensors into constants (no unrolled index) and variables.
    const_mask = [not any(u in ig for u in unroll_labels) for ig in index_groups]

    var_tensors = [t for t, c in zip(tensors, const_mask) if not c]
    var_igs     = [list(ig) for ig, c in zip(index_groups, const_mask) if not c]
    var_and_out = set(i for ig in var_igs for i in ig) | set(out_ig)

    # If any swap crosses the variable/constant boundary (one label appears
    # only in constants, the other only in variables), the pre-contraction
    # cannot handle it because one label would disappear.  Fall back.
    const_only_labels = set(
        i for ig, c in zip(index_groups, const_mask) if c for i in ig
    ) - var_and_out
    has_cross_boundary_swap = False
    if swap is not None:
        for sw_pair in swap:
            in_const = sum(v in const_only_labels for v in sw_pair)
            if 0 < in_const < len(sw_pair):
                has_cross_boundary_swap = True
                break

    if not any(const_mask) or has_cross_boundary_swap:
        # Nothing to pre-compute, or swaps cross the boundary; delegate directly.
        return _contract_with_sliced_unroll(*args, unroll=unroll, optimize=path,
                                            checkpoint_loop=checkpoint_loop, swap=swap, **kwargs)

    # Find connected components of the constant sub-network via Union-Find.
    # Two constant tensors are connected if they share at least one index.
    # Each component is contracted independently, keeping intermediates small.
    const_positions = [k for k, c in enumerate(const_mask) if c]
    parent = {k: k for k in const_positions}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    index_to_first = {}
    for k in const_positions:
        for idx in index_groups[k]:
            if idx in index_to_first:
                px, py = _find(k), _find(index_to_first[idx])
                if px != py:
                    parent[px] = py
            else:
                index_to_first[idx] = k

    comps = {}
    for k in const_positions:
        root = _find(k)
        comps.setdefault(root, []).append(k)

    # Split swaps between constant components and the reduced network.
    # A swap (a, b) goes to a constant component if both labels belong to it.
    comp_index_sets = {}
    for root, positions in comps.items():
        comp_index_sets[root] = set(
            i for k in positions for i in index_groups[k]
        )
    comp_swaps = {root: [] for root in comps}  # swaps assigned to each component
    reduced_swap = []  # swaps for the reduced network
    if swap is not None:
        for sw_pair in swap:
            assigned = False
            for root, idx_set in comp_index_sets.items():
                if all(v in idx_set for v in sw_pair):
                    comp_swaps[root].append(sw_pair)
                    assigned = True
                    break
            if not assigned:
                reduced_swap.append(sw_pair)
    reduced_swap = reduced_swap or None

    # Pre-contract each component once, outside the loop.
    pre_contracted = []  # list of (tensor, ig)
    for comp_root, comp in comps.items():
        comp_tensors = [tensors[k] for k in comp]
        comp_igs     = [list(index_groups[k]) for k in comp]
        c_swap = comp_swaps[comp_root] or None

        # Output indices: those that feed into the variable network or the final output.
        seen = set()
        comp_out_ig = []
        for ig in comp_igs:
            for i in ig:
                if i in var_and_out and i not in seen:
                    comp_out_ig.append(i)
                    seen.add(i)
        comp_out_ig = tuple(comp_out_ig)

        if len(comp) == 1 and c_swap is None:
            pre_contracted.append((comp_tensors[0], comp_igs[0]))
        else:
            comp_interleaved = []
            for t, ig in zip(comp_tensors, comp_igs):
                comp_interleaved.extend([t, ig])
            comp_interleaved.append(comp_out_ig)
            comp_path, _ = get_contraction_path(*comp_interleaved, **path_search_kwargs)
            c_ts, c_inds, c_conjs, c_order, c_ncon_swap = _convert_path_to_ncon_args(
                *comp_interleaved, optimize=comp_path, swap=c_swap
            )
            if checkpoint_loop:
                pre_contracted.append((_ncon_checkpointed(c_ts, c_inds, c_conjs, c_order, c_ncon_swap), list(comp_out_ig)))
            else:
                pre_contracted.append((ncon(c_ts, c_inds, conjs=c_conjs, order=c_order, swap=c_ncon_swap), list(comp_out_ig)))

    # Rebuild the reduced network: variable tensors + one tensor per constant component.
    reduced_interleaved = []
    for t, ig in zip(var_tensors, var_igs):
        reduced_interleaved.extend([t, ig])
    for t, ig in pre_contracted:
        reduced_interleaved.extend([t, ig])
    reduced_interleaved.append(out_ig)

    # Find a contraction path for the reduced network (strip unrolled dims from shape model).
    reduced_path, _ = get_contraction_path(*reduced_interleaved, unroll=unroll, **path_search_kwargs)

    return _contract_with_sliced_unroll(
        *reduced_interleaved, unroll=unroll, optimize=reduced_path,
        checkpoint_loop=checkpoint_loop, swap=reduced_swap, **kwargs
    )


def contract_with_unroll(*args, **kwargs):
    r"""Extension of opt_einsum's contract allowing for index unrolling
    and use of checkpointing over unrolled loop.

    :param args: input to einsum in interleaved format. Explicit index labeling
                 of output is required
    :param unroll: Mapping[Hashable,Sequence[SlicedLeg]] or None
        indices to unroll
    :param checkpoint_loop: if True, each unrolled loop iteration is wrapped in
        :func:`torch.utils.checkpoint.checkpoint`, avoiding storage of masking
        and ncon intermediates across all iterations simultaneously.
    """
    checkpoint_loop = kwargs.pop("checkpoint_loop", False)
    kwargs.pop("who", None)
    kwargs.pop("verbosity", None)
    unroll = kwargs.pop("unroll", None)
    swap = kwargs.pop("swap", None)
    unroll = _validate_and_resolve_unroll(*args, unroll=unroll)

    if unroll is None:
        # convert to ncon call
        ts, inds, conjs, order, ncon_swap = _convert_path_to_ncon_args(*args, swap=swap, **kwargs)
        if checkpoint_loop:
            return _ncon_checkpointed(ts, inds, conjs, order, ncon_swap)
        return ncon(ts, inds, conjs=conjs, order=order, swap=ncon_swap)

    if isinstance(unroll, dict):
        # block-sparse sliced unrolling: unroll is {label: [SlicedLeg, ...]}
        path = kwargs.pop("optimize", None)
        assert path is not None, "optimize (contraction path) must be provided"
        return _contract_with_sliced_unroll(*args, unroll=unroll, optimize=path,
                                            checkpoint_loop=checkpoint_loop, swap=swap, **kwargs)

    raise NotImplementedError(
        "contract_with_unroll: unsupported unroll type. "
        "Pass unroll as a dict mapping labels to lists of SlicedLeg."
    )
