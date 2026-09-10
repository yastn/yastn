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
"""Distributed (multi-node) dispatch for ``_contract_with_sliced_unroll``.

Where :mod:`._oe_blocksparse_mp` parallelises the unrolled-combo sum across a
pool of local worker processes communicating via **CUDA IPC** (node-local),
this module runs the same sum under an **SPMD** model over
``torch.distributed``: every rank runs the identical program, statically owns a
cost-balanced slice of the surviving combos, computes its local partials with
the *unchanged* single-process engine, and reduces the assembled output with a
single collective. IPC handles never leave a node; NCCL carries the reduction
intra- and inter-node.

Why static balancing is enough: combo costs here are moderate and
metadata-predictable, so a cost-aware **LPT** (longest-processing-time-first)
partition keeps ranks within a small factor of each other. The closing
``all_reduce`` is then a cheap barrier rather than a straggler bottleneck.

Two facts make the collective form clean and communication-free at schedule
time (both provided by :mod:`._oe_blocksparse_mp`):

* :func:`_derive_output_structs` derives the full output key set and per-key
  shapes from the input *legs alone* — every rank agrees on the output layout
  with no exchange.
* :func:`_zero_fill_to_full` makes each rank's contribution to the output
  identically shaped, so a single ``all_reduce(SUM)`` over the assembled buffer
  replaces the MP path's gather + per-key sum.

Autograd uses the same checkpoint pattern as the MP path: forward runs under
``no_grad``; backward re-runs the rank's assigned combos with autograd enabled,
splits ``grad_out`` into per-key grads, backpropagates, and ``all_reduce``s the
per-input grads (SUM, not average — the forward output is a sum over combos).

Launch model: the caller starts one process per GPU via ``torchrun`` (or an
equivalent ``dist.init_process_group``) and every rank reaches the
``contract_with_unroll(..., distributed=True)`` call together with a
structurally identical (data-parallel) copy of the inputs.
"""
import logging

import torch
import torch.distributed as dist
from .._profile import nsys_profile

log = logging.getLogger(__name__)


def _dist_world_size(group=None):
    """World size of ``group`` (default group) if distributed is initialised,
    else 1. Used by the router to decide whether the SPMD path is worthwhile."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=group)
    return 1


def _dist_ready(group=None):
    """Return ``(rank, world_size)`` for the SPMD contraction.

    Requires an initialised process group (the caller launches via
    ``torchrun`` / ``dist.init_process_group``). The compute/reduction device is
    NOT chosen here — it follows the caller's input placement (see the
    dispatcher), so a rank that put its tensors on ``cuda:local_rank`` reduces
    there, and a CPU/gloo run reduces on the CPU."""
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "distributed=True requires an initialised torch.distributed process "
            "group. Launch with torchrun (or call dist.init_process_group) before "
            "contract_with_unroll(..., distributed=True).")
    return dist.get_rank(group=group), dist.get_world_size(group=group)


@nsys_profile
def _combo_costs(tensors, ig_list, out_ig, surviving, dim_overrides_per_combo):
    """Per-combo compute-effort estimate (opt_einsum ``opt_cost``, i.e. FLOPs),
    computed from metadata only — no GPU work.

    For each surviving combo the per-axis effective dims (post meta-mask +
    prefilter) collected by ``_metadata_filter_combos(collect_dim_overrides=True)``
    are turned into einsum ``shapes`` and fed through the cached path search.
    Combos that share a dim signature share the (lru-cached) search, so the
    number of actual searches is bounded by the distinct signatures.

    Returns ``{combo_idx: float | None}`` — ``None`` where a cost could not be
    derived (the LPT scheduler treats those as unit cost)."""
    from .oe_blocksparse import (
        _build_template_interleaved,
        _preprocess_interleaved_to_expr_and_shapes,
        _get_contraction_path_cached,
    )
    template = _build_template_interleaved(list(tensors), list(ig_list), out_ig)
    expr, _ = _preprocess_interleaved_to_expr_and_shapes(*template)
    costs = {}
    for n in surviving:
        do = (dim_overrides_per_combo or {}).get(n) or {}
        try:
            _, shapes = _preprocess_interleaved_to_expr_and_shapes(*template, dim_overrides=do)
            path, info = _get_contraction_path_cached(expr, shapes)
            costs[n] = float(info.opt_cost)
        except Exception as e:  # noqa: BLE001 — cost is advisory; fall back to unit
            log.debug("combo %s cost estimate failed (%s); using unit cost", n, e)
            costs[n] = None
    return costs


def _lpt_partition(surviving, costs, world_size):
    """Longest-processing-time-first greedy bin-packing of ``surviving`` combos
    into ``world_size`` bins by estimated ``costs``.

    Deterministic given identical inputs, so every rank computes the *same*
    assignment with no communication; rank ``r`` then runs bin ``r``. Combos
    with unknown cost are weighted as unit. Empty bins (more ranks than combos)
    are valid — that rank contributes an all-zero buffer to the reduction."""
    import heapq
    bins = [[] for _ in range(world_size)]
    if world_size <= 1:
        bins[0] = list(surviving)
        return bins

    def cost_of(n):
        v = costs.get(n)
        return v if v is not None else 1.0

    # (load, tie-break bin index) min-heap; ties broken by index for determinism.
    heap = [(0.0, i) for i in range(world_size)]
    heapq.heapify(heap)
    # Sort by cost desc, then combo index for a stable, rank-independent order.
    for n in sorted(surviving, key=lambda n: (-cost_of(n), n)):
        load, i = heapq.heappop(heap)
        bins[i].append(n)
        heapq.heappush(heap, (load + cost_of(n), i))
    return bins


def _reconstruct_inputs(input_data_tensors, input_meta_list, cfg, requires_grad):
    """Rebuild yastn tensors from (data, meta) pairs on this rank.

    When ``requires_grad`` the ``_data`` is replaced by a fresh detached leaf
    (mirrors the MP worker's backward re-run) so the caller can read
    ``leaf.grad`` after ``torch.autograd.backward``. Returns ``(tensors,
    leaves)`` where ``leaves`` is ``None`` unless ``requires_grad``."""
    from . import Tensor
    tensors = [Tensor.from_dict({**m, "data": data}, config=cfg)
               for data, m in zip(input_data_tensors, input_meta_list)]
    if not requires_grad:
        return tensors, None
    leaves = []
    for t in tensors:
        leaf = t._data.detach().clone().requires_grad_(True)
        t._data = leaf
        leaves.append(leaf)
    return tensors, leaves


def _interleave(tensors, ig_list, out_ig):
    inter = []
    for t, ig in zip(tensors, ig_list):
        inter.append(t)
        inter.append(ig)
    inter.append(out_ig)
    return inter


def _run_local_partials(tensors, meta):
    """Run this rank's assigned combo bin through the unchanged single-process
    engine, returning ``{output_pos_key: partial yastn.Tensor}``. Grad state is
    dictated by the surrounding context (no_grad in forward, enabled in
    backward)."""
    from .oe_blocksparse import _contract_with_sliced_unroll
    engine_kwargs = dict(meta["ncon_kwargs"])
    engine_kwargs["_precomputed_pf_trim"] = meta["pf_trim_per_combo"]
    engine_kwargs["_precomputed_dim_overrides"] = (
        meta["dim_overrides_per_combo"] if meta["per_combo_path"] else None)
    interleaved = _interleave(tensors, meta["ig_list"], meta["out_ig"])
    return _contract_with_sliced_unroll(
        *interleaved, unroll=meta["unroll"], optimize=meta["optimize"],
        swap=meta["swap"], _combo_indices=meta["my_bin"], _return_partials=True,
        mp_workers_per_device=0, checkpoint_loop=meta["checkpoint_loop"],
        **engine_kwargs)


def _assemble_local_full(partials, meta):
    """Assemble this rank's per-key partials into a buffer with the *full*
    output struct (zeros where this rank produced no partial), so every rank's
    buffer is identically laid out and a single ``all_reduce(SUM)`` yields the
    assembled output.

    Missing keys are padded with zeros of the cached per-key size; the complete
    key set is then blocked with the SAME ``block`` + ``drop_leg_history`` the
    forward pass uses, so the result matches ``full_struct`` exactly."""
    from . import Tensor
    from ..initialize import block as yastn_block
    from ._oe_blocksparse_mp import _zero_fill_to_full
    cfg, dtype, device = meta["cfg"], meta["dtype"], meta["device"]
    per_key_struct = meta["per_key_struct"]
    full_keys = sorted(per_key_struct.keys())

    per_key_tensors = {}
    for k in full_keys:
        if k in partials:
            per_key_tensors[k] = _zero_fill_to_full(partials[k], per_key_struct[k], cfg)
        else:
            zdata = torch.zeros(per_key_struct[k]["size"], dtype=dtype, device=device)
            per_key_tensors[k] = Tensor.from_dict(
                {**per_key_struct[k], "data": zdata}, config=cfg)

    if meta["common_legs_axes"] is None:
        return per_key_tensors[()]._data
    assembled = yastn_block(per_key_tensors, common_legs=meta["common_legs_axes"])
    return assembled.drop_leg_history()._data


def _split_grad_per_key(grad_out_data, meta):
    """Split ``grad_out_data`` (grad wrt the assembled output) into per-key
    grads by replaying the linear ``block`` step under autograd with zero
    leaves. ``block`` is linear in its inputs, so the per-key grads are
    independent of the leaf values."""
    if meta["common_legs_axes"] is None:
        return {(): grad_out_data.detach()}
    from . import Tensor
    from ..initialize import block as yastn_block
    cfg, dtype, device = meta["cfg"], meta["dtype"], meta["device"]
    per_key_struct = meta["per_key_struct"]
    full_keys = sorted(per_key_struct.keys())
    with torch.enable_grad():
        leaves = {}
        per_key_tensors = {}
        for k in full_keys:
            leaf = torch.zeros(per_key_struct[k]["size"], dtype=dtype,
                               device=device, requires_grad=True)
            leaves[k] = leaf
            per_key_tensors[k] = Tensor.from_dict(
                {**per_key_struct[k], "data": leaf}, config=cfg)
        assembled = yastn_block(per_key_tensors, common_legs=meta["common_legs_axes"])
        assembled.drop_leg_history()._data.backward(grad_out_data.detach())
    return {k: leaves[k].grad.detach() for k in full_keys
            if leaves[k].grad is not None}


class _DistSlicedUnrollFunction(torch.autograd.Function):
    """SPMD analogue of :class:`._oe_blocksparse_mp._MultiprocSlicedUnrollFunction`.

    Forward: each rank computes its combo bin's partials (no_grad), assembles a
    full-output-shaped buffer (zeros for untouched keys), and ``all_reduce``s it
    — every rank ends with the assembled output.
    Backward: each rank splits ``grad_out`` into per-key grads, re-runs its bin
    with autograd, backpropagates, and ``all_reduce``s the per-input grads.
    """

    @staticmethod
    def forward(ctx, *all_args):
        meta = all_args[-1]
        input_data_tensors = all_args[:-1]
        ctx.n_inputs = len(input_data_tensors)
        ctx.meta = meta

        tensors, _ = _reconstruct_inputs(
            input_data_tensors, meta["input_meta_list"], meta["cfg"], requires_grad=False)
        with torch.no_grad():
            partials = _run_local_partials(tensors, meta)
            local_full = _assemble_local_full(partials, meta)
        dist.all_reduce(local_full, op=dist.ReduceOp.SUM, group=meta["group"])

        ctx.save_for_backward(*input_data_tensors)
        return local_full

    @staticmethod
    def backward(ctx, grad_out_data):
        meta = ctx.meta
        input_data_tensors = ctx.saved_tensors

        grad_per_key = _split_grad_per_key(grad_out_data, meta)

        # A custom Function.backward runs with grad tracking DISABLED, so the
        # local combo re-run and its backward must be re-enabled explicitly (the
        # MP path avoids this only because its re-run happens in a worker
        # process, outside any Function.backward).
        from ._oe_blocksparse_mp import _zero_fill_to_full
        with torch.enable_grad():
            tensors, leaves = _reconstruct_inputs(
                input_data_tensors, meta["input_meta_list"], meta["cfg"], requires_grad=True)
            partials = _run_local_partials(tensors, meta)

            out_tensors, grad_tensors = [], []
            for key, partial in partials.items():
                if key not in grad_per_key:
                    continue
                full_partial = _zero_fill_to_full(partial, meta["per_key_struct"][key], meta["cfg"])
                out_tensors.append(full_partial._data)
                grad_tensors.append(grad_per_key[key])
            if out_tensors:
                torch.autograd.backward(out_tensors, grad_tensors)

        grads = [leaf.grad.detach() if leaf.grad is not None
                 else torch.zeros_like(leaf) for leaf in leaves]
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=meta["group"])
        return (*grads, None)


def _contract_with_sliced_unroll_dist(*args, unroll, optimize, checkpoint_loop=False,
                                      swap=None, group=None, **kwargs):
    """SPMD dispatcher for ``_contract_with_sliced_unroll``.

    Every rank: prefilters + schedules identically (no comms), runs its combo
    bin through the single-process engine, and reduces the assembled output via
    ``all_reduce``. Supports single-key (no output unroll) and multi-key
    (output-unrolled) contractions and autograd (checkpoint pattern)."""
    from .oe_blocksparse import _metadata_filter_combos
    from ._oe_blocksparse_mp import _derive_output_structs, _config_descriptor
    from ._initialize import make_config
    from . import Tensor, YastnError

    tensors = args[0:2 * (len(args) // 2):2]
    ig_list = list(args[1:2 * (len(args) // 2):2])
    out_ig = args[-1]
    parent_config = tensors[0].config

    if "torch" not in getattr(parent_config.backend, "BACKEND_ID", ""):
        raise RuntimeError("distributed=True requires a torch backend.")

    rank, world_size = _dist_ready(group)
    # Reduction device follows the caller's input placement: SPMD ranks put
    # their (identical) inputs on their own device (e.g. cuda:local_rank), and a
    # CPU/gloo run keeps everything on the CPU. NCCL requires the input to live
    # on the local GPU; that is the caller's responsibility.
    device = str(tensors[0].device)
    cfg = make_config(**{**_config_descriptor(parent_config), "default_device": str(device)})

    per_combo_path = bool(kwargs.get("per_combo_path", False))

    # Metadata-only prefilter + per-combo effective dims — deterministic and
    # identical on every rank (no data, no GPU work).
    surviving, pf_trim_per_combo, dim_overrides_per_combo = _metadata_filter_combos(
        tensors, ig_list, out_ig, unroll, optimize, swap, collect_dim_overrides=True)
    if not surviving:
        # Raised identically on all ranks *before* any collective — no deadlock.
        raise YastnError("No valid charge sectors found for contraction.")

    costs = _combo_costs(tensors, ig_list, out_ig, surviving, dim_overrides_per_combo)
    my_bin = _lpt_partition(surviving, costs, world_size)[rank]
    log.info("dist rank %d/%d: %d/%d combos, est. load %.3e",
             rank, world_size, len(my_bin), len(surviving),
             sum((costs.get(n) or 1.0) for n in my_bin))

    per_key_struct, full_struct, common_legs_axes = _derive_output_structs(
        tensors, ig_list, out_ig, unroll, surviving, cfg)

    # Split each input into (data on this rank's device, picklable meta).
    input_data_tensors, input_meta_list = [], []
    for t in tensors:
        d = t.to_dict(level=1)
        data = d.pop("data")
        if str(data.device) != str(device):
            data = data.to(device)
        input_data_tensors.append(data)
        input_meta_list.append(d)

    meta = {
        "cfg": cfg,
        "device": device,
        "dtype": input_data_tensors[0].dtype,
        "group": group,
        "ig_list": ig_list,
        "out_ig": out_ig,
        "unroll": unroll,
        "optimize": optimize,
        "swap": swap,
        "ncon_kwargs": kwargs,
        "per_combo_path": per_combo_path,
        "input_meta_list": input_meta_list,
        "per_key_struct": per_key_struct,
        "full_struct": full_struct,
        "common_legs_axes": common_legs_axes,
        "my_bin": my_bin,
        "checkpoint_loop": checkpoint_loop,
        "pf_trim_per_combo": pf_trim_per_combo,
        "dim_overrides_per_combo": dim_overrides_per_combo,
    }

    out_data = _DistSlicedUnrollFunction.apply(*input_data_tensors, meta)
    return Tensor.from_dict({**full_struct, "data": out_data}, config=cfg)
