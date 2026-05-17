"""Multiprocessing dispatch for `_contract_with_sliced_unroll`.

Spawns a pool of persistent worker processes (one or more per device) and
dispatches the unrolled-combo sum across them. Workers are GIL-free; on small
launch-overhead-bound workloads (block-sparse iPEPS energy with chi=98, D=7)
this gives near-linear speedup vs the threaded path, which is GIL-bound.

Autograd: a custom torch.autograd.Function uses the **checkpoint pattern** —
forward in workers is no_grad; on backward, workers re-run their assigned
combos with autograd enabled, call backward per output_pos_key, and ship
per-input gradient data tensors back via CUDA IPC.

Supports both:
- Single-key contractions (no output-unrolled labels). The result has a single
  block; workers return ``{(): partial_data}`` and parent uses ``merged[()]``.
- Multi-key contractions (output-unrolled labels). Workers return
  ``{key: partial_data}`` for each key they touched; parent assembles via
  ``yastn_block`` and, in backward, re-runs the block step in autograd-enabled
  mode to split ``grad_out_data`` back into per-key gradients before shipping.
"""
import atexit
import torch
import torch.multiprocessing as _mp


# Module-level pool registry: (devices_tuple, mp_workers_per_device,
# config_descriptor_hash) -> _PersistentWorkerPool
_pool_registry = {}


def _config_descriptor(config):
    return {
        'backend': config.backend.BACKEND_ID,
        'sym': config.sym.SYM_ID,
        'default_device': config.default_device,
        'default_dtype': config.default_dtype,
        'fermionic': config.fermionic,
        'default_fusion': config.default_fusion,
        'force_fusion': config.force_fusion,
        'tensordot_policy': config.tensordot_policy,
    }


def _serialize_yastn(t):
    """yastn.Tensor -> dict for IPC. Data is detached; metadata is picklable."""
    d = t.to_dict(level=1)
    d['data'] = d['data'].detach()
    return d


def _deserialize_yastn(d, config):
    from . import Tensor
    return Tensor.from_dict(d, config=config)


def _build_cache_key(input_dicts, unroll, ig_list, out_ig, optimize, swap, per_combo_path):
    """Stable key for the assembly-info cache. ``per_combo_path`` participates
    because cached entries carry ``dim_overrides_per_combo`` only when it was
    True at insertion time; reusing a False-time entry for a True-time call
    would leave workers with a missing payload."""
    input_keys = []
    for d in input_dicts:
        st = d['struct']
        if isinstance(st, dict):
            input_keys.append((st.get('s'), tuple(map(tuple, st.get('t', ()))),
                               tuple(map(tuple, st.get('D', ()))), st.get('n')))
        else:
            input_keys.append((st.s, st.t, st.D, st.n))
    unroll_key = tuple(sorted(
        (str(k), tuple((sl.t, sl.D) for sl in sls)) for k, sls in unroll.items()
    ))
    return (tuple(input_keys), unroll_key,
            tuple(tuple(ig) for ig in ig_list), tuple(out_ig),
            str(optimize), str(swap), bool(per_combo_path))


def _compute_common_legs_axes(partials_dict, unroll, out_ig):
    """Return (common_legs_axes, ndim_out) for yastn_block of partials, or
    (None, None) if no output-unrolled labels (single-key case).
    Mirrors the assembly logic in _contract_with_sliced_unroll.
    """
    out_ig_list = list(out_ig)
    blocked_axes = sorted(
        out_ig_list.index(u) for u in unroll.keys() if u in out_ig_list
    )
    if not blocked_axes:
        return None, None
    if not partials_dict:
        return None, None
    first_partial = next(iter(partials_dict.values()))
    ndim_out = first_partial.ndim_n
    return [ax for ax in range(ndim_out) if ax not in blocked_axes], ndim_out


def _meta_only(yastn_dict):
    return {k: v for k, v in yastn_dict.items() if k != 'data'}


def _per_device_input_replicas(input_data_tensors, worker_devs, original_device):
    r"""Replicate input data tensors once per unique worker device.

    Workers on the same device receive IPC handles to the same per-device
    buffer, so N workers on cuda:1 share one cuda:1 replica instead of
    creating N independent copies inside ``Tensor.from_dict``.
    """
    original_device = str(original_device)
    data_per_dev = {original_device: [d.detach() for d in input_data_tensors]}
    for dev in {str(d) for d in worker_devs}:
        if dev == original_device:
            continue
        data_per_dev[dev] = [d.detach().to(dev) for d in input_data_tensors]
    return data_per_dev


def _patch_worker_kwargs(ncon_kwargs, assigned, pf_trim_per_combo, dim_overrides_per_combo):
    """Build a per-worker copy of ncon_kwargs carrying just this worker's
    slice of precomputed prefilter data, so the worker's
    ``_contract_with_sliced_unroll`` can skip re-running ``_meta_combo_check``
    / ``_post_trim_label_dims`` for combos the parent already prefiltered."""
    if pf_trim_per_combo is None and dim_overrides_per_combo is None:
        return ncon_kwargs
    patched = dict(ncon_kwargs)
    if pf_trim_per_combo is not None:
        patched['_precomputed_pf_trim'] = {n: pf_trim_per_combo[n]
                                           for n in assigned if n in pf_trim_per_combo}
    if dim_overrides_per_combo is not None:
        patched['_precomputed_dim_overrides'] = {n: dim_overrides_per_combo[n]
                                                  for n in assigned if n in dim_overrides_per_combo}
    return patched


def _zero_fill_to_full(partial, full_struct_dict, cfg):
    """Return a yastn Tensor with full_struct_dict's struct, equal to partial
    on partial's blocks and zero elsewhere. Autograd-tracked through partial.
    Used by workers to align per-key partial sums to the cached per-key struct.
    """
    from . import Tensor
    full_size = (full_struct_dict['struct']['size']
                 if isinstance(full_struct_dict['struct'], dict)
                 else full_struct_dict['struct'].size)
    zeros_data = torch.zeros(full_size,
                             dtype=partial._data.dtype,
                             device=partial._data.device)
    zero_tensor = Tensor.from_dict({**full_struct_dict, 'data': zeros_data},
                                   config=cfg)
    return zero_tensor + partial


def _worker_main(rank, gpu_dev, config_desc, cmd_q, res_q):
    """Worker process entry point. Loops on cmd_q until 'shutdown'."""
    import os, sys
    import torch
    if str(gpu_dev).startswith('cuda'):
        gpu_idx = int(str(gpu_dev).split(':')[1])
        torch.cuda.set_device(gpu_idx)
    from ._initialize import make_config
    from .oe_blocksparse import _contract_with_sliced_unroll
    # Override default_device to this worker's assigned GPU so yastn tensors
    # reconstructed from IPC are placed on the right device (Tensor.from_dict
    # honors config.default_device).
    cfg = make_config(**{**config_desc, 'default_device': str(gpu_dev)})

    while True:
        try:
            msg = cmd_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        cmd = msg[0]
        if cmd == 'shutdown':
            break
        try:
            if cmd in ('forward', 'backward'):
                (txn_id, serialized_inputs, ig_list, out_ig, unroll, optimize,
                 swap, ncon_kwargs, assigned_indices, grad_per_key,
                 per_key_struct, checkpoint_loop) = msg[1:]
                inputs = [_deserialize_yastn(d, cfg) for d in serialized_inputs]
                if cmd == 'backward':
                    for t in inputs:
                        t._data = t._data.detach().clone().requires_grad_(True)

                interleaved = []
                for t, ig in zip(inputs, ig_list):
                    interleaved.append(t)
                    interleaved.append(ig)
                interleaved.append(out_ig)

                if cmd == 'forward':
                    with torch.no_grad():
                        partials = _contract_with_sliced_unroll(
                            *interleaved, unroll=unroll, optimize=optimize, swap=swap,
                            _combo_indices=assigned_indices,
                            _return_partials=True,
                            mp_workers_per_device=0,
                            checkpoint_loop=checkpoint_loop,
                            **ncon_kwargs,
                        )
                    if per_key_struct is None:
                        # Raw mode (cache miss): no zero-fill — different
                        # workers may produce partials with different charge
                        # sectors. Parent handles via yastn '+'.
                        out = {k: _serialize_yastn(v) for k, v in partials.items()}
                    else:
                        # Zero-fill mode (cache hit): all workers' per-key
                        # tensors share identical shape so parent can sum
                        # raw data tensors without yastn '+'.
                        out = {}
                        for k, p in partials.items():
                            full_p = _zero_fill_to_full(p, per_key_struct[k], cfg)
                            out[k] = _serialize_yastn(full_p)
                    res_q.put(('forward_done', rank, txn_id, out))
                else:  # backward
                    partials = _contract_with_sliced_unroll(
                        *interleaved, unroll=unroll, optimize=optimize, swap=swap,
                        _combo_indices=assigned_indices,
                        _return_partials=True,
                        mp_workers_per_device=0,
                        checkpoint_loop=checkpoint_loop,
                        **ncon_kwargs,
                    )
                    out_tensors = []
                    grad_tensors = []
                    worker_device = inputs[0]._data.device
                    for key, partial in partials.items():
                        if key not in grad_per_key:
                            continue
                        full_partial = _zero_fill_to_full(partial, per_key_struct[key], cfg)
                        out_tensors.append(full_partial._data)
                        g = grad_per_key[key]
                        if g.device != worker_device:
                            g = g.to(worker_device)
                        grad_tensors.append(g)
                    if out_tensors:
                        torch.autograd.backward(out_tensors, grad_tensors)
                    grads = [t._data.grad.detach() if t._data.grad is not None
                             else torch.zeros_like(t._data.detach())
                             for t in inputs]
                    res_q.put(('backward_done', rank, txn_id, grads))
        except Exception:
            import traceback
            res_q.put((cmd + '_err', rank, msg[1] if len(msg) > 1 else None,
                       traceback.format_exc()))


class _PersistentWorkerPool:
    def __init__(self, devices, n_per_device, config_desc):
        ctx = _mp.get_context('spawn')
        self.devices = list(devices)
        self.n_per_device = n_per_device
        self.config_desc = config_desc
        self.cmd_qs = []
        self.res_q = ctx.Queue()
        self.procs = []
        self._next_txn = 0
        rank = 0
        self.worker_devs = []
        for dev in self.devices:
            for _slot in range(n_per_device):
                cmd_q = ctx.Queue()
                p = ctx.Process(
                    target=_worker_main,
                    args=(rank, dev, config_desc, cmd_q, self.res_q),
                    daemon=False,
                )
                p.start()
                self.cmd_qs.append(cmd_q)
                self.procs.append(p)
                self.worker_devs.append(dev)
                rank += 1
        self.n_workers = rank
        self._struct_cache = {}

    def allocate_txn(self):
        self._next_txn += 1
        return self._next_txn

    def shutdown(self):
        for q in self.cmd_qs:
            try:
                q.put(('shutdown',))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


def _get_or_create_pool(devices, n_per_device, config):
    desc = _config_descriptor(config)
    key = (tuple(str(d) for d in devices), int(n_per_device),
           tuple(sorted(desc.items())))
    if key not in _pool_registry:
        _pool_registry[key] = _PersistentWorkerPool(devices, n_per_device, desc)
    return _pool_registry[key]


@atexit.register
def _shutdown_all_pools():
    for pool in list(_pool_registry.values()):
        pool.shutdown()
    _pool_registry.clear()


class _MultiprocSlicedUnrollFunction(torch.autograd.Function):
    """Custom autograd.Function dispatching unrolled combos to a worker pool.

    Inputs: (*input_data_tensors, meta_bundle).
    Forward: workers compute per-key partials (no_grad); parent merges per key,
    optionally calls yastn_block, returns assembled out_data.
    Backward: parent re-runs yastn_block on saved merged data with autograd
    enabled, extracts per-key grads via local backward, ships per-key grads to
    workers; workers re-run their combos with autograd, call per-key partial
    backward, return per-input grads. Parent sums input grads across workers.
    """

    @staticmethod
    def forward(ctx, *all_args):
        meta = all_args[-1]
        input_data_tensors = all_args[:-1]
        n_inputs = len(input_data_tensors)

        pool = meta['pool']
        worker_assignments = meta['worker_assignments']
        ig_list = meta['ig_list']
        out_ig = meta['out_ig']
        unroll = meta['unroll']
        optimize = meta['optimize']
        swap = meta['swap']
        ncon_kwargs = meta['ncon_kwargs']
        input_meta_list = meta['input_meta_list']
        per_key_struct = meta['per_key_struct']
        common_legs_axes = meta['common_legs_axes']
        parent_config = meta['parent_config']

        txn_id = pool.allocate_txn()
        ctx.txn_id = txn_id

        original_device = meta['original_device']
        data_per_dev = _per_device_input_replicas(
            input_data_tensors, pool.worker_devs, original_device)

        pf_trim_per_combo = meta.get('pf_trim_per_combo')
        dim_overrides_per_combo = meta.get('dim_overrides_per_combo')

        # Dispatch forward to workers
        for w_idx in range(pool.n_workers):
            assigned = worker_assignments[w_idx]
            if not assigned:
                continue
            dev = str(pool.worker_devs[w_idx])
            serialized_inputs = [
                {**m, 'data': data} for data, m in zip(data_per_dev[dev], input_meta_list)
            ]
            worker_kwargs = _patch_worker_kwargs(
                ncon_kwargs, assigned, pf_trim_per_combo, dim_overrides_per_combo)
            pool.cmd_qs[w_idx].put((
                'forward', txn_id, serialized_inputs, ig_list, out_ig,
                unroll, optimize, swap, worker_kwargs, assigned, None,
                per_key_struct, meta['checkpoint_loop'],
            ))

        n_active = sum(1 for a in worker_assignments if a)
        per_worker_partials = []  # list of {key: serialized yastn dict}
        for _ in range(n_active):
            msg = pool.res_q.get()
            if msg[0] != 'forward_done':
                raise RuntimeError(f"worker forward failed: {msg}")
            _, _rank, _txn, partials_dict = msg
            per_worker_partials.append(partials_dict)

        if per_key_struct is None:
            # Cache-miss path: workers returned RAW partials. Merge via yastn '+'
            # which handles charge-sector union across workers, then derive
            # struct from the merged result and populate the cache so future
            # calls hit the zero-fill fast path.
            from . import Tensor
            merged_yastn = {}
            for w_partials in per_worker_partials:
                for key, ydict in w_partials.items():
                    t = Tensor.from_dict(ydict, config=parent_config)
                    if key in merged_yastn:
                        merged_yastn[key] = merged_yastn[key] + t
                    else:
                        merged_yastn[key] = t
            if not merged_yastn:
                from . import YastnError
                raise YastnError("No valid charge sectors found for contraction.")

            common_legs_axes, _ndim = _compute_common_legs_axes(merged_yastn, unroll, out_ig)
            if common_legs_axes is None:
                assembled = merged_yastn[()]
            else:
                from ..initialize import block as yastn_block
                assembled = yastn_block(merged_yastn, common_legs=common_legs_axes)
                assembled = assembled.drop_leg_history()

            per_key_struct = {k: _meta_only(v.to_dict(level=1))
                              for k, v in merged_yastn.items()}
            full_struct = _meta_only(assembled.to_dict(level=1))
            pool._struct_cache[meta['cache_key']] = (per_key_struct, full_struct,
                                                     common_legs_axes,
                                                     meta['surviving'],
                                                     meta['pf_trim_per_combo'],
                                                     meta['dim_overrides_per_combo'])
            # Update meta in place so backward (and the wrap step in
            # _contract_with_sliced_unroll_mp) see the populated struct.
            meta['per_key_struct'] = per_key_struct
            meta['common_legs_axes'] = common_legs_axes
            meta['full_struct'] = full_struct

            merged_keys = sorted(merged_yastn.keys())
            merged_data = {k: merged_yastn[k]._data for k in merged_keys}
            out_data = assembled._data
        else:
            # Cache-hit path: workers zero-filled, all per-key tensors share
            # shape so we can sum raw torch data tensors directly.
            merged_data = {}
            for w_partials in per_worker_partials:
                for key, ydict in w_partials.items():
                    d = ydict['data']
                    if str(d.device) != str(original_device):
                        d = d.to(original_device)
                    if key in merged_data:
                        merged_data[key] = merged_data[key] + d
                    else:
                        merged_data[key] = d
            if not merged_data:
                from . import YastnError
                raise YastnError("No valid charge sectors found for contraction.")
            merged_keys = sorted(merged_data.keys())
            if common_legs_axes is None:
                out_data = merged_data[()]
            else:
                from . import Tensor
                from ..initialize import block as yastn_block
                per_key_tensors = {}
                for k in merged_keys:
                    tdict = {**per_key_struct[k], 'data': merged_data[k]}
                    per_key_tensors[k] = Tensor.from_dict(tdict, config=parent_config)
                assembled = yastn_block(per_key_tensors, common_legs=common_legs_axes)
                assembled = assembled.drop_leg_history()
                out_data = assembled._data

        ctx.save_for_backward(*input_data_tensors,
                              *(merged_data[k] for k in merged_keys))
        ctx.n_inputs = n_inputs
        ctx.merged_keys = merged_keys
        ctx.meta = meta

        return out_data

    @staticmethod
    def backward(ctx, grad_out_data):
        meta = ctx.meta
        pool = meta['pool']
        worker_assignments = meta['worker_assignments']
        ig_list = meta['ig_list']
        out_ig = meta['out_ig']
        unroll = meta['unroll']
        optimize = meta['optimize']
        swap = meta['swap']
        ncon_kwargs = meta['ncon_kwargs']
        input_meta_list = meta['input_meta_list']
        per_key_struct = meta['per_key_struct']
        common_legs_axes = meta['common_legs_axes']
        parent_config = meta['parent_config']
        n_inputs = ctx.n_inputs
        merged_keys = ctx.merged_keys

        saved = ctx.saved_tensors
        input_data_tensors = saved[:n_inputs]
        merged_data_saved = saved[n_inputs:]

        # Extract per-key gradient: rerun the block step with autograd-enabled
        # leaves to split grad_out_data back into per-key chunks.
        if common_legs_axes is None:
            # Single-key: per-key grad is grad_out_data unchanged
            grad_per_key = {(): grad_out_data.detach()}
        else:
            from . import Tensor
            from ..initialize import block as yastn_block
            with torch.enable_grad():
                leaves = {k: merged_data_saved[i].detach().clone().requires_grad_(True)
                          for i, k in enumerate(merged_keys)}
                per_key_tensors = {}
                for k in merged_keys:
                    tdict = {**per_key_struct[k], 'data': leaves[k]}
                    per_key_tensors[k] = Tensor.from_dict(tdict, config=parent_config)
                assembled = yastn_block(per_key_tensors, common_legs=common_legs_axes)
                assembled = assembled.drop_leg_history()
                assembled._data.backward(grad_out_data.detach())
            grad_per_key = {k: leaves[k].grad.detach() for k in merged_keys
                            if leaves[k].grad is not None}

        # Dispatch backward to workers with per-key grads.
        # Note: do NOT pre-replicate inputs per device here. Workers must
        # ``.detach().clone()`` the data anyway (for autograd-enabled re-run),
        # so each worker creates its own clone regardless. Holding a parent-
        # side per-device pre-replica alive throughout backward (PyTorch IPC
        # requires the producer to outlive consumers) only adds one extra
        # input-sized buffer per remote device — which can OOM on tight
        # remote GPUs while not buying the IPC dedup that helps in forward.
        txn_id = pool.allocate_txn()
        pf_trim_per_combo = meta.get('pf_trim_per_combo')
        dim_overrides_per_combo = meta.get('dim_overrides_per_combo')
        for w_idx in range(pool.n_workers):
            assigned = worker_assignments[w_idx]
            if not assigned:
                continue
            serialized_inputs = []
            for data, m in zip(input_data_tensors, input_meta_list):
                d = {**m, 'data': data.detach()}
                serialized_inputs.append(d)
            worker_kwargs = _patch_worker_kwargs(
                ncon_kwargs, assigned, pf_trim_per_combo, dim_overrides_per_combo)
            pool.cmd_qs[w_idx].put((
                'backward', txn_id, serialized_inputs, ig_list, out_ig,
                unroll, optimize, swap, worker_kwargs, assigned, grad_per_key,
                per_key_struct, meta['checkpoint_loop'],
            ))

        n_active = sum(1 for a in worker_assignments if a)
        per_worker_input_grads = []
        for _ in range(n_active):
            msg = pool.res_q.get()
            if msg[0] != 'backward_done':
                raise RuntimeError(f"worker backward failed: {msg}")
            _, _rank, _txn, grads = msg
            per_worker_input_grads.append(grads)

        # Sum per-input grads across workers. Workers may live on different
        # GPUs, so gather each contribution to the original (input) device
        # before adding.
        original_device = meta['original_device']
        summed = []
        for k in range(n_inputs):
            acc = per_worker_input_grads[0][k]
            if str(acc.device) != str(original_device):
                acc = acc.to(original_device)
            for w_grads in per_worker_input_grads[1:]:
                g = w_grads[k]
                if str(g.device) != str(original_device):
                    g = g.to(original_device)
                acc = acc + g
            summed.append(acc)

        return (*summed, None)


def _contract_with_sliced_unroll_mp(*args, unroll, optimize, checkpoint_loop=False,
                                    swap=None, devices=None, mp_workers_per_device,
                                    **kwargs):
    """Multiprocess dispatcher for _contract_with_sliced_unroll. Supports both
    single-key (no output unroll) and multi-key (output-unrolled) cases."""
    import os
    print(f"[mp_unroll pid={os.getpid()}] entered with "
          f"mp_workers_per_device={mp_workers_per_device} "
          f"devices={devices} n_unroll_labels={len(unroll)}", flush=True)
    tensors = args[0:2 * (len(args) // 2):2]
    ig_list = list(args[1:2 * (len(args) // 2):2])
    out_ig = args[-1]

    parent_config = tensors[0].config
    original_device = str(tensors[0].device)
    if devices is None:
        devices = [original_device]
    else:
        if not isinstance(devices, (list, tuple)):
            devices = [devices]
        devices = list(dict.fromkeys(str(d) for d in devices))

    pool = _get_or_create_pool(devices, mp_workers_per_device, parent_config)

    per_combo_path = bool(kwargs.get("per_combo_path", False))
    # Cache: (per_key_struct, full_struct, common_legs_axes, surviving_combos,
    #         pf_trim_per_combo, dim_overrides_per_combo)
    cache_key = _build_cache_key([t.to_dict(level=1) for t in tensors],
                                 unroll, ig_list, out_ig, optimize, swap, per_combo_path)
    cached = pool._struct_cache.get(cache_key)
    if cached is not None:
        # Cache holds prefilter results too, so workers always get the
        # precomputed payload without re-running _metadata_filter_combos.
        (per_key_struct, full_struct, common_legs_axes,
         surviving, pf_trim_per_combo, dim_overrides_per_combo) = cached
    else:
        per_key_struct = None
        full_struct = None
        common_legs_axes = None
        from .oe_blocksparse import _metadata_filter_combos
        surviving, pf_trim_per_combo, dim_overrides_per_combo = _metadata_filter_combos(
            tensors, ig_list, out_ig, unroll, optimize, swap,
            collect_dim_overrides=per_combo_path)

    # Distribute SURVIVING combo indices round-robin across workers
    worker_assignments = [[] for _ in range(pool.n_workers)]
    for i, combo_idx in enumerate(surviving):
        worker_assignments[i % pool.n_workers].append(combo_idx)

    input_data_tensors = []
    input_meta_list = []
    for t in tensors:
        d = t.to_dict(level=1)
        input_data_tensors.append(d.pop('data'))
        input_meta_list.append(d)

    meta = {
        'pool': pool,
        'worker_assignments': worker_assignments,
        'ig_list': ig_list,
        'out_ig': out_ig,
        'unroll': unroll,
        'optimize': optimize,
        'swap': swap,
        'ncon_kwargs': kwargs,
        'input_meta_list': input_meta_list,
        'per_key_struct': per_key_struct,
        'common_legs_axes': common_legs_axes,
        'full_struct': full_struct,
        'parent_config': parent_config,
        'original_device': original_device,
        'cache_key': cache_key,
        'surviving': surviving,
        'checkpoint_loop': checkpoint_loop,
        'pf_trim_per_combo': pf_trim_per_combo,
        'dim_overrides_per_combo': dim_overrides_per_combo,
    }

    out_data = _MultiprocSlicedUnrollFunction.apply(*input_data_tensors, meta)
    # Function.forward populates meta['full_struct'] on cache miss.
    full_struct = meta['full_struct']
    from . import Tensor
    return Tensor.from_dict({**full_struct, 'data': out_data}, config=parent_config)
