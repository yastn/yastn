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
import logging
import queue as _queue
from collections import OrderedDict
import torch
import torch.multiprocessing as _mp


log = logging.getLogger(__name__)

# Module-level pool registry: (devices_tuple, mp_workers_per_device,
# config_descriptor_hash) -> _PersistentWorkerPool
_pool_registry = {}

# Bound the per-pool assembly-recipe cache so long, shape-evolving runs don't
# accumulate one entry per distinct structure without limit (LRU eviction).
_STRUCT_CACHE_MAXSIZE = 512

# How often a result-drain waits before checking whether an assigned worker has
# died without posting (turns an infinite res_q.get() hang into an error).
_WORKER_POLL_SECONDS = 30.0


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
        'profile': config.profile,
    }


def _serialize_yastn(t):
    """yastn.Tensor -> dict for IPC. Data is detached; metadata is picklable."""
    d = t.to_dict(level=1)
    d['data'] = d['data'].detach()
    return d


def _deserialize_yastn(d, config):
    from . import Tensor
    return Tensor.from_dict(d, config=config)


def _struct_identity(t):
    """Hashable, allocation-free structural fingerprint of a tensor under the
    leg-first model: its ``struct`` (legs / n / isdiag), any pending transpose,
    and its meta/hard fusion histories. These are exactly the fields ``to_dict``
    compares for structural equality, and each is already a hashable
    NamedTuple/tuple in memory (``get_blocks`` and the ``_meta_*`` routines
    lru_cache on ``struct`` / ``hfs``), so no ``to_dict``/serialization
    round-trip is needed to key on structure.

    Lossless: two tensors share this identity iff they are structurally
    identical, so a dict/lru keyed on it cannot false-hit — unlike the earlier
    key, which projected the serialized struct down and could collide."""
    return (t.struct, t.trans, t.hfs, t.mfs)


def _build_cache_key(tensors, unroll, ig_list, out_ig, optimize, swap, per_combo_path):
    """Stable, cheap key for the assembly-info cache, built from in-memory
    hashable metadata (no ``to_dict``/``_freeze`` round-trip). ``per_combo_path``
    participates because cached entries carry ``dim_overrides_per_combo`` only
    when it was True at insertion time; reusing a False-time entry for a
    True-time call would leave workers with a missing payload.

    Each input contributes its full leg-first structural fingerprint, so any
    difference in leg structure, pending transpose or fusion history misses the
    cache. Config/sym are fixed per pool, so they need not enter the key."""
    input_keys = tuple(_struct_identity(t) for t in tensors)
    unroll_key = tuple(sorted(
        (str(k), tuple((sl.t, sl.D) for sl in sls)) for k, sls in unroll.items()
    ))
    return (input_keys, unroll_key,
            tuple(tuple(ig) for ig in ig_list), tuple(out_ig),
            str(optimize), str(swap), bool(per_combo_path))


def _meta_only(yastn_dict):
    return {k: v for k, v in yastn_dict.items() if k != 'data'}


def _zeros_meta(config, legs, n):
    """Serialized ``to_dict(level=1)`` metadata (minus ``data``) for
    ``zeros(legs, n)`` WITHOUT allocating the full block-data buffer.

    ``zeros`` -> ``_fill_tensor`` computes the struct via ``get_blocks`` (pure
    metadata) and only then allocates ``_init_block(bl.size)``. We stop before
    that allocation: the ``Tensor`` shell already carries a 1-element placeholder
    buffer, so a large output costs no full-size zero buffer per cache miss."""
    from . import Tensor
    from ._auxiliary import _unpack_legs, get_blocks
    ulegs, mfs = _unpack_legs(legs)
    s = tuple(lg.s for lg in ulegs)
    hfs = tuple(lg.hf for lg in ulegs)
    a = Tensor(config=config, s=s, n=n, isdiag=False, mfs=mfs, hfs=hfs)
    legs_tD = tuple(lb._replace(t=lg.t, D=lg.D) for lb, lg in zip(a.struct.legs, ulegs))
    bl = get_blocks(config.sym, a.struct._replace(legs=legs_tD))
    a.struct = bl.struct
    d = _meta_only(a.to_dict(level=1))
    d['size'] = bl.size  # to_dict read size from the 1-element placeholder buffer
    return d


def _derive_output_structs(tensors, ig_list, out_ig, unroll, surviving, config):
    """Derive ``(per_key_struct, full_struct, common_legs_axes)`` from input
    legs alone — NO block contraction on data.

    In a tensor-network contraction the output legs are exactly the free
    (uncontracted) input legs gathered in ``out_ig`` order; an output-unrolled
    leg is additionally restricted to its per-key slice.

    Single-key (no output-unrolled leg): the output struct is the full gathered
    legs, built metadata-only via :func:`_zeros_meta` (no data buffer).

    Multi-key (output-unrolled): build a zero skeleton per *surviving* output
    key and assemble them with the *same* ``block`` + ``drop_leg_history`` the
    forward pass uses, so ``full_struct`` matches the assembled ``out_data``
    exactly — including when an output slice has no surviving combo (absent from
    both). Abelian charge conservation gives the output charge as the fused
    input charge.

    Skeletons are in canonical form (identity pending-transpose). ``ncon`` may
    return a value-identical result carrying a *lazy* transpose, so a worker
    partial's native ``struct``/``trans`` can differ while ``get_legs`` and the
    value agree — a representation choice, not a correctness difference; yastn
    arithmetic aligns them when the worker zero-fills to this struct. (No
    per-operand conjugation is assumed, which the dispatcher does not pass.)"""
    from ..initialize import zeros as yastn_zeros
    from ._legs import Leg
    sym = config.sym
    out_ig_list = list(out_ig)

    # Each output label is a free leg carried by exactly one (tensor, axis).
    label_src = {}
    for k, ig in enumerate(ig_list):
        for a, lbl in enumerate(ig):
            if lbl in out_ig_list:
                label_src[lbl] = (k, a)
    out_legs_full = [tensors[k].get_legs(a) for k, a in (label_src[L] for L in out_ig_list)]

    n_out = sym.add_charges(*(t.struct.n for t in tensors))

    unroll_labels = list(unroll.keys())
    sizes = [len(unroll[u]) for u in unroll_labels]
    label_pos = {u: i for i, u in enumerate(unroll_labels)}
    output_unroll_axes = {out_ig_list.index(u): u for u in unroll_labels if u in out_ig_list}
    blocked_axes = sorted(output_unroll_axes.keys())
    ndim_out = len(out_ig_list)
    common_legs_axes = ([ax for ax in range(ndim_out) if ax not in blocked_axes]
                        if blocked_axes else None)

    if common_legs_axes is None:
        # Single-key: metadata-only, no full zero buffer (the large-output case).
        d = _zeros_meta(config, out_legs_full, n_out)
        return {(): d}, d, None

    def _unravel(n):
        # combo index -> per-unroll-label slice index, in itertools.product order
        # (last label varies fastest); avoids materializing the full product.
        idx = [0] * len(sizes)
        for i in range(len(sizes) - 1, -1, -1):
            idx[i] = n % sizes[i]
            n //= sizes[i]
        return idx

    # Multi-key: a zero skeleton per surviving output key, assembled via the
    # SAME block()+drop_leg_history the forward pass uses so full_struct matches
    # out_data exactly.
    per_key_tensors = {}
    for n in surviving:
        idx = _unravel(n)
        opk = tuple(idx[label_pos[output_unroll_axes[ax]]] for ax in blocked_axes)
        if opk in per_key_tensors:
            continue
        legs = list(out_legs_full)
        for i, ax in enumerate(blocked_axes):
            sl = unroll[output_unroll_axes[ax]][opk[i]]
            legs[ax] = Leg(sym=config, s=out_legs_full[ax].s, t=sl.t, D=sl.D)
        per_key_tensors[opk] = yastn_zeros(config=config, legs=legs, n=n_out)

    per_key_struct = {opk: _meta_only(t.to_dict(level=1)) for opk, t in per_key_tensors.items()}
    from ..initialize import block as yastn_block
    assembled = yastn_block(per_key_tensors, common_legs=common_legs_axes).drop_leg_history()
    full_struct = _meta_only(assembled.to_dict(level=1))
    return per_key_struct, full_struct, common_legs_axes


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
    # Leg-first: to_dict(level=1) serializes the struct as {'legs','n','isdiag'}
    # (no 'size' — it is derived on demand via get_blocks) and emits the total
    # data size at the top level as full_struct_dict['size'] (== a.size). The
    # old struct['size']/struct.size lookups both went stale under leg-first.
    full_size = full_struct_dict['size']
    zeros_data = torch.zeros(full_size,
                             dtype=partial._data.dtype,
                             device=partial._data.device)
    zero_tensor = Tensor.from_dict({**full_struct_dict, 'data': zeros_data},
                                   config=cfg)
    return zero_tensor + partial


def _install_parent_death_signal():
    """Linux: ask the kernel to SIGTERM this worker if the parent dies, so a
    parent killed by signal (no atexit -> _shutdown_all_pools never runs) does
    not leak non-daemon workers holding CUDA contexts. Best-effort / no-op
    elsewhere."""
    try:
        import os, signal, ctypes
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        # Race: if the parent already died before prctl, we were reparented and
        # will never get the signal — exit now.
        if os.getppid() == 1:
            os._exit(1)
    except Exception:
        pass


def _worker_main(rank, gpu_dev, config_desc, cmd_q, res_q,
                 log_queue=None, log_level=logging.INFO, logger_levels=None):
    """Worker process entry point. Loops on cmd_q until 'shutdown'."""
    import torch
    _install_parent_death_signal()
    # Route this worker's logging (notably get_contraction_path path-info
    # reports at INFO level) through the parent's QueueListener. Spawn-mode
    # children inherit no logging config, so without this every log.info call
    # in a worker is silently dropped.
    from .._mp_logging import install_worker_log_handler
    install_worker_log_handler(log_queue, log_level,
                               tag=f"oe_mp rank {rank} dev {gpu_dev}",
                               logger_levels=logger_levels)
    if str(gpu_dev).startswith('cuda'):
        gpu_idx = int(str(gpu_dev).split(':')[1])
        torch.cuda.set_device(gpu_idx)
    from ._initialize import make_config
    from .oe_blocksparse import _contract_with_sliced_unroll
    # Override default_device to this worker's assigned GPU so yastn tensors
    # reconstructed from IPC are placed on the right device (Tensor.from_dict
    # honors config.default_device).
    cfg = make_config(**{**config_desc, 'default_device': str(gpu_dev)})
    log.info("worker ready")

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

                    # Zero-fill each partial to its per-key output struct so all
                    # workers' per-key tensors share identical shape and the
                    # parent sums raw data tensors directly (no yastn '+').
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
        from .._mp_logging import (
            start_parent_log_listener, parent_log_level, snapshot_logger_levels)
        ctx = _mp.get_context('spawn')
        self.devices = list(devices)
        self.n_per_device = n_per_device
        self.config_desc = config_desc
        self.cmd_qs = []
        self.res_q = ctx.Queue()
        self.procs = []
        self._next_txn = 0
        # Multiprocess-safe logging: parent owns the only real handler via a
        # QueueListener; workers push records onto self.log_queue. When this
        # pool is created inside another pool's worker, that worker's own
        # QueueHandler is reused as the listener target, forwarding records up
        # the chain. See yastn._mp_logging.
        self.log_queue, self.log_listener = start_parent_log_listener(ctx)
        log_level = parent_log_level()
        logger_levels = snapshot_logger_levels()
        rank = 0
        self.worker_devs = []
        for dev in self.devices:
            for _slot in range(n_per_device):
                cmd_q = ctx.Queue()
                p = ctx.Process(
                    target=_worker_main,
                    args=(rank, dev, config_desc, cmd_q, self.res_q,
                          self.log_queue, log_level, logger_levels),
                    daemon=False,
                )
                p.start()
                self.cmd_qs.append(cmd_q)
                self.procs.append(p)
                self.worker_devs.append(dev)
                rank += 1
        self.n_workers = rank
        self._struct_cache = OrderedDict()

    def cache_get(self, key):
        """LRU read of the assembly-recipe cache."""
        try:
            self._struct_cache.move_to_end(key)
            return self._struct_cache[key]
        except KeyError:
            return None

    def cache_put(self, key, value):
        """LRU write; evict the oldest entry past the size bound."""
        self._struct_cache[key] = value
        self._struct_cache.move_to_end(key)
        while len(self._struct_cache) > _STRUCT_CACHE_MAXSIZE:
            self._struct_cache.popitem(last=False)

    def get_result(self, active_worker_idxs):
        """Blocking get from ``res_q`` with a liveness guard: if no message
        arrives within ``_WORKER_POLL_SECONDS`` and an assigned worker has died
        (CUDA OOM kill, segfault) without posting, raise instead of hanging
        forever. Legitimately long contractions just keep waiting."""
        while True:
            try:
                return self.res_q.get(timeout=_WORKER_POLL_SECONDS)
            except _queue.Empty:
                dead = [(i, self.procs[i].exitcode) for i in active_worker_idxs
                        if not self.procs[i].is_alive()]
                if dead:
                    raise RuntimeError(
                        f"MP worker(s) died before posting results "
                        f"(rank, exitcode): {dead} — likely CUDA OOM kill or segfault")

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
        # Stop the listener only after workers are joined, so any record they
        # emitted has been enqueued before the listener drains and stops.
        from .._mp_logging import stop_parent_log_listener
        stop_parent_log_listener(self.log_listener)


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

    Inputs: (*input_data_tensors, meta_bundle). The per-key output structs are
    derived from input legs upfront (see _derive_output_structs), so:
    Forward: workers compute per-key partials (no_grad) and zero-fill each to its
    per-key struct; parent sums raw data per key, optionally calls yastn_block,
    returns assembled out_data.
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
        active_idxs = [i for i, a in enumerate(worker_assignments) if a]
        per_worker_partials = []  # list of {key: serialized yastn dict}
        while len(per_worker_partials) < n_active:
            msg = pool.get_result(active_idxs)
            kind, _rank, _txn, payload = msg
            if _txn != txn_id:
                # leftover from an earlier transaction that raised mid-drain
                log.warning("discarding stale %s message from txn %s", kind, _txn)
                continue
            if kind != 'forward_done':
                raise RuntimeError(f"worker forward failed: {msg}")
            per_worker_partials.append(payload)

        # Workers zero-filled each partial to its per-key output struct, so all
        # per-key tensors share shape and we sum raw torch data tensors directly
        # (gathering each worker's contribution back to the original device).
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
        active_idxs = [i for i, a in enumerate(worker_assignments) if a]
        per_worker_input_grads = []
        while len(per_worker_input_grads) < n_active:
            msg = pool.get_result(active_idxs)
            kind, _rank, _txn, payload = msg
            if _txn != txn_id:
                # leftover from an earlier transaction that raised mid-drain
                log.warning("discarding stale %s message from txn %s", kind, _txn)
                continue
            if kind != 'backward_done':
                raise RuntimeError(f"worker backward failed: {msg}")
            per_worker_input_grads.append(payload)

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
    log.info("mp_unroll dispatch: mp_workers_per_device=%s devices=%s "
             "n_unroll_labels=%d", mp_workers_per_device, devices, len(unroll))
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
    cache_key = _build_cache_key(tensors,
                                 unroll, ig_list, out_ig, optimize, swap, per_combo_path)
    cached = pool.cache_get(cache_key)
    if cached is not None:
        # Cache holds prefilter results too, so workers always get the
        # precomputed payload without re-running _metadata_filter_combos.
        (per_key_struct, full_struct, common_legs_axes,
         surviving, pf_trim_per_combo, dim_overrides_per_combo) = cached
    else:
        from .oe_blocksparse import _metadata_filter_combos
        surviving, pf_trim_per_combo, dim_overrides_per_combo = _metadata_filter_combos(
            tensors, ig_list, out_ig, unroll, optimize, swap,
            collect_dim_overrides=per_combo_path)
        # Derive the assembly structs from input legs (no data contraction);
        # workers always zero-fill and the cache is a pure performance memo.
        per_key_struct, full_struct, common_legs_axes = _derive_output_structs(
            tensors, ig_list, out_ig, unroll, surviving, parent_config)
        pool.cache_put(cache_key, (per_key_struct, full_struct, common_legs_axes,
                                   surviving, pf_trim_per_combo, dim_overrides_per_combo))

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
        'checkpoint_loop': checkpoint_loop,
        'pf_trim_per_combo': pf_trim_per_combo,
        'dim_overrides_per_combo': dim_overrides_per_combo,
    }

    out_data = _MultiprocSlicedUnrollFunction.apply(*input_data_tensors, meta)
    # full_struct is derived from input legs in the dispatcher (above).
    full_struct = meta['full_struct']
    from . import Tensor
    return Tensor.from_dict({**full_struct, 'data': out_data}, config=parent_config)
