# Copyright 2025 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================
"""[PROTOTYPE] AD-aware distributed CTM step.

``_env_ctm_dist_mp.py`` parallelizes the *forward* CTM step but detaches
tensors at the worker boundary -- backward through it is broken. This
module is a from-scratch reimplementation that mirrors the same three-
stage decomposition (halves -> projectors -> env update) but **preserves
the autograd graph in the main process** by wrapping each per-site per-
stage computation in a custom ``torch.autograd.Function``.

Pattern (same for all three stages)
-----------------------------------
1. ``apply(...)`` builds a serialized payload, dispatches a ``stageN_fwd``
   task to a worker, and synchronously gets back output data tensors +
   metas.
2. ``apply(...)`` then constructs an ad-hoc ``torch.autograd.Function``
   whose ``forward`` returns the precomputed output tensors (after
   ``save_for_backward`` of the inputs) and whose ``backward``
   dispatches a ``stageN_bwd`` task to the *same* worker. The worker
   re-runs the per-site computation under ``torch.enable_grad()`` and
   returns input gradients via ``torch.autograd.grad``.
3. Cross-site projector flow (Stage 2 outputs feed Stage 3 inputs at
   neighboring sites) is handled by main-process autograd: the same
   yastn Tensor object is consumed by multiple Stage 3 calls, so
   gradients accumulate naturally on its ``_data`` and route back to
   the producing Stage 2 call.

Worker stays alive across CTM calls (and across the Neumann loop in
``FixedPoint.backward``), so spawn cost is paid once per optimization.

Scope
-----
- Stages 1, 2, 3 implemented (workers + main-side Functions).
- ``update_core_AD_`` orchestrates one CTM move via the three stages.
- ``update_AD_`` runs a multi-move sweep (one CTM step in
  ``fixed_point_iter`` is one sweep with the standard 'hv' moves).
- ``fp_update_`` is the drop-in for ``env_in.update_(...)`` inside
  ``fixed_point_iter`` -- opt-in via ``ctm_opts_fp['fp_devices']``.
- DoublePepsTensor handling: serialized as bra+ket pair (op currently
  unsupported -- raises if encountered).
"""
from __future__ import annotations

import atexit
import logging
import queue
import threading
import time


log = logging.getLogger(__name__)

_pool = None  # singleton, keyed on (tuple(devices), tuple(sorted(config_desc)))


# ===========================================================================
# Pool
# ===========================================================================

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


class _DistCTMPool:
    """Persistent spawn-mode worker pool, one worker per device."""

    # res_q poll interval (seconds): how often wait() wakes up to probe
    # worker liveness. Short enough that a dead worker is detected within
    # a few seconds, long enough that a healthy wait pays negligible CPU.
    _POLL_INTERVAL = 5.0

    def __init__(self, devices, config_desc):
        import sys
        import torch.multiprocessing as _mp

        mctx = _mp.get_context('spawn')
        self.devices = list(devices)
        self.cmd_qs = []
        self.res_q = mctx.Queue()
        self.procs = []
        self._next_txn = 0
        self._results = {}
        self._lock = threading.Lock()

        parent_sys_path = list(sys.path)
        for rank, dev in enumerate(self.devices):
            cmd_q = mctx.Queue()
            p = mctx.Process(
                target=_worker_main,
                args=(rank, dev, config_desc, cmd_q, self.res_q,
                      parent_sys_path),
                daemon=False,
            )
            p.start()
            self.cmd_qs.append(cmd_q)
            self.procs.append(p)
        self.n_workers = len(self.devices)

    def submit(self, rank, payload):
        with self._lock:
            self._next_txn += 1
            txn = self._next_txn
        self.cmd_qs[rank].put((txn, *payload))
        return txn

    def drop_graphs(self, rank, graph_ids):
        """Fire-and-forget: tell a worker to release its saved_graphs
        entries. Worker does NOT respond on res_q (would otherwise leak
        stranded entries we never wait on). Safe at interpreter shutdown
        -- swallow any exception from a closed queue.
        """
        try:
            self.cmd_qs[rank].put((0, 'drop_graphs', list(graph_ids)))
        except Exception:
            pass

    def wait(self, txn, total_timeout=None):
        """Block until worker returns ``txn``.

        Polls ``res_q`` with a finite timeout so we can probe worker
        liveness in between -- a dead worker won't ever produce a
        result, so without this we'd hang forever. Out-of-order results
        are stashed for later wait() calls. Optional ``total_timeout``
        bounds wall time per call (None = wait until worker dies or
        result arrives).
        """
        with self._lock:
            if txn in self._results:
                return self._results.pop(txn)
        deadline = (None if total_timeout is None
                    else time.monotonic() + total_timeout)
        while True:
            try:
                msg = self.res_q.get(timeout=self._POLL_INTERVAL)
            except queue.Empty:
                dead = [r for r, p in enumerate(self.procs)
                        if not p.is_alive()]
                if dead:
                    exitcodes = [self.procs[r].exitcode for r in dead]
                    raise RuntimeError(
                        f"dist_ctm_AD worker(s) died: ranks={dead} "
                        f"exitcodes={exitcodes}; check stderr for trace")
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(
                        f"dist_ctm_AD wait txn={txn} timed out after "
                        f"{total_timeout}s")
                continue
            r_txn = msg[0]
            if r_txn == txn:
                return msg[1:]
            with self._lock:
                self._results[r_txn] = msg[1:]

    def shutdown(self):
        for q in self.cmd_qs:
            try:
                q.put((0, 'shutdown'))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


def get_or_create_pool(devices, config):
    """Return the (cached) singleton pool. Re-create when devices OR
    config (dtype/sym/backend/etc.) differs from the cached pool --
    workers were spawned with a snapshot of config and silently
    serving the wrong dtype is far worse than the spawn cost.
    """
    global _pool
    desc = _config_descriptor(config)
    key = (tuple(str(d) for d in devices),
           tuple(sorted(desc.items())))
    if _pool is not None and getattr(_pool, '_key', None) != key:
        log.warning(
            "dist_ctm_AD pool: devices or config changed, re-creating workers")
        _pool.shutdown()
        _pool = None
    if _pool is None:
        _pool = _DistCTMPool(list(devices), desc)
        _pool._key = key
    return _pool


@atexit.register
def _shutdown_pool_on_exit():
    global _pool
    if _pool is not None:
        _pool.shutdown()
        _pool = None


# ===========================================================================
# Saved-graph cleanup
# ===========================================================================

class _SavedGraphHandle:
    """Cleanup anchor that releases workers' saved activations when the
    main-side autograd graph is freed.

    Workers retain (input_leaves, outputs) under graph_ids for the life
    of the matching main-side autograd graph (so backward can run
    autograd.grad against the saved graph without re-forwarding). Once
    the main graph is freed -- typically when the loss tensor and
    intermediate outputs go out of scope at the end of
    ``FixedPoint.backward`` -- the worker's copies are dead weight.

    We attach an instance of this class as ``ctx._cleanup_handle`` on
    the autograd Function ctx. When ctx is garbage-collected (= the
    backward node is freed), this object's ``__del__`` fires and sends
    a fire-and-forget ``drop_graphs`` cmd to each rank that holds
    saved activations for this batch. Without this, ``saved_graphs``
    grows unboundedly across an optimization and OOMs the workers.
    """

    def __init__(self, pool, rank_to_graph_ids):
        self._pool = pool
        # Copy to dict-of-list-of-int so we don't keep references to
        # any other transient data alive via this handle.
        self._rank_to_gids = {r: list(gids)
                              for r, gids in rank_to_graph_ids.items()
                              if gids}

    def __del__(self):
        # Robust against interpreter shutdown / pool already-destroyed:
        # the worker-side dict is local, and a missing pool / closed
        # queue is handled by ``pool.drop_graphs`` which itself
        # swallows exceptions.
        try:
            pool = self._pool
            if pool is None:
                return
            for rank, gids in self._rank_to_gids.items():
                pool.drop_graphs(rank, gids)
        except Exception:
            pass


# ===========================================================================
# Tensor (de)serialization helpers
# ===========================================================================

_META_CACHE = {}              # key -> level=1 meta dict (no 'data')
_CACHE_LIMIT = 8192


def _shape_key(t):
    """Hashable identifier for a yastn.Tensor's structural meta. Two
    Tensors with the same struct/slices/hfs/mfs share the same level=1
    meta dict (modulo the data field), so they share a cache entry."""
    return (t.struct, t.slices, t.hfs, t.mfs)


def _split_tensor(t):
    """yastn.Tensor -> (data, meta_l1_without_data).

    The level=1 meta is identical for all Tensors with the same struct
    structure, so we cache it per shape key. The dict produced by
    ``to_dict`` is treated as immutable (workers receive a fresh copy
    via pickle, and downstream uses don't mutate). This skips a
    significant amount of dict construction at chi=72.
    """
    key = _shape_key(t)
    cached = _META_CACHE.get(key)
    if cached is not None:
        return t._data, cached
    d = t.to_dict(level=1)
    data = d.pop('data')
    if len(_META_CACHE) < _CACHE_LIMIT:
        _META_CACHE[key] = d
    return data, d


def _split_dpt(dpt):
    """DoublePepsTensor -> (bra_data, bra_meta, ket_data, ket_meta, extra).

    Uses the same per-shape meta cache as ``_split_tensor`` -- in
    typical PEPS, bra and ket are the same Tensor (or share shape), so
    one cache lookup covers both legs.
    """
    if dpt.op is not None:
        raise NotImplementedError(
            "AD-aware distributed CTM does not yet support DoublePepsTensor.op")
    bra_data, bra_meta = _split_tensor(dpt.bra)
    ket_data, ket_meta = _split_tensor(dpt.ket)
    extra = {'transpose': dpt.trans, 'swaps': dict(dpt.swaps)}
    return bra_data, bra_meta, ket_data, ket_meta, extra


def _serialize_args(args):
    """Walk a list of args (tensors / DPTs / Sites / None), build
    a 'skeleton' describing types and positions plus a flat list of
    differentiable data tensors and matching metas.

    Skeleton is a list with one entry per arg position; entries are
    one of:
        ('None',)
        ('Site', site_obj)
        ('Tensor', meta) -- consumes 1 data tensor
        ('DPT', extra, bra_meta, ket_meta) -- consumes 2 (bra, ket)
    """
    from yastn.yastn.tensor import Tensor
    from yastn.yastn.tn.fpeps._doublePepsTensor import DoublePepsTensor
    from yastn.yastn.tn.fpeps import Site

    skeleton = []
    flat_data = []
    for a in args:
        if a is None:
            skeleton.append(('None',))
        elif isinstance(a, Tensor):
            data, meta = _split_tensor(a)
            skeleton.append(('Tensor', meta))
            flat_data.append(data)
        elif isinstance(a, DoublePepsTensor):
            bra_data, bra_meta, ket_data, ket_meta, extra = _split_dpt(a)
            skeleton.append(('DPT', extra, bra_meta, ket_meta))
            flat_data.append(bra_data)
            flat_data.append(ket_data)
        elif isinstance(a, Site):
            skeleton.append(('Site', a))
        else:
            raise TypeError(
                f"_serialize_args: unrecognized arg type {type(a)}: {a!r}")
    return skeleton, flat_data


def _deserialize_args(skeleton, flat_data, cfg):
    """Rebuild args from skeleton + flat data tensors (on worker side)."""
    from yastn.yastn.tensor import Tensor
    from yastn.yastn.tn.fpeps._doublePepsTensor import DoublePepsTensor

    args = []
    di = 0
    for slot in skeleton:
        kind = slot[0]
        if kind == 'None':
            args.append(None)
        elif kind == 'Site':
            args.append(slot[1])
        elif kind == 'Tensor':
            meta = slot[1]
            d = dict(meta)
            d['data'] = flat_data[di]
            di += 1
            args.append(Tensor.from_dict(d, config=cfg))
        elif kind == 'DPT':
            _, extra, bra_meta, ket_meta = slot
            bra_d = dict(bra_meta)
            bra_d['data'] = flat_data[di]
            di += 1
            ket_d = dict(ket_meta)
            ket_d['data'] = flat_data[di]
            di += 1
            bra = Tensor.from_dict(bra_d, config=cfg)
            ket = Tensor.from_dict(ket_d, config=cfg)
            args.append(DoublePepsTensor(bra=bra, ket=ket,
                                          trans=extra['transpose'],
                                          swaps=extra['swaps']))
        else:
            raise RuntimeError(f"unknown skeleton kind: {kind}")
    if di != len(flat_data):
        raise RuntimeError(
            f"skeleton/flat_data mismatch: consumed {di}, have {len(flat_data)}")
    return args


def _send_payload(t):
    """Prepare a tensor for the mp.Queue.

    CUDA: kept as torch.Tensor; mp's reducer ships it via CUDA IPC.
    CPU:  routed through numpy (torch's CPU reducer uses shared memory
          which aliases sender/receiver storage; the caching allocator
          can recycle the sender's tensor mid-flight and corrupt the
          receiver's view).

    ``resolve_conj()`` clears the complex-conjugate flag (some reducer
    paths refuse it).  Producer-stream ordering is handled by
    torch.multiprocessing's CUDA reducer: when the tensor is pickled
    for the queue, the reducer records an event on the producer's
    current stream and ships the IPC event handle alongside the
    storage handle.  The receiver's first access stream-waits on that
    event before reading.  No host-side ``cuda.synchronize`` is
    required.
    """
    import torch
    t = t.detach().resolve_conj()
    if t.is_cuda:
        return t.contiguous()
    return t.cpu().numpy()


def _send_payload_tuple(ts):
    return tuple(_send_payload(t) for t in ts)


def _recv_payload(p, device=None):
    """Receive a payload and return a contiguous, independently-owned tensor.

    For the CUDA-IPC path, ``clone()`` is critical so the receiver owns
    its storage (otherwise the producer's caching allocator could
    recycle the bytes underneath us once the IPC view is dropped).
    The clone is launched on the receiver's current stream and is
    automatically stream-ordered after the producer's IPC event
    (inserted by torch.multiprocessing's CUDA reducer at storage
    rebuild time).  Downstream user kernels on the same stream see
    the clone without any host-side ``cuda.synchronize`` -- the GPU
    enforces the ordering through stream events.

    Cross-device moves use blocking ``.to(device)`` (non_blocking H2D
    to unpinned CPU memory has been observed to return before the
    copy completes in some PyTorch builds).
    """
    import torch
    if isinstance(p, torch.Tensor):
        out = p.contiguous().clone()
    else:
        out = torch.from_numpy(p.copy()).contiguous()
    if device is not None:
        out = out.to(device)
    return out


# ===========================================================================
# Worker entrypoint
# ===========================================================================

def _worker_main(rank, device, config_desc, cmd_q, res_q, parent_sys_path):
    import os, sys, torch

    for p in reversed(parent_sys_path):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    if str(device).startswith('cuda'):
        gpu_idx = int(str(device).split(':')[1])
        torch.cuda.set_device(gpu_idx)

    from yastn.yastn.tensor._initialize import make_config
    cfg = make_config(**{**config_desc, 'default_device': str(device)})

    print(f"[dist_ctm_AD pid={os.getpid()} rank={rank} dev={device}] worker ready",
          flush=True)

    # Holds activations for in-flight forwards until their matching
    # backward runs. graph_id (= forward txn) -> (input_leaves, outputs).
    saved_graphs = {}

    while True:
        try:
            msg = cmd_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        txn, cmd, *args = msg
        if cmd == 'shutdown':
            break
        if cmd == 'drop_graphs':
            # Fire-and-forget cleanup: main never waits on this, so we
            # deliberately do not put on res_q.
            for gid in args[0]:
                saved_graphs.pop(gid, None)
            continue
        try:
            if cmd.endswith('_fwd'):
                stage = cmd[:-len('_fwd')]
                compute_fn = _STAGE_COMPUTE[stage]
                res = _stage_fwd_worker_impl(
                    cfg, device, args, compute_fn, txn, saved_graphs)
                res_q.put((txn, cmd + '_done', *res))
            elif cmd.endswith('_bwd'):
                res = _stage_bwd_via_saved(args, saved_graphs, device)
                res_q.put((txn, cmd + '_done', res))
            else:
                res_q.put((txn, 'err', f'unknown cmd: {cmd}'))
        except Exception:
            import traceback
            res_q.put((txn, 'err', traceback.format_exc()))


# ===========================================================================
# Per-stage worker forward implementations
# ===========================================================================

def _stage12_compute(args, extra):
    """Per-site fused halves -> projectors compute.

    args = the 16 env tensors for one site's 2x2 patch.
    extra = {'move': str, 'pairs': tuple[str, ...],
             'opts_svd': dict, 'kwargs': dict}.

    Returns a flat tuple ``(p1_pair0, p2_pair0, p1_pair1, p2_pair1, ...)``
    with two projectors per pair, in the order given by ``pairs``.

    Fusing halves and projectors into one worker call keeps a single
    autograd graph rooted at the 16 env leaves (halves are intermediate
    saved tensors).  This avoids the previous design's two-graph state
    -- one per stage, both alive in worker memory -- and the IPC
    round-trip that shipped halves between the stages.
    """
    from yastn.yastn.tn.fpeps.envs._env_contractions import (
        halves_4x4_lhr, halves_4x4_tvb,
    )
    from yastn.yastn.tn.fpeps.envs._env_ctm_dist_mp import (
        projectors_move_rh, projectors_move_lh,
        projectors_move_tv, projectors_move_bv,
    )

    move = extra['move']
    if any(x in move for x in 'lrh'):
        h1, h2 = halves_4x4_lhr(args)
    else:
        h1, h2 = halves_4x4_tvb(args)

    PROJ = {
        'rh': projectors_move_rh, 'lh': projectors_move_lh,
        'tv': projectors_move_tv, 'bv': projectors_move_bv,
    }
    opts_svd = extra['opts_svd']
    kwargs = extra.get('kwargs', {})
    outs = []
    for pair in extra['pairs']:
        p1, p2 = PROJ[pair](h1, h2, opts_svd, **kwargs)
        outs.append(p1)
        outs.append(p2)
    return tuple(outs)


def _stage3_compute(args, extra):
    """args = the 12 mixed args from update_env_fetch_args (with
    None/Site preserved); extra = {'move': str}.

    Returns whatever update_env_dir returns (a tuple of yastn Tensors,
    possibly with None entries for boundary cases).
    """
    from yastn.yastn.tn.fpeps.envs._env_contractions import update_env_dir
    return update_env_dir(extra['move'], *args)


_STAGE_COMPUTE = {
    'stage12': _stage12_compute,
    'stage3': _stage3_compute,
}


def _serialize_outputs(outs):
    """Tuple of yastn Tensors / None -> (output_data, output_metas).

    None entries are preserved as (None, None). Each non-None output
    is split into a torch.Tensor data payload (via _send_payload, which
    routes CPU through numpy to avoid shared-memory aliasing) and a
    level=1 meta dict.
    """
    output_data, output_metas = [], []
    for o in outs:
        if o is None:
            output_data.append(None)
            output_metas.append(None)
            continue
        d = o.to_dict(level=1)
        data = d.pop('data')
        output_data.append(_send_payload(data))
        output_metas.append(d)
    return output_data, output_metas


def _stage_fwd_worker_impl(cfg, device, args, compute_fn, graph_id,
                           saved_graphs):
    """Worker forward.

    When ``autograd`` is True, runs ``compute_fn`` under
    ``enable_grad()`` and stashes the live graph keyed by ``graph_id``
    so the matching backward (``_stage_bwd_via_saved``) can call
    ``torch.autograd.grad`` against the saved leaves and outputs --
    no re-forward needed.  The output ``_data.resolve_conj()`` view is
    also saved because main-side cotangents arrive conj-resolved
    (forced by ``_send_payload``); yastn ops sometimes leave the bit
    set, which would cause a sign mismatch in ``autograd.grad``.

    When ``autograd`` is False, runs ``compute_fn`` under
    ``no_grad`` -- inputs are not marked ``requires_grad`` and no
    saved-for-backward graph is built.  Used during CTMRG convergence
    sweeps where main is also in ``torch.no_grad()`` and no backward
    will fire.
    """
    import torch

    autograd, extra, skeleton, flat_data_payload = args

    if autograd:
        flat_data = [_recv_payload(p, device).requires_grad_(True)
                     for p in flat_data_payload]
        yargs = _deserialize_args(skeleton, flat_data, cfg)
        with torch.enable_grad():
            outs = compute_fn(yargs, extra)
            out_data_resolved = [None if o is None else o._data.resolve_conj()
                                 for o in outs]
        saved_graphs[graph_id] = (flat_data, outs, out_data_resolved)
    else:
        flat_data = [_recv_payload(p, device) for p in flat_data_payload]
        yargs = _deserialize_args(skeleton, flat_data, cfg)
        with torch.no_grad():
            outs = compute_fn(yargs, extra)
    return _serialize_outputs(outs)


# ===========================================================================
# Worker backward: autograd.grad against the saved graph
# ===========================================================================

def _stage_bwd_via_saved(args, saved_graphs, device):
    """args = (graph_id, cot_payload).

    Looks up the graph saved by the matching forward dispatch and runs
    ``torch.autograd.grad`` directly -- no re-forward.
    ``retain_graph=True`` so that subsequent backward calls (the
    Neumann loop) on the same graph keep working.
    """
    import torch

    graph_id, cot_payload = args
    if graph_id not in saved_graphs:
        raise RuntimeError(
            f"saved graph for graph_id={graph_id} not found "
            f"(expected from prior forward dispatch)")
    flat_data, outs, out_data_resolved = saved_graphs[graph_id]

    out_data_pairs = []
    cot_data = []
    for out, out_resolved, (cot_p, out_meta) in zip(
            outs, out_data_resolved, cot_payload):
        if out is None or out_meta is None:
            continue
        out_data_pairs.append(out_resolved)
        cot_data.append(_recv_payload(cot_p, device))

    if not out_data_pairs:
        return [_send_payload(torch.zeros_like(t)) for t in flat_data]

    grads = torch.autograd.grad(
        outputs=out_data_pairs,
        inputs=flat_data,
        grad_outputs=cot_data,
        retain_graph=True,
        allow_unused=True,
    )
    grads_out = []
    for g, t in zip(grads, flat_data):
        gt = g if g is not None else torch.zeros_like(t)
        grads_out.append(_send_payload(gt))
    return grads_out


# ===========================================================================
# Main-side: per-site stage applies
# ===========================================================================

def _find_ref_config(args):
    from yastn.yastn.tensor import Tensor
    from yastn.yastn.tn.fpeps._doublePepsTensor import DoublePepsTensor
    for a in args:
        if isinstance(a, Tensor):
            return a.config
        if isinstance(a, DoublePepsTensor):
            return a.bra.config
    raise RuntimeError("no tensor in args to source config from")


def _make_batched_stage_fn(pool, stage):
    """Build a torch.autograd.Function that handles a whole stage's
    batch of per-site dispatches in one pair of (forward, backward).

    Why one Function per batch (not per site): autograd's engine
    traverses Function nodes serially. If each per-site call were its
    own Function, only one worker is busy at a time during backward.
    Bundling all per-site calls of a stage into one Function lets that
    Function's backward submit ALL per-site backward tasks to workers
    concurrently, exactly as the forward already does. This unlocks
    backward parallelism that autograd cannot give us on its own.
    """
    import torch

    class _BatchedFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, batch_meta, output_data, *all_input_data):
            ctx.batch_meta = batch_meta
            ctx.n_inputs = len(all_input_data)
            ctx.out_dev = (all_input_data[0].device if all_input_data
                           else output_data[0].device)

            # Workers retain (input_leaves, outputs) under sm['graph_id']
            # for the life of this ctx; backward sends only cotangents.
            # Attach a cleanup handle so when ctx is GC'd (autograd graph
            # freed), workers drop their saved activations.
            rank_to_gids = {}
            for sm in batch_meta:
                gid = sm.get('graph_id')
                if gid is not None:
                    rank_to_gids.setdefault(sm['rank'], []).append(gid)
            if rank_to_gids:
                ctx._cleanup_handle = _SavedGraphHandle(pool, rank_to_gids)

            return tuple(t.to(ctx.out_dev) for t in output_data)

        @staticmethod
        def backward(ctx, *cotangents):
            n_inputs = ctx.n_inputs
            out_dev = ctx.out_dev
            flat_grads = [None] * n_inputs  # per-input grad accumulator

            # Phase 1: submit all per-site backward tasks concurrently.
            # Send only (graph_id, cotangents) -- workers' saved graphs
            # hold the input leaves.
            txns = []
            for sm in ctx.batch_meta:
                site_cotangents = tuple(cotangents[g] for g in
                                         sm['output_global_indices'])
                cot_payload = [
                    (_send_payload(c), m)
                    for c, m in zip(site_cotangents, sm['output_metas'])
                ]
                payload = (stage + '_bwd', sm['graph_id'], cot_payload)
                txns.append(pool.submit(sm['rank'], payload))

            # Phase 2: collect and accumulate per-input grads.
            for sm, txn in zip(ctx.batch_meta, txns):
                msg = pool.wait(txn)
                if msg[0] != stage + '_bwd_done':
                    raise RuntimeError(f"{stage} bwd failed: {msg}")
                grads_payload = msg[1]
                for local_i, gi in enumerate(range(*sm['input_range'])):
                    g = _recv_payload(grads_payload[local_i], out_dev)
                    if flat_grads[gi] is None:
                        flat_grads[gi] = g
                    else:
                        flat_grads[gi] = flat_grads[gi] + g

            return (None, None) + tuple(flat_grads)

    return _BatchedFn


def _stage_apply_batched(pool, stage, jobs):
    """Dispatch many per-site stage tasks concurrently and finalize.

    ``jobs``: list of (rank, extra, args) tuples. Returns a list of
    yastn-output lists (one per job, preserving order).

    Forward phase: submit all per-site fwd tasks concurrently, wait
    for all results. Backward parallelism is unlocked by bundling all
    per-site calls into one ``_BatchedStageFn`` so that the Function's
    backward can fan out to workers in parallel (autograd alone would
    walk Function nodes serially).

    The grad-enabled state at dispatch time is threaded through to
    workers: when main is in ``torch.no_grad()`` (CTMRG convergence
    sweeps), workers also run under ``no_grad``, skip
    ``requires_grad_(True)`` on inputs, and don't stash a graph -- and
    main skips the ``_BatchedFn.apply`` wrap entirely.
    """
    import torch
    from yastn.yastn.tensor import Tensor

    autograd_enabled = torch.is_grad_enabled()

    # ---- Forward submit phase: dispatch all per-site jobs concurrently.
    submitted = []
    for rank, extra, args in jobs:
        skeleton, flat_data = _serialize_args(args)
        payload = (stage + '_fwd', autograd_enabled, extra, skeleton,
                   _send_payload_tuple(flat_data))
        txn = pool.submit(rank, payload)
        submitted.append((rank, extra, args, skeleton, flat_data, txn))

    # ---- Forward wait phase: collect per-job results into batch arrays
    # and build the index maps the batched Function's backward needs.
    all_input_data = []          # flat list of input tensors across jobs
    all_output_data_live = []    # flat list of output tensors (no Nones)
    batch_meta = []              # one dict per job
    job_output_metas_full = []   # per-job, with None entries for boundaries
    job_live_global_idx = []     # per-job: live-output-position -> global idx (or None)
    job_args = []

    for rank, extra, args, skeleton, flat_data, txn in submitted:
        msg = pool.wait(txn)
        if msg[0] != stage + '_fwd_done':
            raise RuntimeError(f"{stage} fwd failed: {msg}")
        output_payload = msg[1]
        output_metas = msg[2]
        output_data = [None if a is None else _recv_payload(a)
                       for a in output_payload]

        # Register this job's inputs in the global flat list.
        in_start = len(all_input_data)
        all_input_data.extend(flat_data)
        in_end = len(all_input_data)

        # Register live outputs globally; remember positions.
        live_global = []          # one per output position; None for boundary
        live_metas_only = []      # for backward: order of live outputs
        live_global_indices = []  # only the non-None entries from above
        for od, om in zip(output_data, output_metas):
            if om is None:
                live_global.append(None)
            else:
                gidx = len(all_output_data_live)
                all_output_data_live.append(od)
                live_global.append(gidx)
                live_metas_only.append(om)
                live_global_indices.append(gidx)

        batch_meta.append({
            'rank': rank,
            'skeleton': skeleton,
            'bwd_payload_extra': extra,
            'input_range': (in_start, in_end),
            'output_metas': live_metas_only,
            'output_global_indices': live_global_indices,
            # graph_id == forward dispatch's txn. Worker saved its
            # autograd graph under this key; backward looks it up.
            'graph_id': txn,
        })
        job_output_metas_full.append(output_metas)
        job_live_global_idx.append(live_global)
        job_args.append(args)

    # ---- Wrap the whole batch in one autograd Function (or short-
    #      circuit under no_grad, in which case workers built no graphs
    #      and there is nothing for backward to consume).
    n_live = len(all_output_data_live)
    if n_live > 0:
        if autograd_enabled:
            Fn = _make_batched_stage_fn(pool, stage)
            out_tensors_flat = Fn.apply(
                batch_meta, tuple(all_output_data_live), *all_input_data)
            if not isinstance(out_tensors_flat, tuple):
                out_tensors_flat = (out_tensors_flat,)
        else:
            out_dev = (all_input_data[0].device if all_input_data
                       else all_output_data_live[0].device)
            out_tensors_flat = tuple(t.to(out_dev)
                                     for t in all_output_data_live)
    else:
        out_tensors_flat = ()

    # ---- Reconstruct per-job yastn Tensor lists.
    out_lists = []
    for j, output_metas in enumerate(job_output_metas_full):
        out_yastn = []
        live_global = job_live_global_idx[j]
        ref_config = _find_ref_config(job_args[j])
        for i, m in enumerate(output_metas):
            if m is None:
                out_yastn.append(None)
                continue
            data_t = out_tensors_flat[live_global[i]]
            d = dict(m)
            d['data'] = data_t
            out_yastn.append(Tensor.from_dict(d, config=ref_config))
        out_lists.append(out_yastn)

    return out_lists


# ===========================================================================
# Top-level orchestration
# ===========================================================================

def update_core_AD_(env, move, opts_svd, devices, **kwargs):
    """One CTM move, AD-aware distributed.

    Mirrors ``_env_ctm_dist_mp.py::_update_core_D_`` but builds an
    autograd graph in main. Mutates ``env`` in place; the new tensors
    have ``_data`` carrying autograd tracking, so backward through
    ``env``'s output env tensors flows to the original input tensors.
    """
    from yastn.yastn.tn.fpeps.envs._env_contractions import (
        update_env_fetch_args,
    )

    if move not in 'hv':
        raise NotImplementedError(
            f"AD-aware distributed CTM prototype: "
            f"only moves 'h' and 'v' are supported; got '{move}'")

    pool = get_or_create_pool(devices, env.config)

    # Don't mutate caller's dict; default 'tol' if neither tol nor tol_block set.
    opts_svd = dict(opts_svd)
    if 'tol' not in opts_svd and 'tol_block' not in opts_svd:
        opts_svd['tol'] = 1e-14

    # 'h'/'v' moves use one site group containing all sites.
    sites_proj = list(env.sites())

    # Round-robin over workers per site (deterministic given env.sites() order).
    site_to_rank = {}

    def _rank_for(site):
        if site not in site_to_rank:
            site_to_rank[site] = len(site_to_rank) % pool.n_workers
        return site_to_rank[site]

    # Forward all caller kwargs to the projector path -- mirrors the
    # non-AD _env_ctm_dist_mp.py. proj_corners / svd_with_truncation
    # consume what they need (truncation_f, svd_policy, svds_thresh,
    # use_qr, fix_signs, verbosity, profiling_mode, ...) and ignore
    # the rest. Earlier this filtered to {'use_qr'} only and silently
    # dropped svd_policy / truncation_f, changing FP CTM behaviour.
    kw_proj = dict(kwargs)

    # ----- Fused halves + projectors per site.  One worker job holds a
    # single autograd graph from the 16 env leaves through halves to
    # the (p1, p2) projectors of both pairs, eliminating the previous
    # design's two-graph worker state and the halves IPC round-trip.
    # 'h' move -> pairs (rh, lh); 'v' move -> (tv, bv).
    pairs = ('rh', 'lh') if move == 'h' else ('tv', 'bv')
    stage12_jobs = []
    for site in sites_proj:
        tl, tr, bl, br = (env.nn_site(site, d=d)
                          for d in ((0, 0), (0, 1), (1, 0), (1, 1)))
        ts_args = [
            env[tl].l, env[tl].tl, env[tl].t, env.psi[tl],
            env[bl].b, env[bl].bl, env[bl].l, env.psi[bl],
            env[tr].t, env[tr].tr, env[tr].r, env.psi[tr],
            env[br].r, env[br].br, env[br].b, env.psi[br],
        ]
        extra = {'move': move, 'pairs': pairs,
                 'opts_svd': opts_svd, 'kwargs': kw_proj}
        stage12_jobs.append((_rank_for(site), extra, ts_args))
    stage12_results = _stage_apply_batched(pool, 'stage12', stage12_jobs)

    for site, outs in zip(sites_proj, stage12_results):
        tl, tr, bl, br = (env.nn_site(site, d=d)
                          for d in ((0, 0), (0, 1), (1, 0), (1, 1)))
        for k, pair in enumerate(pairs):
            p1, p2 = outs[2 * k], outs[2 * k + 1]
            if pair == 'rh':
                env.proj[tr].hrb, env.proj[br].hrt = p1, p2
            elif pair == 'lh':
                env.proj[tl].hlb, env.proj[bl].hlt = p1, p2
            elif pair == 'tv':
                env.proj[tl].vtr, env.proj[tr].vtl = p1, p2
            elif pair == 'bv':
                env.proj[bl].vbr, env.proj[br].vbl = p1, p2

    env._trivial_projectors_(move, sites_proj)

    # ----- Stage 3: per-site env update. Dispatch runs on a snapshot of
    # env tensors, and writes only happen in the assignment loop after
    # all dispatches have returned -- so reads/writes don't race.
    sub_moves = 'lr' if move == 'h' else 'tb'
    stage3_jobs = []
    stage3_meta = []
    for site in sites_proj:
        for mv in sub_moves:
            args = list(update_env_fetch_args(site, env, mv))
            stage3_jobs.append((_rank_for(site), {'move': mv}, args))
            stage3_meta.append((site, mv))
    stage3_results = _stage_apply_batched(pool, 'stage3', stage3_jobs)
    for (site, mv), outs in zip(stage3_meta, stage3_results):
        _assign_stage3_outs(env, site, mv, outs)

    return env


def _assign_stage3_outs(env, site, mv, outs):
    """Place Stage 3 outputs back into env.

    Mirrors the assignment in _env_ctm_dist_mp.py::_update_core_D_.
    """
    res_main, res_corner_a, res_corner_b = outs + [None] * (3 - len(outs))
    if mv == 'l':
        if res_main is not None:
            env[site].l = res_main
        if res_corner_a is not None:
            env[site].tl = res_corner_a
        if res_corner_b is not None:
            env[site].bl = res_corner_b
    elif mv == 'r':
        if res_main is not None:
            env[site].r = res_main
        if res_corner_a is not None:
            env[site].tr = res_corner_a
        if res_corner_b is not None:
            env[site].br = res_corner_b
    elif mv == 't':
        if res_main is not None:
            env[site].t = res_main
        if res_corner_a is not None:
            env[site].tl = res_corner_a
        if res_corner_b is not None:
            env[site].tr = res_corner_b
    elif mv == 'b':
        if res_main is not None:
            env[site].b = res_main
        if res_corner_a is not None:
            env[site].bl = res_corner_a
        if res_corner_b is not None:
            env[site].br = res_corner_b


def update_AD_(env, opts_svd, moves='hv', method='2x2', devices=None,
               **kwargs):
    """One CTM sweep, AD-aware distributed."""
    if devices is None:
        raise ValueError("update_AD_ requires devices=...")
    for mv in moves:
        update_core_AD_(env, mv, opts_svd, devices, **kwargs)
    return env


def iterate_AD_(env, opts_svd, moves='hv', method='2x2', max_sweeps=1,
                devices=None, **kwargs):
    """Generator: forward CTMRG convergence loop, AD-aware distributed.

    Drop-in replacement for ``_env_ctm_dist_mp.iterate_D_`` at the
    multi-device call site in ``FixedPoint.get_converged_env``. Yields
    a ``CTMRG_out`` after each sweep so the outer loop can check
    convergence the same way it does for ``iterate_D_``.

    Wins over ``iterate_D_``:
      - Persistent worker pool (one spawn cost across the whole run,
        not per ``fp_ctmrg`` call).
      - Per-rank command queues (no shared-task-queue contention).
      - Batched per-stage dispatch.

    Runs each sweep under ``torch.no_grad()`` because this is the
    forward CTMRG convergence loop -- no backward fires through it.
    The autograd machinery in ``update_AD_`` still runs (workers
    capture per-stage graphs), but the corresponding ``_BatchedFn``
    ctxs are GC'd at the end of each sweep, firing the saved-graph
    cleanup so workers don't accumulate activations across sweeps.
    """
    import torch
    from yastn.yastn.tn.fpeps.envs._env_ctm import CTMRG_out

    if devices is None or len(devices) < 1:
        raise ValueError("iterate_AD_ requires devices=...")

    # Strip control kwargs that don't belong on update_AD_'s projector path.
    # corner_tol / iterator_step belong to the iterate_D_ generator API;
    # checkpoint_move is a backward-graph option, irrelevant for no_grad fwd.
    update_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ('iterator_step', 'corner_tol',
                                   'iterator', 'checkpoint_move')}

    for sweep in range(1, max_sweeps + 1):
        with torch.no_grad():
            update_AD_(env, opts_svd=opts_svd, moves=moves,
                       method=method, devices=devices, **update_kwargs)
        yield CTMRG_out(sweeps=sweep, max_dsv=None,
                        max_D=env.max_D(), converged=False)


# ===========================================================================
# Drop-in replacement for env.update_(...) inside fixed_point_iter
# ===========================================================================

def fp_update_(env_in, ctm_opts_fp):
    """Honor ``ctm_opts_fp['fp_devices']`` to switch between serial and
    distributed-AD updates.
    """
    fp_devices = ctm_opts_fp.get('fp_devices', None)
    opts = {k: v for k, v in ctm_opts_fp.items() if k != 'fp_devices'}
    if fp_devices is None:
        return env_in.update_(**opts)
    return update_AD_(env_in, devices=fp_devices, **opts)
