# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================
"""Multiprocess-safe logging for spawn-mode worker pools.

Spawn-mode children inherit no logging configuration, so any ``log.info``
in a worker is silently dropped: the worker's root logger defaults to
``WARNING`` with no handlers, so ``INFO`` records (notably opt_einsum
contraction path-info reports) never reach a handler. Naively reattaching a
``FileHandler`` in each worker is worse -- it races on the shared file.
Mixing one truncating writer (the parent opened the log ``filemode='w'``)
with several appending writers (``filemode='a'`` in each worker) corrupts
ordering and overwrites lines, because the parent writes at its own
advancing offset while workers always jump to ``EOF``.

This module centralizes logging via the standard ``QueueHandler`` /
``QueueListener`` pattern:

* The parent owns the only real handler(s). A background ``QueueListener``
  thread drains a shared queue and emits each record through those handlers.
  Because the parent's own direct logging and the listener thread share the
  same handler instance (which holds an internal lock), writes are
  serialized -- no corruption, correct ordering.
* Each worker routes its root logger through a single ``QueueHandler`` onto
  that queue. Multi-line records (e.g. ``get_contraction_path`` path-info
  blocks) survive intact because they are emitted by one process.

The pattern composes under nesting: if a worker that already has a
``QueueHandler`` creates its own pool, that worker becomes the "parent" for
the inner pool and its ``QueueHandler`` is reused as the listener target, so
inner-worker records are forwarded up the chain to the top-level file.

Pool usage (parent side, in ``__init__`` and ``shutdown``)::

    self.log_queue, self.log_listener = start_parent_log_listener(ctx)
    log_level = parent_log_level()
    # ... pass (self.log_queue, log_level) to each spawned worker ...

    def shutdown(self):
        ...  # join workers FIRST so no late record is enqueued post-stop
        stop_parent_log_listener(self.log_listener)

Worker usage (first thing in ``_worker_main``)::

    install_worker_log_handler(log_queue, log_level,
                               tag=f"<pool> rank {rank} dev {dev}")

This module is intentionally stateless (functions only) so it is safe even
if imported under more than one module name in the doubly-nested ``yastn``
package layout.
"""
import logging
import logging.handlers


def parent_log_level():
    """Effective level of the parent root logger, falling back to ``INFO``.

    Captured in the parent at pool-creation time and handed to workers so a
    worker filters records (e.g. drops ``DEBUG``) consistently with the
    parent before anything is enqueued.
    """
    lvl = logging.getLogger().level
    return lvl if lvl else logging.INFO


def snapshot_logger_levels():
    """Capture ``{name: level}`` for every logger with an explicit
    (non-``NOTSET``) level in this process.

    Handed to spawn workers alongside the root level so they replay the
    parent's *per-logger* configuration on top of the root level -- e.g. a
    single module bumped to ``DEBUG`` (``--log_oe_path`` raises
    ``yastn.yastn.tensor.oe_blocksparse``) becomes visible in workers too,
    while everything else stays at the root level. Returns a plain dict
    (picklable for ``Process`` args).
    """
    levels = {}
    for name, lg in logging.Logger.manager.loggerDict.items():
        # loggerDict also holds PlaceHolder objects, which have no level.
        if isinstance(lg, logging.Logger) and lg.level != logging.NOTSET:
            levels[name] = lg.level
    return levels


def start_parent_log_listener(ctx):
    """Create a cross-process log queue and start a ``QueueListener``.

    Runs in the parent. The listener drains the returned queue into the
    parent root logger's *existing* handlers (so worker records land in the
    same destination as the parent's own logs, through the same lock).

    :param ctx: the multiprocessing context the pool spawns workers with
        (e.g. ``torch.multiprocessing.get_context('spawn')``); the queue is
        created from it so it is compatible with the spawned children.
    :returns: ``(log_queue, listener)``. If the parent root logger has no
        handlers configured, returns ``(None, None)`` -- callers must then
        skip worker-side handler installation so workers do not enqueue
        records onto a queue nobody drains.
    """
    handlers = list(logging.getLogger().handlers)
    if not handlers:
        return None, None
    log_queue = ctx.Queue()
    listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True)
    listener.start()
    return log_queue, listener


def stop_parent_log_listener(listener):
    """Stop a listener started by :func:`start_parent_log_listener`.

    Idempotent and exception-safe (callable at interpreter shutdown). Call
    this only after joining the workers, so every record they emitted has
    been enqueued before the listener drains and stops.
    """
    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass


def install_worker_log_handler(log_queue, level=logging.INFO, tag=None,
                               logger_levels=None):
    """Route a spawn worker's root logger through ``log_queue``.

    Clears any inherited handlers, attaches a single ``QueueHandler``, and
    sets the root level so ``log.info`` records are enqueued (and ``DEBUG``
    is dropped at the worker when ``level`` is ``INFO``).

    :param log_queue: the queue from :func:`start_parent_log_listener`, or
        ``None`` (no-op -- the parent had no handlers to forward to).
    :param level: root level for the worker (typically the parent's level).
    :param tag: short identifier (e.g. ``"oe_mp rank 2 dev cuda:1"``) baked,
        together with the originating pid, into every message so records
        from concurrent workers stay attributable in the merged log. The
        parent's listener applies the parent's own format on top.
    :param logger_levels: optional ``{name: level}`` from
        :func:`snapshot_logger_levels`, replayed after the root level so the
        worker mirrors the parent's per-logger config (e.g. one module at
        ``DEBUG`` while the root stays at ``INFO``).
    """
    if log_queue is None:
        return
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    qh = logging.handlers.QueueHandler(log_queue)
    prefix = f"[{tag}] " if tag else ""
    # QueueHandler.prepare() runs self.format(record) before pickling, so the
    # tag + pid become part of the message text that the parent's listener
    # later emits. ``%(process)d`` resolves to the record's originating pid
    # and is preserved across the queue.
    qh.setFormatter(logging.Formatter(prefix + "pid %(process)d: %(message)s"))
    root.addHandler(qh)
    root.setLevel(level)
    if logger_levels:
        for name, lvl in logger_levels.items():
            logging.getLogger(name).setLevel(lvl)
