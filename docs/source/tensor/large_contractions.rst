Large contractions
==================

See also :ref:`tensor contractions <tensor/algebra:tensor contractions>` for
the basic building blocks (:func:`yastn.ncon`, :func:`yastn.tensordot`, ...).

The module :mod:`yastn.tensor.oe_blocksparse` provides a *sliced* (also called
*unrolled*) block-sparse contraction engine on top of :func:`yastn.ncon`. It is
aimed at contractions whose single-shot intermediate tensors are too large to
fit in memory, and/or that should be spread across several GPUs. The companion
module :mod:`yastn.tensor._oe_blocksparse_mp` implements the multi-device
(multiprocess) dispatch of the same computation.

Motivation
----------

A tensor-network contraction is specified in opt_einsum's *interleaved* format
with an **explicit** output index group::

    contract_with_unroll(T1, ig1, T2, ig2, ..., out_ig, unroll=..., optimize=...)

Two knobs turn this from a plain ``ncon`` into a large-contraction engine:

* **Slicing / unrolling.** One or more index labels are *unrolled*: their leg is
  partitioned into non-overlapping :class:`~yastn.SlicedLeg` pieces. The
  contraction is then evaluated as a loop over the Cartesian product of these
  pieces (*combos*). Each iteration masks the relevant tensors down to a single
  slice, so every intermediate produced by ``ncon`` is strictly smaller than the
  full-leg intermediate. This trades extra compute (the loop) for a lower
  **peak memory** footprint, and exposes the loop as an axis of parallelism.

* **Multi-device dispatch.** The independent combos of the unroll loop can be
  distributed round-robin across a pool of persistent worker processes pinned to
  different devices (e.g. several GPUs), with autograd supported through a
  checkpoint-style custom :class:`torch.autograd.Function`.

Both *contracted* labels (summed away) and *output* labels (kept in the result)
may be unrolled. Partials from contracted-only combos are summed with ``+``;
partials that differ along an output-unrolled axis are grouped by their position
along that axis and reassembled into the final tensor with :func:`yastn.block`.

Slicing a leg
-------------

A :class:`~yastn.SlicedLeg` selects a subset of a leg's charge sectors, each
optionally restricted to a contiguous slice inside that sector's block
dimension. A list of non-overlapping :class:`~yastn.SlicedLeg` objects whose
union covers the whole leg is a valid *partition*; iterating over it and summing
the partial contractions reproduces the full result.

The ``unroll`` argument maps index labels to such partitions:

.. code-block:: python

    import yastn

    # partition the leg carrying label 'k' into charge-sector-sized pieces
    unroll = {'k': yastn.make_sliced_legs(A.get_legs(axis_of_k))}

    # or let YASTN slice a leg uniformly into segments of at most `size`
    # simply by passing an int (resolved against the network's legs):
    unroll = {'k': 256}   # equivalent to slice_leg_uniform(leg, 256)

Two partitioning helpers are provided:

* :func:`yastn.make_sliced_legs` — one :class:`~yastn.SlicedLeg` per charge
  sector (the finest partition; the simplest, and the one an integer maps to
  when ``size`` exceeds every sector).
* :func:`yastn.tensor.oe_blocksparse.slice_leg_uniform` — contiguous segments of
  at most ``size`` elements; a segment may span several charge sectors.

Entry-point API
---------------

.. autofunction:: yastn.get_contraction_path

.. autofunction:: yastn.contract_with_unroll

Supporting types used to build the ``unroll`` argument:

.. autoclass:: yastn.SlicedLeg
   :members:

.. autofunction:: yastn.make_sliced_legs

.. autofunction:: yastn.tensor.oe_blocksparse.slice_leg_uniform

Single-device execution
-----------------------

:func:`yastn.contract_with_unroll` resolves the ``unroll`` argument (integers are
expanded to :class:`~yastn.SlicedLeg` partitions), optionally searches a
contraction path with :func:`yastn.get_contraction_path`, and then either falls
back to a plain :func:`yastn.ncon` (when ``unroll is None``) or drives the sliced
loop in ``_contract_with_sliced_unroll``.

The loop is preceded by a **metadata-only prefilter** (``_metadata_filter_combos``):
for every combo it applies the slice masks *to the tensor metadata only* (no GPU
work), drops combos whose masked tensors have no surviving blocks, and runs
:func:`ncon_prefilter` to record, per surviving combo, which blocks each operand
must keep (``pf_trim``) and — when ``per_combo_path`` is set — the effective
per-axis dimensions used to tune a combo-specific path. Only the surviving combos
enter the actual contraction loop.

Inside the loop, each combo (``_contract_single_combo``) masks its operands with
cached diagonal :meth:`~yastn.Tensor.apply_mask` tensors, trims blocks according
to ``pf_trim`` (:func:`_filter_tensor_blocks`), runs :func:`yastn.ncon`, and
accumulates the partial into a bucket keyed by its position along the
output-unrolled axes. Optionally the whole loop body is wrapped in
:func:`torch.utils.checkpoint.checkpoint` (``checkpoint_loop=True``) so masking
and ``ncon`` intermediates are recomputed on backward instead of being kept for
every iteration. Finally, single-key results are returned directly, while
output-unrolled results are reassembled with :func:`yastn.block` and stripped of
their fusion history via :meth:`~yastn.Tensor.drop_leg_history`.

.. graphviz::
   :caption: Single-device (serial) control/data flow of ``contract_with_unroll``.

   digraph single_device {
       rankdir=TB;
       node [shape=box, fontname="Helvetica", fontsize=10, margin="0.12,0.06"];
       edge [fontname="Helvetica", fontsize=9];

       entry   [label="contract_with_unroll(*args, unroll, optimize=None, ...)",
                style=filled, fillcolor="#e8f0fe"];
       resolve [label="_validate_and_resolve_unroll\l(int -> [SlicedLeg] via slice_leg_uniform)\l"];
       path    [label="get_contraction_path\l(opt_einsum path; per-combo slice dims)\l",
                style=filled, fillcolor="#e8f0fe"];
       branch  [label="unroll ?", shape=diamond, style=filled, fillcolor="#fff3cd"];
       plain   [label="ncon(...)\l(plain contraction, no slicing)\l"];
       route   [label="device routing\l(devices + mp_workers_per_device)\l", shape=diamond,
                style=filled, fillcolor="#fff3cd"];
       mp      [label="_contract_with_sliced_unroll_mp\l(multi-device path)\l",
                style=filled, fillcolor="#fde2e2"];

       prefilter [label="_metadata_filter_combos  (metadata only, no GPU)\lper combo: meta-mask -> empty-block skip -> ncon_prefilter\l=> surviving combos, pf_trim, dim_overrides\l"];

       subgraph cluster_loop {
           label="combo loop over surviving combos  (_contract_single_combo)";
           style=dashed; color="#8ab4f8"; fontname="Helvetica"; fontsize=10;
           pick   [label="choose path\l(shared `optimize` or per-combo)\l"];
           mask   [label="apply_mask(slice) on unrolled axes\l(cached diagonal masks)\l"];
           trim   [label="_filter_tensor_blocks(pf_trim)"];
           ncon   [label="ncon(masked operands)\l(optional torch.utils.checkpoint)\l"];
           accum  [label="accumulate into partials[output_pos_key]"];
           pick -> mask -> trim -> ncon -> accum;
       }

       assemble [label="output-unrolled labels ?", shape=diamond, style=filled, fillcolor="#fff3cd"];
       single   [label="partials[()]\l(single block)\l"];
       block    [label="yastn.block(partials)\l+ drop_leg_history\l"];
       result   [label="result : yastn.Tensor", style=filled, fillcolor="#d7f5dd"];

       entry -> resolve;
       resolve -> path [label="optimize is None"];
       resolve -> branch [label="optimize given"];
       path -> branch;
       branch -> plain [label="None"];
       branch -> route [label="dict"];
       route -> mp [label="multi-device"];
       route -> prefilter [label="serial"];
       prefilter -> pick;
       accum -> assemble;
       assemble -> single [label="no"];
       assemble -> block  [label="yes"];
       plain -> result;
       single -> result;
       block -> result;
   }

Multi-device execution
----------------------

When ``devices`` names more than one device (or a single device with
``mp_workers_per_device >= 2``), ``_contract_with_sliced_unroll`` hands off to
``_contract_with_sliced_unroll_mp`` in :mod:`yastn.tensor._oe_blocksparse_mp`.
A **persistent pool** of ``spawn``-ed worker processes (``mp_workers_per_device``
per device) is created once per ``(devices, workers, config)`` and reused across
calls; each worker loops on a command queue and pins itself to its assigned
device.

The dispatcher runs the same metadata prefilter as the serial path and, in
addition, derives the output tensor's structure directly from the input legs
(``_derive_output_structs``) — no data contraction is needed to know the result's
blocks. Both results are memoized per pool, keyed on a structural fingerprint of
the inputs. The surviving combos are distributed round-robin into per-worker
assignments, and the computation is driven through a custom
:class:`torch.autograd.Function` (``_MultiprocSlicedUnrollFunction``) that
implements a **checkpoint pattern**:

* **Forward.** Input data is replicated once per unique worker device (shared via
  CUDA IPC), each worker contracts its assigned combos under
  :func:`torch.no_grad`, zero-fills every partial to its per-key output struct,
  and ships the partials back. The parent sums per-key partials (gathering to the
  original device) and, for output-unrolled contractions, reassembles them with
  :func:`yastn.block`.

* **Backward.** The parent re-runs only the (cheap) :func:`yastn.block` assembly
  with autograd enabled to split the output gradient into per-key gradients, then
  asks each worker to *re-run* its combos with autograd on and call
  :func:`torch.autograd.backward` locally. Workers return per-input gradient data,
  which the parent sums across workers. No live autograd graph crosses the process
  boundary; the forward is replayed on backward instead.

Robustness details handled by the pool: a Linux parent-death signal so orphaned
workers do not leak CUDA contexts, a liveness guard that turns a dead-worker hang
into an error, and multiprocess-safe logging that forwards worker log records to
the parent.

.. graphviz::
   :caption: Multi-device dispatch of the sliced unroll loop (forward + backward).

   digraph multi_device {
       rankdir=TB;
       node [shape=box, fontname="Helvetica", fontsize=10, margin="0.12,0.06"];
       edge [fontname="Helvetica", fontsize=9];

       entry [label="_contract_with_sliced_unroll_mp(*args, unroll, optimize,\ldevices, mp_workers_per_device)\l",
              style=filled, fillcolor="#fde2e2"];
       pool  [label="_get_or_create_pool(devices, n_per_device, config)\l(persistent, spawn; workers loop on cmd_q / res_q)\l"];
       cache [label="cache lookup (struct fingerprint)\lmiss: _metadata_filter_combos  => surviving, pf_trim, dim_overrides\l      _derive_output_structs => per_key_struct, full_struct\l"];
       assign [label="round-robin surviving combos -> worker_assignments"];
       apply  [label="_MultiprocSlicedUnrollFunction.apply(*input_data, meta)",
               style=filled, fillcolor="#e8f0fe"];

       subgraph cluster_fwd {
           label="FORWARD"; style=dashed; color="#8ab4f8"; fontname="Helvetica"; fontsize=10;
           f_rep  [label="replicate inputs per device (CUDA IPC)"];
           f_disp [label="cmd_q.put('forward', assigned, ...)"];
           f_work [label="worker (no_grad):\l_contract_with_sliced_unroll(_return_partials)\l-> per-key partials -> zero-fill to per_key_struct\l-> res_q.put('forward_done')\l", style=filled, fillcolor="#eef1f5"];
           f_sum  [label="parent: sum per-key data across workers\l(single-key -> merged[()];  multi-key -> yastn.block + drop_leg_history)\l=> out_data\l"];
           f_rep -> f_disp -> f_work -> f_sum;
       }

       subgraph cluster_bwd {
           label="BACKWARD"; style=dashed; color="#f5b78a"; fontname="Helvetica"; fontsize=10;
           b_split [label="parent: rerun yastn.block w/ autograd\l=> grad_per_key\l"];
           b_disp  [label="cmd_q.put('backward', assigned, grad_per_key)"];
           b_work  [label="worker (enable_grad): re-run combos, zero-fill,\ltorch.autograd.backward(partials, grads)\l-> res_q.put('backward_done', input_grads)\l", style=filled, fillcolor="#eef1f5"];
           b_sum   [label="parent: sum per-input grads across workers"];
           b_split -> b_disp -> b_work -> b_sum;
       }

       out    [label="Tensor.from_dict(full_struct, out_data)", style=filled, fillcolor="#d7f5dd"];

       entry -> pool -> cache -> assign -> apply;
       apply -> f_rep;
       f_sum -> out;
       out -> b_split [label="backward()", style=dashed, constraint=false];
   }

Memory management
-----------------

The sliced loop keeps the *peak* size of any single intermediate bounded, but on
CUDA the PyTorch caching allocator can still **fragment**: it holds memory that
is *reserved but unallocated* in segments that also contain live blocks, so a
later large contiguous allocation fails even though the totals look sufficient.
``torch.cuda.empty_cache()`` only returns segments that are *entirely* free, so
it cannot cure this class of fragmentation — only ``expandable_segments:True``
(virtual-memory remapping of partial segments) can.

Two further sets of knobs matter here, documented elsewhere but worth pointing at because
the unroll loop amplifies both:

* Every :func:`yastn.ncon` inside the loop fuses and unfuses legs, so
  ``YASTN_FUSE_SCATTER_THRESH`` and ``YASTN_FUSE_SCATTER_CHUNK`` —
  see :ref:`tensor/algebra:gpu execution: hybrid scatter/loop` — apply to each combo.
  ``YASTN_FUSE_SCATTER_CHUNK`` is the one to reach for if the *index build* itself is the
  memory problem, since it bounds that build's scratch allocation.
* The metadata caches (:ref:`tensor/caching:caching`) are populated per process, so each
  spawned worker keeps its own. The fusion-index cache holds device-resident tensors;
  :func:`yastn.clear_cache` releases them if the accumulated indices become significant
  next to the contraction's own working set.

The knobs below tune allocator behaviour for the ``torch`` / ``torch_cutensor``
CUDA backends (they are inert on CPU / NumPy):

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Variable / argument
     - Scope
     - Purpose
   * - ``PYTORCH_ALLOC_CONF``
     - all processes, at allocator init
     - PyTorch's own global allocator config; the place to set
       ``expandable_segments:True``. Inherited by spawned workers.
   * - ``YASTN_OE_ALLOC_CONF`` (or ``contract_with_unroll(alloc_conf=...)``)
     - each spawned worker, at startup
     - Runtime-mutable allocator knobs (``garbage_collection_threshold``,
       ``max_split_size_mb``, ``roundup_*``) applied *per worker*, before its
       first allocation. The keyword argument wins over the env var.
   * - ``YASTN_OE_CUDA_CACHE_RELEASE_LEVEL``
     - ``torch_cutensor`` contraction
     - How often ``empty_cache()`` is called during the loop (``0``–``3``).
   * - ``YASTN_OE_OOM_RETRY``
     - ``torch`` CUDA contraction
     - ``1`` retries a contraction op once, after ``empty_cache()``, on a CUDA
       out-of-memory error (default ``0``).

Ideal case — ``expandable_segments:True``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the host kernel supports it, the cleanest option is PyTorch's expandable
segments, set **globally** through PyTorch's own environment variable::

    export PYTORCH_ALLOC_CONF=expandable_segments:True

Each process's allocator reads this once at initialization, and ``spawn``-ed
workers inherit the same environment — so their allocators pick it up too, with
no per-worker configuration. Expandable segments let the allocator hand back
*partial* segments, eliminating the fragmentation that plain ``empty_cache``
cannot. In this case leave both ``YASTN_OE_ALLOC_CONF`` and
``YASTN_OE_CUDA_CACHE_RELEASE_LEVEL`` **unset**.

Fallback — kernel without ``expandable_segments`` support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sharing CUDA tensors between processes with ``expandable_segments:True`` needs
the ``pidfd_open`` syscall. On kernels (or containers) that lack it, the
multi-device path **crashes** during CUDA IPC with an error naming
``pidfd_open``. This is deliberate — YASTN does *not* silently work around it, so
the failure clearly tells you this host cannot combine expandable segments with
multiprocess dispatch. Drop ``expandable_segments`` and use the two per-worker
knobs below instead.

**Per-worker allocator settings.** Apply runtime-mutable caching-allocator knobs
(see `Optimizing memory usage with PYTORCH_ALLOC_CONF
<https://docs.pytorch.org/docs/2.13/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf>`_)
to each worker at startup::

    export YASTN_OE_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

The key knob is ``garbage_collection_threshold``: it makes the allocator
proactively reclaim unused cached blocks once reserved memory exceeds the given
fraction, keeping the reserved-but-unallocated pile from growing and
fragmenting. ``max_split_size_mb`` additionally stops large free blocks from
being carved up for small requests, preserving big contiguous regions for the
large intermediates. Only runtime-mutable keys take effect; ``backend`` and
``expandable_segments`` are fixed at allocator init and are ignored here. The
same string can be passed programmatically, which wins over the environment::

    contract_with_unroll(..., alloc_conf="garbage_collection_threshold:0.8")

(Applies to the spawned workers of the multi-device path. A single-process
serial run has no workers; use ``PYTORCH_ALLOC_CONF`` for it, which the main
process reads at init.)

**Cache-release cadence.** On the ``torch_cutensor`` backend, YASTN can call the
*blocking* ``torch.cuda.empty_cache()`` at chosen points to return fully-free
segments to the driver. A release point fires only when the level is at least
its tag (each level adds to the ones below):

======  ===========================================================
Level   Effect
======  ===========================================================
``0``   never (default)
``1``   once per contraction
``2``   + after every combo (one contracted slice of the network)
``3``   + after every ``tensordot`` inside ``ncon``
======  ===========================================================

::

    export YASTN_OE_CUDA_CACHE_RELEASE_LEVEL=2

This should be experimented with; a reasonable starting point is **level 2**
(clean up after every combo). Remember that ``empty_cache`` cannot defragment
free space interleaved with live allocations — that is what
``expandable_segments:True`` is for. A malformed value is treated as ``0`` with a
one-time warning.

..    Both graphs are rendered by :mod:`sphinx.ext.graphviz`, which shells out to the
..    Graphviz ``dot`` executable at build time. If the graphs are missing from the
..    built HTML, install Graphviz (``conda install graphviz`` or
..    ``apt-get install graphviz``) so ``dot`` is on ``PATH``.
