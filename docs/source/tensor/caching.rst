Caching
=======

Why there is a cache
--------------------

YASTN separates a tensor operation into two stages: computing the **metadata** — which
blocks of the operands meet, how their charges combine, where each block lands in the
output buffer — and then moving the **data** according to that plan. The metadata stage is
pure Python over the tensor's structural descriptors (``struct``, fusion history ``hfs``,
signatures), which are hashable and, crucially, do *not* depend on the numerical values
inside the blocks.

That makes the metadata stage cacheable. Tensor-network algorithms overwhelmingly repeat
the *same* block structure step after step — DMRG sweeps, CTMRG iterations, or the combo
loop of :ref:`tensor/large_contractions:large contractions` all contract tensors whose
charge sectors are fixed while their entries change. Each metadata builder is therefore
wrapped in :func:`functools.lru_cache` keyed on structure, so the Python bookkeeping is
paid once and every later call with the same structure is a dictionary lookup.

The consequence worth remembering: the cache is what keeps YASTN's Python overhead from
scaling with the number of iterations. If a hot loop shows an unexpectedly low hit rate,
something is perturbing the block structure between steps.

Controlling the caches
----------------------

.. autofunction:: yastn.set_cache_maxsize

.. autofunction:: yastn.clear_cache

.. autofunction:: yastn.get_cache_info

Reading the statistics
----------------------

:func:`yastn.get_cache_info` returns a ``dict`` mapping a short name to that cache's
:class:`functools.CacheInfo` — a named tuple ``(hits, misses, maxsize, currsize)``:

.. code-block:: python

    import yastn

    yastn.set_cache_maxsize(maxsize=1024)
    ...  # run a few iterations of your algorithm
    info = yastn.get_cache_info()
    print(info["fuse_hard"])       # CacheInfo(hits=..., misses=..., maxsize=1024, currsize=...)

    # a healthy steady-state loop is dominated by hits
    hits, misses = info["fuse_hard"].hits, info["fuse_hard"].misses
    print(f"hit rate {hits / (hits + misses):.2%}")

A ``currsize`` sitting at ``maxsize`` together with a climbing ``misses`` count means the
cache is thrashing — the working set of distinct structures is larger than the cache, and
raising ``maxsize`` will help. Persistent misses at low ``currsize`` mean the structures
themselves keep changing, which no cache size will fix.

The entries, grouped by the area they serve:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Area
     - Keys
   * - Fusion
     - ``fuse_hard``, ``unfuse_hard``, ``intersect_hfs``, ``combine_leg_structure``
   * - Contraction
     - ``tensordot_f2m``, ``tensordot_fc``, ``tensordot_nf``, ``tensordot_cutensor_cpu``,
       ``tensordot_cutensor_gpu``, ``broadcast``, ``mask``, ``trace``, ``vdot``,
       ``swap_gate``, ``swap_gate_charge``, ``ncon``
   * - Algebra
     - ``addition``
   * - Block structure
     - ``get_blocks``, ``get_blocks_charges_all``, ``get_trimmed_struct_engine``,
       ``get_trimmed_struct_engine_gpu``
   * - Backend (``torch`` only)
     - ``pack_transpose_and_merge_params``

The key set depends on the backend
----------------------------------

The tensor-side keys above are always present. Backend-side keys appear only once that
backend has actually been imported: a backend module registers its own caches when it is
loaded, so ``yastn`` itself never has to import a backend — and never forces an optional
dependency such as ``torch`` on users of another backend. Code that inspects a particular
backend key should therefore use ``.get()``:

.. code-block:: python

    fusion_params = yastn.get_cache_info().get("pack_transpose_and_merge_params")
    if fusion_params is not None:      # torch backend is loaded
        ...

:func:`yastn.set_cache_maxsize` still covers backends loaded *later*: the requested size is
remembered and applied at registration time, so the ordering of ``set_cache_maxsize`` and
:func:`yastn.make_config` does not matter.

Memory held by the caches
-------------------------

Most entries are small — tuples of integers and short NumPy index arrays. The exception is
the torch backend's ``pack_transpose_and_merge_params``, which caches the index maps used
by the GPU fuse/unfuse path (see
:ref:`tensor/algebra:gpu execution: hybrid scatter/loop`). Those index tensors are moved
onto the data's device on first use and stay there, so the entry pins GPU memory
proportional to the size of the fused buffers it has seen.

:func:`yastn.clear_cache` is the lever for releasing them — worth calling between phases of
a calculation that use very different tensor sizes, and worth knowing about when several
worker processes each accumulate their own copy.
