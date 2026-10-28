Algebra with YASTN tensors
==========================

Basic algebra operations with symmetric tensors
-----------------------------------------------

See examples at :ref:`examples/tensor/algebra:basic algebra operations`.

Symmetric tensors can be added to and multiplied by a scalar
through the usual operations ``+``, ``-``, ``*``, and ``/``.
Element-wise raising to a power is done with the standard power operation ``**``.

Simple element-wise operations
------------------------------

.. automethod:: yastn.Tensor.__abs__
.. automethod:: yastn.Tensor.real
.. automethod:: yastn.Tensor.imag
.. automethod:: yastn.Tensor.sqrt
.. automethod:: yastn.Tensor.rsqrt
.. automethod:: yastn.Tensor.reciprocal
.. automethod:: yastn.Tensor.exp
.. automethod:: yastn.Tensor.clip

.. automethod:: yastn.Tensor.__mul__
.. automethod:: yastn.Tensor.__pow__
.. automethod:: yastn.Tensor.__truediv__

.. autofunction:: yastn.Tensor.__add__
.. autofunction:: yastn.Tensor.__sub__
.. autofunction:: yastn.add

.. autofunction:: yastn.Tensor.__lt__
.. autofunction:: yastn.Tensor.__gt__
.. autofunction:: yastn.Tensor.__le__
.. autofunction:: yastn.Tensor.__ge__
.. autofunction:: yastn.Tensor.bitwise_not

.. autofunction:: yastn.allclose


Tensor contractions
-------------------

See examples at :ref:`examples/tensor/algebra:tensor contractions`.

Tensor contractions are the main building blocks of tensor network algorithms.
The functions below facilitate the computation of

	* `Trace`: :math:`B_{jl}= \sum_{i} T_{ijil}` or, using Einstein's summation convention,
	  repeated indices as :math:`B_{jl} = T_{ijil}`.
	* `Contractions`: in the usual form :math:`C_{abc} = A_{aijb}{\times}B_{cij}` and also
	  outer products :math:`M_{abkl} = A_{ak}{\times}B_{bl}`.

or composition of such operations over several tensors.

.. automethod:: yastn.Tensor.__matmul__

.. autofunction:: yastn.tensordot
.. autofunction:: yastn.vdot
.. autofunction:: yastn.broadcast
.. autofunction:: yastn.apply_mask
.. autofunction:: yastn.trace
.. autofunction:: yastn.einsum
.. autofunction:: yastn.ncon
.. autofunction:: yastn.swap_gate
.. autofunction:: yastn.fkron

.. note::

   Contractions are metadata-heavy: matching blocks between operands and planning the
   output is pure Python work that :doc:`caching </tensor/caching>` reuses across calls
   (``tensordot_f2m``, ``tensordot_fc``, ``tensordot_nf``, ``tensordot_cutensor_*``,
   ``ncon``, ...). In a loop that contracts the same block structure repeatedly, these
   caches are what keeps the Python cost flat.

   Depending on ``tensordot_policy``, :func:`yastn.tensordot` also fuses legs into matrices
   before calling into the backend. On a CUDA device that fusion runs through the path
   described in :ref:`tensor/algebra:gpu execution: hybrid scatter/loop`, so the knobs
   documented there apply to contractions as well as to explicit
   :meth:`~yastn.Tensor.fuse_legs` calls.


Non-Abelian contractions with cuTENSOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``backend='torch_cutensor'`` and a CUDA device, non-Abelian contractions
(including ``SU2`` and ``SU2xU1``) use channel-aware reduced-block metadata.
Each nonzero gluing contribution becomes a separate contracted section of a
block-sparse matrix product. The input packing applies its recoupling coefficient,
and ``tensordot_bs_v2`` sums the products into their output channels.
No expansion into the magnetic basis is needed. PyTorch autograd propagates
through the packing, cuTENSOR contraction, and output embedding, including sums
of gradients for reused input blocks.

This path is selected independently of ``tensordot_policy``. Channel planning
runs on the CPU and is cached; ``meta_tensordot_policy`` still controls the
Abelian cuTENSOR path. Non-Abelian tensors on CPU use the ordinary PyTorch kernel.

Packing duplicates reduced blocks that contribute more than once. Products are
partitioned into batches before packing, and each batch's result is accumulated
into the final output, including channels shared by multiple batches.
``YASTN_CUTENSOR_PACK_CHUNK_BYTES`` sets the positive byte budget for packed
A+B+C per call (default: 1 GiB). Large CUDA contractions reduce that budget to
at most one eighth of available memory after allowing for the final output.
Budgets are rounded down to a power of two in elements to reuse metadata.
This bounds packed storage, not the final output or cuTENSOR workspace.
A single product exceeding the budget uses Torch's streamed block kernel
without packing. A cuTENSOR allocation failure also retries the affected batch
with that kernel; other runtime errors propagate unchanged.

Compact device metadata is cached. Gather indices and coefficient expansion are limited to
chunks of at most ``2**20`` elements, and backward recomputes these maps instead
of retaining full maps in the autograd graph. Dtype promotion happens while
packing, without a separate promoted copy of the whole input. Large noncanonical
inputs can recouple directly into packing, subject to a composition cost limit.
Autograd saves only the original operands needed for the requested derivatives,
not their packed copies. Backward rebuilds one opposite operand's packing per
batch, contracts with cuTENSOR, and accumulates at promoted precision before
converting to the input dtype. This trades extra packing work for lower retained
memory. Output embedding also uses chunks and writes into a single
zero-initialized buffer. Combining
many block products into one call can reduce launch overhead, but packing and
cuTENSOR planning can outweigh that benefit for small contractions. Measure the
workload with ``python autotune/bench_nonabelian_cutensor.py`` on the target GPU;
the script checks values and gradients before reporting warmed-up timings.
Its memory counter measures PyTorch allocations. The bundled TAPP revision also
uses PyTorch's allocator for workspace; externally allocated workspace in other
versions is not included.

The ordinary PyTorch contraction kernel materializes permuted blocks on demand,
with at most 8 MiB cached per operand. Larger blocks are temporary and can be
recomputed when reused. Backward accumulates one input block at a time and writes
it directly into the original layout and dtype, avoiding full permuted gradient
buffers. Only operands needed for requested gradients are saved. These bounds
reduce temporary memory at the cost of potentially repeating permutations and
dtype conversions when blocks do not fit in the cache.


Transposition
-------------

See examples at :ref:`examples/tensor/algebra:transposition`.

.. autofunction:: yastn.transpose
.. autoproperty:: yastn.Tensor.T
.. autofunction:: yastn.moveaxis
.. autoproperty:: yastn.Tensor.H


Fusion of legs (reshaping)
--------------------------

See examples at :ref:`examples/tensor/algebra:fusion (reshaping)`.

Fusion of several vector spaces :math:`V_1,V_2,\ldots,V_n` creates a new vector space as direct product :math:`W=V_1 \otimes V_2 \otimes \ldots \otimes V_n`,
which is then indexed by a single index of dimension :math:`\prod_i {\rm dim}(V_i)`.
Here multiplication depends on abelian symmetry, as the resulting total dimension is a sum of dimensions for effective charges.
The inverse operation can split the fused space into its original constituents.

For dense tensors, the operation corresponds to reshaping.

Fusion can be used to vary compression between (unfused) symmetric tensors with many small non-zero blocks and tensors with several fused spaces having just few, but large non-zero blocks.

.. automethod:: yastn.Tensor.fuse_legs
.. automethod:: yastn.Tensor.unfuse_legs
.. automethod:: yastn.Tensor.add_leg
.. automethod:: yastn.Tensor.remove_leg
.. automethod:: yastn.Tensor.drop_leg_history


GPU execution: hybrid scatter/loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hard fusion (:meth:`~yastn.Tensor.fuse_legs` with ``mode='hard'``) and its inverse
:meth:`~yastn.Tensor.unfuse_legs` are not just bookkeeping: they physically transpose and
copy every block into (or out of) the fused buffer. On the ``torch`` backend this move has
two implementations, and which one runs is chosen per call.

On **CPU** it is always a loop over blocks, one copy per block.

On **GPU** a per-block loop is a poor fit when a tensor has many small blocks: each block
costs a kernel launch, and thousands of tiny launches dominate the actual data movement.
The alternative — building an explicit index map and doing a single ``scatter``/``gather``
over the whole buffer — pays a one-off cost to construct that map, then moves everything in
one kernel. Neither wins everywhere: the loop is bandwidth-optimal for large blocks (no
index build at all), the scatter wins for many small ones.

YASTN therefore **splits each call by block size** rather than choosing globally. Blocks
with at least ``YASTN_FUSE_SCATTER_THRESH`` elements go through the loop; smaller blocks are
processed in bounded tiles with compact scatter (fusing) or gather (unfusing). Three regimes fall out:

* **all blocks large** — pure per-block loop; no index map is built,
* **all blocks small** — tiled scatter/gather over the whole buffer,
* **mixed** — the large blocks loop while the small ones use tiled compact indices
  that cover only the real blocks.

Unfusing additionally falls back to the loop when the destination blocks do not tile the
output buffer exactly.

Only block metadata is cached: ``pack_transpose_and_merge_params`` holds the CPU
parameters and ``device_merge_parameters`` holds separate copies for each device.
Element-sized maps are built one tile at a time and rebuilt during backward,
including for mixed blocks. See :ref:`tensor/caching:caching`;
:func:`yastn.clear_cache` releases the cached metadata.

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Variable
     - Default
     - Effect
   * - ``YASTN_FUSE_SCATTER_THRESH``
     - ``65536`` (``2**16``)
     - Per-block element count separating the loop (``>=`` threshold) from the
       scatter/gather (``<`` threshold). The default is roughly the point at which a single
       block saturates the memory interface of a datacentre GPU, which is also where the
       loop's launch cost and the index build cost cross over. Lower it to push more blocks
       through the loop, raise it to push more through the scatter.
   * - ``YASTN_FUSE_SCATTER_CHUNK``
     - unset
     - Unset selects a tile from the workload and available GPU memory. The policy
       targets eight tiles, starting at ``2**20`` elements for large operations,
       with scratch limited to the smaller of 1 GiB and one eighth of the estimated
       memory remaining after output allocation. Memory pressure can select smaller
       tiles. A **positive** value overrides this policy for index construction and
       data movement, including the mixed-block path and backward.
       ``0`` is the escape hatch: force the per-block loop even on GPU, which is also the
       baseline to A/B against when checking whether the scatter path is helping.

Both variables are read on every call, so they can be changed at runtime without
reimporting YASTN. They have no effect on CPU or on the NumPy backend. A negative or
non-integer value is reported through :mod:`warnings` and the default is used instead — an
invalid setting never raises.

All regimes produce bit-identical results; the choice is purely one of performance.


Non-Abelian fusion-tree transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Permuting non-Abelian legs requires a change of fusion basis as well as a dense
axis permutation. The channel matching and F/R coefficients for adjacent swaps
and three-leg F moves are cached by symmetry, tensor structure, and permutation.
Changing numerical tensor entries reuses these plans without caching their values.

The PyTorch CUDA backend batches transformations with at least 32 small
contributions (each below 65536 elements) using weighted gather/scatter kernels.
Larger contributions and smaller plans use block loops. Temporary index maps are
built in tiles of at most ``2**20`` contributions' elements, and backward applies
the adjoint map while summing gradients for reused input blocks. This accumulation
can change floating-point rounding; it preserves the fusion transformation within
numerical precision. CPU and NumPy use the same cached coefficients with block loops.

Hard fusion and unfusion also cache compact block-normalization weights on the
PyTorch device. They expand those weights there instead of repeatedly transferring
an element-sized array from the host. This path uses ordinary PyTorch and does not
require cuTENSOR.


Conjugation of symmetric tensors
--------------------------------

See examples at :ref:`examples/tensor/algebra:conjugation of symmetric tensors`.

.. automethod:: yastn.Tensor.conj
.. automethod:: yastn.Tensor.conj_blocks
.. automethod:: yastn.Tensor.flip_signature
.. automethod:: yastn.Tensor.flip_charges


Tensor norms
------------

.. autofunction:: yastn.linalg.norm


Spectral decompositions and truncation
--------------------------------------

See examples at :ref:`examples/tensor/decomposition:decompositions of symmetric tensors`.

``qr(..., compute_q=False)`` returns only ``R``. NumPy and Torch without
gradients omit construction of ``Q``. When Torch needs a derivative, the
backend keeps the reduced QR required by backward but does not assemble or
unfuse a separate output ``Q`` tensor.

``svd_with_truncation`` selects singular vectors while the factors are still
fused matrices, before phase fixing and unfusion. It uses the same solver and
global truncation rules as ``svd`` followed by ``truncation_mask``; no extra SVD
pass is introduced. This reduces temporary output storage, but the exact SVD
backward still retains the full singular vectors it requires.

Use ``D_block_min`` to protect selected symmetry sectors during truncation::

    U, S, V = a.svd_with_truncation(
        axes=(0, 1), D_total=21, tol=1e-10,
        D_block_min={(0,): 1, (1,): 1})

For SU2, when both sectors are present, this keeps at least one spin-0 and one
spin-1/2 irrep, consuming at least three dense states from ``D_total``.
Each minimum counts reduced states
(complete irreps), and charge keys refer to the resulting spectrum ``S``.
An integer minimum applies to every sector present; unspecified dictionary
entries have minimum zero. Entries for absent sectors are ignored; this also
applies to sectors omitted by a partial solver. Each minimum is capped at the
available sector dimension: requesting three reduced states from a sector with
only two retains both. Required states are selected first, even below
``tol`` or ``tol_block``, and the remaining capacity is filled in spectral order.
``D_total`` and ``D_block`` remain hard limits: conflicts with these effective
minima raise ``YastnError``. The same option is available in
``eigh_with_truncation`` and ``truncation_mask``, including through algorithm
options forwarded to these functions. With ``hermitian=True``, minima also
apply to conjugate sectors, whose states are kept in equal-rank pairs; their
shared minimum is capped at the smaller sector dimension.
For Abelian symmetries positive minima cannot be combined with
``largest_gap`` or ``eps_multiplet``; non-Abelian symmetries already ignore
these heuristics. The default ``D_block_min=0`` retains the previous behavior.

.. autofunction:: yastn.linalg.svd
.. autofunction:: yastn.linalg.svd_with_truncation
.. autofunction:: yastn.linalg.qr
.. autofunction:: yastn.linalg.eig
.. autofunction:: yastn.linalg.eigh
.. autofunction:: yastn.linalg.eigh_with_truncation
.. autofunction:: yastn.linalg.truncation_mask
.. autofunction:: yastn.linalg.entropy

.. _tensor-aux:

Auxiliary
---------

Eliminating individual blocks

.. automethod:: yastn.Tensor.remove_zero_blocks
.. automethod:: yastn.Tensor.remove_random_blocks

Methods called by :doc:`Krylov-based algorithms</tensor/krylov>`.

.. automethod:: yastn.Tensor.expand_krylov_space
