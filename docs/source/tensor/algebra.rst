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
collected into one compact scatter (fusing) or gather (unfusing). Three regimes fall out:

* **all blocks large** — pure per-block loop; no index map is built,
* **all blocks small** — a single lean scatter/gather over the whole buffer,
* **mixed** — the hybrid kernel: the large blocks loop while the small ones ride one
  scatter built over compact indices that cover only the real blocks.

Unfusing additionally falls back to the loop when the destination blocks do not tile the
output buffer exactly.

The index maps are keyed on structure and cached, so a repeated fusion pattern builds them
once — see :ref:`tensor/caching:caching`, entry ``pack_transpose_and_merge_params``. That
entry holds device-resident tensors; :func:`yastn.clear_cache` releases them.

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
     - Tile size for building the index map. Unset means a single tile of ``2**27``
       elements. A **positive** value tiles the build, bounding its peak scratch memory.
       ``0`` is the escape hatch: force the per-block loop even on GPU, which is also the
       baseline to A/B against when checking whether the scatter path is helping.

Both variables are read on every call, so they can be changed at runtime without
reimporting YASTN. They have no effect on CPU or on the NumPy backend. A negative or
non-integer value is reported through :mod:`warnings` and the default is used instead — an
invalid setting never raises.

All regimes produce bit-identical results; the choice is purely one of performance.


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
