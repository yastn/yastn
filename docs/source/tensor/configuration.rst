YASTN configuration
===================

All YASTN tensors have to be provided with **configuration**, which defines:

   #. linear algebra backend

   #. :doc:`abelian symmetry group</tensor/symmetry>`

   #. default data type (``float64``, ``complex128``) and device (provided it is supported by backend) of a tensor

   #. fermionic statistics (controlling action of :meth:`yastn.swap_gate`)

The configuration can be provided as a Python module,
`types.SimpleNamespace <https://docs.python.org/3/library/types.html#types.SimpleNamespace>`_,
`typing.NamedTuple <https://docs.python.org/3/library/typing.html#typing.NamedTuple>`_, or similar, which defines the following members.

* required: ``backend``, ``sym``,
* optional: ``default_device``, ``default_dtype``, ``fermionic``, ``default_fusion``,  ``force_fusion``, ``tensordot_policy``, ``meta_tensordot_policy``, ``lazy_threshold``.

The configuration can be conveniently generated using

.. autofunction:: yastn.make_config

Below is an example of configuration defined as a plain Python module,
using NumPy backend and :math:`U(1)` symmetry.

.. code-block:: python

   import yastn.backend.backend_np as backend
   from yastn.sym import sym_U1 as sym

   default_device: str = 'cpu'
   default_dtype: str = 'float64'
   fermionic = False
   default_fusion: str = 'hard'
   force_fusion: str = None

Environment variables
---------------------

The following runtime knobs are read from the process environment. Defaults
preserve the standard behavior; values are case-insensitive where noted in the
implementation.

.. list-table::
   :header-rows: 1
   :widths: 38 16 46

   * - Variable
     - Default
     - Effect
   * - ``YASTN_CUTENSOR_PACK_CHUNK_BYTES``
     - ``16 GiB``
     - Byte budget for packed A+B+C in one non-Abelian cuTENSOR call. Larger
       values reduce batching overhead at a higher peak.
   * - ``YASTN_CUTENSOR_PACK_DIVISOR``
     - ``4``
     - On CUDA, also cap the packing budget at available memory divided by this
       value. Smaller values use more memory.
   * - ``YASTN_CUTENSOR_PACK_CHUNK_ELEMENTS``
     - ``2**20``
     - Element chunk for pack/unpack index maps. ``YASTN_MEMORY_OVERRIDE=1``
       lifts the default to ``2**28``.
   * - ``YASTN_MEMORY_OVERRIDE``
     - ``0``
     - ``1`` disables the packing-budget divisor, forces the per-block fusion
       loop, disables NNN append chunking, and disables CUDA cache releases.
   * - ``YASTN_SCALED_MERGE``
     - ``0``
     - ``1`` selects per-record weighted fusion
       (``transpose_and_merge_scaled``). It saves a full scaled source buffer,
       but is slower when there are many metadata records.
   * - ``YASTN_FUSE_SCATTER_CHUNK``
     - unset
     - Fusion scatter tile size. Unset selects an adaptive tile; ``0`` forces
       the per-block loop; a positive integer fixes the tile size.
   * - ``YASTN_FUSE_SCATTER_THRESH``
     - ``65536``
     - Blocks with at least this many elements use the per-block loop; smaller
       blocks use compact scatter/gather.
   * - ``YASTN_META_CUTENSOR``
     - ``AUTO``
     - Force the non-Abelian metadata matcher to ``CPU`` or ``GPU``.
   * - ``YASTN_META_CUTENSOR_GPU_MIN``
     - ``4096``
     - Minimum row count before the GPU metadata matcher is used.
   * - ``YASTN_NNN_APPEND_CHUNK_BYTES``
     - ``16 GiB``
     - Input chunk budget for NNN boundary-vector append.
   * - ``YASTN_OE_ALLOC_CONF``
     - unset
     - Per-worker PyTorch allocator settings for ``ncon``/unroll.
   * - ``YASTN_OE_CUDA_CACHE_RELEASE_LEVEL``
     - ``0``
     - Frequency (``0``--``3``) of ``empty_cache()`` calls inside the
       ``torch_cutensor`` contraction loop.
   * - ``YASTN_OE_OOM_RETRY``
     - ``0``
     - ``1`` retries a CUDA contraction once after ``empty_cache()`` on OOM
       (torch CUDA path only).
   * - ``YASTN_PROFILE``
     - ``0``
     - ``1`` enables NVTX profiling ranges when NVTX is available.
