Algebra with YASTN tensors
==========================

Basic algebra operations with symmetric tensors
-----------------------------------------------

See examples at :ref:`examples/tensor/algebra:basic algebra operations`.

Symmetric tensors can be added and multiplied by a scalar
through usual operations ``+``, ``-``, ``*``, ``/``.
Element-wise raising to a power is done by the standard power operation ``**``.

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
.. autofunction:: yastn.linear_combination

.. autofunction:: yastn.Tensor.__lt__
.. autofunction:: yastn.Tensor.__gt__
.. autofunction:: yastn.Tensor.__le__
.. autofunction:: yastn.Tensor.__ge__
.. autofunction:: yastn.Tensor.bitwise_not

.. autofunction:: yastn.allclose


Tensor contractions
-------------------

See examples at :ref:`examples/tensor/algebra:tensor contractions`.

Tensor contractions are the main building block of tensor network algorithms.
Functions below facilitate the computation of

	* `Trace`: :math:`B_{jl}= \sum_{i} T_{ijil}` or using Einstein's summation convention
	  for repeated indices :math:`B_{jl} = T_{ijil}`.
	* `Contractions`: in the usual form :math:`C_{abc} = A_{aijb}{\times}B_{cij}` and also
	  outer products :math:`M_{abkl} = A_{ak}{\times}B_{bl}`

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


Transposition
-------------

See examples at :ref:`examples/tensor/algebra:transposition`.

.. autofunction:: yastn.transpose
.. autoproperty:: yastn.Tensor.T
.. autofunction:: yastn.moveaxis


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
.. autofunction:: yastn.linalg.eigh
.. autofunction:: yastn.linalg.eigh_with_truncation
.. autofunction:: yastn.linalg.truncation_mask
.. autofunction:: yastn.linalg.truncation_mask_multiplets
.. autofunction:: yastn.linalg.entropy

.. _tensor-aux:

Auxliary
--------

Methods called by :doc:`Krylov-based algorithms</tensor/krylov>`.

.. automethod:: yastn.Tensor.expand_krylov_space
.. automethod:: yastn.Tensor.linear_combination
