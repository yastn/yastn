Algebra with YAST tensors
=========================

Basic algebra operations with symmetric tensors
-----------------------------------------------

Symmetric tensors can be added and multiplied by a scalar
through usual operations ``+``, ``-``, ``*``, ``/``.
You can also raise each element of tensor to some power using
standard power operation ``**``.

See examples: :ref:`examples/tensor/algebra:basic algebra operations`.

Simple element-wise operations
------------------------------

.. automethod:: yast.Tensor.__abs__ 
.. automethod:: yast.Tensor.real
.. automethod:: yast.Tensor.imag
.. automethod:: yast.Tensor.sqrt
.. automethod:: yast.Tensor.rsqrt
.. automethod:: yast.Tensor.reciprocal
.. automethod:: yast.Tensor.exp


.. autofunction:: yast.apxb


Tensor contractions
-------------------

Tensor contractions are the main building block of tensor network algorithms.
Functions below facilitate the computation of

	* `Trace`: :math:`B_{jl}= \sum_{i} T_{ijil}` or using Einstein's summation convention
	  for repeated indices :math:`B_{jl} = T_{ijil}`.
	* `Contractions`: in the usual form :math:`C_{abc} = A_{aijb} \times B_{cij}` and also
	  outer products :math:`M_{abkl} = A_{ak} \times B_{bl}`

or composition of such operations over several tensors.


See examples: :ref:`examples/tensor/algebra:tensor contractions`.

.. automethod:: yast.Tensor.__matmul__

.. autofunction:: yast.tensordot
.. autofunction:: yast.vdot
.. autofunction:: yast.trace
.. autofunction:: yast.einsum
.. autofunction:: yast.ncon


Transposition
-------------

See examples: :ref:`examples/tensor/algebra:transposition`.

.. automethod:: yast.Tensor.transpose
.. automethod:: yast.Tensor.move_leg
.. automethod:: yast.Tensor.moveaxis


Fusion of legs (reshaping)
--------------------------

Fusion of several vector spaces :math:`V_1,V_2,\ldots,V_n` creates a new vector space as direct product :math:`W=V_1 \otimes V_2 \otimes \ldots \otimes V_n`, which is then indexed by a single index of dimension :math:`\prod_i dim(V_i)`. The inverse operation can split the fused space into its original constituents.

For dense tensors, this operation corresponds to reshaping.

Fusion can be used to vary compression between (unfused) symmetric tensors with many small non-zero blocks and tensors with several fused spaces having just few, but large non-zero blocks.

See examples: :ref:`examples/tensor/algebra:fusion (reshaping)`.

.. automethod:: yast.Tensor.fuse_legs
.. automethod:: yast.Tensor.unfuse_legs
.. automethod:: yast.Tensor.add_leg
.. automethod:: yast.Tensor.remove_leg

Conjugation of symmetric tensors
--------------------------------

See examples: :ref:`examples/tensor/algebra:conjugation of symmetric tensors`.

.. automethod:: yast.Tensor.conj
.. automethod:: yast.Tensor.conj_blocks
.. automethod:: yast.Tensor.flip_signature


Tensor norms
------------

.. automodule:: yast.linalg
	:noindex:
	:members: norm


Spectral decompositions
-----------------------

.. automodule:: yast.linalg
	:noindex:
	:members: svd, svd_with_truncation, qr, eigh, eigh_with_truncation,
           	truncation_mask, truncation_mask_multiplets
