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

.. automethod:: yast.Tensor.__mul__
.. automethod:: yast.Tensor.__pow__
.. automethod:: yast.Tensor.__truediv__

.. autofunction:: yast.Tensor.__add__
.. autofunction:: yast.Tensor.__sub__
.. autofunction:: yast.apxb

.. autofunction:: yast.Tensor.__lt__
.. autofunction:: yast.Tensor.__gt__
.. autofunction:: yast.Tensor.__le__
.. autofunction:: yast.Tensor.__ge__
.. autofunction:: yast.Tensor.bitwise_not


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
.. autofunction:: yast.broadcast
.. autofunction:: yast.apply_mask
.. autofunction:: yast.trace
.. autofunction:: yast.einsum
.. autofunction:: yast.ncon
.. autofunction:: yast.swap_gate


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
.. automethod:: yast.Tensor.drop_leg_history

Conjugation of symmetric tensors
--------------------------------

See examples: :ref:`examples/tensor/algebra:conjugation of symmetric tensors`.

.. automethod:: yast.Tensor.conj
.. automethod:: yast.Tensor.conj_blocks
.. automethod:: yast.Tensor.flip_signature
.. automethod:: yast.Tensor.flip_charges


Tensor norms
------------

.. autofunction:: yast.linalg.norm


Spectral decompositions and truncation
--------------------------------------

.. autofunction:: yast.linalg.svd 
.. autofunction:: yast.linalg.svd_with_truncation
.. autofunction:: yast.linalg.qr
.. autofunction:: yast.linalg.eigh
.. autofunction:: yast.linalg.eigh_with_truncation
.. autofunction:: yast.linalg.truncation_mask
.. autofunction:: yast.linalg.truncation_mask_multiplets
.. autofunction:: yast.linalg.entropy
