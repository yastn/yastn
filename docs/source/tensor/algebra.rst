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

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: __abs__, real, imag, sqrt, rsqrt, reciprocal, exp


.. automodule::	yast
	:noindex:
	:members: apxb


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

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: __matmul__

.. automodule:: yast
	:noindex:
	:members: tensordot, vdot, trace, einsum, ncon


Transposition
-------------

See examples: :ref:`examples/tensor/algebra:transposition`.

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: transpose, move_leg, moveaxis


Fusion of legs (reshaping)
--------------------------

Fusion of several vector spaces :math:`V_1,V_2,\ldots,V_n` creates a new vector space as a direct sum :math:`W=V_1 \oplus V_2 \oplus \ldots \oplus V_n`, which is then indexed by a single index of dimension :math:`\sum_i dim(V_i)`. The inverse operation can split the fused space into its original constituents.

For dense tensors, this operation corresponds to reshaping.

Fusion can be used to vary compression between (unfused) symmetric tensors with many small non-zero blocks and tensors with several fused spaces having just few, but large non-zero blocks.

See examples: :ref:`examples/tensor/algebra:fusion (reshaping)`.

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: fuse_legs, unfuse_legs


Conjugation of symmetric tensors
--------------------------------

See examples: :ref:`examples/tensor/algebra:conjugation of symmetric tensors`.

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: conj, conj_blocks, flip_signature


Tensor norms
------------

.. automodule:: yast.linalg
	:noindex:
	:members: norm

Spectral decompositions
-----------------------

.. automodule:: yast.linalg
	:noindex:
	:members: svd, svd_with_truncation, qr, eigh, eigh_with_truncation