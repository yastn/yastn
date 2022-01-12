Algebra with YAST tensors
=========================

Basic algebra operations with symmetric tensors
-----------------------------------------------

Symmetric tensors can be added and multiplied by a scalar
through usual operations ``+``, ``-``, ``*``, ``/``.
You can also raise each element of tensor to some power using 
standard power operation ``**``. 

See examples: :ref:`examples/algebra:basic algebra operations`.

Simple element-wise operations
------------------------------

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: __abs__, real, imag, sqrt, rsqrt, reciprocal, exp


.. automodule::	yast
	:noindex:
	:members: apxb


Transposition
-------------

See examples: :ref:`examples/algebra:transposition`.

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: transpose, move_leg, moveaxis


Fusion of legs (reshaping)
--------------------------

See examples: :ref:`examples/algebra:fusion`.

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: fuse_legs, unfuse_legs


Conjugation of symmetric tensors
--------------------------------

See examples: :ref:`examples/algebra:conjugation of symmetric tensors`.

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
	:members: svd, svd_lowrank, qr, eigh