Algebra with YAST tensors
=========================

Basic algebra operations with symmetric tensors
-----------------------------------------------

Symmetric tensors can be added and multiplied by a scalar
through usual operations ``+``, ``-``, ``*``, ``/``.
You can also raise each element of tensor to some power using 
standard power operation ``**``. 


Simple element-wise operations
------------------------------

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: absolute, real, imag, sqrt, rsqrt, reciprocal, exp


.. automodule::	yast
	:noindex:
	:members: apxb


Transposition
-------------

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: transpose, moveaxis


Conjugation of symmetric tensors
--------------------------------

.. autoclass:: yast.Tensor
	:exclude-members: __init__, __new__
	:noindex:
	:members: conj, conj_blocks, flip_signature