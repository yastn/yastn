Algebra with MPS/MPO objects
=========================

Copying the object 
---------------------------------

If you want to use the object as an independent element you shuold creates a copy of it and doesn't depend on the original version any more.

See examples: :ref:`examples/init:Copying`.

.. autoclass:: yast.Mps
	:noindex:
	:members: copy, clone


Addition
---------------------------------

#example for addition, checks for nr_phys 

See examples: :ref:`examples/init:clone, detach or copy tensors`.

.. autoclass:: yast.Mps
	:noindex:
	:members: add, apxb


Multiplication
---------------------------------

#multiplication by a number and multiplication by an operator.? + test

See examples: :ref:`examples/init:clone, detach or copy tensors`.

.. autoclass:: yast.Tensor
	:noindex:
	:exclude-members: __init__, __new__
	:members: x_a_times_b


Canonical form
---------------------------------

The MPS/MPO can be put in the left-/right-canonical form using QR or SVD decomposition. The SVD decomposition allows additional truncation up to the Schmidt vectors up to custom weight.

See examples: :ref:`examples/mps:canonical form`.

.. autoclass:: yast.Mps
	:noindex:
	:members: truncate_sweep, canonical_sweep
