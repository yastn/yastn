Matrix product algebra
=========================

Copying an object
---------------------------------

Simple assignment of the element under a new name does not create an independent copy. Therefore all changes in the original version will be reflected in the new variable.
To make an independent copy you should use :code:`mp_new = mp_old.copy()`

.. autoclass:: yamps.Mps
	:noindex:
	:members: copy, clone


See examples here :ref:`examples/mps/mps:Copying`.

.. todo:: what about shallow and deep copy? should I delete copy/clone?


Addition
---------------------------------

In order to make a direct sum of two matrix products make sure they have the same symmetries and length. Only then they can be added to each other alond distinguished axis defined by `common_legs`.
The addition of two matrix products can be done using `apxb()`, where you can additionally specify the prefactor which will be multiply to the second Mps you give.

.. automodule:: yamps
	:members: apxb

The addition of any number of matrix products can be done using `add()`, where you can additionally specify the list of prefactors which will be multiplied to each Mps.

.. automodule:: yamps
	:members: add

See examples here :ref:`examples/mps/mps:Addition`.

.. todo:: check tests for symmetric Mps


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
