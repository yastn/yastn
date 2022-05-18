Matrix product algebra
=========================

Copying an object
---------------------------------

Simple assignment of the element under a new name does not create an independent copy. Therefore all changes in the original version will be reflected in the new variable.
To make an independent copy you should use :code:`mp_new = mp_old.copy()`

.. autoclass:: yamps.Mps
	:noindex:
	:exclude-members: __init__, __new__
	:members: copy, clone

See examples here :ref:`examples/mps/mps:Copying`.

.. todo:: what about shallow and deep copy? should I delete copy/clone?


Addition
---------------------------------

In order to make a direct sum of two matrix products make sure they have the same symmetries and length. Only then they can be added to each other alond distinguished axis defined by `common_legs`.
The addition of two matrix products can be done using `apxb()`, where you can additionally specify the prefactor which will be multiply to the second Mps you give.

.. automodule:: yamps
	:noindex:
	:members: apxb

The addition of any number of matrix products can be done using `add()`, where you can additionally specify the list of prefactors which will be multiplied to each Mps.

.. automodule:: yamps
	:noindex:
	:members: add

See examples here :ref:`examples/mps/mps:Addition`.

.. todo:: check tests for symmetric Mps


Multiplication
---------------------------------

In order to multiply two Mps-s you need to know interpretation of their legs. That includes which legs of their individual tensors should be contracted and which lie along the code for the enw Mps and thus should be fused to a new leg. 
Additionally, you can multiply the product of Mps-s by setting a prefactor to by any number.

.. automodule:: yamps
	:noindex:
	:members: x_a_times_b

See examples here :ref:`examples/mps/mps:Multiplication`.

.. todo:: do I really need additional test for Mps with symmetries?


Canonical form
---------------------------------


The cannonical form of the matrix product can be obtaining by subesqent QR or SVD decomposition. In the 1D objects you can choose it to be put into left or right canonical version depending of the parameter `to` to be `last` or `first`.

QR decomposision exhibits better performence while procedure beeing made exactly.

.. autoclass:: yamps.Mps
	:noindex:
	:exclude-members: __init__, __new__
	:members: canonize_sweep

See examples: :ref:`examples/mps/mps:Canonical form by QR decomposition`.

On the other hand singular values decomposision (SVD) additionally allows for truncating Schmidt vectors exceeding set truncation tolerance or maximal bond dimension. The truncation happens on each site of the Mps after persorming SVD of the tensor.

.. autoclass:: yamps.Mps
	:noindex:
	:exclude-members: __init__, __new__
	:members: truncate_sweep

See examples: :ref:`examples/mps/mps:Canonical form by SVD decomposition`.

.. todo:: this probably should be changed after splitting SVD to pure svd nd truncation
