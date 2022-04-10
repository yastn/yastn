Examples for `yamps` module
=====================================

Filling matrix product with tensors
----------------------------

The `yamps.Mps` can be filled by hand by directly assigning `yast.Tensor` to `A`, e.g. :code:`psi.A[3] = <example tensor>`, where `psi` is an example `yamps.Mps` object.

.. literalinclude:: /../../tests/mps/test_initialization.py
        :pyobject: test_assign_block

For symmetric `yamps.Mps` we havee to make sure that the blocks throughout the chain have consistent bond dimension.
That means that the blocks of one tensor define the structure of its neighbours.

.. literalinclude:: /../../tests/mps/ops_dense.py
        :pyobject: mps_random

.. literalinclude:: /../../tests/mps/ops_Z2.py
        :pyobject: mps_random

.. literalinclude:: /../../tests/mps/ops_U1.py
        :pyobject: mps_random


Automatically generated matrix product
----------------------------

XX Hamiltonian generated automatically for the case with no symmetries imposed. 

.. literalinclude:: /../../tests/mps/ops_dense.py
        :pyobject: mpo_gen_XX

XX Hamiltonian generated automatically for the case with imposed U(1) symmetry for the tensors. 

.. literalinclude:: /../../tests/mps/ops_U1.py
        :pyobject: mpo_gen_XX



.. FOR ALGEBRA 

Copying
------------------------

.. literalinclude:: /../../tests/mps/test_copy.py


Addition
------------------------

.. literalinclude:: /../../tests/mps/test_addition.py


Multiplication
------------------------

.. literalinclude:: /../../tests/mps/test_multiplication.py


.. QR and SVD

Canonical form by QR decomposition
---------------------------------------

.. literalinclude:: /../../tests/mps/test_canonical.py


Canonical form by SVD decomposition
----------------------------------------

.. literalinclude:: /../../tests/mps/test_truncate_svd.py


.. outside world
Save and load
----------------------------------------

.. literalinclude:: /../../tests/mps/test_save_load.py


.. algorithms
DMRG
-----

.. literalinclude:: /../../tests/mps/test_dmrg.py

TDVP
-----

.. literalinclude:: /../../tests/mps/test_tdvp.py
