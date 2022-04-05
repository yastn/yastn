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


Canonical form
------------------------

Matrix-product state/operator can be brought to the canonical form using QR decomposition. In the 1D objects you can choose it to be put into left or right canonical version.

.. literalinclude:: /../../tests/mps/test_canonical.py

The other way to get a canonical form of the object is to use the SVD decomposition. As the SVD progress throuth 1D chain you can choose to keep Schmidt vectors of custom coefficient.

.. literalinclude:: /../../tests/mps/test_truncate_svd.py


Copying
------------------------

.. literalinclude:: /../../tests/mps/test_copy.py

