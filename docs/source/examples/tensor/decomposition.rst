Decompositions of symmetric tensors
===================================

The examples below demonstrate common decomposition routines for symmetric tensors.

.. code-block:: python

   import yastn
   import pytest
   config_kwargs = {"backend": "np"}


SVD decompositions and truncation
---------------------------------

The example below demonstrates SVD-based decomposition and truncation of a symmetric tensor.

.. literalinclude:: /../../tests/tensor/test_svd.py
   :pyobject: test_svd_truncate


QR decompositions
-----------------

The example below takes a tensor :code:`a` with four legs, decomposes it using QR, and contracts the resulting Q and R tensors back into :code:`a`.

.. literalinclude:: /../../tests/tensor/test_qr.py
   :pyobject: run_qr_combine


Combining with scipy.sparse.linalg.eigs
---------------------------------------

Calculate the dominant eigenvector of a transfer matrix by employing the Krylov-based eigs method available in SciPy.
Tensor operations can be passed to other SciPy methods in a similar way, although this is currently limited to the NumPy backend.

.. code-block:: python

   import numpy as np
   from scipy.sparse.linalg import eigs, LinearOperator

.. literalinclude:: /../../tests/tensor/test_eigs_scipy.py
   :pyobject: test_eigs_simple
