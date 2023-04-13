Decompositions of symmetric tensors
===================================

QR decompositions
-----------------

The function below takes tensor :code:`a` with 4 legs, decompose it using QR and contracts the resulting Q and R tensors back into :code:`a`.

.. literalinclude:: /../../tests/tensor/test_qr.py
   :pyobject: run_qr_combine


SVD decompositions and truncation
---------------------------------

.. literalinclude:: /../../tests/tensor/test_svd.py
   :pyobject: test_svd_truncate


scipy.sparse.linalg.eigs
------------------------

Calculate the dominant eigenvector of a transfer matrix by employing the Krylov-base eigs method available in SciPy.
Tensor operations can be similarly passed to other SciPy methods, though this is limited to the NumPy backend.

.. literalinclude:: /../../tests/tensor/test_eigs_scipy.py
   :pyobject: test_eigs_simple
