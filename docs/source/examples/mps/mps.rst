Examples for mps
=====================================

Canonical form
------------------------

Matrix-product state/operator can be brought to the canonical form using QR decomposition. In the 1D objects you can choose it to be put into left or right canonical version.

.. literalinclude:: /../../tests/mps/test_canonical.py

The other way to get a canonical form of the object is to use the SVD decomposition. As the SVD progress throuth 1D chain you can choose to keep Schmidt vectors of custom coefficient.

.. literalinclude:: /../../tests/mps/test_truncate_svd.py


Copying
------------------------

.. literalinclude:: /../../tests/mps/test_copy.py

