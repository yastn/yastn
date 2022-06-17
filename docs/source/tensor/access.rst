Accessing YAST tensors
======================

Direct access to blocks
-----------------------

Blocks of YAST tensor can be simply accessed as ``dict`` i.e. with ``[]`` operator.

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: TestSyntaxBlockAccess.test_syntax_block_access

.. automethod:: yast.Tensor.__getitem__ 

Converting to dense tensors, scalars
------------------------------------

.. automethod:: yast.Tensor.to_dense
.. automethod:: yast.Tensor.to_nonsymmetric
.. automethod:: yast.Tensor.to_numpy
.. automethod:: yast.Tensor.to_raw_tensor
.. automethod:: yast.Tensor.to_number
.. automethod:: yast.Tensor.item
