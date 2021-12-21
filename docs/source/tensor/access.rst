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

.. autoclass:: yast.Tensor
	:noindex:
	:exclude-members: __init__, __new__
	:members: to_dense, to_nonsymmetric, to_numpy, to_raw_tensor, 
		to_number, item
