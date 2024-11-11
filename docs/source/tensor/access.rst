Accessing YASTN tensors
=======================

Direct access to blocks
-----------------------

Blocks of YASTN tensor can be simply accessed in the same way as
standard dictionary. For example,

.. code-block:: python

   import yastn
   config_kwargs = {"backend": "np"}

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :language: python
   :pyobject: test_syntax_block_access

.. automethod:: yastn.Tensor.__getitem__

Converting to dense tensors, scalars
------------------------------------

.. automethod:: yastn.Tensor.to_dense
.. automethod:: yastn.Tensor.to_nonsymmetric
.. automethod:: yastn.Tensor.to_numpy
.. automethod:: yastn.Tensor.to_raw_tensor
.. automethod:: yastn.Tensor.to_number
.. automethod:: yastn.Tensor.item
