Accessing YASTN tensors
=======================

Direct access to blocks
-----------------------

Blocks of a YASTN tensor can be accessed in the same way as a
standard dictionary. See the example at :ref:`examples/tensor/init:Direct access to blocks`.

.. automethod:: yastn.Tensor.__getitem__

Converting to dense tensors, scalars
------------------------------------

.. automethod:: yastn.Tensor.to_dense
.. automethod:: yastn.Tensor.to_nonsymmetric
.. automethod:: yastn.Tensor.to_numpy
.. automethod:: yastn.Tensor.to_raw_tensor
.. automethod:: yastn.Tensor.to_number
.. automethod:: yastn.Tensor.item
