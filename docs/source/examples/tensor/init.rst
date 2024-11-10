Basic tensor initialization and creation operations
===================================================

In all examples, start with importing the repository and
setting configuration options to employ a numpy backend (a default option).
Other configuration options will be taken as :ref:`default. <tensor/configuration:YASTN configuration>`

.. code-block:: python

   import yastn
   config_kwargs = {"backend": "np"}


Create tensors from scratch
---------------------------

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: test_syntax_tensor_creation_operations

Create empty tensor and fill it block by block
----------------------------------------------

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: test_syntax_create_empty_tensor_and_fill


Clone, detach or copy tensors
-----------------------------

.. literalinclude:: /../../tests/tensor/test_autograd.py
   :pyobject: test_clone_copy


Serialization of symmetric tensors
==================================

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: test_syntax_tensor_export_import_operations
