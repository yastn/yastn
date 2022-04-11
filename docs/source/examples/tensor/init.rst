Basic tensor initialization and creation operations
===================================================


Configuration of the symmetries
-------------------------------------------

.. The matrix product `yamps.Mps` is build using tensors `yast.Tensor`. The symmetry of the tensors determines the symmetry of the full matrix product.
.. The `yamps.Mps` has to be initialised using particular symmetries.
.. #configuration of the symmetries, see test/configs, comment on the default flags /is that dictionary/, 


Configuration of backend
-------------------------------------------

.. backed is should be added in the configuration, 


Create tensors from scratch
---------------------------

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: TestSyntaxTensorCreation.test_syntax_tensor_creation_operations

Create empty tensor and fill it block by block
----------------------------------------------

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: TestSyntaxTensorCreation.test_syntax_create_empty_tensor_and_fill


Clone, detach or copy tensors
-----------------------------

.. literalinclude:: /../../tests/tensor/test_autograd.py
   :pyobject: TestSyntaxAutograd.test_clone_copy


Serialization of symmetric tensors
==================================

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: TestSyntaxTensorExportImport.test_syntax_tensor_export_import_operations
