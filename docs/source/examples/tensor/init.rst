Basic tensor initialization and creation operations
===================================================

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
