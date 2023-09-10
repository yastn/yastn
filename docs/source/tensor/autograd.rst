Autograd support
================

The algebra of symmetric tensors is composed from usual tensor algebra now
operating over many blocks (dense tensors). Hence, by composition,
the operations on symmetric tensors can be straightforwardly differentiated
if the individual operations on dense tensors support autograd.

YASTN supports autograd through selected backends which provide
this capability for dense tensor algebra, for example PyTorch backend.

You can activate autograd on YASTN tensor

.. automethod:: yastn.Tensor.requires_grad_


The operations on tensor are then recorded for later differentiation.

.. literalinclude:: /../../tests/tensor/test_autograd.py
   :pyobject: TestSyntaxAutograd.test_requires_grad