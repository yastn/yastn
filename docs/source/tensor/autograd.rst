Autograd support
================

The algebra of symmetric tensors is composed from usual tensor algebra now 
operating over many blocks (dense tensors). Hence, by composition, 
the operations on symmetric tensors can be straightforwardly differentiated 
if the individual operations on dense tensors support autograd.    

YAST supports autograd through selected backends which provide 
this capability for dense tensor algebra, for example PyTorch backend. 

You can activate autograd on YAST tensor

.. automethod:: yast.Tensor.requires_grad_