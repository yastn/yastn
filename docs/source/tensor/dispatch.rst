Dispatching contractions
========================

We employ dedicated methods for special contractions that appear in MPS and PEPS algorithms.
Employing those methods in the algorithms provides a dispatching mechanism to
utilize sparse tensor objects, e.g., :class:`yastn.tn.fpeps.DoublePepsTensor`.

.. autofunction:: yastn.Tensor._attach_01
.. autofunction:: yastn.Tensor._attach_12
.. autofunction:: yastn.Tensor._attach_23
.. autofunction:: yastn.Tensor._attach_30
