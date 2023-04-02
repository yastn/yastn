Create Leg (vector space)
=========================

Tensor are :ref:`multilinear maps from product of vector spaces <theory/tensor/basics:tensors>`.
In YASTN, the `legs` of the tensor represent individual vector spaces.

To create a Leg, use

.. autoclass:: yastn.Leg
    :members: conj, tD, history
    :special-members: __getitem__
    :exclude-members: __init__, __new__

.. autofunction:: yastn.random_leg
.. autofunction:: yastn.leg_union
