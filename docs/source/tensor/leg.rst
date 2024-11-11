Create Leg (vector space)
=========================

Tensors are :ref:`multilinear maps from product of vector spaces <theory/tensor/basics:tensors>`.
In YASTN, the ''legs`` of the tensor represent individual vector spaces.

The spaces of the :class:`yastn.Tensor` are characterized by a structure :class:`yastn.Leg`.

.. autoclass:: yastn.Leg
    :members: conj, tD, history
    :special-members: __getitem__
    :exclude-members: __init__, __new__

.. autofunction:: yastn.leg_product
.. autofunction:: yastn.leg_undo_product
.. autofunction:: yastn.random_leg
.. autofunction:: yastn.leg_union
