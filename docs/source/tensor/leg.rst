Create Leg (vector space)
=========================

Tensors are :ref:`multilinear maps from a product of vector spaces <theory/tensor/basics:tensors>`.
In YASTN, the **legs** of a tensor represent individual vector spaces.

The spaces of a :class:`yastn.Tensor` are characterized by the structure :class:`yastn.Leg`.

.. autoclass:: yastn.Leg
    :members: conj, tD, history
    :special-members: __getitem__
    :exclude-members: __init__, __new__

.. autofunction:: yastn.leg_product
.. autofunction:: yastn.undo_leg_product
.. autofunction:: yastn.gaussian_leg
.. autofunction:: yastn.legs_union
