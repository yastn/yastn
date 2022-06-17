Create Leg (vector space)
=========================

Tensor are multilinear maps from product of vector spaces.
In YAST, the `legs` of the tensor represent individual vector spaces.

To create a Leg, use

.. autoclass:: yast.Leg
    :members: conj, tD
    :special-members: __getitem__
    :exclude-members: __init__, __new__
