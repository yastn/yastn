Predefined YASTN tensors
========================

In YASTN there are a number of predefined bases for typical models. These can be used to generate a model of given physical definition. 
For more general case a custom operator basis class can be defined. 

Spin-1/2 and Pauli matrices
---------------------------

.. autoclass:: yastn.operators.Spin12
    :members: I, x, y, z, sz, sx, sy, sp, sm, vec_z, vec_x, vec_y, space


Spin-1
------

.. autoclass:: yastn.operators.Spin1
    :members: I, sz, sx, sy, sp, sm, vec_z, vec_x, vec_y, vec_s, g, space


Spinless fermions
-----------------

.. autoclass:: yastn.operators.SpinlessFermions
    :members: I, n, cp, c, vec_n, space


Spinful fermions
-----------------

.. autoclass:: yastn.operators.SpinfulFermions
    :members: I, n, cp, c, vec_n, space
