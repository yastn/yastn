Predefined YASTN tensors
========================

In YASTN, we predefined several classes containing sets of standard operators,
which can be employed in simulations of many typical physical models.

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
