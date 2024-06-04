Creating PEPS
=============

Initializing empty PEPS
-----------------------

PEPS is an instance of a class :class:`yastn.tn.fpeps.Peps`.
It extends lattice geometry, associating each unique lattice site with a tensor.

.. autoclass:: yastn.tn.fpeps.Peps
    :members: copy, clone, save_to_dict, transfer_mpo

Initializing product PEPS
-------------------------

.. autofunction:: yastn.tn.fpeps.product_peps

Examples are given in
:ref:`Purification<examples/fpeps/ntu:Thermal expectation value of spinful fermi sea>`.


Import and export PEPS
----------------------

PEPS can be saved as Python dict after serialization by
:meth:`yastn.tn.fpeps.Peps.save_to_dict` and deserialized back using :meth:`yastn.tn.fpeps.load_from_dict`.

.. autofunction:: yastn.tn.fpeps.load_from_dict


Double-layer PEPS
-----------------

A special class supports 2-layer PEPS, which appears in contraction of a product of two PEPS states :math:`\langle \psi | \phi \rangle`.
Outputs :class:`yastn.tn.fpeps.DoublePepsTensor` for a given site while accessed with :code:`[]`.
It is an auxiliary class, that is initialized by environments during contraction of PEPS with physical legs.

.. autoclass:: yastn.tn.fpeps.Peps2Layers



Calculation of expectation values of interests requires contraction of PEPS with its conjugate.
This amounts to contraction of PEPS network composed of reduced tensor :math:`a` which is
obtained by tracing over the physical index in tensors :math:`A` and it's conjugate :math:`A^{\dagger}`.
In YASTN, this is supported by :class:`yastn.tn.fpeps.DoublePepsTensor`.
Note that in the following diagram the virtual legs of the peps tensor are labelled by
:math:`t` (top), :math:`l` (left), :math:`b` (bottom), and :math:`r` (right) in an anticlockwise fashion.
For the conjugate tensor, similarly, they are labelled by :math:`{t'}`, :math:`{l'}`, :math:`{b'}` and :math:`{r'}`.
Swap gates are placed where the legs cross. This gives a simple structure for the contracted tensors on
the :math:`2D` lattice, respecting the global fermionic order.

::

                      t' t
                       \ \
                        | \
                       /  _\_____
                      /  |       |                            t' t
                   /--|--|   A   |-------\                     \ \
                  /   |  |_______|        \                   __\_\__
             l --/    |    |      \        \-- r         l --|       |-- r
                      |    |    __ \               ===       |   a   |
             l'--\    |   _|___|_ \ \      /-- r'        l'--|_______|-- r'
                  \   |  |       | \ \    /                      \ \
                   \--|--|   A'  |--\-\--/                        \ \
                      \  |_______|   \ \                          b' b
                       \         \    \ \
                        \________/     \ \
                                       b' b




Double PEPS Tensor
------------------

The auxiliary class which allows treating top and bottom PEPS tensors---to be contracted along
physical dimensions---as a single tensor of rank-4 for various operations.

It provides a dispatching mechanism for efficient contraction in construction of enlarge corners in CTMRG or boundary MPS algorithms.
Equivalent operations in :code:`yastn.Tensor` are :ref:`here<tensor/dispatch:Dispatching contractions>`.

.. autoclass:: yastn.tn.fpeps.DoublePepsTensor
    :members: ndim, get_shape, get_legs, transpose, conj, clone, copy, _attach_01, _attach_12, _attach_23, _attach_30, fuse_layers
