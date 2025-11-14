Creating PEPS
=============

Initializing empty PEPS
-----------------------

PEPS is an instance of a class :class:`yastn.tn.fpeps.Peps`, utilizing a lattice geometry setup
(e.g., SquareLattice or CheckerboardLattice). Each unique lattice site is associated with a tensor,
following the layout specified by the lattice geometry.

::

      # leg ordering of PEPS tensors is
      # top, left, bottom, right, physical

                  top `0th`
                     ╲
                      ╲
                  ┌────┴────┐
    left `1st` ───┤ A_{x,y} ├─── right `3rd`
                  └──┬────┬─┘
                     |     ╲
                     |      ╲
                phys `4th`  bottom `2nd`

.. autoclass:: yastn.tn.fpeps.Peps
    :members: copy, clone, shallow_copy, save_to_dict, transfer_mpo


Initializing product PEPS
-------------------------

.. autofunction:: yastn.tn.fpeps.product_peps

See examples at :ref:`quickstart/kibble_zurek:Kibble-Zurek quench in 2D transverse-field Ising model`.


Import and export PEPS
----------------------

PEPS instances can be saved as Python dict after serialization by
:meth:`yastn.tn.fpeps.Peps.to_dict`. This enables saving the PEPS structure,
lattice geometry, and tensor data for storage or transfer. A PEPS object can later be
deserialized back using :meth:`yastn.tn.fpeps.Peps.from_dict` or a general :meth:`yastn.from_dict`.

.. autofunction:: yastn.tn.fpeps.Peps.to_dict
.. autofunction:: yastn.tn.fpeps.Peps.from_dict

Finite (small) PEPS can be contracted into a tensor.

.. automethod:: yastn.tn.fpeps.Peps.to_tensor


Double-layer PEPS
-----------------

Calculation of expectation values of interests requires contraction of PEPS with its conjugate.
This amounts to the contraction of the PEPS network composed of reduced rank-4 tensors :math:`a`, which is
obtained by tracing over the physical index in tensors :math:`A` and it's conjugate :math:`A^{\dagger}`.
It is supported by a special class :class:`yastn.tn.fpeps.Peps2Layers`, that outputs tensor :math:`a`
in the form of :class:`yastn.tn.fpeps.DoublePepsTensor` while accessed with :code:`[]` for a given site.

.. autoclass:: yastn.tn.fpeps.Peps2Layers


Double PEPS Tensor
------------------

The auxiliary class allows treating top and bottom PEPS tensors---to be contracted along
physical dimensions---as a single tensor of rank-4 for various operations.
It provides a dispatching mechanism for efficient contraction in construction of enlarge corners in CTMRG or boundary MPS algorithms.

::

                        t't
                         ╲╲
                         ╱ ╲
                        ╱   ╲                                     t't
                       ╱ ┌───┴───┐                                 ╲╲
                   /──┼──┤   A   ├─────\                            ╲╲
                  ╱   |  └─┬────┬┘      ╲                         ┌──┴┴───┐
            l ───/    |    |     ╲       \─── r     ════     l ───┤   a   ├─── r
            l'───\    |    |   ___╲      /─── r'    ════     l'───┤       ├─── r'
                  ╲   |  ┌─┴──┴──┐╲╲    ╱                         └──┬┬───┘
                   \──┼──┤   A'  ├─╲╲──/                              ╲╲
                      ╲  └─┬─────┘  ╲╲                                 ╲╲
                       ╲  ╱          ╲╲                                b'b
                        ‾‾            ╲╲
                                      b'b


Note that in the following diagram the virtual legs of the peps tensor are labelled by
:math:`t` (top), :math:`l` (left), :math:`b` (bottom), and :math:`r` (right) in an anticlockwise fashion.
For the conjugate tensor, similarly, they are labelled by :math:`{t'}`, :math:`{l'}`, :math:`{b'}` and :math:`{r'}`.
Swap gates are placed where the legs cross. This gives a simple structure for the contracted tensors on
the :math:`2D` lattice, respecting the global fermionic order.

.. autoclass:: yastn.tn.fpeps.DoublePepsTensor
    :members: ndim, get_shape, get_legs, transpose, conj, clone, copy, fuse_layers, tensordot
