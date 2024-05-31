PEPS and its initialization
===========================

Initializing empty PEPS
-----------------------

PEPS is an instance of a class :class:`yastn.tn.fpeps.Peps`,
which includes information about lattice geometry associating each unique site with a tensor.

.. autoclass:: yastn.tn.fpeps.Peps
    :members: save_to_dict, copy, clone, transfer_mpo

Initializing PEPS lattice
-------------------------

.. autofunction:: yastn.tn.fpeps.product_peps

Examples are given in
:ref:`Purification<examples/fpeps/ntu:Thermal expectation value of spinful fermi sea>`,

.. autofunction:: yastn.tn.fpeps.load_from_dict


Double-layer PEPS
-----------------

A special class supports 2-layer PEPS.
It appears in calculating a product of two PEPS states :math:`\langle \psi | \phi \rangle`
It outputs :class:`yastn.tn.fpeps.DoublePepsTensor` while getting a tensor with [].
It is an auxiliary class, that is initialized by environments during contraction of PEPS with physical legs.

.. autoclass:: yastn.tn.fpeps.Peps2Layers


Double PEPS Tensor
------------------

The auxiliary class which allows treating top and bottom PEPS tensors, to be contracted along physical dimensions,
as a single tensor for various operations. The attribute transpose indicates (virtual) transposition of the tensor.

The key functions `_attach_01`, `_attach_12`, `_attach_23`, and `_attach_30` append the top and bottom
PEPS tensor to a given four-legged enlarge-corner tensor at a specific position, respectively,
top-right, top-left, bottom-right, and bottom-left,
These functions support CTMRG contraction of double-layer PEPS,
or integration with MPS methods, e.g., for boundary-MPS contraction.

.. autoclass:: yastn.tn.fpeps.DoublePepsTensor
    :members: ndim, get_shape, get_legs, transpose, conj, clone, copy, _attach_01, _attach_12, _attach_23, _attach_30, fuse_layers
