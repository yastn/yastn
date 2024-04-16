Initializing PEPS
=================

PEPS is an instance of :class:`yastn.tn.fpeps.Peps` which assumes information about geometry the lattice through :class:`yastn.tn.fpeps.SquareLattice`. 
We have control over :code:`dims` specifying dimensions of the lattice and :code:`boundary`, which could be `infinite`,  
open boundary conditions `obc`, or with cylindrical geometry `geometry`. It supports two-dimensional tensor networks, defined by dict of rank-5 and rank-6 :class:`yastn.Tensor`-s for PEPS
with just the physical index and with an additional ancillary index, respectively.

For purification, the state is initialized as a maximally entangled state between the physical and auxilliary degrees of freedom:

.. autofunction:: yastn.tn.fpeps.product_peps

More details can be found here:
:ref:`Purification<theory/fpeps/purification:Purification>`

Examples are given in
:ref:`Purification<examples/fpeps/ntu:Thermal expectation value of spinful fermi sea>`,

The state can also be initialized in a product state with a user-defined dictionary containing the lattice sites
as the keys and the basis vector states (yastn.Tensor) as the values. If only a single yastn.Tensor is provided,
it gets repeated across the lattice.

.. autofunction:: yastn.tn.fpeps.product_peps
