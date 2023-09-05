Initializing PEPS
=================

PEPS are instances of class :class:`yastn.tn.fpeps.Peps`. It is inherited from :class:`yastn.tn.fpeps.Lattice`
which has the following information about the lattice :code:`lattice`: can be 'checkerboard' or 'rectangle',
:code:`dims` specifying dimensions of the lattice and :code:`boundary`, which could be `infinite` or `finite`.
It supports two-dimensional tensor networks, defined by dict of rank-5 and rank-6 :class:`yastn.Tensor`-s for PEPS
with just the physical index and with an additional ancillary index, respectively.

For purification, the state is initialized as a maximally entangled state between the physical and auxilliary degrees of freedom:

.. autofunction:: yastn.tn.fpeps._initialization_peps.initialize_peps_purification

More details can be found here:
:ref:`Purification<theory/fpeps/purification:Purification>`

Examples are given in
:ref:`Purification<examples/fpeps/ntu:Thermal expectation value of spinless fermi sea>`,
:ref:`Purification<examples/fpeps/ntu:Thermal expectation value of spinful fermi sea>`,

