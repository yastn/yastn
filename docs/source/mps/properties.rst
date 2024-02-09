Properties
==========

Geometry
--------

The number of sites of MPS/MPO :code:`psi` can be read as :code:`psi.N` or :code:`len(psi)`.
Iterating though sites is supported by

.. automethod:: yastn.tn.mps.MpsMpoOBC.sweep

Symmetry
--------

The symmetry of the MPS/MPO is inherited from the symmetry of
:meth:`yastn.Tensor<yastn.Tensor>`'s forming it. The symmetries
of those tensors, and other `config` settings, have to be matching
as they are contracted along the virtual dimension.
As such, the global MPS/MPO symmetry is a consequence of the symmetry of local tensors.
Config of MPS/MPO :code:`psi` can be accesed via property :code:`psi.config`.

Index convention
----------------

MPS is built from tensors with a single physical index and two virtual indices.
MPO tenors have an additional physical index. The number of physical indices
can be accessed through the property :code:`psi.nr_phys`.

The index convention for MPS/MPO tensors is: `0th` index corresponds to the left virtual space,
i.e, in the direction of the first MPS/MPO site. `1st`` index is physical (ket), and `2nd` index
corresponds to the right virtual space, i.e., in the direction of the last MPS/MPO site.
For MPO, the last `3rd`` index is also physical (bra). We typically assume the signatures
:math:`s=(-1, +1, +1)` for MPS tensors and :math:`s=(-1, +1, +1, -1)` for MPO tensors.

::

    # indices of the individual tensors in MPS
             ___
    (-1) 0--|___|--2 (+1)
              |
              1 (ket; +1)

    # indices of the individual tensors in MPO. The physical indices
    # are ordered consistently with the usual matrix notation,
    # i.e., O = \sum_{ij} O_ij |i><j|

              3 (bra; -1)
             _|_
    (-1) 0--|___|--2 (+1)
              |
              1 (ket; +1)


Physical and virtual spaces
---------------------------

In the case of MPS/MPO with no explicit symmetry (dense tensors), the virtual space :math:`V_{j,j+1}`
for :math:`(j,j+1)`` bond is :math:`\mathbb{R}^{D_{j,j+1}}` or :math:`\mathbb{C}^{D_{j,j+1}}`.
Typically, the maximal permitted dimension of these virtual spaces is the control parameter for matrix product representation.
For symmetric MPS/MPO the virtual space is a direct sum of simple spaces, dubbed *sectors*, labeled by symmetry charges *t*

.. math::
    V_{j,j+1} = \bigoplus_{t} V^t_{j,j+1},\quad \textrm{where}\quad V^t_{j,j+1} = \mathbb{R}^{D^t_{j,j+1}}\ \textrm{or}\ \mathbb{C}^{D^t_{j,j+1}}.

Typically, the effective control parameter is the maximal total dimension
:math:`D_{j,j+1}=\sum_t D^t_{j,j+1}`, appearing in some methods as a parameter :code:`D_total`.
A number of functions allow extracting this information directly from MPS/MPO:

.. automethod:: yastn.tn.mps.MpsMpoOBC.get_bond_dimensions
.. automethod:: yastn.tn.mps.MpsMpoOBC.get_bond_charges_dimensions
.. automethod:: yastn.tn.mps.MpsMpoOBC.get_virtual_legs
.. automethod:: yastn.tn.mps.MpsMpoOBC.get_physical_legs
