Properties
==========

Symmetry
--------

The symmetry of the MPS/MPO is inherited from the symmetry
of :meth:`yastn.Tensor<yastn.Tensor>`'s forming the matrix product.
The symmetries of those tensors have to be the same, as they are contracted along the virtual dimension,
implying that the global MPS/MPO symmetry is a consequence of the local symmetry of :meth:`yastn.Tensor<yastn.Tensor>`.


Index convention
----------------

MPS is built from tensors with a single physical index and two virtual indices.
MPO tenors have an additional physical index. The number of physical indices can be accessed through the property :code:`nr_phys`.

The index convention for MPS/MPO tensors is `0th` index corresponds to the left virtual space, i.e, in the direction of the first MPS/MPO site.
`1st`` index is physical, and `2nd` index corresponds to the right virtual space, i.e. in the direction of the last MPS/MPO site.
For MPO, the last `3rd`` index is also physical.
We typically assume the signatures :math:`s=(-1, +1, +1)` for MPS tensors and :math:`s=(-1, +1, +1, -1)` for MPO tensors.

::

    # indices of the individual tensors in MPS
             ___
    (-1) 0--|___|--2 (+1)
              |
              1 (ket; +1)

    # indices of the individual tensors in MPO. The physical indices
    # are ordered as for usual quantum-mechanical operator,
    # i.e., O = \sum_{ij} O_ij |i><j|

              3 (bra; -1)
             _|_
    (-1) 0--|___|--2 (+1)
              |
              1 (ket; +1)


Physical and virtual spaces
---------------------------

In the case of MPS/MPO with no explicit symmetry, the virtual space :math:`V_{j,j+1}`
for *(j,j+1)* bond is simple :math:`\mathbb{R}^{D_{j,j+1}}`
or :math:`\mathbb{C}^{D_{j,j+1}}` space.
Typically, the maximal permited dimension of these virtual spaces is the control parameter for matrix product representation.
For symmetric MPS/MPO the virtual space is a direct sum of simple spaces, dubbed *sectors*, labeled by symmetry charges *t*

.. math::
    V_{j,j+1} = \oplus_{t} V^t_{j,j+1},\quad \textrm{where}\quad V^t_{j,j+1} = \mathbb{R}^{D^t_{j,j+1}}\ \textrm{or}\ \mathbb{C}^{D^t_{j,j+1}}.

Typically, the effective control parameter is the maximal total dimension
:math:`D_{j,j+1}=\sum_t D^t_{j,j+1}`, appearing in some methods as parameter :code:`D_total`.
To get the profile of the total bond dimensions along MPS/MPO :code:`psi` call :code:`psi.get_bond_dimensions()`,
and for a more detailed information resolved by charge sectors :code:`psi.get_bond_charges_dimensions()`.

.. autofunction:: yastn.tn.mps.MpsMpo.get_bond_dimensions

.. autofunction:: yastn.tn.mps.MpsMpo.get_bond_charges_dimensions


Schmidt values and entropy profile
----------------------------------

The Schmidt values are computed by performing bipartition of the MPS/MPO across
each of the bonds. This amounts to SVD decomposition with respect to a bond
with all sites to the left being in left-canonical form and all sites to the right being in right-canonical form
(see :ref:`Canonical form<theory/mps/basics:canonical form>`).

Given MPS/MPO :code:`psi` you can get the profile for Schmidt values across all bonds by calling :code:`psi.get_Schmidt_values()`,
and von Neumann or Renyi entropies that follow from those Schmidt values by running :code:`psi.get_entropy()`.

.. autofunction:: yastn.tn.mps.MpsMpo.get_Schmidt_values

.. autofunction:: yastn.tn.mps.MpsMpo.get_entropy
