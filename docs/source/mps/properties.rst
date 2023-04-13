Properties
==========

Symmetry
--------

The symmetry of the MPS/MPO is inherited from the symmetry
of :meth:`yastn.Tensor<yastn.Tensor>`'s forming the matrix product.
The symmetries of those tensors have to be the same, as they are contracted along the virtual dimension,
implying that the global MPS/MPO symmetry is a consequence of the local symmetry of :meth:`yastn.Tensor<yastn.Tensor>`.

Physical and virtual spaces
---------------------------

MPS is built from tensors with a single physical index and two virtual indices.
MPO tenors have an additional physical index. The number of physical indices
can be accessed through the property :code:`yastn.tn.mps.MpsMps.nr_phys`.

In the case of MPS/MPO with no explicit symmetry, the virtual space :math:`V_{j,j+1}`
for *j*-to-*j+1* bond is simple :math:`\mathbb{R}^{D_{j,j+1}}`
or :math:`\mathbb{C}^{D_{j,j+1}}` space.
Typically, the maximal permited dimension of these virtual spaces is the control parameter for matrix product representation.

For symmetric MPS/MPO the virtual space has instead a structure of a direct sum of simple spaces, dubbed *sectors*, labeled by symmetry charges *t*

.. math::
    V_{j,j+1} = \oplus_{t} V^t_{j,j+1},\quad \textrm{where}\quad V^t_{j,j+1} = \mathbb{R}^{D^t_{j,j+1}}\ \textrm{or}\ \mathbb{C}^{D^t_{j,j+1}}.

Note that for a product of abelian symmetries, the *t*'s are :code:`tuple(int)`.
Therefore, symmetric MPS/MPO ansatz is specified by the content of all its virtual spaces.
Typically, the effective control parameter is the maximal total dimension
:math:`D_{j,j+1}=\sum_t D^t_{j,j+1}`, appearing in some methods as parameter :code:`D_total`.

To get the profile of the total bond dimensions along MPS/MPO :code:`psi`
call :code:`psi.get_bond_dimensions()`.

.. autofunction:: yastn.tn.mps.MpsMpo.get_bond_dimensions

More detailed information resolved by charge sectors is given by
:code:`psi.get_bond_charges_dimensions()`.

.. autofunction:: yastn.tn.mps.MpsMpo.get_bond_charges_dimensions


Index convention
----------------

The index convention for MPS/MPO tensors is *0th* index
(leftmost, i.e, in the direction of the first MPS/MPO site)
corresponds to the left virtual space.
*1st* index is physical, and *2nd* index
(rightmost, i.e. in the direction of the last MPS/MPO site)
corresponds to the right virtual space.
For MPO, the last *3rd* index is also physical.

::

    # indices of the individual tensors in MPS
        ___
    0--|___|--2
         |
         1 (ket)

    # indices of the individual tensors in MP0. The physical indices
    # are ordered as for usual quantum-mechanical operator, i.e., O = \sum_{ij} O_ij |i><j|

         3 (bra)
        _|_
    0--|___|--2
         |
         1 (ket)


Schmidt values and entropy profile
----------------------------------

The Schmidt values are computed by performing bipartition of the MPS/MPO across
each of the bonds. This amounts to SVD decomposition wrt. to a bond
with all sites to the left being in left-canonical form and all sites to the right being in right-canonical form
(see :ref:`Canonical form<theory/mps/basics:canonical form>`).

Given MPS/MPO :code:`psi` you can get the profile for Schmidt values across bonds
by calling :code:`psi.get_Schmidt_values()`.

.. autofunction:: yastn.tn.mps.MpsMpo.get_Schmidt_values

The Von Neumann or Renyi entropy profile (a function of Schmidt values)
can be obtained by running :code:`psi.get_entropy()`.

.. autofunction:: yastn.tn.mps.MpsMpo.get_entropy

