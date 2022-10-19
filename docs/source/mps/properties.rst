Properties
==========

Symmetry
--------

The symmetry of the `YAMPS` MPS or MPO is inherted from the symmetry 
of :meth:`yast.Tensor<yast.Tensor>`'s which form the matrix product. The symmetries of the tensors have to be consistent along virtual dimensions implying that the global MPS/MPO symmetry is a consequence of the local symmetry of :meth:`yast.Tensor<yast.Tensor>`.

Physical and virtual spaces
---------------------------

MPS is built from tensors with single physical index and two virtual indices. 
MPO tenors have an additional physical index. The number of physical indices
can be accessed through the property :code:`yamps.MpsMps.nr_phys`.

The index convention for MPS/MPO tensors in `YAMPS` is:
0-th index (leftmost) corresponds to the left virtual space,
1-st index is physical, for MPO the 2-nd index is also physical, 
and finally the highest index (rightmost) corresponds to the right virtual space. 

::
    
    # indices of the individual tensors in MPS
        ___
    0--|___|--2
         |
         1 (ket)

    # indices of the individual tensors in MP0. The physical indices 
    # are ordered as for usual quantum-mechanical operator, i.e., O = \sum_{ij} O_ij |i><j|)

         2 (bra)
        _|_
    0--|___|--3
         |
         1 (ket)

The index convention can be accessed through properties 

    * :code:`yamps.MpsMpo.left` for position of left virtual index
    * :code:`yamps.MpsMpo.right` for position of right virtual index
    * :code:`yamps.MpsMpo.phys` for position(s) of physical indices given as :code:`tuple(int)`.

In case of MPS/MPO with no explicit symmetry, the virtual space :math:`V_{j,j+1}` 
for *j*-to-*j+1* bond is simple :math:`\mathbb{R}^{D_{j,j+1}}` 
or :math:`\mathbb{C}^{D_{j,j+1}}` space. Typically, the maximal dimension allowed 
dimension of these virtual spaces is the control parameter for matrix product representation. 

For symmetric MPS/MPO the virtual space has instead a structure of direct sum of simple spaces, dubbed *sectors*, labeled by symmetry charges *c* 

.. math::
    V_{j,j+1} = \oplus_{c} V^c_{j,j+1},\quad \textrm{where}\quad V^c_{j,j+1} = \mathbb{R}^{D^c_{j,j+1}}\ \textrm{or}\ \mathbb{C}^{D^c_{j,j+1}}.

Note that for a product of abelian symmetries, the *c*'s are :code:`tuple(int)`. 
Therefore, symmetric MPS/MPO ansatz is specified by the content of all its virtual spaces.
Typically, the effective control parameter is the maximal total dimension 
:math:`D_{j,j+1}=\sum_c D^c_{j,j+1}`. 

To get the profile of the total bond dimensions along `YAMPS` MPS/MPO :code:`A` 
call :code:`A.get_bond_dimensions()`.

.. autofunction:: yamps.MpsMpo.get_bond_dimensions

More detailed information resolved by charge sectors is given by 
:code:`A.get_bond_charges_dimensions()`.

.. autofunction:: yamps.MpsMpo.get_bond_charges_dimensions


Schmidt values and entropy profile
----------------------------------

The Schmidt values are extracted at each bond separately. In order to get them, we do SVD decomposition with all sites to the left being in left-canonical form and all sites to the right being in right-canonical form 
(see :ref:`Canonical form<theory/mps/basics:canonical form>`). 

Given MPS/MPO :code:`A` you can get the profile for Schmidt values for all bipartitions by calling :code:`A.get_Schmidt_values()`.

.. autofunction:: yamps.MpsMpo.get_Schmidt_values

The Von Neumann or Renyi entropy profile (a function of Schmidt values) 
can be extracted by running  :code:`A.get_entropy()`. 

.. autofunction:: yamps.MpsMpo.get_entropy

