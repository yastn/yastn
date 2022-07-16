Properties
==========


`YAMPS` properties
-----------------------

The symmetry of the `YAMPS` object is determined by the symmetry of :meth:`yast.Tensor`'s forming matrix product. The symmetries of the tensors have to be consistent along virtual dimensions implying that global MPS/MPO symmetry is a consequence of 
local symmetry for :meth:`yast.Tensor`.

The :code:`Mps` and :code:`Mpo` differ by the number of physical dimensions. You can check what is the interpretation of the `YAMPS` object :code:`A` by checking if :code:`A.nr_phys` is `1` implying MPS or `2` implying MPO. 

Each tensor of the `YAMPS` object has two virtual dimensions and `1` or `2`  physical dimensions. The tensors assigned to `YAMPS` object are assumed a left-most leg of index `0` as left virtual dimension and 
the right-most leg of the highest index as right virtual dimension. 

::
    
    # indicies in the individual tensor in MPS
        ___
    0--|___|--2
         |
         1 (ket)

    # indicies in the individual tensor in MP0
    # (the physical indicies are ordered as for a usual matrix)

         2 (bra)
        _|_
    0--|___|--3
         |
         1 (ket)


You can always check what are left virtal legs by running :code:`A.left` (right virtal :code:`A.right`) and physical legs by running :code:`A.phys` which gives a tuple of indicies.


Virtual dimension
-----------------

The structure of virtual bond dimension is a control parameter for matrix product representation. In order to get the profile of the total (all blocks) bond dimension in the `YAMPS` object for :code:`A` run `Ds = A.get_bond_dimensions()`.

.. autofunction:: yamps.MpsMpo.get_bond_dimensions

More detailed information is given by `A.get_bond_charges_dimensions()` where we get the information on each block in tensors.

.. autofunction:: yamps.MpsMpo.get_bond_charges_dimensions


Schmidt values and entropy profile
----------------------------------

The Schmidt values are extracted at each bond separately. In order to get it we do SVD decomposition with all sites to the left being in left-canonical form and all sites to the right being in right-canonical form. 
You can get the profile for Schmidt values for all bipartition by :code:`SV = A.get_Schmidt_values()`.

.. autofunction:: yamps.MpsMpo.get_Schmidt_values

The entropy profile can be extracted by running :code:`en = A.get_entropy()`. As dafault we get von Neumann entropy (:math:`\alpha=1`). You can get the Renyi entropy of any order :math:`\alpha` by adding a flag :code:`alpha=<desired order>`.

.. autofunction:: yamps.MpsMpo.get_entropy

