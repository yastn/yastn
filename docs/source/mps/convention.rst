YAMPS Conventions
=================

Index convention
----------------

The index convention for MPS/MPO tensors in `YAMPS` is
    
    i) *0th* index (leftmost) corresponds to the left virtual space,
    ii) *1st* index is physical, and *2nd* index (rightmost) corresponds to the right virtual space. 
    iii) For MPO the last *3rd* index is also physical.

::
    
    # indices of the individual tensors in MPS
        ___
    0--|___|--2
         |
         1 (ket)

    # indices of the individual tensors in MP0. The physical indices 
    # are ordered as for usual quantum-mechanical operator, i.e., O = \sum_{ij} O_ij |i><j|)

         3 (bra)
        _|_
    0--|___|--2
         |
         1 (ket)

The index convention can be accessed through properties 

    * :code:`yast.tn.mps.MpsMpo.left` for position of left virtual index
    * :code:`yast.tn.mps.MpsMpo.right` for position of right virtual index
    * :code:`yast.tn.mps.MpsMpo.phys` for position(s) of physical indices given as :code:`tuple(int)`.