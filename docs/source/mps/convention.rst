Conventions
===========

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