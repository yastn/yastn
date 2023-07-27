Neighborhood tensor update (NTU) algorithm
===========================================

Neighborhood Tensor Update (NTU) (see Ref [2] for details) can be regarded as a special case of a cluster update (see Ref [1]),
where the number of neighboring lattice sites taken into account during truncation makes for a refining parameter. The cluster update
interpolates between a local truncation — as in the simple update (SU)[4]—and the full update (FU) [5] that takes into account all 
correlations in the truncated state. The NTU cluster includes the neighboring sites only as the metric tensor to compute the Frobenius 
norm in :ref:`Optimization of iPEPS<theory/fpeps/basics:Optimization of iPEPS>`. 

In the diagram below, we have a checkerboard lattice with alternating tensors :math:`A` and :math:`B`
in the :math:`2D` square lattice. The tensors :math:`A'` and :math:`B'` in the center are highlighted as 
they have been updated by a NN :math:`2`-site gate of SVD rank :math:`r`. The procedure for
truncating the bond dimension back to :math:`D` involves calculating the Frobenius norm. Ideally, in case of iPEPS
the whole infinite lattice should contribute in calculation of the norm. This being practically impossible, CTMRG
is often used to construct environmental tensors which approximate the infinite environment around the updated bond.
This is done in the Full Update procedure. Due to matrix inversions involved in CTMRG, the metric tensor loses its
hermiticty often rendering the algorithm unstable and also very expensive. A slightly less accurate but computationally
cheaper and stable way is just to use the NN sites surrounding the updated bond to calculate the metric tensor.

::

                  
                       \                  \
                        \                  \
                      __________         __________
                     |          |       |          |
                  ---|    B     |-- D --|    A     |---         
              \      |__________|       |__________|            
               \         |  \               |   \           \
             _________   |  _\______       _|____\_       ___\_____
            |         |     ||     || rxD ||      ||     |         |
         ---|    B    |-----||  A' ||-----||  B'  ||-----|    A    |---
            |_________|     ||_____||     ||______||     |_________| 
                |  \            \                \            |   \
                |   \        ____\______         _\________   |    \
                            |           |       |          |
                         ---|     A     |-- D --|     B    |--- 
                            |___________|       |__________|
                                |  \               |  \
                                |   \              |   \
                              

The NTU error can be calculated numerically exactly via parallelizable tensor contractions. The
description of the algorithm in Ref[2] and in Appendix B of Ref [3]. That exactness warrants
that the error measure is Hermitian and non-negative own to the numerical precision.

The least square optimization processes used is in 

.. autofunction:: yastn.tn.fpeps.evolution._routines


References & Related works
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. "Algorithms for finite projected entangled pair states", M. Lubasch, J. I. Cirac, and M.-C. Banyuls, `Phys. Rev. B 90, 064425 (2014) <https://arxiv.org/abs/1405.3259>`_
2. "Finite-temperature tensor network study of the Hubbard model on an infinite square lattice" Aritra Sinha, Marek M. Rams, Piotr Czarnik, and Jacek Dziarmaga, `Phys. Rev. B 106, 195105 (2022) <https://arxiv.org/abs/2209.00985>`_
3. "Time evolution of an infinite projected entangled pair state: Neighborhood tensor update", Jacek Dziarmaga, `Phys. Rev. B 104, 094411 (2021) <https://arxiv.org/abs/2107.06635>`_
4. "Accurate Determination of Tensor Network State of Quantum Lattice Models in Two Dimensions", H. C. Jiang, Z. Y. Weng, and T. Xiang, `Phys. Rev.Lett. 101, 090603 (2008) <https://arxiv.org/abs/0806.3719>`_
5. "Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions", J. Jordan, R. Orus, G. Vidal, F. Verstraete, and J. I. Cirac, `Phys. Rev. Lett. 101, 250602 (2008) <https://arxiv.org/abs/cond-mat/0703788>`_


