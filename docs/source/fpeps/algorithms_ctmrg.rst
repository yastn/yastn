Corner transfer matrix renormalization group (CTMRG) algorithm
===============================================================

Introduction 
^^^^^^^^^^^^

Nishino and Okunishi[1] first deployed CTMRG by extending the DMRG framework to give variational approximations for 
Baxter's corner matrices of the vertex model. Their pioneering work demonstrated the potential of combining DMRG with 
variational techniques to tackle complex problems in quantum systems.
The subsequent development of CTMRG (Corner Transfer Matrix Renormalization Group) beyond the realm of C4v symmetric tensors
was accomplished by Orus and Vidal[2], with further refinements by Corboz[3]. Referred to as directional CTM, this extension enabled 
the application of CTMRG techniques to nonsymmetric tensors. However, this extension also introduced additional challenges 
due to the loss of symmetry on the virtual indices.

Idea
^^^^^
The core idea behind CTMRG, both in the symmetric and nonsymmetric cases, remains the same. The method approximates the 
contraction of an infinite tensor network by utilizing a finite set of environment tensors with a fixed rank :math:`\chi`.
These environment tensors undergo a Renormalization Group (RG) procedure, iteratively converging towards their fixed-point forms. 
The ultimate goal is to recover the thermodynamic limit properties as the rank :math:`\chi \rightarrow \infty`. 
In the corner transfer matrix method, the infinite environment of a given site (or sites in a unit cell) in a 2D tensor network is approximated by a 
combination of four corner :math:`C_{nw},C_{sw},C_{ne},C_{se}` and four transfer :math:`T_{n},T_{w},T_{e},T_{s}` tensors of finite size, as depicted in the 
following figure. The single-site observable :math:`O` is placed between the site and its conjugated attached via the physical indices.
The CTMRG tensors when contracted provides the reduced density matrix with which the expectation value is to be computed.

::

     ________       ________       ________                                             \
    |        |     |        | \chi|        |                                             \
    |  C_nw  |-----|  T_n   |-----|  C_ne  |                                            __\____ 
    |________|     |________|     |________|                                           |       |
         |             |              |                          |                   --|   A   |---
         |             |              |                     _____|_____                |_______|   
     ____|___       ___|____       ___|____                |           |                   |  \
    |        |     |        | D^2 |        |               |           |                   |   \   
    |  T_w   |-----|   O    |-----|   T_e  |           ----|     O     |----  =         ___|___ \
    |________|     |________|     |________|               |           |             O |       |
         |              |             |                    |___________|               |       |
         | \chi         |             |                          |                     |_______|
     ____|___       ____|___       ___|____                      |                         |    /
    |        |     |        |     |        |                                               |   /              
    |   C_sw |-----|  T_s   |-----|  C_se  |                                             __|__/_
    |________|     |________|     |________|                                            |       |         
                                                                                     ---|   A'  |---
                                                                                        |_______|
                                                                                           /
                                                                                          /  
                                                                                         /


The convention for ordering the indices in the CTMRG environment tensors are given below:




::

     ___________                         ___________                        ___________
    |           |                       |           |                      |           |
    |   C_nw    |-----1           0-----|   T_n     |-----2          0-----|    C_ne   |
    |___________|                       |___________|                      |___________|
          |                                   |                                  |
          |                                   |                                  |          
          0                                   1                                  1

          2                                   0                                  0
          |                                   |                                  |
     _____|_____                         _____|_____                        _____|_____
    |           |                       |           |                      |           |
    |   T_w     |-----1           1-----|     O     |-----3          1-----|    T_e    |
    |___________|                       |___________|                      |___________|
          |                                   |                                  |
          |                                   |                                  |
          0                                   2                                  2

          1                                   1                                  0        
          |                                   |                                  |
     _____|_____                         _____|_____                         ____|_____
    |           |                       |           |                       |          |
    |   C_sw    |-----0           2-----|   T_s     |-----0           1-----|   C_se   |
    |___________|                       |___________|                       |__________|




We implement one-step of the Corboz version [3] of CTMRG:

.. autofunction:: yastn.tn.fpeps.ctm._ctm_iteration_routines.CTM_it

References & Related works
^^^^^^^^^^^^^^^^^^^^^^^^^^
1. “Corner Transfer Matrix Renormalization Group Method”, T. Nishino and K. Okunishi, `Journal of the Physical Society of Japan 65, 891 (1996) <https://arxiv.org/abs/cond-mat/9507087>`_
2. "Simulation of two dimensional quantum systems on an infinite lattice revisited: corner transfer matrix for tensor contraction", R. Orus, G. Vidal, `Phys. Rev. B 80, 094403 (2009) <https://arxiv.org/abs/0905.3225>`_
3. "Competing States in the t-J Model: Uniform d-Wave State versus Stripe State (Supplemental Material)", P. Corboz, T. M. Rice, and M. Troyer, `Phys. Rev. Lett. 113, 046402 (2014) <https://arxiv.org/abs/1402.2859>`_