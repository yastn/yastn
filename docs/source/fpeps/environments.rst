Environments
============

Contractions of PEPS lattice are supported by environment classes. Those include:
    * :class:`yastn.tn.fpeps.EnvNTU` supports a family of NTU approximations of the bond metric for time evolution.
    * :class:`yastn.tn.fpeps.EnvCTM` CTMRG for finite or infinite lattices. Supports local expectation values, bond metric, etc.
    * :class:`yastn.tn.fpeps.EnvBoundaryMps` Contraction based on BoundaryMPS in a finite lattice supports expectation values, including long-range correlations, sampling, etc.
    * :class:`yastn.tn.fpeps.EnvApproximate` Supports larger clusters for approximate contraction of bond metric.


Neighberhood tensor update environments
---------------------------------------

.. autoclass:: yastn.tn.fpeps.EnvNTU
    :members: bond_metric


Corner transfer matrix renormalization group (CTMRG)
----------------------------------------------------

Introduction
""""""""""""

Nishino and Okunishi first deployed CTMRG [1] by extending the DMRG framework to give variational approximations for
Baxter's corner matrices of the vertex model. Their pioneering work demonstrated the potential of combining DMRG with
variational techniques to tackle complex problems in quantum systems.
The subsequent development of CTMRG (Corner Transfer Matrix Renormalization Group) beyond the realm of C4v symmetric tensors
was accomplished by Orus and Vidal [2], with further refinements by Corboz [3]. Referred to as directional CTM, this extension enabled
the application of CTMRG techniques to nonsymmetric tensors. However, this extension also introduced additional challenges
due to the loss of symmetry on the virtual indices.

Idea
""""

The core idea behind CTMRG, both in the symmetric and nonsymmetric cases, remains the same. The method approximates the
contraction of an infinite tensor network by utilizing a finite set of environment tensors with a fixed rank :math:`\chi`.
These environment tensors undergo a Renormalization Group (RG) procedure, iteratively converging towards their fixed-point forms.
The ultimate goal is to recover the thermodynamic limit properties as the rank :math:`\chi \rightarrow \infty`.
In the corner transfer matrix method, the infinite environment of a given site (or sites in a unit cell) in a 2D tensor network is approximated by a
combination of four corner :math:`C_{nw},C_{sw},C_{ne},C_{se}` and four transfer :math:`T_{n},T_{w},T_{e},T_{s}` tensors of finite size, as depicted in the
following figure. The single-site observable :math:`O` is placed between the site and its conjugated attached via the physical indices.
The CTMRG tensors, when contracted, provides the reduced density matrix with which the expectation value is to be computed.


::

     _______     _______     _______                       \
    |       |   |       |chi|       |                     __\____
    |  C_nw |---|  T_n  |---|  C_ne |                    |       |
    |_______|   |_______|   |_______|                  --|   A   |--
        |           |           |                        |_______|
     ___|___     ___|___     ___|___                         |  \
    |       |   |       |D^2|       |       ___|___       ___|___
    |  T_w  |---|   O   |---|  T_e  |      |       |     |       |
    |_______|   |_______|   |_______|    --|   O   |-- = |       |
        |chi        |           |          |_______|     |_______|
     ___|___     ___|___     ___|___           |             |
    |       |   |       |   |       |                     ___|_/_
    |  C_sw |---|  T_s  |---|  C_se |                    |       |
    |_______|   |_______|   |_______|                  --|   A'  |--
                                                         |_______|
                                                           /
                                                          /


The convention for ordering the indices in the CTMRG environment tensors are given below:


::

     _______             _______             _______
    |       |           |       |           |       |
    |  C_nw |--1     0--|  T_n  |--2     0--|  C_ne |
    |_______|           |_______|           |_______|
        |                   |                   |
        0                   1                   1

        2                   0                   0
     ___|___             ___|___             ___|___
    |       |           |       |           |       |
    |  T_w  |--1     1--|   O   |--3     1--|  T_e  |
    |_______|           |_______|           |_______|
        |                   |                   |
        0                   2                   2

        1                   1                   0
     ___|___             ___|___             ___|___
    |       |           |       |           |       |
    |  C_sw |--0     2--|  T_s  |--0     1--|  C_se |
    |_______|           |_______|           |_______|



.. autoclass:: yastn.tn.fpeps.EnvCTM

We implement a single iteration of the Corboz version [3] of CTMRG:

.. autofunction:: yastn.tn.fpeps.EnvCTM.update_

Through this method of EnvCTM class we perform an iteration of CTMRG.
One can stop the CTM after a fixed number of iterations. Stopping criteria can also be set based on
the convergence of one or more observables, for example, total energy.
Once the CTMRG converges, it is straightforward to obtain one-site and two-site
observables.

One-site observables for all lattice sites can be calculated using the function

.. autofunction:: yastn.tn.fpeps.EnvCTM.measure_1site

and all unique neighbor two-point correlators can be calculated using the function

.. autofunction:: yastn.tn.fpeps.EnvCTM.measure_nn


Boundary MPS
------------

.. autoclass:: yastn.tn.fpeps.EnvBoundaryMps


Approximate cluster update
--------------------------

.. autoclass:: yastn.tn.fpeps.EnvApproximate
