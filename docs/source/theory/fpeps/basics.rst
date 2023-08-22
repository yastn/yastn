==============
Basic concepts
==============

Projected Entangled Pair States (PEPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Projected Entangled Pair State (PEPS) [1] is a two-dimensional (`2D`) tensor network that, by construction, satisfies area law of entanglement entropy in `2D`.
It allows efficient simulations of ground and thermal states of many-body quantum systems in `2D` with their respective area laws.
PEPS is constructed from a set of tensors located on a `2D` lattice.
Each tensor have physical leg, corresponding to physical degrees of freedom of the lattice site, and virtual legs (bonds) connecting it to neighboring tensors.
Here we implement a 2D rectangular lattice of size :math:`L_{x} \times L_{y}`, with sites labeled by coordinates :math:`(x,y)` as shown below:


::

       # coordinates of the underlying 2D lattice

        ----------->
       |
       |    (0,0)      (0,1)      (0,2)      ...      (0, L_y-1)
       |
       |    (1,0)      (1,1)      (1,2)      ...      (1, L_y-1)
      \|/
              .           .                              .
              .                     .                    .
              .                               .          .

          (L_x-1,0)  (L_x-1,1)  (L_x-1,2)    ...    (L_x-1,L_y-1)


Each tensor :math:`A_{(x,y)}` in PEPS is a rank-:math:`5` tensor of size :math:`D_{(x-1,y),(x,y)} \times D_{(x,y-1),(x,y)} \times D_{(x,y),(x+1,y)} \times D_{(x,y),(x,y+1)} \times d_{(x,y)}`.


::

      # individual tensor of the PEPS lattice

                                 D_(x-1,y),(x,y)
                                         \
                                       ___\_______
                                      |           |
         A_(x,y)  =  D_(x,y-1),(x,y)--|   (x,y)   |--D_(x,y),(x,y+1)
                                      |___________|
                                         |      \
                                         |       \
                                      d_(x,y)  D_(x,y),(x+1,y)


The module :code:`yastn.tn.fpeps` implements operations on PEPS both for a finite system with open boundary conditions (OBC) and in the thermodynamic limit.
A schematic diagram of PEPS with OBC is given below.

::

      # PEPS for 6 sites arranged in a (L_x, L_y) = (2, 3) lattice with OBC.
      # For OBC, dimensions of virtual legs on the edge of the lattice are one.

           1                         1                         1
            \                         \                         \
          ___\_____                 ___\_____                 ___\_____
         |         |               |         |               |         |
      1--|  (0,0)  |-D_(0,0),(0,1)-|  (0,1)  |-D_(0,1),(0,2)-|  (0,2)  |--1
         |_________|               |_________|               |_________|
           |     \                   |     \                   |     \
           |      \                  |      \                  |      \
       d_(0,0)  D_(0,0),(1,0)    d_(0,1)  D_(0,1),(1,1)    d_(0,2)  D_(0,2),(1,2)
                    \                         \                         \
                  ___\_____                 ___\_____                 ___\_____
                 |         |               |         |               |         |
              1--|  (1,0)  |-D_(1,0),(1,1)-|  (1,1)  |-D_(1,1),(1,2)-|  (1,2)  |--1
                 |_________|               |_________|               |_________|
                   |     \                   |     \                   |    \
                   |      \                  |      \                  |     \
                d_(1,0)    1              d_(1,1)    1              d_(1,2)   1





Implementation of fermionic anticommutation rules in PEPS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We follow the recipe introduced by Corboz et al. in Ref. [2].
It relies on (a) working with parity preserving tensors and (b) supplementing lines (legs) crossings, in a planar projection of the network, with fermionic swap gates :meth:`yastn.swap_gate`.
The distribution of crossings (swap gates) follows from a chosen fermionic order, but their application can be made local for contracting the network.
The leading numerical cost of contracting fermionic and bosonic (or spin) lattices are the same, with swap gates adding a subleading overhead.
The module :code:`yastn.tn.fpeps` handles both fermionic and bosonic statistics, controlled by :code:`fermionic` flag in the :ref:`tensor configuration <tensor/configuration:yastn configuration>`.
We use the name fpeps to emphasize the incorporation of fermionic statistics in the module.

Here we employ the fermionic order demonstrated in a 3x3 PEPS example below.
Parity-preserving tensors permit moving the lines over tensors, changing the placement of the swap gates without affecting the result.

::


              ________            ________            ________
             |        |          |        |          |        |
             |        |----------|        |--------- |        |
             |________|          |________|          |________|
               |    \              |    \              |    \
               |   __\_____        |   __\_____        |   __\_____
               |  |        |       |  |        |       |  |        |
      |Psi> =  |  |        |-------|--|        |-------|--|        |
               |  |________|       |  |________|       |  |________|
               |    |    \         |    |    \         |    |    \
               |    |   __\_____   |    |   __\_____   |    |   __\_____
               |    |  |        |  |    |  |        |  |    |  |        |
               |    |  |        |--|----|--|        |--|----|--|        |
               |    |  |________|  |    |  |________|  |    |  |________|
               |    |    |         |    |    |         |    |    |
               |    |    |         |    |    |         |    |    |

            --------------------------------------------------------->
                                                   fermionic order



Infinite PEPS (iPEPS)
^^^^^^^^^^^^^^^^^^^^^

Although finite PEPS is widely used, some of the best results, arguably, have been obtained with infinite PEPS (iPEPS) [3].
It operates directly in the thermodynamic limit describing a system with translational invariance.
In iPEPS ansatz is formed by a unit cell of tensors repeated all over infinite lattice.
A common example is a checkerboard lattice, which has two tensors A and B in a 2 by 2 unit cell.

::

       # Checkerboard ansatz for iPEPS
                 .               .
                  .               .
                 __\____         __\____
                |       |       |       |
          ... --|   A   |-- D --|   B   |-- ...
                |_______|       |_______|
                   |   \          |    \
                   |    D         |     D
                         \               \
                        __\____         __\____
                       |       |       |       |
                ...  --|   B   |-- D --|   A   |-- ...
                       |_______|       |_______|
                         |    \          |    \
                         |     .         |     .
                                .               .


Time evolution
^^^^^^^^^^^^^^
An ubiquitous problem is simulation of real or imaginary time evolution.
For evolution generated by a local Hamiltonian :math:`H`, we employ Suzuki-Trotter decomposition.
The time evolution operator :math:`\exp(-\beta H)`, here in the imaginary time, is approximated by
a product of local gates applied `n` times with a small time step :math:`d\beta = \beta / n`.

:math:`\exp(-\beta \hat{H}) =  [\exp(-d\beta \hat{H})]^{n} \approx [\prod_{bond} \exp(-d\beta \hat{H}_{bond})]^{n}`

For Hamiltonians with nearest neighbor interactions, the operators are typically two-site gates applied to the physical index of the PEPS tensors. For checkerboard lattice,
there are four unique bonds : AB horizontal, BA horizontal, AB vertical, BA vertical. Typically we use 2nd order Suzuki-Trotter method where our application of
an operator :math:`U(d\beta) = \exp(-d\beta H)` on the iPEPS network with a checkerboard ansatz go like:

:math:`\exp(-d\beta H) = U_{ab}^{hor}(d\beta H)U_{ba}^{hor}U_{ab}^{ver}(d\beta H)U_{ba}^{ver}U_{ba}^{ver}(d\beta H)U_{ab}^{ver}U_{ba}^{hor}(d\beta H)U_{ab}^{hor}(d\beta H)`

The gates increase the virtual bond dimension of the PEPS by a factor which is equal to the svd rank of the gate. So if the SVD rank of a gate is :math:`r`, then after application of the gate,
the bond dimension becomes :math:`r \times D`.

::


        # action of gate on horizontal A-B bond in iPEPS

               .                  .
                .                  .
                 .                  .
                  \                  \
                   \                  \
                 __________         __________
                |          |       |          |
         ...  --|    A     |-- D --|    B     |-- ...
                |__________|       |__________|
                   |  \               |   \
                   |   \              |    \
                   |\   D            /|     D
                   |\\   \          //|      \
                   ||\\ __\_______ //||       \
                   ||//    \  r    \\||        \
                   |//      \       \\|         \
                   |/        \       \|          \
                   |     ___________  |       ___________
                        |           |       |          |
                ...  -- |     B     |-- D --|     A    |-- ...
                        |___________|       |__________|
                          |   \               |   \
                          |    \              |    \
                                .                   .
                                 .                   .
                                  .                   .


So they have to be truncated back to :math:`D` to prevent blowing up of the algorithm (owing to required computational
resources piling up with application of each such bond). In 1D, truncation by SVD is optimal because of the canonical structure of MPS.
However, because of loops in PEPS, it cannot be brought to a canonical form and we have to use optimization techniques on top of SVD.



Truncation of PEPS bond dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimization of iPEPS involves minimizing the Frobenius norm of two structures (a) PEPS with a gate applied to a bond which increased its bond dimension to :math:`D' = r \times D` and (b) a new PEPS with
:math:`D` as in the following figure:

::

               .                  .                                                      .                  .
                .                  .                                                      .                  .
        (a)      .                  .                                         (b)          .                  .
                  \                  \                                                      \                  \
                   \                  \                                                      \                  \
                 __________         __________                                            __________          _________
                |          |       |          |                                          |           |       |         |
         ...  --|    A'    |-------|    B'    |-- ...                             ...  --|     A''   |-- D --|    B''  |-- ...
                |__________| r x D |__________|                                          |___________|       |_________|
                   |  \               |   \                                                 |   \               |   \
                   |   \              |    \                                                |    \              |    \
                   |    D             |     D                                               |     D             |     D
                   |     \            |      \                                              |      \            |      \
                          \                   \                          ~                          \                   \
                        ___________         __________                   ~                       _________         _________
                       |           |       |          |                                         |         |       |         |
                ... -- |     B     |-- D --|     A    |-- ...                            ...  --|    B    |-- D --|    A    |-- ...
                       |___________|       |__________|                                         |_________|       |_________|
                          |   \               |   \                                                |   \             |   \
                          |    \              |    \                                               |    \            |    \
                          |     .             |     .                                              |     .           |     .
                          |      .            |      .                                             |      .          |      .
                                  .                   .                                                    .                 .


Note that this figures are part of an infinite PEPS structure extending in all four directions. If we denote the wavefunction representing the PEPS in fig. (a)
by :math:`|\Psi(A',B')\rangle` and the wavefunction representing the PEPS in fig. (b) as :math:`|\Psi(A'',B'')\rangle`, then the Frobenius norm is denoted by
:math:`d(A',B';A'',B'') = || |\Psi(A'',B'')\rangle - |\Psi(A',B')\rangle ||^{2}`. Minimization of the Frobenius norm is done with respect to a metric tensor.
The state-of-the-art optimization method in this context is the so-called Full Update. Although the Full Update has been immensely succesful in calculating ground states of
2D models, it has been found to be expensive and somewhat unstable for thermal states. In YASTN, we use the fermionic version of the newly developed optimization technique
called the Neighborhood Tensor Update (NTU) to calculate the thermal states of the Fermi Hubbard Model. For details see :ref:`NTU<fpeps/algorithms_NTU>`


iPEPS contraction; CTMRG
^^^^^^^^^^^^^^^^^^^^^^^^

The exact contraction of an iPEPS is exponentially hard. The state-of-the-art technique for calculating the norm and expectation values
is the Corner Transfer Matrix Renormalization Group.
`Corner Transfer matrix renormalization group`
(:ref:`CTMRG<fpeps/algorithms_ctmrg:corner transfer matrix renormalization group (ctmrg) algorithm>`)
is an algorithm that calculates :math:`4` corner and :math:`4` transfer matrices surrounding each unique tensor in the unit cell.
These :math:`4` corner and :math:`4` transfer matrices basically replaces the infinite environment surrounding the lattice site.


References & Related works
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. “Renormalization algorithms for Quantum-Many Body Systems in two and higher dimensions”, Frank Verstraete and Juan I. Cirac, `arXiv:cond-mat/0407066(2004) <https://arxiv.org/abs/cond-mat/0407066>`_
2. "Simulation of strongly correlated fermions in two spatial dimensions with fermionic projected entangled-pair states", Philippe Corboz, Román Orús, Bela Bauer, and Guifré Vidal, `Phys. Rev. B 81, 165104 (2010) <https://arxiv.org/abs/0912.0646>`_
3. “Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions”, J. Jordan, R. Orus, G. Vidal, F. Verstraete, and J. I. Cirac, `Phys. Rev. Lett. 101, 250602 (2008) <https://arxiv.org/abs/cond-mat/0703788>`_
4. "Finite-temperature tensor network study of the Hubbard model on an infinite square lattice", Aritra Sinha, Marek M. Rams, Piotr Czarnik, and Jacek Dziarmaga, `Phys. Rev. B 106, 195105 (2022) <https://arxiv.org/abs/2209.00985>`_
5. "Time evolution of an infinite projected entangled pair state: Neighborhood tensor update", Jacek Dziarmaga, `Phys. Rev. B 104, 094411 (2021) <https://arxiv.org/abs/2107.06635>`_
