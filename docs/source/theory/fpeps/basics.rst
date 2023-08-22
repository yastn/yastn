==============
Basic concepts
==============

Projected Entangled Pair States (PEPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
                   __\____         __\____
                  |       |       |       |
            ... --|   B   |-- D --|   A   |-- ...
                  |_______|       |_______|
                    |    \          |    \
                    |     .         |     .
                           .               .


Time evolution
^^^^^^^^^^^^^^

The simulation of time evolution is ubiquitous problem.
We focus on real- or imaginary-time evolution generated by a local Hamiltonian :math:`H`, and, for simplicity, discuss here a checkerboard iPEPS lattice.
Within the Suzuki-Trotter decomposition, the time evolution operator :math:`\exp(-d\beta H)`, for a small time step :math:`d\beta`, here in the imaginary time,
is approximated a product of local two-site gates.

For a checkerboard iPEPS lattice and a Hamiltonians with nearest-neighbor interactions,
:math:`H = \sum_{bond} H_{bond},`
there are four unique groups of disjoint bonds: AB horizontal, BA horizontal, AB vertical, BA vertical.
The corresponding two-site gates :math:`U_{bond} = \exp(-d\beta H_{bond} / 2)`, and
a typical 2nd-order Suzuki-Trotter approximation gives

:math:`\exp(-d\beta H) \approx U_{AB}^{hor} U_{BA}^{hor} U_{AB}^{ver} U_{BA}^{ver} U_{BA}^{ver} U_{AB}^{ver} U_{BA}^{hor} U_{AB}^{hor}`.

We allow for an ambiguous notation here, as a gate for all, say, AB horizontal bonds is a product of local two-site commuting gates on each bond, that can be applied simultaneously.
Each gate increases the virtual bond dimension of PEPS tensors by a factor equal to the SVD rank of the gate `r`.

::

      # Action of a two-site gate on horizontal A-B bond in iPEPS

               .               .
                .               .
             ____\__         ____\__
            |       |       |       |
      ... --|   A   |-- D --|   B   |-- ...
            |_______|       |_______|
              |    \          |    \
              |\    D        /|     D
              ||\____\__r___/||      \
              ||/     \     \||       \
              |/       \     \|        \
              |     ____\__   |     ____\__
                   |       |       |       |
             ... --|   B   |-- D --|   A   |-- ...
                   |_______|       |_______|
                     |   \           |   \
                     |    .          |    .
                           .               .

To keep the PEPS representation compact, each application of the gate has to be followed by a truncation procedure to reduce the virtual bond dimension back to :math:`D`.


Truncation of the PEPS bond dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In 1D, for a canonical structure of MPS, a local truncation based on SVD singular values is optimal in a Frobenius norm.
However, a loopy structure of PEPS prevents a canonical form, and a successful algorithm requires using optimization techniques on top of SVD.
The aim is to minimize the Frobenius norm of (a) the PEPS after application of the gate with virtual bond dimension increased to :math:`D' = r \times D` and (b) a new PEPS with
the bond dimension truncated back to :math:`D`:

::

      (a)  .               .               (b)  .               .
            .               .                    .               .
           __\____         __\____              __\____         __\____
          |       |       |       |            |       |       |       |
      ..--|   A'  |-r x D-|   B'  |--..    ..--|  A''  |-- D --|  B''  |--..
          |_______|       |_______|            |_______|       |_______|
             |   \          |    \       ~~~     |   \           |   \
             |    D         |     D      ~~~     |    D          |    D
                 __\____         __\____             __\____         __\____
                |       |       |       |           |       |       |       |
            ..--|   B   |-- D --|   A   |--..   ..--|   B   |-- D --|   A   |--..
                |_______|       |_______|           |_______|       |_______|
                  |    \          |    \              |    \          |    \
                  |     .         |     .             |     .         |     .
                         .               .                   .               .

Note that these figures are part of an infinite PEPS structure extending in all directions.
We denote the wavefunction in (a) by :math:`|\Psi(A',B')\rangle` and in (b) as :math:`|\Psi(A'',B'')\rangle`.
The Frobenius norm is denoted by :math:`d(A',B';A'',B'') = || |\Psi(A'',B'')\rangle - |\Psi(A',B')\rangle ||^{2}`
The aim is to minimalize it with respect to two isoleted tensors :math:`A''` and :math:`B''` with the metric tensor representing the rest of the lattice.
The state-of-the-art optimization method in this context is the so-called Full Update, which is however numerically expensive and might be unstable in some applications.
In YASTN, we implement a recently developed Neighborhood Tensor Update (:ref:`NTU<fpeps/algorithms_NTU:Neighborhood tensor update (NTU) algorithm>`) [4] that approximates the metric tensor by contracting a small cluster of neighboring tensors.


Corner transfer matrix renormalization group (CTMRG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exact contraction of an iPEPS is exponentially hard.
The state-of-the-art approximate technique for calculating expectation values is the Corner Transfer Matrix Renormalization Group (:ref:`CTMRG<fpeps/algorithms_ctmrg:corner transfer matrix renormalization group (ctmrg) algorithm>`).
CTMRG iteratively finds the environment of each tensor, representing the rest of the infinite lattice, in the form of four corner tensors and edge tensors transfer matrices surrounding each unique tensor in the unit cell.


References & Related works
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. “Renormalization algorithms for Quantum-Many Body Systems in two and higher dimensions”, Frank Verstraete and Juan I. Cirac, `arXiv:cond-mat/0407066(2004) <https://arxiv.org/abs/cond-mat/0407066>`_
2. "Simulation of strongly correlated fermions in two spatial dimensions with fermionic projected entangled-pair states", Philippe Corboz, Román Orús, Bela Bauer, and Guifré Vidal, `Phys. Rev. B 81, 165104 (2010) <https://arxiv.org/abs/0912.0646>`_
3. “Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions”, J. Jordan, R. Orus, G. Vidal, F. Verstraete, and J. I. Cirac, `Phys. Rev. Lett. 101, 250602 (2008) <https://arxiv.org/abs/cond-mat/0703788>`_
4. "Finite-temperature tensor network study of the Hubbard model on an infinite square lattice", Aritra Sinha, Marek M. Rams, Piotr Czarnik, and Jacek Dziarmaga, `Phys. Rev. B 106, 195105 (2022) <https://arxiv.org/abs/2209.00985>`_
5. "Time evolution of an infinite projected entangled pair state: Neighborhood tensor update", Jacek Dziarmaga, `Phys. Rev. B 104, 094411 (2021) <https://arxiv.org/abs/2107.06635>`_
