==============
Basic concepts
==============

Projected Entangled Pair States (PEPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Projected Entangled Pair State (PEPS) [1,2] is a tensor network appearing typically in the context of two-dimensional (`2D`) systems,
that by construction satisfies the area law of entanglement entropy for such systems.
It allows efficient simulations of ground and thermal states of many-body quantum systems in `2D` with their respective area laws [3].
We employ PEPS constructed from a set of tensors located on a `2D` lattice.
Each tensor has a physical leg, corresponding to the physical degrees of freedom of the lattice site, and virtual legs (bonds) connecting it to neighboring tensors.
Here we implement a `2D` rectangular lattice of size :math:`L_{x} \times L_{y}`, with sites labeled by coordinates :math:`(x,y)` as shown below:


::

       # coordinates of the underlying 2D lattice

        ----------->
       |
       |  (0,0)      (0,1)      (0,2)      ...      (0, L_y-1)
       |
       |  (1,0)      (1,1)      (1,2)      ...      (1, L_y-1)
      \|/
            .          .                               .
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


The hermitian conjugate of the above tensor :math:`A_{x,y}^{\dag}` is represented as its mirror image 

::
        # conjugate tensor in PEPS

   

               

                                            d_(x,y)  D_(x,y),(x+1,y)
                                              |     /
                                             _|____/______
                                            |             |
          A_(x,y)^{\dag} = D_(x,y-1),(x,y)--|    (x,y)    |-- D_(x,y),(x,y+1)
                                            |_____________|
                                                 /
                                                /
                                          D_(x-1,y),(x,y)


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

We follow the recipe introduced by Corboz et al. in Ref. [4].
It relies on (a) working with parity preserving tensors and (b) supplementing lines (legs) crossings, in a planar projection of the network, with fermionic swap gates :meth:`yastn.swap_gate`.
The distribution of crossings (swap gates) follows from a chosen fermionic order, but their application can be made local for contracting the network.
The leading numerical cost of contracting fermionic and bosonic (or spin) lattices are the same, with swap gates adding a subleading overhead.
The module :code:`yastn.tn.fpeps` handles both fermionic and bosonic statistics, controlled by :code:`fermionic` flag in the :ref:`tensor configuration <tensor/configuration:yastn configuration>`.
We use the name fpeps to emphasize the incorporation of fermionic statistics in the module.

Here we employ the fermionic order demonstrated in a :math:`3\times 3` PEPS example below.
Parity-preserving tensors permit moving the lines over tensors, changing the placement of the swap gates without affecting the result.

Time evolution
^^^^^^^^^^^^^^

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



Time evolution
^^^^^^^^^^^^^^

The simulation of time evolution of a quantum state is an ubiquitous problem.
We focus on real- or imaginary-time evolution generated by a local Hamiltonian :math:`H`.
For simplicity, we discuss here a PEPS defined on a :math:`2 \times 2` lattice.
Within the Suzuki-Trotter decomposition, the time evolution operator :math:`\exp(-d\beta H)`, for a small time step :math:`d\beta`, here in the imaginary time,
is approximated by a product of local two-site gates.

For a Hamiltonian with nearest-neighbor interactions definded on a :math:`2 \times 2` lattice, :math:`H = \sum_{bond} H_{bond},`
there are four disjoint bonds: :math:`A_{1}A_{2}` horizontal, :math:`A_{3}A_{4}` horizontal, :math:`A_{1}A_{3}` vertical, :math:`A_{2}A_{4}` vertical.
The corresponding two-site gates :math:`U_{bond} = \exp(-d\beta H_{bond} / 2)`, and a typical 2nd-order Suzuki-Trotter approximation gives

:math:`\exp(-d\beta H) \approx U_{A_{1}A_{2}}^{hor} U_{A_{3}A_{4}}^{hor} U_{A_{1}A_{3}}^{ver} U_{A_{2}A_{4}}^{ver} U_{A_{2}A_{4}}^{ver} U_{A_{1}A_{3}}^{ver} U_{A_{3}A_{4}}^{hor} U_{A_{1}A_{2}}^{hor}`.

Each gate increases the virtual bond dimension of PEPS tensors by a factor equal to the SVD rank of the gate `r`.

In `1D`, canonical structure of MPS makes truncation of a single bond dimension based on SVD singular values optimal in a Frobenius norm.
However, a loopy structure of PEPS prevents a canonical form, and a successful algorithm requires using optimization techniques on top of SVD.
The aim is to minimize the Frobenius norm of (a) the PEPS after application of the gate with virtual bond dimension increased to
:math:`D = r \times D` and (b) a new PEPS with the bond dimension truncated back to :math:`D`:

      # Action of a two-site gate on horizontal A_1-A_2 bond in the PEPS.
      # Line crossing indicates application of a swap gate.
             _______         _______
            |       |       |       |
            |  A_1  |-- D --|  A_2  |
            |_______|       |_______|
              |    \          |    \
              |\    D        /|     D
              ||\____\__r___/||      \
              ||/     \     \||       \
              |/       \     \|        \
              |     ____\__   |     ____\__
                   |       |       |       |
                   |  A_3  |-- D --|  A_4  |
                   |_______|       |_______|
                     |               |
                     |               |


To keep the PEPS representation compact, each application of the gate has to be followed by a truncation procedure to reduce the virtual bond dimension back to :math:`D`.


Truncation of the PEPS bond dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In `1D`, canonical structure of MPS makes truncation of a single bond dimension based on SVD singular values optimal in a Frobenius norm.
However, a loopy structure of PEPS prevents a canonical form, and a successful algorithm requires using optimization techniques on top of SVD.
The aim is to minimize the Frobenius norm of (a) the PEPS after application of the gate with virtual bond dimension increased to
:math:`D = r \times D` and (b) a new PEPS with the bond dimension truncated back to :math:`D`:

::

      (a)                                (b)
           _______         _______              _______         _______
          |       |       |       |            |       |       |       |
          |  A_1' |-r x D-|  A_2' |            |  A_1''|-- D --|  A_2''|
          |_______|       |_______|            |_______|       |_______|
             |   \          |    \       ~~~     |   \           |   \
             |    D         |     D      ~~~     |    D          |    D
                 __\____         __\____             __\____         __\____
                |       |       |       |           |       |       |       |
                |  A_3  |-- D --|  A_4  |           |  A_3  |-- D --|  A_4  |
                |_______|       |_______|           |_______|       |_______|
                  |               |                    |               |
                  |               |                    |               |


We denote the wavefunction in (a) by :math:`|\Psi(A_1',A_2')\rangle` and in (b) as :math:`|\Psi(A_1'',A_2'')\rangle`.
The Frobenius norm is denoted by :math:`d(A_1',A_2';A_1'',A_2'') = || |\Psi(A_1'',A_2'')\rangle - |\Psi(A_1',A_2')\rangle ||^{2}`
The aim is to minimalize it with respect to two isolated tensors :math:`A_{1}''` and :math:`A_{2}''` with the metric tensor representing the
rest of the lattice. In the minimal example above, it would just correspond to :math:`A_{3}` and :math:`A_{4}`.
More generally, a state-of-the-art optimization method in this context is the so-called Full Update [5], employing the Corner Transfer Matrix Renormalization Group to
obtain an environment of tensors to be optimized. It is however numerically expensive and might be unstable in some applications.
In YASTN, we implement a Neighborhood Tensor Update (:ref:`NTU<fpeps/algorithms_NTU:Neighborhood tensor update (NTU) algorithm>`) [5] that approximates
the metric tensor by contracting a small cluster of neighboring tensors.


Infinite PEPS (iPEPS)
^^^^^^^^^^^^^^^^^^^^^

Although finite PEPS is widely used, some of the best results have been obtained with infinite PEPS (iPEPS) [6].
It operates directly in the thermodynamic limit describing a system with translational invariance.
In iPEPS ansatz is formed by a unit cell of tensors repeated all over an infinite lattice.
A common example is a checkerboard lattice, which has two tensors A and B in a :math:`2\times 2` unit cell.

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



Corner transfer matrix renormalization group (CTMRG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exact contraction of a PEPS is exponentially hard [7]. It can be done site by site using the reduced tensor
:math:`a` which is obtained by tracing over the physical index in tensors :math:`A` and it's conjugate :math:`A^{\dag}`. Note that 
in the following digram the virtual legs of the peps tensor are labelled by :math:`t`(top), :math:`l`(left), :math:`b`(bottom), and :math:`r`(right) in 
an anticlockwise fashion. For the conjufate tensor similarly, it is labelled by :math:`t'`, :math:`l'`, :math:`b'` and :math:`r'`.

::

                           t'  t
                            \  \ 
                             \  \ 
                              \  \ 
                               \  \__
                               |     \ 
                               |      \
                               |       \
                               |        \
                               |      ___\______
                               |     |          |
                           ----------|    A     |-----------                         t' t
                          |    |     |__________|           |                         \  \
                          |    |            |    \          |                        __\__\__ 
                  l ______|    |            |     \         |______ r           l --|        |-- r
                    ______     |            |    _ \         ______ r'    =         |   a    |
                  l'      |    |            |   | \ \      |                    l'--|________|-- r'
                          |    |         ___|___|_ \ \     |                            \  \
                          |    |        |         | \ \    |                             \  \
                           ----|--------|   A'    |--\-\----                             b'  b
                               |        |_________|   \ \
                               |                \      \ \
                               |                 \      \ \
                               |                  \      \ \
                               |___________________\      \ \
                                                          b'  b



Swap gates are placed where the legs cross. By index bending and locally introducing the swap gates, we can 
get a simple structure for the contracted tensors on the :math:`2D` lattice while respecting the fermionic order. 
The state-of-the-art approximation technique for calculating expectation values in the case of iPEPS is the
Corner Transfer Matrix Renormalization Group (:ref:`CTMRG<fpeps/algorithms_ctmrg:corner transfer matrix renormalization group (ctmrg) algorithm>`).

CTMRG iteratively finds the environment of each tensor, representing the rest of the infinite lattice, in the form of four corner
tensors and edge tensors transfer matrices surrounding each unique tensor in the unit cell.


References & Related works
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. "Renormalization algorithms for Quantum-Many Body Systems in two and higher dimensions”, F. Verstraete and J. I. Cirac, `arXiv:cond-mat/0407066(2004) <https://arxiv.org/abs/cond-mat/0407066>`_
2. "A practical introduction to tensor networks: Matrix product states and projected entangled pair states", R. Orus, `Annals of Physics 349, 117 (2014) <https://arxiv.org/abs/1306.2164>`_
3. "Entanglement and tensor network states", J. Eisert, `arXiv:1308.3318 [quant-ph] (2013), <https://arxiv.org/abs/1308.3318>`_
4. "Simulation of strongly correlated fermions in two spatial dimensions with fermionic projected entangled-pair states", Philippe Corboz, Román Orús, Bela Bauer, and Guifré Vidal, `Phys. Rev. B 81, 165104 (2010) <https://arxiv.org/abs/0912.0646>`_
5. "Time evolution of an infinite projected entangled pair state: Neighborhood tensor update", Jacek Dziarmaga, `Phys. Rev. B 104, 094411 (2021) <https://arxiv.org/abs/2107.06635>`_
6. “Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions”, J. Jordan, R. Orus, G. Vidal, F. Verstraete, and J. I. Cirac, `Phys. Rev. Lett. 101, 250602 (2008) <https://arxiv.org/abs/cond-mat/0703788>`_
7. "On entropy growth and the hardness of simulating time evolution", N. Schuch, M. M. Wolf, K. G. H. Vollbrecht and J. I. Cirac, `New Journal of Physics 10(3), 033032 (2008) <https://arxiv.org/abs/0801.2078>`_