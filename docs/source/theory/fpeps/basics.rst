Basic concepts
==============

Projected Entangled Pair States (PEPS)
--------------------------------------

The Projected Entangled Pair State (PEPS) [1,2] is a tensor network ansatz
typically appearing in the context of two-dimensional (`2D`) systems.
By construction, it satisfies the area law of entanglement entropy for such systems.
It allows efficient simulations of ground and thermal states
of many-body quantum systems in `2D` with their respective area laws [3].

We employ PEPS constructed from a set of tensors located on a `2D` lattice.
Each tensor has a physical leg, corresponding to the physical degrees of freedom of the lattice site,
and virtual legs (bonds) connecting it to neighboring tensors.
Here we implement a `2D` square lattice of size :math:`L_{x} \times L_{y}`,
with sites labeled by coordinates :math:`(x,y)` as shown below:


::

       # coordinates of the underlying 2D lattice

        ----------->
       |
       |  (0,0)      (0,1)      ...       (0,L_y-1)
       |
       |  (1,0)      (1,1)      ...       (1,L_y-1)
      \|/
            .          .                    .
            .                               .
            .                    .          .

        (L_x-1,0)  (L_x-1,1)    ...   (L_x-1,L_y-1)


Each tensor :math:`A_{(x,y)}` in PEPS is a rank-:math:`5` tensor of size
:math:`D_{(x-1,y),(x,y)} \times D_{(x,y-1),(x,y)} \times D_{(x,y),(x+1,y)} \times D_{(x,y),(x,y+1)} \times d_{(x,y)}`.

::

      # individual tensor of the PEPS lattice

                                 D_(x-1,y),(x,y)
                                         \
                                       ___\_____
                                      |         |
         A_(x,y)  =  D_(x,y-1),(x,y)--|  (x,y)  |--D_(x,y),(x,y+1)
                                      |_________|
                                        |      \
                                        |       \
                                     d_(x,y)  D_(x,y),(x+1,y)


The hermitian conjugate of the above tensor :math:`A_{x,y}^{\dagger}` is represented as its mirror image

::

      # conjugate tensor in PEPS

                                        d_(x,y)  D_(x,y),(x+1,y)
                                           |      /
                                          _|_____/_
                                         |         |
      A_(x,y)^\dagger = D_(x,y-1),(x,y)--|  (x,y)  |--D_(x,y),(x,y+1)
                                         |_________|
                                             /
                                            /
                                        D_(x-1,y),(x,y)


The module :code:`yastn.tn.fpeps` implements operations on PEPS both for
a finite system with open boundary conditions (OBC) and in the thermodynamic limit.
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


Fermionic anticommutation rules in PEPS
---------------------------------------

We follow the recipe introduced by Corboz et al. in Ref. [4].
It relies on (a) working with parity preserving tensors and (b) supplementing lines (legs)
crossings in a planar projection of the network with fermionic swap gates :meth:`yastn.swap_gate`.
The distribution of crossings (swap gates) follows from a chosen fermionic order,
but their application can be made local for contracting the network.
The leading numerical cost of contracting fermionic and bosonic (or spin) lattices are the same,
with swap gates adding a subleading overhead.
The module :code:`yastn.tn.fpeps` handles both fermionic and bosonic statistics,
controlled by :code:`fermionic` flag in the :ref:`tensor configuration <tensor/configuration:yastn configuration>`.
We use the name *fpeps* to emphasize the incorporation of fermionic statistics in the module.

Here we employ the fermionic order demonstrated in a :math:`3\times 3` PEPS example below.
Parity-preserving tensors permit moving the lines over tensors,
changing the placement of the swap gates without affecting the result.

::

              ____         ____         ____
             |____|-------|____|-------|____|
               |  \         |  \         |  \
               |  _\__      |  _\__      |  _\__
               | |____|-----|-|____|-----|-|____|
      |Psi> =  |   |  \     |   |  \     |   |  \
               |   |  _\__  |   |  _\__  |   |  _\__
               |   | |____|-|---|-|____|-|---|-|____|
               |   |   |    |   |   |    |   |   |
               |   |   |    |   |   |    |   |   |

               ---------------------------------->
                                 fermionic order


Infinite PEPS (iPEPS)
---------------------

Although finite PEPS is widely used, some of the best results have been obtained with infinite PEPS (iPEPS) [5].
It operates directly in the thermodynamic limit describing a system with translational invariance.
iPEPS ansatz is formed by a unit cell of tensors repeated all over an infinite lattice.
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


Time evolution
--------------

The simulation of time evolution of a quantum state is an ubiquitous problem.
We focus on real- or imaginary-time evolution generated by a local Hamiltonian :math:`H`.
For simplicity, we discuss here a PEPS defined on a :math:`2 \times 2` lattice.
Within the Suzuki-Trotter decomposition, the time evolution operator :math:`\exp(-d\beta H)`
for a small time step :math:`d\beta`, here in the imaginary time,
is approximated by a product of local two-site gates.

For a Hamiltonian with nearest-neighbor interactions definded on
a :math:`2 \times 2` lattice, :math:`H = \sum_{\rm bond} H_{\rm bond},` there are four disjoint bonds:
two horizontal :math:`1{-}2`, :math:`3{-}4`, and two vertical :math:`1{-}3`, :math:`2{-}4`.
The corresponding two-site gates :math:`U_{\rm bond} = \exp(-d\beta H_{\rm bond} / 2)`,
and a typical 2nd-order Suzuki-Trotter approximation gives

:math:`\exp(-d\beta H) \approx U_{1{-}2}^{\rm hor} U_{3{-}4}^{\rm hor} U_{1{-}3}^{\rm ver} U_{2{-}4}^{\rm ver} U_{2{-}4}^{\rm ver} U_{1{-}3}^{\rm ver} U_{3{-}4}^{\rm hor} U_{1{-}2}^{\rm hor}`.

Each gate increases the virtual bond dimension of PEPS tensors by a factor equal to the SVD rank of the gate `r`.

::

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


To keep the PEPS representation compact, each application of the gate has to be followed by
a truncation procedure to reduce the virtual bond dimension back to :math:`D`.

In `1D`, the canonical structure of the MPS makes the local truncation of bond dimension
based on SVD singular values globally optimal in a Frobenius norm.
However, a loopy structure of PEPS impedes the utilization of canonical forms,
and a successful algorithm requires using optimization techniques on top of SVD.
The aim is to minimize the Frobenius norm of: PEPS after the application of the Trotter gate
whose virtual bond dimension is now increased to :math:`r \times D` in (a),
and a new PEPS with the bond dimension truncated back to :math:`D` in (b).

::

      (a)                                  (b)
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


We denote the wavefunction in (a) by :math:`\Psi(A_1',A_2')` and in (b) as :math:`\Psi(A_1'',A_2'')`.
The normalized Frobenius norm of the difference is

:math:`d(A_1',A_2';A_1'',A_2'') = || \Psi(A_1',A_2') - \Psi(A_1'',A_2'') || / || \Psi(A_1',A_2') ||,`

which informs on truncation error. The aim is to minimalize it with respect to the two isolated tensors
:math:`A_{1}''` and :math:`A_{2}''` in the metric tensor representing the rest of the lattice.
In the minimal example above, it just corresponds to :math:`A_{3}` and :math:`A_{4}`.
More generally, a standard method in this context is the so-called Full Update scheme [5],
typically employing the Corner Transfer Matrix Renormalization Group to obtain environmental tensors.
It is, however, numerically expensive and might be unstable in some applications.

YASTN allows for a flexible selection of employed environment approximation.
In particular, we implement a Neighborhood Tensor Update (NTU) [6]
and its generalizations, that approximate the metric tensor by
numerically-exact contraction of a small cluster of neighboring tensors.

Minimization is performed via least-square optimization processes, where
one iterates between two truncated tensors, updating one with the other kept fixed.
An initial guess follows from Environment Assisted Truncation [7].


Neighborhood tensor update (NTU)
--------------------------------

Neighborhood Tensor Update can be regarded as a special case of a cluster update, see Refs [9,10],
where the number of neighboring lattice sites taken into account during truncation makes for a refining parameter.
The cluster update interpolates between a local truncation as in the simple update (SU) [8]
and the full update (FU) [5] that attempts to account for all correlations in the truncated state.
The NTU cluster includes the neighboring sites only as the metric tensor
to compute the Frobenius norm in :ref:`time evolution algorithm<theory/fpeps/basics:Time evolution>`.
In the diagram below, we have a checkerboard lattice with alternating tensors :math:`A` and :math:`B`
in the `2D` square lattice. The tensors :math:`A'` and :math:`B'` in the center are highlighted as
they have been updated by a NN :math:`2`-site gate of SVD-rank :math:`r`. The :code:`NN` environment
use only the sites directly surrounding the updated bond to calculate the metric tensor.

::

                  \             \
                  _\_____       _\_____
                 |       |     |       |
              ---|   B   |--D--|   A   |---
                 |_______|     |_______|
          \         |   \         |   \             \
         __\____    |  __\____    |  __\____       __\____
        |       |     ||     ||     ||     ||     |       |
     ---|   B   |--D--||  A' ||=   =||  B' ||--D--|   A   |---
        |_______|     ||_____||     ||_____||     |_______|
           |   \        |   \         |   \         |   \
           |    \       |  __\____    |  __\____    |    \
                          |       |     |       |
                       ---|   A   |--D--|   B   |---
                          |_______|     |_______|
                            |    \        |    \
                            |     \       |     \


It is calculated numerically exactly, warranting that the bond metric tensor is
Hermitian and non-negative down to the numerical precision.
A family of such environments is supported by :class:`yastn.tn.fpeps.EnvNTU`.


Corner transfer matrix renormalization group (CTMRG)
----------------------------------------------------

Calculation of expectation values of interests requires contraction of PEPS with its conjugate.
This amounts to contraction of PEPS network composed of reduced tensor :math:`a` which is
obtained by tracing over the physical index in tensors :math:`A` and it's conjugate :math:`A^{\dagger}`.
In YASTN, this is supported by :class:`yastn.tn.fpeps.DoublePepsTensor`.
Note that in the following diagram the virtual legs of the peps tensor are labelled by
:math:`t` (top), :math:`l` (left), :math:`b` (bottom), and :math:`r` (right) in an anticlockwise fashion.
For the conjugate tensor, similarly, they are labelled by :math:`{t'}`, :math:`{l'}`, :math:`{b'}` and :math:`{r'}`.
Swap gates are placed where the legs cross. This gives a simple structure for the contracted tensors on
the :math:`2D` lattice, respecting the global fermionic order.

::

                      t' t
                       \ \
                        | \
                       /  _\_____
                      /  |       |                            t' t
                   /--|--|   A   |-------\                     \ \
                  /   |  |_______|        \                   __\_\__
             l --/    |    |      \        \-- r         l --|       |-- r
                      |    |    __ \               ===       |   a   |
             l'--\    |   _|___|_ \ \      /-- r'        l'--|_______|-- r'
                  \   |  |       | \ \    /                      \ \
                   \--|--|   A'  |--\-\--/                        \ \
                      \  |_______|   \ \                          b' b
                       \         \    \ \
                        \________/     \ \
                                       b' b



The exact contraction of a PEPS is exponentially hard and one has to use efficient approximate contraction schemes.
One of the state-of-the-art for calculating expectation values in the case of PEPS employs
the Corner Transfer Matrix Renormalization Group (CTMRG) [11,12,13]. It iteratively finds
the environment of each unique tensor in the lattice, representing the rest of the lattice
in the form of four corner tensors and four edge tensors, for further description see
:ref:`CTMRG<fpeps/environments:Corner transfer matrix renormalization group (CTMRG)>`.
They are used to calculate the expectation values by contracting tensors---
with operators of interest acting on physical legs---and their environments.


Purification
------------

The thermal state for a Hamiltonian :math:`H` and inverse temperature :math:`\beta = 1/(k_B T)`
is given by :math:`\rho_{\beta} = \exp(-\beta H) / Z`, with :math:`Z = \text{Tr}(\exp(-\beta H))`.
Since in tensor networks pure states are more amenable to proper representation and manipulation,
we often choose to embed our thermal density matrix in a pure state by adding
an ancillary Hilbert space to the system Hilbert space. The thermal density matrix is obtained by
tracing out the ancilla degrees of freedom. The technique is outlined as follows.

We start with the system at infinite temperature, :math:`\beta=0`, where all states are equally probable.
This is described as a maximally mixed density matrix :math:`\rho_0`.
With the local basis :math:`\ket{e_{n}}` of dimension  :math:`d`, where for simplicity
we assume that the full Hilbert space of a many-body system is a product of identical local Hilbert spaces,

:math:`\rho_0 = \prod_{\rm sites} \sum_{n} \frac{1}{d} \ket{e_{n}}\bra{e_{n}}`.

A purified wave-function :math:`\ket{\psi_{0}}` at infinite temperature is
a maximally entangled state between the system and ancillary degrees of freedom,
where the latter is spanned by the same basis :math:`\ket{e_{n}}` as system Hilbert space
:math:`\ket{\psi_{0}} = \prod_{\rm sites} \frac{1}{\sqrt{d}} \sum_{n=1}^{d}\ket{e_{n}} \ket{e_{n}}`.
The state at finite temperature :math:`\beta` is obtained by evolving :math:`\ket{\psi_{0}}` in
imaginary time with operator :math:`U = \exp(-\frac{\beta}{2}H)` acting on system degrees of freedom:

:math:`\ket{\psi_{\beta}} = \exp\left(-\frac{\beta}{2} H \right) \ket{\psi_{0}}`

Now, to recover the thermal density matrix of the system, we take
the trace over the ancillary degrees of freedom of the total density matrix

:math:`\rho_{\beta} \sim \exp(-\beta H) \sim \text{Tr}_{\rm ancillas} \ket{\psi_{\beta}} \bra{\psi_{\beta}}`.

In YASTN, legs corresponding to system space and an ancilla space are always fused to
form one physical PEPS leg. During numerical simulations, the Hamiltonian acting on
system degrees of freedom is augmented with an identity operator acting on ancillas.


References & Related works
--------------------------

1. "Renormalization algorithms for Quantum-Many Body Systems in two and higher dimensions”, F. Verstraete and J. I. Cirac, `arXiv:cond-mat/0407066 (2004) <https://arxiv.org/abs/cond-mat/0407066>`_
2. "A practical introduction to tensor networks: Matrix product states and projected entangled pair states", R. Orus, `Ann. Phys. 349, 117 (2014) <https://arxiv.org/abs/1306.2164>`_
3. "Entanglement and tensor network states", J. Eisert, `arXiv:1308.3318 (2013), <https://arxiv.org/abs/1308.3318>`_
4. "Simulation of strongly correlated fermions in two spatial dimensions with fermionic projected entangled-pair states", Philippe Corboz, Román Orús, Bela Bauer, and Guifré Vidal, `Phys. Rev. B 81, 165104 (2010) <https://arxiv.org/abs/0912.0646>`_
5. “Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions”, J. Jordan, R. Orus, G. Vidal, F. Verstraete, and J. I. Cirac, `Phys. Rev. Lett. 101, 250602 (2008) <https://arxiv.org/abs/cond-mat/0703788>`_
6. "Time evolution of an infinite projected entangled pair state: Neighborhood tensor update", Jacek Dziarmaga, `Phys. Rev. B 104, 094411 (2021) <https://arxiv.org/abs/2107.06635>`_
7. "Finite-temperature tensor network study of the Hubbard model on an infinite square lattice" Aritra Sinha, Marek M. Rams, Piotr Czarnik, and Jacek Dziarmaga, `Phys. Rev. B 106, 195105 (2022) <https://arxiv.org/abs/2209.00985>`_
8. “Accurate Determination of Tensor Network State of Quantum Lattice Models in Two Dimensions”, H. C. Jiang, Z. Y. Weng, and T. Xiang, `Phys. Rev. Lett. 101, 090603 (2008) <https://arxiv.org/abs/0806.3719>`_
9. "Algorithms for finite projected entangled pair states", M. Lubasch, J. I. Cirac, and M.-C. Banyuls, `Phys. Rev. B 90, 064425 (2014) <https://arxiv.org/abs/1405.3259>`_
10. "Cluster update for tensor network states", L. Wang and F. Verstraete, `arXiv:1110.4362 (2011) <https://arxiv.org/abs/1110.4362>`_
11. “Corner Transfer Matrix Renormalization Group Method”, T. Nishino and K. Okunishi, `J. Phys. Soc. Jpn. 65, 891 (1996) <https://arxiv.org/abs/cond-mat/9507087>`_
12. "Simulation of two dimensional quantum systems on an infinite lattice revisited: corner transfer matrix for tensor contraction", R. Orus, G. Vidal, `Phys. Rev. B 80, 094403 (2009) <https://arxiv.org/abs/0905.3225>`_
13. "Competing States in the t-J Model: Uniform d-Wave State versus Stripe State (Supplemental Material)", P. Corboz, T. M. Rice, and M. Troyer, `Phys. Rev. Lett. 113, 046402 (2014) <https://arxiv.org/abs/1402.2859>`_
