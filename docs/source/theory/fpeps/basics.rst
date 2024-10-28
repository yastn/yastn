Basic concepts
==============

Projected Entangled Pair States (PEPS)
--------------------------------------

The Projected Entangled Pair State (PEPS) :ref:`[1] <ref1>` :ref:`[2] <ref2>` is a tensor network ansatz
typically appearing in the context of two-dimensional (2D) systems.
By construction, it satisfies the area law of entanglement entropy for such systems.
It allows efficient simulations of ground and thermal states
of many-body quantum systems in 2D with their respective area laws :ref:`[3] <ref3>`.

We employ PEPS constructed from a set of tensors located on a 2D square lattice.
Each tensor has a physical leg, corresponding to the physical degrees of freedom of the lattice site,
and virtual legs (bonds) connecting it to neighboring tensors.
Here we implement a 2D square lattice of size :math:`L_{x}{\times}L_{y}`,
with sites labeled by coordinates :math:`(x,y)` as shown below:


::

       # coordinates of the underlying 2D lattice

        ----------->
       |
       |   (0,0)     (0,1)   ...     (0,Ly-1)
       |
       |   (1,0)     (1,1)   ...     (1,Ly-1)
      \|/
             .         .               .
             .                         .
             .                .        .

        (Lx-1,0)  (Lx-1,1)   ...  (Lx-1,Ly-1)


Each tensor :math:`A_{(x,y)}` in PEPS is a rank-:math:`5` tensor defined as follows:

- **Four virtual bond dimensions** connecting neighboring tensors:
    - :math:`D_{(x-1,y),(x,y)}`: bond dimension connecting to the left neighbor,
    - :math:`D_{(x,y-1),(x,y)}`: bond dimension connecting below,
    - :math:`D_{(x,y),(x+1,y)}`: bond dimension connecting to the right,
    - :math:`D_{(x,y),(x,y+1)}`: bond dimension connecting above.

- **One physical dimension** :math:`d_{(x,y)}`, associated with the lattice site's local degree of freedom.

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


The hermitian conjugate of the above tensor :math:`A_{x,y}^{\dagger}` is represented as its mirror image of the tensor structure,
with **element-wise complex conjugation** of each entry in :math:`A_{(x,y)}`. This means that each element 
:math:`a_{ijkl} \rightarrow \overline{a_{ijkl}}` while the bond dimensions remain the same.

::

      #   # Conjugate tensor in PEPS with element-wise complex conjugation

                                        d_(x,y)  D_(x,y),(x+1,y)
                                           |      /
                                          _|_____/_
                                         |         |
      A_(x,y)^\dagger = D_(x,y-1),(x,y)--|  (x,y)  |--D_(x,y),(x,y+1)
                                         |_________|
                                             /
                                            /
                                        D_(x-1,y),(x,y)


The module :code:`yastn.tn.fpeps` supports PEPS ansatz both for
a finite system with open boundary conditions (OBC) and in the thermodynamic limit.
A schematic diagram of PEPS with OBC is given below.

::

      # PEPS for 6 sites arranged in a (Lx, Ly) = (2, 3) lattice with OBC.
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

We follow the recipe introduced by Corboz et al. in Ref. :ref:`[4] <ref4>`.
This approach relies on two main techniques:
(a) using parity-preserving tensors, which ensure that each tensor respects fermion parity, and
(b) adding fermionic swap gates through :meth:`yastn.swap_gate` at line (leg) crossings in a 
planar projection of the network.

In PEPS, the ordering of fermionic operators impacts their anticommutation properties, which are essential for accurate
simulations of fermionic systems. We establish a **fermionic order** to guide the application of swap gates, with each 
swap gate ensuring correct anticommutation for fermionic crossings. These crossings in the 2D plane project the 3D fermionic
ordering onto a 2D layout, where fermionic swap gates manage the antisymmetry. 

In terms of numerical cost, contracting fermionic and bosonic (or spin) PEPS networks is comparable. The swap gates introduce 
only a subleading overhead, making this approach efficient. The module :code:`yastn.tn.fpeps` handles both fermionic and bosonic
statistics, controlled by the :code:`fermionic` flag in the :ref:`tensor configuration <tensor/configuration:yastn configuration>`. 
We use the name :code:`fpeps` to emphasize the incorporation of fermionic statistics in the module.

Below, we illustrate the fermionic order in a :math:`3{\times}3` PEPS example. Using parity-preserving tensors allows flexibility in 
the placement of swap gates, as tensor parity invariance permits line crossings over or under the tensors without changing the physical results.

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

In this 2D representation, physical lines are placed on one edge of each tensor, allowing for a consistent and 
localized application of swap gates to uphold fermionic anticommutation, supporting efficient network contraction.



Infinite PEPS (iPEPS)
---------------------

While finite PEPS is widely used, infinite PEPS (iPEPS) :ref:`[5] <ref5>` has shown strong performance, especially 
in capturing properties directly in the thermodynamic limit with translational invariance. In iPEPS, a unit
cell of tensors is repeated over an infinite lattice.

A common setup is a **checkerboard lattice** with a :math:`2{\times}2` unit cell, containing two tensors, :math:`A` and :math:`B`, 
which alternate across the lattice. Each tensor represents local degrees of freedom. The **bond dimension** :math:`D` (typically same for all bonds) 
controls the maximum entanglement between neighboring tensors, defining the parameter for the computational cost.

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
For simplicity, we discuss here a PEPS defined on a :math:`2{\times}2` lattice with open boundaries.
Within the Suzuki-Trotter decomposition, the time evolution operator :math:`\exp(-d\beta H)`
for a small time step :math:`d\beta`, here in the imaginary time,
is approximated by a product of local two-site gates.

For a Hamiltonian with nearest-neighbor interactions, we define :math:`H` in terms of bond Hamiltonians
:math:`H_{\langle i,j \rangle}`, where :math:`\langle i,j \rangle` refers to a bond between neighboring 
sites (or tensors) :math:`A_i` and :math:`A_j`. On a :math:`2{\times}2` lattice with sites labeled :math:`1, 2, 3,` 
and :math:`4`, there are four disjoint bonds:

- Two horizontal bonds, :math:`H_{\langle 1,2 \rangle}` and :math:`H_{\langle 3,4 \rangle}`

- Two vertical bonds, :math:`H_{\langle 1,3 \rangle}` and :math:`H_{\langle 2,4 \rangle}`.

The corresponding two-site gates are :math:`U_{\langle i,j \rangle} = \exp(-d\beta H_{\langle i,j \rangle} / 2)`. Using a second-order Suzuki-Trotter approximation, the time evolution operator can be expressed as:

:math:`\exp(-d\beta H) \approx U_{\langle 1,2 \rangle}^{\text{hor}} U_{\langle 3,4 \rangle}^{\text{hor}} U_{\langle 1,3 \rangle}^{\text{ver}} U_{\langle 2,4 \rangle}^{\text{ver}} U_{\langle 2,4 \rangle}^{\text{ver}} U_{\langle 1,3 \rangle}^{\text{ver}} U_{\langle 3,4 \rangle}^{\text{hor}} U_{\langle 1,2 \rangle}^{\text{hor}}`.

Each gate application increases the virtual bond dimension of the PEPS tensors by a factor equal to the SVD rank of the gate `r`.

::

      # Action of a two-site gate on horizontal 1-2 bond in the PEPS.
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

In 1D systems, Matrix Product States (MPS) benefit from a **canonical form**, which enables
optimal truncation of bond dimensions using Singular Value Decomposition (SVD).
This truncation is globally optimal in the Frobenius norm because the canonical form
decouples sections of the MPS, allowing each bond to be truncated independently without 
impacting the global accuracy of the state. However, in PEPS, the two-dimensional structure
introduces loops, which hinder the use of canonical forms and make simple SVD-based truncation suboptimal.
A successful algorithm requires using optimization techniques on top of SVD to manage truncation effectively.
The aim is to minimize the Frobenius norm of: (a) PEPS after the application of the Trotter gate
whose virtual bond dimension is now increased to :math:`r{\times}D`,
and (b) a new PEPS with the bond dimension truncated back to :math:`D`.

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

which informs on truncation errors. The aim is to minimalize it with respect to the two isolated tensors
:math:`A_{1}''` and :math:`A_{2}''` in the metric tensor representing the rest of the lattice.
In the minimal example above, the latter just corresponds to :math:`A_{3}` and :math:`A_{4}`.
More generally, a standard method in this context is the so-called Full Update scheme :ref:`[5] <ref5>`,
typically employing the Corner Transfer Matrix Renormalization Group to obtain environmental tensors
approximating the rest of the lattice. It is, however,
numerically expensive and might be unstable in some applications.

YASTN allows for a flexible selection of employed environment approximation.
In particular, we implement a Neighborhood Tensor Update (NTU) scheme :ref:`[6] <ref6>`,
that approximate the metric tensor by numerically-exact contraction
of a small cluster of neighboring tensors.

Minimization is performed via least-square optimization processes, where
one iterates between two truncated tensors, updating one with the other kept fixed.
An initial guess follows from Environment Assisted Truncation :ref:`[7] <ref7>`,
improving upon a simple non-canonical SVD initialization.


Neighborhood tensor update (NTU)
--------------------------------

Neighborhood Tensor Update can be regarded as a special case of a cluster update, see Refs. :ref:`[9] <ref9>` and :ref:`[10] <ref10>`,
where the number of neighboring lattice sites taken into account during truncation makes for a refining parameter.
The cluster update interpolates between a local truncation as in the simple update (SU) :ref:`[8] <ref8>`
and the full update (FU) :ref:`[5] <ref5>` that attempts to account for all correlations in the truncated state.
The NTU cluster includes only the neighboring sites that can be contracted numerically exactly to obtain the metric tensor
employed in the Frobenius norm in :ref:`time evolution algorithm<theory/fpeps/basics:Time evolution>`.

In the diagram below, we have a checkerboard lattice with alternating tensors :math:`A` and :math:`B`
in the 2D square lattice. The tensors :math:`A'` and :math:`B'` in the center are highlighted as
they have been updated by a NN :math:`2`-site gate of SVD-rank :math:`r`. The :code:`NN` environment
uses only the sites directly surrounding the updated bond to calculate the metric tensor.

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


By construction, the metric tensor for the bond is always Hermitian and non-negative, ensuring numerical stability. A 
family of such environments is supported by :class:`yastn.tn.fpeps.EnvNTU`.


Corner transfer matrix renormalization group (CTMRG)
----------------------------------------------------

Calculation of expectation values of interests requires network contraction.
The exact contraction of a PEPS is exponentially hard, and
one has to use efficient approximate schemes in practice.
One of the state-of-the-art employs the Corner Transfer Matrix Renormalization Group (CTMRG).
Nishino and Okunishi first deployed CTMRG :ref:`[11] <ref11>` by extending the DMRG framework to give variational approximations for
Baxter's corner matrices of the vertex model. The subsequent development of CTMRG beyond the realm of :math:`C_{4v}` symmetric tensors
was accomplished by Orus and Vidal :ref:`[12] <ref12>`, with further refinements by Corboz :ref:`[13] <ref13>`.

The core idea behind CTMRG, both in the symmetric and nonsymmetric cases, remains the same.
The method approximates the contraction of the network by associating each lattice site
with a set of environmental tensors, where the approximation quality is controlled by the CTMRG bond dimension, :math:`\chi`,
which limits the size of these tensors. These environment tensors undergo a renormalization group procedure, iteratively converging towards their fixed-point forms.
The renormalization procedure involves:

- **Iterative Absorption and Truncation**: Initial corner and transfer tensors define the environment. During each iteration, tensors are contracted, decomposed and truncated to the bond dimension :math:`\chi\)`, balancing accuracy with efficiency.

- **Fixed-Point Convergence**: Over successive iterations, the environment tensors converge towards a stable fixed-point form, capturing the lattice environment accurately while maintaining computational feasibility.

In a 2D square lattice, the environment is represented by a combination of four corner :math:`C_{nw},C_{sw},C_{ne},C_{se}`
and four transfer :math:`T_{n},T_{w},T_{e},T_{s}` tensors of finite size, as depicted in the following figure. Tensor :math:`a` in the diagram 
below results from contracting a single-site PEPS tensor :math:`A` and its conjugate :math:`A^\dagger` over the physical dimension.

::

     _______     _______     _______
    |       |   |       |chi|       |
    |  C_nw |---|  T_n  |---|  C_ne |
    |_______|   |_______|   |_______|
        |           |           |
     ___|___     ___|___     ___|___
    |       |   |       |D^2|       |
    |  T_w  |---|   a   |---|  T_e  |
    |_______|   |_______|   |_______|
        |chi        |           |
     ___|___     ___|___     ___|___
    |       |   |       |   |       |
    |  C_sw |---|  T_s  |---|  C_se |
    |_______|   |_______|   |_______|


They are used to calculate expectation values by contracting PEPS site tensors and their environments.
When calculating expectation values, tensor :math:`a` is supplemented by any operators acting on the physical legs to account for observables.


Purification
------------

The thermal state for a Hamiltonian :math:`H` and inverse temperature :math:`\beta = 1/(k_B T)`
is given by :math:`\rho_{\beta} = \exp(-\beta H) / Z`, where :math:`Z = \text{Tr}(\exp(-\beta H))` is the partition function.
Since in tensor networks, pure states are more amenable to representation and manipulation,
we often embed our thermal density matrix in a pure state by adding
an ancillary Hilbert space to the system Hilbert space. The thermal density matrix is then obtained by
tracing out the ancilla degrees of freedom. This approach is outlined as follows.

We start with the system at infinite temperature, :math:`\beta=0`, where all states are equally probable.
This is described as a maximally mixed density matrix :math:`\rho_0`.
With the local basis :math:`\ket{e_{n}}` of dimension :math:`d`, where for simplicity
we assume that the full Hilbert space of a many-body system is a product of identical local Hilbert spaces,

:math:`\rho_0 = \prod_{\rm sites} \sum_{n} \frac{1}{d} \ket{e_{n}}\bra{e_{n}}`.

A purified wave-function :math:`\ket{\psi_{0}}` at infinite temperature is
a maximally entangled state between the system and ancillary degrees of freedom,
where the latter is spanned by the same basis :math:`\ket{e_{n}}` as the system Hilbert space:
:math:`\ket{\psi_{0}} = \prod_{\rm sites} \frac{1}{\sqrt{d}} \sum_{n=1}^{d}\ket{e_{n}} \ket{e_{n}}`.
The state at finite temperature :math:`\beta` is then obtained by evolving :math:`\ket{\psi_{0}}` in
imaginary time with operator :math:`U = \exp(-\frac{\beta}{2}H)` acting on the system degrees of freedom:

:math:`\ket{\psi_{\beta}} = \exp\left(-\frac{\beta}{2} H \right) \ket{\psi_{0}}`

To recover the thermal density matrix of the system, we take
the trace over the ancillary degrees of freedom of the total density matrix:

:math:`\rho_{\beta} = \frac{1}{Z} \text{Tr}_{\rm ancillas} \ket{\psi_{\beta}} \bra{\psi_{\beta}}`,

where :math:`Z = \text{Tr}(\exp(-\beta H))` ensures normalization.

In YASTN, legs corresponding to system space and ancilla space are always fused to
form one physical PEPS leg. During numerical simulations, the Hamiltonian acting on the system degrees of 
freedom is augmented with an identity operator acting on the ancillas. This means the Hamiltonian acts 
only on the system space, represented as:

:math:`H_{\text{total}} = H \otimes I_{\text{ancilla}},`

where :math:`H` is the Hamiltonian on the system Hilbert space, and :math:`I_{\text{ancilla}}` is the identity on the ancilla space. 
This setup ensures that evolution in imaginary time affects only the system's degrees of freedom.



References & Related Works
--------------------------

.. _ref1:

[1] "Renormalization algorithms for Quantum-Many Body Systems in two and higher dimensions”, F. Verstraete and J. I. Cirac. Available at: `arXiv:cond-mat/0407066 (2004) <https://arxiv.org/abs/cond-mat/0407066>`_

.. _ref2:

[2] "A practical introduction to tensor networks: Matrix product states and projected entangled pair states", R. Orus, `Ann. Phys. 349, 117 (2014) <https://arxiv.org/abs/1306.2164>`_

.. _ref3:

[3] "Entanglement and tensor network states", J. Eisert, `arXiv:1308.3318 (2013) <https://arxiv.org/abs/1308.3318>`_

.. _ref4:

[4] "Simulation of strongly correlated fermions in two spatial dimensions with fermionic projected entangled-pair states", P. Corboz, R. Orús, B. Bauer, and G. Vidal, `Phys. Rev. B 81, 165104 (2010) <https://arxiv.org/abs/0912.0646>`_

.. _ref5:

[5] “Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions”, J. Jordan, R. Orus, G. Vidal, F. Verstraete, and J. I. Cirac, `Phys. Rev. Lett. 101, 250602 (2008) <https://arxiv.org/abs/cond-mat/0703788>`_

.. _ref6:

[6] "Time evolution of an infinite projected entangled pair state: Neighborhood tensor update", Jacek Dziarmaga, `Phys. Rev. B 104, 094411 (2021) <https://arxiv.org/abs/2107.06635>`_

.. _ref7:

[7] "Finite-temperature tensor network study of the Hubbard model on an infinite square lattice" Aritra Sinha, M. M. Rams, P. Czarnik, and J. Dziarmaga, `Phys. Rev. B 106, 195105 (2022) <https://arxiv.org/abs/2209.00985>`_

.. _ref8:

[8] “Accurate Determination of Tensor Network State of Quantum Lattice Models in Two Dimensions”, H. C. Jiang, Z. Y. Weng, and T. Xiang, `Phys. Rev. Lett. 101, 090603 (2008) <https://arxiv.org/abs/0806.3719>`_

.. _ref9:

[9] "Algorithms for finite projected entangled pair states", M. Lubasch, J. I. Cirac, and M.-C. Bañuls, `Phys. Rev. B 90, 064425 (2014) <https://arxiv.org/abs/1405.3259>`_

.. _ref10:

[10] "Cluster update for tensor network states", L. Wang and F. Verstraete, `arXiv:1110.4362 (2011) <https://arxiv.org/abs/1110.4362>`_

.. _ref11:

[11] “Corner Transfer Matrix Renormalization Group Method”, T. Nishino and K. Okunishi, `J. Phys. Soc. Jpn. 65, 891 (1996) <https://arxiv.org/abs/cond-mat/9507087>`_

.. _ref12:

[12] "Simulation of two dimensional quantum systems on an infinite lattice revisited: corner transfer matrix for tensor contraction", R. Orus, G. Vidal, `Phys. Rev. B 80, 094403 (2009) <https://arxiv.org/abs/0905.3225>`_

.. _ref13:

[13] "Competing States in the t-J Model: Uniform d-Wave State versus Stripe State (Supplemental Material)", P. Corboz, T. M. Rice, and M. Troyer, `Phys. Rev. Lett. 113, 046402 (2014) <https://arxiv.org/abs/1402.2859>`_
