Basic concepts
==============

Matrix product state (MPS)
--------------------------

Numerical simulations of quantum many-body systems prove to be hard due to the size of
the matrix representation of the system, which grows exponentially with the number of elementary system components.
In particular, if the local Hilbert space for the *j*-th component, :math:`\mathcal{H}_j`,
has dimension :math:`d_j` (e.g., :math:`d_j=2` for a qubit and :math:`d_j=3` for a qutrit), then :math:`N`
sites will have the total Hilbert space :math:`\mathcal{H} = \bigotimes_{j=0}^{N-1} \mathcal{H}_j` of dimension
:math:`d = \prod_{j=0}^{N-1} d_j`.
Bipartite tensor networks, such as matrix product states, can be conceptually derived
sequentially separating groups of components using singular value decomposition (:meth:`SVD<yastn.linalg.svd>`).
To form a **matrix product state**, SVD at first divides `0`-th site from `1-to-(N-1)`-th,
then `0-to-1` from `2-to-(N-1)`-th, and so on until the state is decomposed into a product of `N` tensors.

.. math::
    \Psi^{\sigma_0,\sigma_1\dots \sigma_{N-1}} \in \mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1} \xrightarrow{SVD}{\sum_{j_0,j_1\dots j_{N-1}} \, A^{\sigma_0}_{,j_0} A^{\sigma_1}_{j_0,j_1} \dots A^{\sigma_{N-2}}_{j_{N-2},j_{N-1}} A^{\sigma_{N-1}}_{j_{N-1},}}

A single tensor :math:`A_j` is a rank-3 array of size :math:`D_{j-1,j}{\times}d_j{\times}D_{j,j+1}`,
where :math:`d_j` is the dimension of local Hilbert space, i.e., a *physical* dimension,
and :math:`D_{j-1,j}` and :math:`D_{j,j+1}` are *virtual* bond dimensions between
sites :math:`j-1` and :math:`j` and between :math:`j` and :math:`j+1`, respectively.
The bond dimension is introduced by spectral decomposition and allows encoding correlations
between physical degrees of freedom. An example of an MPS tensor is shown in the diagram below.

::

    # individual tensor in the MPS
                      ___
    A_j = D_{j-1,j}--|___|--D_{j,j+1}
                       |
                      d_j

:code:`yastn.tn.mps` implements operations on one-dimensional MPS with open boundary conditions.
The schematic picture for general MPS is shown below. Notice that for open boundary conditions,
we always have edge tensor with dimension :math:`1\times d_0{\times}D_{0,1}`
on the left edge and :math:`D_{N-2,N-1}{\times}d_{N-1}{\times}1` on the right edge, i.e. terminal
bond dimensions are :math:`D_{-1,0}=1` and :math:`D_{N-1,N}=1`.

::

        # MPS for N=5 sites with open boundary conditions
           ___           ___           ___           ___           ___
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-1
            |             |             |             |             |
           d_0           d_1           d_2           d_3           d_4

The above discussion presents the top-to-bottom construction of the MPS.
However, one can also treat MPS as an ansatz, that provides a good approximation of weakly entangled states.
The paradigmatic example is energy minimization finding the best approximation of the ground state of a system.
The MPS ansatz with fixed virtual dimensions defines the manifold of states we can reach. Virtual bond dimension
controls the quality of the approximation, where the exact representation is recovered for :math:`D_{j,j+1}\rightarrow\infty`.

The MPS can be equipped with symmetries by making its tensors symmetric.
Note that common virtual spaces of neighboring MPS tensors have to match to allow their contraction.


Matrix product operator (MPO)
-----------------------------

**Matrix product operator** is an efficient representation of an operator acting in the space of :math:`N` sites,
equivalent to a square matrix of dimension :math:`d = \prod_{j=0}^{N-1} d_j`,
by a product of :math:`N` tensors with two physical and two virtual indices.
The concept of MPO is analogous to :ref:`MPS <theory/mps/basics:Matrix product state (MPS)>`.

A single tensor is a rank-4 array of size :math:`D_{j-1,j}{\times}d_j{\times}D_{j,j+1}{\times}d^j` as depicted below.

::

        # individual tensor in the MPO

                    d^j
                    _|_
        D_{j-1,j}--|___|--D_{j,j+1}
                     |
                    d_j


The *physical* dimensions :math:`d_j` and :math:`d^j` represent ket and bra vector spaces, respectively.
As for MPS, the *virtual* bond dimension :math:`D_{k,k+1}` connects sites :math:`k` and :math:`k+1`.
The MPO ansatz is suitable to represent various operators, e.g., Hamiltonians, observables,
density matrices, transfer matrices, etc.

The full MPO is diagrammatically depicted below. Notice, that for open boundary conditions,
terminal bond dimensions are :math:`D_{-1,0}=1` and :math:`D_{N-1,N}=1`.

::

        # MPO for N=5 sites with open boundary conditions

           d_0           d_1           d_2           d_3           d_4
           _|_           _|_           _|_           _|_           _|_
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-1
            |             |             |             |             |
           d_0           d_1           d_2           d_3           d_4



The MPO operators can accommodate fermionic statistics.
For fermionic operators, :meth:`swap gates<yastn.swap_gate>` can be used to incorporate the antisymmetric nature of fermionic operators into MPO,
being equivalent to the introduction of a Jordan-Wigner string.


Canonical form
--------------

The practical success of matrix product states is closely tied to their canonical forms,
accompanied by exact and efficient procedures to transform between them.
The canonical form is defined with respect to a specific position in the MPS.
Let us focus on a canonical form with respect to the *j*-to-*j+1*-th bond of the MPS,
which we represent diagrammatically below.

::

        # canonical form of the MPS
           _________________                         ___________________
          |                 |   __________________  |                   |
          | L_{0,1,...,j-1} |--|_\Lambda_{j-1,j}_|--| R_{j,j+1,...,N-1} |
          |_________________|                       |___________________|
                |||...|                                     |||...|
         d_0 x d_1 x...x d_{j-1}                       d_j x ... x d_{N-1}


In the canonical form, the MPS is split as in the Schmidt decomposition (or SVD) resulting in
:math:`D_{j-1,j}` pairs of left and right Schmidt vectors, :math:`|L_{0,1,\ldots,j-1}\rangle` and :math:`|R_{j,j+1,\ldots,N-1}\rangle` (composed from respective MPS tensors),
weighted by Schmidt values :math:`\Lambda_{j-1,j}`.
More generally, instead of a diagonal positive matrix :math:`\Lambda_{j-1,j}`,
one often works with a central matrix (block) :math:`C_{j-1,j}` that can be obtained through :meth:`QR<yastn.linalg.qr>` decomposition.
Keeping the canonical form, we ensure efficient compression and globally optimal truncation of a specific bond.

The left and right Schmidt vectors, forming columns of the matrix :math:`L_{0,1,\ldots,j-1}` and rows of the matrix :math:`R_{j,j+1,\ldots,N-1}` are orthonormal.
It implies that the overlaps :math:`L^\dagger L=I_L` and  :math:`R R^\dagger=I_R` (where physical indices are contracted) result in left (right) identity
matrices :math:`I_{L(R)}` on virtual indices.
Canonical decomposition is also an integral element of common :ref:`MPS algorithm<theory/mps/basics:Algorithms>`, including energy minimization with
DMRG or time evolution with TDVP, allowing to avoid treating the state norm operator explicitly and to form a projection on MPS tangent space.

If the left environment of every MPS tensor is unitary, i.e., for all corresponding left vectors :math:`L_j^\dagger L_j=I_L`, then we say that MPS is in the *left canonical form*.
Similarly, if the same holds for all right environments, then we say that MPS is in the *right canonical form*.
In the intermediate situation, where all environments to the left of some bond and all right environments to the right of the bond are unitary,
we say MPS is in a *mixed canonical* form with respect to that bond.

.. note::
        In :code:`yastn.tn.mps` we refer to 0-th site as :code:`'first'`, and N-1-th site as :code:`'last'`.
        Namely, left-canonical MPS is canonized to the last site, and right-canonical MPS is canonized to the first site.


Algorithms
----------

:ref:`Density matrix renormalization group (DMRG)<mps/algorithms_dmrg:Density matrix renormalization group (DMRG)>`
is an algorithm searching for the MPS, which extremizes the expectation value of a hermitian operator written as MPO, usually the Hamiltonian.

:ref:`Time-dependent variational principle (TDVP)<mps/algorithms_tdvp:Time-dependent variational principle (TDVP)>`
allows for a variational approximation of the evolution of an MPS state :math:`\Psi(0)` under a Hamiltonian MPO
:math:`\hat H` for time :math:`t`, :math:`\Psi(t)=e^{- u t \hat H} \Psi(0)`, where :math:`u` is real or imaginary unit.


Measurements
------------

Scalar product :math:`\langle\Phi|\Psi\rangle`, written in bra-ket notation, where :math:`|\Psi\rangle` is the MPS and
:math:`\langle\Phi|` is a conjugation of the MPS. This overlap can be calculated for an arbitrary pair of vectors of matching physical indices.
After contracting physical and virtual indices, an overlap gives a scalar value.

::

  # overlap between MPS \Psi and conjugate MPS \Phi^\dagger
                 ___    ___    ___    ___    ___
         \Psi = |___|--|___|--|___|--|___|--|___|
                  |      |      |      |      |
                 _|_    _|_    _|_    _|_    _|_
 \Phi^\dagger = |___|--|___|--|___|--|___|--|___|


The expectation value of operator :math:`\hat O` is calculated as :math:`\langle\Psi|\hat O|\Psi\rangle`.
The overlap can be efficiently calculated for any pair of vectors and operator in the MPO form, :math:`\langle\Phi|\hat O|\Psi\rangle`, provided their physical indices are matching.

::

  # overlap between MPS \Psi and conjugate MPS \Phi^\dagger and MPO \hat O
                 ___    ___    ___    ___    ___
         \Psi = |___|--|___|--|___|--|___|--|___|
                  |      |      |      |      |
                 _|_    _|_    _|_    _|_    _|_
       \hat O = |___|--|___|--|___|--|___|--|___|
                  |      |      |      |      |
                 _|_    _|_    _|_    _|_    _|_
 \Phi^\dagger = |___|--|___|--|___|--|___|--|___|


References
----------

1. "Matrix product states, projected entangled pair states, and variational renormalization group methods for quantum spin systems", F. Verstraete, J.I. Cirac, V. Murga, `Advances in Physics 57, 143-224 (2009) <https://arxiv.org/pdf/0907.2796>`_
2. "The density-matrix renormalization group in the age of matrix product states" U. Schollwoeck, `Annals of Physics 326, 96-192 (2011) <https://arxiv.org/pdf/1008.3477.pdf>`_
3. "The Tensor Networks Anthology: Simulation techniques for many-body quantum lattice systems" P. Silvi, F. Tschirsich, M. Gerster, J. Jünemann, D. Jaschke, M. Rizzi, S. Montangero, `SciPost Phys. Lect. Notes 8 (2019) <https://scipost.org/SciPostPhysLectNotes.8>`_
4. "Tensor network contractions: Methods and applications to quantum many-body systems" S.-J. Ran, E. Tirrito, C. Peng, X. Chen, L. Tagliacozzo, G. Su, M. Lewenstein `Lecture Notes in Physics, Volume 964, (2020) <https://link.springer.com/book/10.1007/978-3-030-34489-4>`_
