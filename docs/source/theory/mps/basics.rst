Basic concepts
==============

Matrix product state (MPS)
--------------------------

Numerical simulations of quantum many-body systems prove to be hard due to the exponentially
growing size of the matrix representation of the system with the number of constituent particles.
In particular, if the local Hilbert space for the *j*-th particle, :math:`\mathcal{H}_j`,
has dimension :math:`d_j` (e.g., :math:`d_j=2` for a qubit and :math:`d_j=3` for a qutrit), then :math:`N`
sites will have the total Hilbert space :math:`\mathcal{H} = \bigotimes_{j=0}^{N-1} \mathcal{H}_j` of dimension
:math:`d = \prod_{j=0}^{N-1} d_j`.
Bipartite tensor networks, such as matrix product states,
introduce a concept of efficient separation of variables which splits groups of particles
performing :ref:`spectral decomposition<tensor/algebra:spectral decompositions and truncation>`,
e.g., singular value decomposition (`SVD <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_).
To form a `matrix product state`, SVD at first isolates `0`-th site from `1-to-(N-1)`-th,
then `0-to-1` from `2-to-(N-1)`-th, and so on until the state is decomposed into a product of `N` tensors.

.. math::
    \Psi^{\sigma_0,\sigma_1\dots \sigma_{N-1}} \in \mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1} \xrightarrow{SVD}{\sum_{j_0,j_1\dots j_{N-1}} \, A^{\sigma_0}_{,j_0} A^{\sigma_1}_{j_0,j_1} \dots A^{\sigma_{N-2}}_{j_{N-2},j_{N-1}} A^{\sigma_{N-1}}_{j_{N-1},}}

A single tensor :math:`A_j` is a rank-3 array of size :math:`D_{j-1,j}{\times}d_j{\times}D_{j,j+1}`,
where :math:`d_j` is, previously mentioned, dimension of local Hilbert space
(or equivalently dimension of local variables) so--called *physical* dimension
and :math:`D_{j-1,j}` and :math:`D_{j,j+1}` are *virtual* bond dimensions shared between
sites :math:`j-1` and :math:`j` and between :math:`j` and :math:`j+1` respectively.
The bond dimension is introduced by the spectral decompositon and can be interpreted as dimension
encoding correlations between physical tensor variables. An example MPS tensor is shown in the diagram below.

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

The above discussion presents top-to-bottom construction of the MPS.
However, one can also treat MPS as an ansatz, that provides a good approximation of weakly entangled states.
The paradigmatic example is energy minimization to find the best approximation of the ground state of a system.
The MPS ansatz with fixed virtual dimensions defines the manifold of states we can reach. Virtual bond dimension
controls the quality of the approximation, where the exact representation is recovered for :math:`D_{j,j+1}\rightarrow\infty`.

Note that the construction of MPS requires common virtual spaces to be matching.
The MPS can be equipped with symmetries by making its individual tensors symmetric.
For symmetric tensors both bond dimensions and blocks have to be matching.

Matrix product operator (MPO)
-----------------------------

*Matrix product operator* is an efficient representation of an operator acting in the space of :math:`N` particles,
in general a square matrix of dimension of dimension :math:`d = \prod_{j=0}^{N-1} d_j`, by a product of :math:`N`
tensors with two physical and two virtual indices.
The concept of MPO is analogous to :ref:`MPS <theory/mps/basics:Matrix product state (MPS)>`.

A single tensor is a rank-4 array of size :math:`D_{j-1,j}{\times}d_j{\times}D_{j,j+1}{\times}d^j` as depicted below.

::

        # individual tensor in the MPO

                    d^j
                    _|_
        D_{j-1,j}--|___|--D_{j,j+1}
                     |
                    d_j


The *physical* dimensions :math:`d_j` and :math:`d^j` represent left (bra) and right (ket) vector spaces.
Similar to MPS, the *virtual* bond dimensions of a tensor are :math:`D_{j-1,j}` and :math:`D_{j,j+1}` shared between
sites :math:`j-1` and :math:`j` and between :math:`j` and :math:`j+1` respectively.
Those also encode correlations between physical tensor variables.
The MPO ansatz is suitable to represent operators, e.g., Hamiltonian, operators of observables, a density matrix,
transfer matrices.

:code:`yastn.tn.mps` implements operations on one-dimensional MPS with open boundary conditions.
The full MPO is diagrammatically depicted below. Notice, that for open boundary conditions,
terminal bond dimensions are :math:`D_{-1,0}=1` and :math:`D_{N-1,N}=1`.

::

        # MPO for N=5 sites with open boundary conditions

           d_0           d_1           d_2           d_3           d_4
           _|_           _|_           _|_           _|_           _|_
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-1
            |             |             |             |             |
           d_0           d_1           d_2           d_3           d_4



The MPO operators can be equipped with :ref:`fermionic order <tensor/configuration:YASTN configuration>`.
For fermionic operators, :meth:`swap gates<yastn.swap_gate>` are used to enforce antisymmetric nature of fermions and proper commutation relation of the operator.

Canonical form
--------------

The practical success of matrix product states is closely related to their canonical forms and the possibility to efficiently transform between them.
The canonical form is defined with respect to specific position in the MPS.
Let us choose to write `canonical form` with respect to the *j*-to-*j+1*-th bond of the MPS.
The diagrammatic representation of the MPS in the canonical form for that case is presented below.

::

        # canonical form of the MPS
           _________________                         ___________________
          |                 |   __________________  |                   |
          | L_{0,1\cdots j} |--|_\Lambda_{j,j+1}_|--| R_{j+1\cdots N-1} |
          |_________________|                       |___________________|
                |||...|                                     |||...|
          {d_0 x d_1...x d_j}                       {d_{j+1} x...x d_{N-1}}


In the canonical form, the MPS is split as in the Schmidt decomposition (or SVD) resulting in
:math:`D_{j,j+1}` pairs of left Schmidt vectors :math:`|L_{0,1\cdots j}\rangle` and right Schmidt vectors :math:`|R_{j+1,j+2\cdots N-1}\rangle`
weighted by Schmidt values :math:`\Lambda_{j,j+1}`.
More generally, instead of a diagonal positive matrix :math:`\Lambda_{j,j+1}`,
one often works with a central matrix (block) :math:`C_{j,j+1}` that can be obtained through :meth:`QR decompositions<yastn.linalg.qr>` decomposition.
Keeping the canonical form we ensure efficient compression and globally optimal truncation of :ref:`spectral decomposition<tensor/algebra:spectral decompositions and truncation>`
for a specific bond.

The left and right Schmidt vectors, forming columns of the matrix :math:`L_{0,1\cdots j}` and rows of the matrix :math:`R_{j+1,j+2\cdots N-1}` are orthonormal.
It implies that the overlaps :math:`L^\dagger L=I_L` and  :math:`R R^\dagger=I_R` (where physical indices are contracted) results in left (right) identity
matrices  :math:`I_{L(R)}` on virtual indices.
Canonical decomposition is also an integral element of every :ref:`MPS algorithm<theory/mps/basics:Algorithms>`, including energy minimization with
DMRG or time evolution with TDVP allowing to avoid treating the state norm operator explicitely and allowing for optimal truncation.

If the right and left overlaps involve part of the MPS, we say it is in a *mixed canonical* form with respect to a bond.
On the other hand, itf for every MPS tensor, the left environment is unitary, i.e., for corresponding left vectors :math:`L_j^\dagger L_j=I_L` on virtuals,
then we say that MPS is in the *left canonical form*.
Similarly is the same holds for right environment then we say that MPS is in the *right canonical form*.

.. note::
        In :code:`yastn.tn.mps` we refer to 0-th site as :code:`'first'`, and N-1-th site as :code:`'last'`.
        Namely, left-canonical MPS is canonized to the last site, and right-canonical MPS is canonized to the first site.


Algorithms
----------

:ref:`Density matrix renormalization group (DMRG)<mps/algorithms_dmrg:Density matrix renormalization group (DMRG)>`
is an algorithm searching for the MPS which extremizes the expectation value of the hermitian operator written as MPO, usually the Hamiltonian.

:ref:`Time-dependent variational principle (TDVP)<mps/algorithms_tdvp:Time-dependent variational principle (TDVP)>`
allows for a variational approximation of the evolution of a state :math:`\Psi(0)` under a Hamiltonian :math:`\hat H`, :math:`\Psi(t)=e^{- u t \hat H} \Psi(0)`.
TDVP can be performed for the evolution of MPS under MPO for a time `t`, where `u` is real or imaginary unit.


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
The overlap can be efficiently calculated for any pair of vectors and operator in the MPO form, :math:`\langle\Phi|\hat O|\Psi\rangle`, provided they are consistent along physical indices.

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


References & Related works
--------------------------

1. "Tensor Network Contractions: Methods and Applications to Quantum Many-Body Systems" Shi-Ju Ran, Emanuele Tirrito, Cheng Peng, Xi Chen, Luca Tagliacozzo, Gang Su, Maciej Lewenstein `Lecture Notes in Physics LNP, volume 964, (2020) <https://link.springer.com/book/10.1007/978-3-030-34489-4>`_
2. "The density-matrix renormalization group in the age of matrix product states" Ulrich Schollwoeck, `Annals of Physics, Volume 326, Issue 1, Pages 96-192, (2011) <https://arxiv.org/pdf/1008.3477.pdf>`_
3. "Time-Dependent Variational Principle for Quantum Lattices" Jutho Haegeman, J. Ignacio Cirac, Tobias J. Osborne, Iztok Pižorn, Henri Verschelde, and Frank Verstraete, `Phys. Rev. Lett. 107, 070601 (2011) <https://arxiv.org/abs/1103.0936v2>`_
4. "The Tensor Networks Anthology: Simulation techniques for many-body quantum lattice systems" Pietro Silvi, Ferdinand Tschirsich, Matthias Gerster, Johannes Jünemann, Daniel Jaschke, Matteo Rizzi, Simone Montangero, `SciPost Phys. Lect. Notes 8 (2019) <https://scipost.org/SciPostPhysLectNotes.8>`_
