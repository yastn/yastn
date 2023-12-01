Basic concepts
==============

Matrix product state (MPS)
--------------------------

Numerical simulations of quantum many-body systems prove to be hard due to the exponentially
growing size of the matrix representation of the system with the number of constituent particles.
In particular, if the local Hilbert space for *j*-th particle, :math:`\mathcal{H}_j`, has dimension :math:`d`
(e.g., :math:`d=2` for qubit and :math:`d=3` for qutrit) then :math:`N` sites
will have the total Hilbert space :math:`\mathcal{H} = \mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1}`
of dimension :math:`d^N`. Bipartite tensor networks, such as matrix product states,
introduce a concept of efficient separation of variables which splits groups of particles,
performing :ref:`spectral decomposition<tensor/algebra:spectral decompositions and truncation>`,
e.g. singular value decomposition (`SVD <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_).
To form a `matrix product state`, SVD at first isolates `0`-th site from `1-to-N-1`-th,
then `0-to-1` from `2-to-N-1`-th, and so on until the state is decomposed into `N` tensors.

.. math::
    \Psi^{\sigma_0,\sigma_1\dots \sigma_{N-1}} \in \mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1} \xrightarrow{SVD}{\sum_{j_0,j_1\dots j_{N-1}} \, A^{\sigma_0}_{,j_0} A^{\sigma_1}_{j_0,j_1} \dots A^{\sigma_{N-2}}_{j_{N-2},j_{N-1}} A^{\sigma_{N-1}}_{j_{N-1},}}

A single tensor :math:`A_j` is a rank-3 array of size :math:`D_{j-1,j} \times d_j \times D_{j,j+1}`.

::

    # individual tensor in MPS
                      ___
    A_j = D_{j-1,j}--|___|--D_{j,j+1}
                       |
                      d_j

The MPS forms a one-dimensional structure with each tensor having a physical dimension :math:`d` (:math:`d_j` for general case when qudits at each site are different) and virtual dimensions
:math:`D_{j-1,j}` connecting *j-1*-th site with *j*-th site. :code:`yastn.tn.mps` implements operations on one-dimensional MPS with open boundary conditions.
The schematic picture for general MPS is shown below. Notice that for open boundary conditions, we always have edge tensor with dimension :math:`1\times d_0 \times D_{0,1}`
on the left edge and :math:`D_{N-2,N-1} \times d_{N-1} \times 1` on the right edge.

::

        # matrix product state for N=5 sites with open boundary conditions
           ___           ___           ___           ___           ___
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-1
            |             |             |             |             |
           d_0           d_1           d_2           d_3           d_4

The above discussion presents top-to-bottom construction. However, one can also think about MPS as an ansatz.
The paradigmatic example is energy minimization to find the best approximation of the ground state of a system.
The MPS ansatz with fixed virtual dimensions fixes the manifold of states we can reach. Virtual bond dimension
controls the quality of the approximation, where the exact representation is recovered for :math:`D\rightarrow\infty`.

The MPS can be equipped with symmetries by making its individual tensors symmetric.
Note that the construction of MPS requires common virtual spaces to be matching.


Matrix product operator (MPO)
-----------------------------

*Matrix product operator* is tensor product representation for an operator on space of :math:`N` particles,
in general a :math:`d^N \times d^N` matrix, by :math:`N` tensors with two physical and two virtual indices.
The concept of MPO is analogous to :ref:`MPS <theory/mps/basics:Matrix product state (MPS)>`.

::

        # individual tensor in MPO

                    d^j
                    _|_
        D_{j-1,j}--|___|--D_{j,j+1}
                     |
                    d_j

It allows to encode operators, e.g., Hamiltonian or other operators associated with expectation values, or a density matrix.
MPO with open boundary conditions has a bond dimension :math:`D=1` on the edges of the MPO chain.

::

        # matrix product operator acting on N=5 sites with open boundary conditions

           d_0           d_1           d_2           d_3           d_4
           _|_           _|_           _|_           _|_           _|_
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-1
            |             |             |             |             |
           d_0           d_1           d_2           d_3           d_4


Canonical form
--------------

The practical success of matrix product states is closely related to the existence of its canonical forms.
Among others, this allow using local :ref:`sepctral decomposition<tensor/algebra:spectral decompositions and truncation>` to perform globally optimal truncation of the MPS on a specific bond.
*Canonical decomposition* is also an integral element of every :ref:`MPS algorithm<theory/mps/basics:Algorithms>`, including energy minimization with DMRG or time evolution with TDVP.
In those algorithms, one locally updates individual MPS tensors, adjusting its canonical form while sweeping through the lattice.

We choose to put MPS in the `canonical form` with respect to the *j*-to-*j+1*-th bond.
This is equivalent to performing Schmidt decomposition of the MPS with :math:`D_{j,j+1}` left Schmidt vectors :math:`|L_{0,1\cdots j}\rangle`,
:math:`D_{j,j+1}` right Schmidt vectors :math:`|R_{j+1,j+2\cdots N-1}\rangle`, and a diagonal matrix of Schmidt values :math:`\Lambda_{j,j+1}`.
More generally, instead of a diagonal positive matrix :math:`\Lambda_{j,j+1}`,
one often works with a central matrix (block) :math:`C_{j,j+1}`, which SVD gives the Schmidt values on the bond.

::

        # canonical form of MPS from SVD
           _________________                         ___________________
          |                 |   __________________  |                   |
          | L_{0,1\cdots j} |--|_\Lambda_{j,j+1}_|--| R_{j+1\cdots N-1} |
          |_________________|                       |___________________|
                |||...|                                     |||...|
          {d_0 x d_1...x d_j}                       {d_{j+1} x...x d_{N-1}}


The left and right Schmidt vectors, forming columns of the matrix :math:`L_{0,1\cdots j}` and rows of the matrix :math:`R_{j+1,j+2\cdots N-1}` are orthonormal.
It implies that :math:`L^\dagger L=I` and  :math:`R R^\dagger=I`, where :math:`I` is an identity matrix on the virtual bond, which we obtain after contracting physical indices.
The virtual bond of MPS can be efficiently truncated by discarding singular values :math:`\Lambda_{j,j+1}` of the smallest magnitude.
If, for every MPS tensor, the left environment is unitary, i.e., for corresponding left vectors :math:`L_j^\dagger L_j=I`, then we say that MPS is in the `left canonical form`.
It can be obtained by consecutive :meth:`QR decompositions<yastn.linalg.qr>` of each MPS tensor, starting from `0`-th, where the unitary part forms a new tensor, and the upper-triangular part becomes a central tensor that gets attached to the subsequent MPS tensor.
Similarly, if for every MPS tensor the right environment is unitary, :math:`R_j R_j^\dagger=I`, then we say that MPS is in the `right canonical form`.
A mixed canonical form with respect to some bond or MPS site interpolates between those two extremes.

.. note::
        In :code:`yastn.tn.mps` we refer to `0`-th site as :code:`'first'`, and `N-1`-th site as :code:`'last'`.
        Namely, left-canonical MPS is canonized to the last site, and right-canonical MPS is canonized to the first site.


Algorithms
----------

:ref:`Density matrix renormalization group (DMRG)<mps/algorithms_dmrg:Density matrix renormalization group (DMRG)>`
is an algorithm searching for the MPS which extremizes the expectation value of the hermitian operator written as MPO, usually the Hamiltonian.

:ref:`Time-dependent variational principle (TDVP)<mps/algorithms_tdvp:Time-dependent variational principle (TDVP)>`
allows for a variational approximation of the evolution of a state :math:`\Psi(0)` under a Hamiltonian :math:`\hat H`, :math:`\Psi(t)=e^{- i t \hat H} \Psi(0)`, with :math:`i` an imaginary unit.
TDVP can be performed for the evolution of MPS under MPO for a time `t`, real or imaginary.


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
