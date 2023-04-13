Basic concepts
--------------

Matrix product state (MPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The numerical simulation of quantum system has been proven to be hard due to the exponentially growing size of the matrix representation of the system. In particular, if the local Hilbert space for *j*-th particle is :math:`\mathcal{H}_j` of dimension *d* 
(e.g., *d=2* for qubit and *d=3* for qutrit) then for *N* particles it will be :math:`\mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1}` 
which is :math:`d^N`. 
Tensor Network representation introduces a concept of efficient separation of variables which will iteratively split `j`-th particle from others by performing :ref:`spectral decomposition<tensor/algebra:spectral decompositions and truncation>` e.g. singular value decomposition 
(`SVD <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_). 
To form a `matrix product state`, SVD at first isolates `0`th from `1-to-N-1`-th and then `1`-st from `2-to-N-1`-th until the state is decomposed into `N` tensors.

.. math::
    \Psi^{\sigma_0,\sigma_1\dots \sigma_{N-1}} \in \mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1} \xrightarrow{SVD}{\sum_{j_0,j_1\dots j_{N-1}} \, A^{\sigma_0}_{,j_0} A^{\sigma_1}_{j_0,j_1} \dots A^{\sigma_{N-2}}_{j_{N-2},j_{N-1}} A^{\sigma_{N-1}}_{j_{N-1},}}


A single tensor :math:`A_j` is a rank-3 array of size :math:`D_{j-1,j} \times d_j \times D_{j,j+1}`. 

::
    
    # individual tensor in MPS
                      ___
    A_j = D_{j-1,j}--|___|--D_{j,j+1}
                       |
                      d_j


The MPS forms a one-dimensional structure with each tensor having a physical dimension *d* (:math:`d_j` for general case when qudits at each site are different) and virtual dimensions 
:math:`D_{j-1,j}` connecting *j-1*-th site with *j*-th site. :code:`yastn.tn.mps` implements operations on one-dimensional MPS with open boundary conditions. 
The schematic picture for general MPS is shown below. Notice that for open boundary condition we always have edge tensor with dimension :math:`1\times d_0 \times D_{0,1}` 
on the left edge and :math:`D_{N-2,N-1} \times d_{N-1} \times` on the right edge.

::

        # matrix product state for N=6 sites with open boundary conditions
           ___           ___           ___           ___           ___           ___  
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-D_{4,5}-|___|-1
            |             |             |             |             |             |   
           d_0           d_1           d_2           d_3           d_4           d_5

The above presents top-to-bottom construction. However, one can also think about MPS as an ansatz for calculation. 
The paradigmatic example is energy minimization to find best approximation of the ground state of a system. 
The ansatz puts restriction on the manifold of states we can reach. 
The manifold can be controlled by changing the virtual dimension achieving exact representation when :math:`D\rightarrow\infty`. 


.. note::
        The MPS can be equipped with symmetries by making its individual tensors *A* symmetric. 
        Note that the construction of MPS requires common virtual spaces to be matching.


Matrix product operator (MPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Matrix product operator* is tensor product representation for an operator on space of *N* particles, in general a :math:`d^N \times d^N` matrix, by `N` tensors with two physical and two virtual indices.
The concept of MPO is analogous to :ref:`MPS <theory/mps/basics:Matrix product state (MPS)>`.

::

        # individual tensor in MPO

                    d^j
                    _|_
        D_{j-1,j}--|___|--D_{j,j+1}
                     |
                    d_j

API allows to encode operators, e.g., Hamiltonian or other operators associated with expectation values, under open boundary condition. 
MPO with open boundary condition has a bond dimension `D=1` on the edges of the MPO chain. 

::

        # matrix product operator acting on N=6 sites with open boundary conditions

           d_0           d_1           d_2           d_3           d_4           d_5
           _|_           _|_           _|_           _|_           _|_           _|_  
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-D_{4,5}-|___|-1
            |             |             |             |             |             |   
           d_0           d_1           d_2           d_3           d_4           d_5


Canonical form
^^^^^^^^^^^^^^

The computational costs associated with matrix product representation are controlled by virtual bond dimensions. That means that one wants to use :ref:`sepctral decomposition<tensor/algebra:spectral decompositions and truncation>` which allow for optimal truncation of the MPS. The truncation should be able to keep the most important components and discard those who are of lesser importance. *Canonical decomposition* is integral form for every :ref:`MPS algorithm<theory/mps/basics:Algorithms>`, including energy minimization with DMRG or time evolution with TDVP. 
In those algorithms we act on each MPS tensor separately and we iteratively adjust their form. We choose to put MPS in the `canonical form` with respect to *j*-to-*j+1*-th bond. In order to do that we split MPS by SVD into two parts representing :math:`D_{j,j+1}` left Schmidt vectors :math:`|L_{0,1\cdots j}\rangle`, :math:`D_{j,j+1}` right Schmidt vectors :math:`|R_{j+1,j+2\cdots N-1}\rangle`, and central matrix :math:`\Lambda_{j,j+1}`.

::

        # canonical form of MPS from SVD
           ____________________                         ____________________ 
          |                    |   _________________   |                    |
        1-|   L_{0,1\cdots j}  |--|_\Lambda_{j,j+1}_|--|  R_{0,1\cdots N-1} |-1
          |____________________|                       |____________________|
                |||...|                                       |||...|
           {d_0 x d_1...x d_j}                          {d_{j+1}...x d_{N-1}}   


The central matrix :math:`\Lambda_{j,j+1}` is real and positive. The left and right Schmidt vectors, interpreted as columns of matrices  
:math:`L_{0,1\cdots j}` and :math:`R_{j+1,j+2\cdots N-1}` respectively, form unitary matrices. Crucial aspect of the canonical form is that their unitarity implies :math:`L^\dagger L=I_{D_{j,j+1}}` and  :math:`R R^\dagger=I_{D_{j,j+1}}`, where :math:`I` is an identity matrix which we obtain after contracting physical indices. The eigenvalues of :math:`\Lambda_{j,j+1}` can be efficiently trucated by discarding elements of smallest magnitude. 
If for every MPS tensor the left environment is unitary, i.e., for corresponding left Schmidt vectors :math:`L_j^\dagger L_j=I`, then we say that MPS is in the `left canonical form`. Similarly, if for every MPS tensor the right environment is unitary, :math:`R_j R_j^\dagger=I`, then we say that MPS is in the `right canonical form`.


Algorithms
^^^^^^^^^^

`Density matrix renormalization group` 
(:ref:`DMRG<mps/algorithms_dmrg:density matrix renormalization group (dmrg) algorithm>`) 
is an algorithm searching for the MPS which extremizes the expectation value of hermitian operator written as MPO, usually the Hamiltonian. 

`Time-dependent variational principle` 
(:ref:`TDVP<mps/algorithms_tdvp:time-dependent variational principle (tdvp) algorithm>`) 
allows for variational approximation of the evolution of a state :math:`\Psi` under a Hamiltonian :math:`\hat H`. 
The state after an evolution over time `t` is :math:`\Psi(t)=e^{- i t \hat H} \Psi`, with :math:`i` an imaginary unit. 
TDVP can be performed for any MPS under MPO for a time `t`, real or imaginary.


Measurements
^^^^^^^^^^^^

Norm of an MPS is equivalent to a norm of a vector and can be written as :math:`tr\{\Psi^\dagger \Psi\}` where :math:`tr\{.\}` is a trace operation, or, in bra-ket notation, :math:`\langle\Psi|\Psi\rangle`, where :math:`|\Psi\rangle` is the MPS and 
:math:`\langle\Psi|` is a conjugation of the MPS. This overlap can be calculated for arbitrary pair of vectors of matching physical indices. After contracting physical and virtual indices, an overlap gives a scalar value.


::

        # overlap between MPS \Psi and conjugate MPS \Phi^\dagger
                 ___    ___    ___    ___    ___    ___  
         \Psi = |___|--|___|--|___|--|___|--|___|--|___|
                  |      |      |      |      |      |       
                 _|_    _|_    _|_    _|_    _|_    _|_
 \Phi^\dagger = |___|--|___|--|___|--|___|--|___|--|___|


The expectation value of operator :math:`\hat O` is calculated as :math:`tr\{\Psi^\dagger \hat O \Psi\}`, or, in bra-ket notation, :math:`\langle\Psi|\hat O|\Psi\rangle`. The expectation overlap can be efficiently calculated for any pair of vectors and any operator in MPO form provided that they 
are consistent along physical indices. After contracting physical and virtual indices, an overlap gives a scalar value.


::

        # overlap between MPS \Psi and conjugate MPS \Phi^\dagger and MPO \hat O
                 ___    ___    ___    ___    ___    ___  
         \Psi = |___|--|___|--|___|--|___|--|___|--|___|
                  |      |      |      |      |      |       
                 _|_    _|_    _|_    _|_    _|_    _|_
       \hat O = |___|--|___|--|___|--|___|--|___|--|___|
                  |      |      |      |      |      |      
                 _|_    _|_    _|_    _|_    _|_    _|_
 \Phi^\dagger = |___|--|___|--|___|--|___|--|___|--|___|


References & Related works
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. "Tensor Network Contractions: Methods and Applications to Quantum Many-Body Systems" Shi-Ju Ran, Emanuele Tirrito, Cheng Peng, Xi Chen, Luca Tagliacozzo, Gang Su, Maciej Lewenstein `Lecture Notes in Physics LNP, volume 964, (2020) <https://link.springer.com/book/10.1007/978-3-030-34489-4>`_
2. "The density-matrix renormalization group in the age of matrix product states" Ulrich Schollwoeck, `Annals of Physics, Volume 326, Issue 1, Pages 96-192, (2011) <https://arxiv.org/pdf/1008.3477.pdf>`_
3. "Time-Dependent Variational Principle for Quantum Lattices" Jutho Haegeman, J. Ignacio Cirac, Tobias J. Osborne, Iztok Pižorn, Henri Verschelde, and Frank Verstraete, `Phys. Rev. Lett. 107, 070601 (2011) <https://arxiv.org/abs/1103.0936v2>`_
4. "The Tensor Networks Anthology: Simulation techniques for many-body quantum lattice systems" Pietro Silvi, Ferdinand Tschirsich, Matthias Gerster, Johannes Jünemann, Daniel Jaschke, Matteo Rizzi, Simone Montangero, `SciPost Phys. Lect. Notes 8 (2019) <https://scipost.org/SciPostPhysLectNotes.8>`_
