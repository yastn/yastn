Basics concepts
===============


Matrix product state
--------------------

The numerical simulation of quantum system has been proven to be hard due to exponentially growing size of matrix representation if the system. In particular, if the local Hilbert space for `j`-th particle is :math:`\mathcal{H}_j`of dimension `d` 
(e.g. `d=2` for qubit and `d=3` for qutrit) then for `N` particles it will be :math:`\mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1}` which is `d^N`. 
Tensor Network representation introduce a concept of efficient separation of variables which will iteratively split `j`-th particle from others by performing :ref:`tensor/algebra:Spectral decompositions` e.g. singular value decomposition (SVD). 
The SVD operation at first isolates `0`-th from `1`-to-`N`-th and then `1`-th from `2`-to-`N`-th until the state is decomposed to `N` tensors forming  `matrix product state` (MPS). 

.. math::

    \Psi \in \mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1} \xrightarrow{SVD}{\sum_{j_0,j_1\dots j_{N-1}} \sum_{\sigma_0,\sigma_1\dots \sigma_{N-1}} \, \Theta_{j_0,j_1\dots j_{N-1}}^{\sigma_0,\sigma_1\dots \sigma_{N-1}} \, A^{\sigma_0}_{,j_0} A^{\sigma_1}_{j_0,j_1} \dots A^{\sigma_{N-2}}_{j_{N-2},j_{N-1}} A^{\sigma_{N-1}}_{j_{N-1},}}

A single tensor `A_j` is an object of the size :math:`D_{j-1,j} \times d_j \times D_{j,j+1}`. 

::
    
    # individual tensor in MPS
                      ___
    A_j = D_{j-1,j}--|___|--D_{j,j+1}
                       |
                      d_j


The MPS forms one-dimensional structure with each tensor having a physical dimension `d` (`d_j` for general case when particles/quidict are different) and virtual dimensions `D_{i,j}` conecting `i`-th particle with `j`-th particle. `YAMPS` allows to perform computation on one dimensional MPS with open boundary conditions. 
The schematic picture for general MPS is visible below. Notice that for open boundary condition we always have edge tensor with dimension :math:`1\times d_0 \times D_{0,1}` on the left corner and :math:`D_{N-2,N-1} \times d_{N-1} \times` on the right edge.

::

        # matrix product state for the open boundary conditions N=6
           ___           ___           ___           ___           ___           ___  
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-D_{4,5}-|___|-1
            |             |             |             |             |             |   
           d_0           d_1           d_2           d_3           d_4           d_5

The above presents top-to-bottom construction however one can also think about MPS as an ansatz for calculation. This ansatz can be used for example to perform energy minisation in order to find an ground state for a system. The ansatz puts restriction on the manifold 
of states we can reach. The manifold can be controlled by changing the virtual dimension achiving exact representation when :math:`D\rightarrow\infty`. 


.. note::
        The symmetry of MPS is a consequence of symmetry of the `Tensor`s building it. Note that contruction of MPS requires common vitual dimensions to be matching.


Matrix product operator
-----------------------

`Matrix product operator` (MPO) is tensor product representation for an operator of which is :math:`d^N \times d^N` into `N` tensors of two physical and two virtual indicies. The concept of MPO is anologues to :ref:`theory/mps/basics:Matrix product state`. 

::

        # individual tensor in MPO

                    d^j
                    _|_
        D_{j-1,j}--|___|--D_{j,j+1}
                     |
                    d_j

`YAMPS` allows to encode an operator, e.g., Hamiltonian or expectation value, under open boundary condition. MPO with open boundary condition has a bound dimension `D=1` on the corners of the MPO chain. 

::

        # matrix product operator for the open boundary conditions N=6
           ___           ___           ___           ___           ___           ___  
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-D_{4,5}-|___|-1
            |             |             |             |             |             |   
           d_0           d_1           d_2           d_3           d_4           d_5


Canonical form 
---------------

The cost of matrix product representation is controled by virtual bond dimensions. That means that one wants to use  :ref:`tensor/algebra:Spectral decompositions` which allow for optimal trucation of the MPS. The truncation should be able to keep 
the most important components and discard those who are of lesser importrance. `Canonical decomposition` is integral form for every MPS :ref:`theory/mps/basics:Algorithms` including energy minisation with DMRG or time evolution with TDVP. 
In those algorithms we are act on each MPS tensor separately iteratively adjusting their form. We choose to put MPS in the `canonical form` with respect to `j`-to-`j+1`-th bond. In order to do that we 
split MPS by singular value decomposition into two parts representing Schmidt vectors :math:`L_{0,1\cdots j}` and :math:`R_{j+1,j+2\cdots N-1}` and central matrix :math:`\Lambda_{j,j+1}`.

::

        # canonical form of MPS from SVD
           ____________________                         ____________________ 
          |                    |   _________________   |                    |
        1-|   L_{0,1\cdots j}  |--|_\Lambda_{j,j+1}_|--|  R_{0,1\cdots N-1} |-1
          |____________________|                       |____________________|
                |||...|                                       |||...|
           {d_0 x d_1...x d_j}                          {d_{j+1}...x d_{N-1}}   


The central matrix :math:`\Lambda_{j,j+1}` is positive defines while :math:`L_{0,1\cdots j}` and :math:`R_{j+1,j+2\cdots N-1}` are unitary. Crucial aspect of the canonical form is that with unitary implies 
:math:`L^\dagger L=I` and  :math:`R^\dagger R=I`, where :math:`I` is an identity matrix which we obtain after contracting phisical indicies. The eigenvalues of :math:`\Lambda_{j,j+1}` can be efficiently trucated by discarging elements of smallest amplitude. 
If for every MPS tensor the left environwent is :math:`L_j^\dagger L_j=I` then we say that MPS in `left canonical form`. Similarly, if for every MPS tensor the right environwent is :math:`R_j^\dagger R_j=I` then we say that MPS in `right canonical form`.


Algorithms
----------

`Density matrix renormalisation group` (:ref:`theory/mps/algorithms:DMRG`) is an algorithm which seach for the MPS which extremize the expectation value for hermitian MPO, which is usually the Hamiltonian (operator for energy of the system). 

`Time-dependent variational principle` (:ref:`theory/mps/algorithms:TDVP`) allows for the evolution of a state :math:`\Psi` under a Hamiltonian :math:`\hat H`. The state after an evolution over time `t` changes the state to :math:`\Psi(t)=e^{- i t \hat H} \Psi`, where :math:`i` an imaginary number. `YAMPS` allows to perform TDVP for any MPS under hermitian MPO for a time `t` which in general can be complex. 



Measurements
------------

Norm of an MPS is equivalent to a norm of a vector and can be written as :math:`tr\{\Psi^\dagger \Psi\}` where :math:`tr\{.\}` is a trace operation or in bra-ket notation :math:`\langle\Psi||\Psi\rangle` where :math:`|\Psi\rangle` is the MPS and 
:math:`\langle\Psi|` is a cojugation of the MPS. This ovelap can be calculated for arbitrary pair of vectors of matching phisical indicies. After contracting phisical and virtual indicies an overlap gives a scalar value.


::

        # overlap between MPS \Psi and conjugate MPS \Phi^\dagger
                 ___    ___    ___    ___    ___    ___  
         \Psi = |___|--|___|--|___|--|___|--|___|--|___|
                  |      |      |      |      |      |       
                 _|_    _|_    _|_    _|_    _|_    _|_
 \Phi^\dagger = |___|--|___|--|___|--|___|--|___|--|___|


The expectation value of the operator of any operator :math:`\hat O` is calculated as :math:`tr\{\Psi^\dagger \hat O \Psi\}` or in bra-ket notation :math:`\langle\Psi|\hat O|\Psi\rangle`. The overlap can be calculated for any pair of vectors and any operator provided that they 
are consistent along physical indicies. After contracting phisical and virtual indicies an overlap gives a scalar value.


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
--------------------------

1. "Tensor Network Contractions: Methods and Applications to Quantum Many-Body Systems" Shi-Ju Ran, Emanuele Tirrito, Cheng Peng, Xi Chen, Luca Tagliacozzo, Gang Su, Maciej Lewenstein `Lecture Notes in Physics LNP, volume 964, (2020) <https://link.springer.com/book/10.1007/978-3-030-34489-4>`_
2. "The density-matrix renormalization group in the age of matrix product states" Ulrich Schollwoeck, `Annals of Physics, Volume 326, Issue 1, Pages 96-192, (2011) <https://arxiv.org/pdf/1008.3477.pdf>`_
3. "Time-Dependent Variational Principle for Quantum Lattices" Jutho Haegeman, J. Ignacio Cirac, Tobias J. Osborne, Iztok Pižorn, Henri Verschelde, and Frank Verstraete, `Phys. Rev. Lett. 107, 070601 (2011) <https://arxiv.org/abs/1103.0936v2>`_
