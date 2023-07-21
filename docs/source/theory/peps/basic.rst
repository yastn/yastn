==============
Basic concepts
==============

Projected Entangled Pair States (PEPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Projected Entangled Pair State (PEPS)[1] is a two-dimensional (2D) extension of Matrix Product States (MPS). It offers a more efficient representation for many-body 
quantum systems in two dimensions. PEPS is constructed from a set of tensors located on a 2D lattice. Each tensor corresponds to a lattice site with physical degrees
of freedom and is connected to its nearest neighbors by virtual bonds.

The 2D lattice of size :math:`Lx \times Ly` is labeled with coordinates :math:`(x,y)` as shown in the table below:


::
        # coordinates of the underlying 2D lattice:

        -------------------------------------------------------- 
        |
        |   (0,0)     (0,1)     (0,2)     ...     (0, L_y)
        |
        |   (1,0)     (1,1)     (1,2)     ...     (1, L_y)
        |          
        |     .          .                            .
        |     .                   .                   .
        |     .                             .         .
        |
        |   (L_x,0)   (L_x,1)   (L_x,2)   ...     (L_x,L_y)
        

Each tensor :math:`A_{x,y}` in the PEPS is a rank-5 array of size 
:math:`D_{(x-1,y),(x,y)} \times D_{(x,y-1),(x,y)} \times D_{(x,y),(x+1,y)} \times D_{(x,y),(x,y+1)} \times d_{(x,y)} `

::
    
    # individual tensor in PEPS       
      
                                D_(x-1,y),(x,y)            
                                            \                                                  
                                          ___\_________                                             
                                         |             |                                       
             A_(x,y) = D_(x,y-1),(x,y) --|    (x,y)    |-- D_(x,y),(x,y+1)                  
                                         |_____________|                                     
                                            |       \ 
                                            |        \   
                                        d_(x,y)  D_(x,y),(x+1,y)


The PEPS forms a two-dimensional structure with each tensor having a physical dimension *d* (:math:`d_{(x,y)}` for general case when
at each site the local Hilbert space is different) and virtual dimensions :math:`D_{(x-1,y),(x,y)}` connecting *(x-1,y)*-th site with *(x,y)*-th site. 
:code:`yastn.tn.fpeps` implements operations PEPS with both open boundary conditions(OBC) and at thermodynamic limit. The schematic picture for 
general PEPS with OBC are given below.
 
::

        # projected entangled pair states for N=6 sites arranged in a L_x=2 times L_y=3 lattice with open boundary conditions
 
              1                                 1                                  1
               \                                 \                                  \  
            ____\_______                       ___\________                       ___\________
           |            |                     |            |                     |            |
        1--|   (0,0)    |--  D_(0,0),(0,1)  --|   (0,1)    |--  D_(0,1),(0,2)  --|    (0,2)   |-- 1
           |____________|                     |____________|                     |____________|
              |      \                           |      \                           |     \   
              |       \                          |       \                          |      \
        d_(0,0)    D_(0,0),(1,0)            d_(0,1)   D_(0,1),(1,1)             d_(0,2)  D_(0,2),(1,2)
                        \                                  \                                 \
                     ____\_______                       ____\______                       ____\_______
                    |            |                     |           |                     |            |
                 1--|   (1,0)    |--  D_(1,0),(1,1)  --|   (1,1)   |--  D_(1,1),(1,2)  --|   (1,2)    |-- 1
                    |____________|                     |___________|                     |____________|
                       |      \                           |     \                           |    | 
                       |       \                          |      \                          |    |                         
                   d_(1,0)      1                      d_(1,1)    1                     d_(1,2)   1        

Notice that for OBC we have left edge tensors with dimension :math:`D_{(x-1,y),(x,y)} \times 1 \times D_{(x,y),(x+1,y)} \times D_{(x,y),(x,y+1)} \times d_{(x,y)}` 
for :math:`y = 0` and :math:`x=1` to :math:`x=L_{x}-2`, right edge tensors with dimension :math:`D_{(x-1,y),(x,y)} \times D_{(x,y-1),(x,y)} \times D_{(x,y),(x+1,y)} \times 1 \times d_{(x,y)}`
for :math:`y = L_y` and :math:`x=1` to :math:`x=L_{x}-2`, top edge tensors with dimension :math:`1 \times D_{(x,y-1),(x,y)} \times D_{(x,y),(x+1,y)} \times D_{(x,y),(x,y+1)} \times d_{(x,y)} ` 
for :math:`x = 0` and :math:`y=1` to :math:`y=L_{y}-2`, bottom edge tensors with dimension :math:`D_{(x-1,y),(x,y)} \times D_{(x,y-1),(x,y)} \times 1 \times D_{(x,y),(x,y+1)} \times d_{(x,y)} ` 
for :math:`x = L_x`and :math:`y=1` to :math:`y=L_{y}-2`, and four corner tensors :math:`A_{(0,0)}, A_{(L_{x}-1,0)}, A_{(L_{x}-1, L_{y}-1)}` and :math:`A_{(0,L_{y}-1)}` with dimensions
:math:`1 \times 1 \times D_{(0,0),(1,0)} \times D_{(0,0),(0,1)}`, :math:`D_{(L_{x}-2,0),(L_{x}-1,0)} \times 1 \times 1 \times D_{(L_{x}-1,0),(L_{x}-1,1)}`, 
:math:`D_{(L_{x}-2,L_{y}-1),(L_{x}-1,L_{y}-1)} \times D_{(L_{x}-1,L_{y}-2),(L_{x}-1,L_{y}-1)} \times 1 \times 1` and :math:`1 \times D_{(0,L_{y}-2),(0,L_{y}-1)} \times D_{(0,L_{y}-1),(1,L_{y}-1)} \times 1`
The PEPS construction satifies area law of entanglement entropy and Hence it can be used to simulate ground states and thermal states which have their corresponding area laws. 



Implementation of Fermionic Anticommutation Rules in PEPS 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Ref.[2] Corboz et al. implemented a fermionic version of the PEPS algorithm. The key additional elements proposed over bosonic PEPS include
(a) working with parity preserving tensors and (b) replacement of line crossings with fermionic swap gates (yastn/tn/tensor/_contractions.swap_gate).
They number of line crossings and hence the swap gates depend on a specific chosen fermionic order. For details see Ref[2].


Fermionic Order (PEPS)
^^^^^^^^^^^^^^^^^^^^^^

In YASTN, we choose the following fermionic order. we demonstrate it in an example for 3x3 PEPS:
For extracting fermionic statistics we have to place a swap gate at each line crossing.

::
       

                        
                  _________                                       _________                                       _________
                 |         |                                     |         |                                     |         |
                 |         |-------------------------------------|         |-------------------------------------|         |                                           
                 |_________|                                     |_________|                                     |_________|
                      |  \                                            |  \                                            |  \
                      |   \                                           |   \                                           |   \
                      |    \                                          |    \                                          |    \
                      |     \                                         |     \                                         |     \
                      |      \                                        |      \                                        |      \
                      |       \                                       |       \                                       |       \
                      |        \ _________                            |        \ _________                            |        \ _________
                      |         |         |                           |         |         |                           |         |         |
                      |         |         |---------------------------|---------|         |---------------------------|---------|         |                                                                     
       |Psi>  =       |         |_________|                           |         |_________|                           |         |_________|
                      |              |  \                             |              |  \                             |              |  \
                      |              |   \                            |              |   \                            |              |   \
                      |              |    \                           |              |    \                           |              |    \
                      |              |     \                          |              |     \                          |              |     \
                      |              |      \                         |              |      \                         |              |      \
                      |              |       \                        |              |       \                        |              |       \
                      |              |        \ _________             |              |        \ _________             |              |        \ _________
                      |              |         |         |            |              |         |         |            |              |         |         |
                      |              |         |         |------------|--------------|---------|         |------------|--------------|---------|         |                                                                                                      
                      |              |         |_________|            |              |         |_________|            |              |         |_________|
                      |              |              |                 |              |              |                 |              |              |
                      |              |              |                 |              |              |                 |              |              |
                      |              |              |                 |              |              |                 |              |              |


                      
                   FERMIONIC ORDER
                --------------------------------------------------------------------------------------------------------------------------------------------------->>


                 



infinite PEPS (iPEPS)
^^^^^^^^^^^^^^^^^^^^^

Although finite PEPS is widely being used, the true success of PEPS lies in its infinite version called the infinite PEPS or iPEPS [3] which works in the thermodynamic limit.

In essence we can have a single tensor repeated all over the lattice with a fixed bond dimension :math:`D`.
::

        # iPEPS with one tensor repeated all over
               .                  .
                .                  .
                 .                  .
                  \                  \
                   \                  \
                 __________         __________
                |          |       |          |
         ...  --|    A     |-- D --|    A     |-- ...
                |__________|       |__________|
                   |  \               |   \
                   |   \              |    \
                        D                   D
                         \                   \
                          \                   \
                        ___________         __________
                       |           |       |          |
                ... -- |     A     |-- D --|     A    |-- ...
                       |___________|       |__________|
                           |  \               |  \
                           |   \              |   \
                                .                  .
                                 .                  .
                                  .                  .

However to stabilize complex orders we need unit cells of more than 1 site. The most common example is that of a checkerboard lattice. It
has two 2 sites A and B in its unit cell and is a great ansatz for toy models which may have for example antiferromagnetic phases. In reference [4] we used a
checkerboard lattice ansatz for iPEPS to desribe the Hubbard model at high and intermediate temperatures at strongly coupling and near to half-filling. 

::

        # Checkerboard ansatz for iPEPS: two sites A and B in the unit cell repeated all over 
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
                        D                   D
                         \                   \
                          \                   \
                        ___________         __________
                       |           |       |          |
                ... -- |     B     |-- D --|     A    |-- ...
                       |___________|       |__________|
                           |  \               |  \
                           |   \              |   \
                                .                  .
                                 .                  .
                                  .                  .


Time Evolution (PEPS)
^^^^^^^^^^^^^^^^^^^^^
Calculating ground states as well as thermal states using purification is usually done through imaginary time evolution.
We apply the imaginary time evolution operator :math:`\exp(-\beta H)` (:math:`H` is the Hamiltonian) to the state. This is done through Suzuki-Trotter decomposition
of the operator into infinitesimal operators to be applied over all the bonds :math:`b` in :math:`n` time steps such that :math:`n = \beta / d\beta`.

:math:`\exp(-\beta \hat{H}) = \exp(-\beta \sum_{bonds} \hat{H}_{bonds}) = (\exp(-d\beta \sum_{bonds} \hat{H}_{bonds}))^{n} \approx (prod \exp(-d\beta \hat{H}_{bonds}))^{n} `

For Hamiltonians with nearest neighbor interactions, the operators are typically two-site gates applied to the physical index of the PEPS tensors. For checkerboard lattice,
there are four unique bonds : AB horizontal, BA horizontal, AB vertical, BA vertical. Typically we use 2nd order Suzuki-Trotter method where our application of 
an operator :math:`U(d\beta) = \exp(-d\beta H)` on the iPEPS network with a checkerboard ansatz go like:

:math:`\exp(-d\beta H) = U_{ab}^{hor}(d\beta H)U_{ba}^{hor}U_{ab}^{ver}(d\beta H)U_{ba}^{ver}U_{ba}^{ver}(d\beta H)U_{ab}^{ver}U_{ba}^{hor}(d\beta H)U_{ab}^{hor}`

The gates increase the virtual bond dimension of the PEPS by a factor which is equal to the svd rank of the gate. So if the SVD rank of a gate is :math:`r`, then after application of the gate, 
the bond dimension of the bond becomes :math:`r \times D`. 

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



Optimization of iPEPS
^^^^^^^^^^^^^^^^^^^^^

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
called the Neighborhood Tensor Update (NTU) to calculate the thermal states of the Fermi Hubbard Model.
:ref:`NTU<peps/algorithms_NTU>`


Contracting the iPEPS
^^^^^^^^^^^^^^^^^^^^^

The exact contraction of an iPEPS is exponentially hard. The state-of-the-art technique for calculating the norm and expectation values 
is the Corner Transfer Matrix Renormalization Group.
`Corner Transfer matrix renormalization group` 
(:ref:`CTMRG<peps/algorithms_ctmrg:corner transfer matrix renormalization group (ctmrg) algorithm>`) 
is an algorithm that calculates 4 corner and 4 transfer matrices surrounding each unique tensor in the unit cell.
These 4 corner and 4 transfer matrices basically replaces the infinite environment surrounding the lattice site.


References & Related works
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. “Renormalization algorithms for Quantum-Many Body Systems in two and higher dimensions”, Frank Verstraete and Juan I. Cirac, `arXiv:cond-mat/0407066(2004) <https://arxiv.org/abs/cond-mat/0407066>`_
2. "Simulation of strongly correlated fermions in two spatial dimensions with fermionic projected entangled-pair states", Philippe Corboz, Román Orús, Bela Bauer, and Guifré Vidal, `Phys. Rev. B 81, 165104 (2010) <https://arxiv.org/abs/0912.0646>`_
3. “Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions”, J. Jordan, R. Orus, G. Vidal, F. Verstraete, and J. I. Cirac, `Phys. Rev. Lett. 101, 250602 (2008) <https://arxiv.org/abs/cond-mat/0703788>`_
4. "Finite-temperature tensor network study of the Hubbard model on an infinite square lattice", Aritra Sinha, Marek M. Rams, Piotr Czarnik, and Jacek Dziarmaga, `Phys. Rev. B 106, 195105 (2022) <https://arxiv.org/abs/2209.00985>`_
5. "Time evolution of an infinite projected entangled pair state: Neighborhood tensor update", Jacek Dziarmaga, `Phys. Rev. B 104, 094411 (2021) <https://arxiv.org/abs/2107.06635>`_
