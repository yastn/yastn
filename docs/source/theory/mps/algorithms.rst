
DMRG
----------

`Density matrix renormalisation group` (DMRG) is the algorithm which finds MPS of the extremal expectation value for the hermitian MPO. In order to find a ground state for the Hamiltonian written in MPO form we 
iteratively adjust each :math:`A_j` tensor in MPS chain. In order to find :math:`A_j` which locally minimizes the energy defined with the effective Hamiltonian :math:`H_{eff}`. The optimisation process is permed throughour the MPS chain untill convergence. 

::

        # optimisatio of the MPS tensor A_j using the effective Hamiltonian H_eff
                                  ___
                                -|A_j|-
                 ___    ___        |        ___    ___    ___
                |___|--|___|-..         ..-|___|--|___|--|___|
                  |      |                   |      |      |      
                 _|_    _|_       _|_       _|_    _|_    _|
        H_eff = |___|--|___|-...-|H_j|-...-|___|--|___|--|___|
                  |      |         |         |      |      |     
                 _|_    _|_                 _|_    _|_    _|
                |___|--|___|-..         ..-|___|--|___|--|___|


The virtual dimension for the MPS anzatz can be adjusted by performing of `2-site` tensor. Here, we merge two MPS tensors, perform DMRG step and split them using SVD with truncation giving a new virtual dimension.
For simplicity the DMRG algorithm uses MPS in the :ref:`theory/mps/basics:Canonical form` which simplifies the norm to :math:`tr\{A_j^\dagger A_j\}` which we fix to `1`.


::

    # optimisatio of the MPS tensor using the effective Hamiltonian

            O-site DMRG                         1-site DMRG                              2-site DMRG
                ___                                ___                                ___    ___
     ______   -|L_j|-   ______          ______   -|A_j|-   ______          ______   -|A_j|--|A_j|-    ______  
    |      |--       --|      |        |      |--   |   --|      |        |      |--   |      |    --|      |
    |      |           |      |        |      |           |      |        |      |                   |      |
    |      |           |      |        |      |    _|_    |      |        |      |    _|_   __|__    |      |
    |      |-----------|      |        |      |---|H_j|---|      |        |      |---|H_j|-|H_j+1|---|      |
    |      |           |      |        |      |     |     |      |        |      |     |      |      |      |
    |      |           |      |        |      |           |      |        |      |                   |      |
    |______|--       --|______|        |______|--       --|______|        |______|--               --|______|


TDVP
-----

`Time-dependent variational principle` (TDVP) algorithm allows to perform real or imaginary time evolution with any operator `\hat O` in the MPO form. The algorithm splits the exponent into small Trotter steps `dt`. The exponentaition :math:`exp(t \hat H_{eff})` is applied for eah tensor :math:`A_j`. 
The effective operator :math:`H_{eff}` as calculated similarly as for :ref:`theory/mps/algorithms:DMRG`. In practive the TDVP step keeps the caconical form for the MPS. This involves spliting a site with SVD and evolving a remnant matrix back in time such taht we do not evolve this term twice. 

