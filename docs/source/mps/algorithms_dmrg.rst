Density matrix renormalization group (DMRG)
===========================================

DMRG is a variational technique devised to obtain a state that extremizes the expectation value of the Hermitian operator.
A typical example is the search for the best MPS approximation of a ground-state of some Hamiltonian in the form of MPO.

A high-level function organizing DMRG simulations is :code:`yastn.tn.mps.dmrg_()`.
See examples at :ref:`examples/mps/algorithms_dmrg:low energy states of the xx model`.

.. autofunction:: yastn.tn.mps.dmrg_


Single-site DMRG
----------------

In the algorithm we `sweep` through the MPS, starting from the initial guess :code:`psi`,
optimizing each :class:`yastn.Tensor` :math:`A_j` one by one while keeping
all other tensors fixed (alternating least squares).
At each step, the best tensor :math:`A_j` is found by minimizing the energy
of the local effective Hamiltonian :math:`H_{\rm eff}`. DMRG with :code:`method='1site'`
works with single-site effective Hamiltonians.

::

    # local optimization of the MPS tensor A_j
    # using the effective Hamiltonian H_eff
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

After the local problem is solved and a new :math:`A_j` minimizing :math:`H_{\rm eff}` is found,
the MPS is orthogonalized to keep it in the :ref:`canonical form<theory/mps/basics:Canonical form>`.
In the simplest formulation of '1site' DMRG algorithm, the virtual dimension/spaces are fixed.


Two-site DMRG
-------------

The virtual spaces of the MPS can be adjusted by performing DMRG :code:`method='2site'`.
In this approach, we (i) build effective Hamiltonian :math:`H_{\rm eff}` of two neighbouring sites,
(ii) solve for its ground state, (iii) split the resulting tensor back into
two sites with SVD, and then move to next pair. By doing SVD decomposition we can dynamically
adjust virtual space of MPS according to parameters in :code:`opts_svd`.

::

  # local optimization of the MPS tensors A_j and A_j+1 using
  # the effective Hamiltonian H_eff

           H_eff of 2-site DMRG
              ___    _____
   ______   -|A_j|--|A_j+1|-    ______
  |      |--   |      |      --|      |
  |      |                     |      |
  |      |    _|_   __|__      |      |
  |      |---|H_j|-|H_j+1|-----|      |
  |      |     |      |        |      |
  |      |                     |      |
  |______|--                 --|______|

The next pair will take one of the sites from the previous step while the other one
is orthogonalized to maintain :ref:`canonical form<theory/mps/basics:Canonical form>` of the MPS.

Projecting out selected MPS
---------------------------

The optimization can be performed in the restricted subspace, where contributions
from some MPSs are projected out. An alternative approach that we utilize
is adding penalty terms in the directions of those MPSs.
This allows one to search for a few excited states of the Hamiltonian.
The list of MPS to project out is given as :code:`project=[lower_E_MPS, ...]`.
