Density matrix renormalization group (DMRG) algorithm
=====================================================

DMRG is a variational technique devised
to obtain a state which extremizes the expectation value of Hermitian operator.
A typical example is search for the best MPS approximation of a ground-state
of some Hamiltonian in form of MPO.

A high-level function organizing DMRG simulations is :code:`yastn.tn.mps.dmrg_()`. For examples, see :ref:`examples/mps/algorithms:dmrg`.

.. autofunction:: yastn.tn.mps.dmrg_


Single-site DMRG
----------------

In the algorithm we `sweep` through the MPS, starting from the initial guess :code:`psi`,
optimizing each :class:`yastn.Tensor` :math:`A_j` one by one while keeping
all other tensors fixed (alternating least squares).
At each step, the best :math:`A_j` is then found by minimizing the energy
of the local effective Hamiltonian :math:`H_{eff}`. This DMRG variant, called :code:`method='1site'`,
works with single-site effective :math:`H_{eff}`.

::

    # local optimization of the MPS tensor A_j using the effective Hamiltonian H_eff
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

After the local problem is solved and a new :math:`A_j` minimizing :math:`H_{eff}` is found,
the MPS is orthogonalized to keep it in the :ref:`canonical form<theory/mps/basics:Canonical form>`.
In the simplest formulation of ``1site`` DMRG algorithm, the virtual dimension/spaces are fixed.


Two-site DMRG
-------------

The virtual spaces of the MPS can be adjusted by performing :code:`method='2site'` DMRG.
In this approach, we (i) build effective Hamiltonian :math:`H_{eff}` of two neighbouring sites,
(ii) solve for its ground state, and (iii) finally split the resulting tensor back into two sites with SVD
and then move to next pair. By doing SVD decomposition we can dynamically
adjust virtual space of MPS according to :code:`opts_svd`.

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
from some MPSs are projected out. This can be useful when searching for
excited states. List of MPS to project out is given as :code:`project=[lower_E_MPS, ...]`.
