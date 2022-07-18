Density matrix renormalisation group (DMRG) algorithm
=======================================================

:ref:`theory/mps/algorithms:DMRG` is a variational technique devised to obtain a state which extremises the expectation values for hermitian operator. A typical example is solving for ground-state of the Hamiltonian. The Hamiltonian is 
written as MPO and the state is in MPS ansatz.

In the algorithm  we `sweep` throgh the MPS initial guess :code:`psi` making the optimisation of each :meth:`yast.Tensor` one by one locally solving for the smallest energy for effective Hamiltonian derived from MPO :code:`H`. 
The local optimisation can take tensors one after one which is :code:`version='1site'`. Find the best approximation, orthogonalise to keep :ref:`theory/mps/basics:Canonical form` of the MPS. In `1site` version we are confined in particular MPS ansatz with
fixed bond dimension. 

Alternatively, we can perform local optimisation for a pair of neightbouring tensors at one which is :code:`version='2site'`. In this approach, we take two neighbouring sites, find the best approximation, split sites with SVD and move to another pair.
The next pair will take one of sites from our previous pair while the other one is orthogonalised to keep :ref:`theory/mps/basics:Canonical form` of the MPS. 
By doing SVD decomposition we can dynamically adjust MPS ansatz according to :code:`opts_svd`. 

The convergence of the DMRG is controlled either by the expectational values in iteration of the variational step :code:`converge='energy'` or by the Schmidt values  :code:`converge='schmidt'` which is more sensitive parameter. 
The DMRG algorithm sweeps thought the lattice :code:`max_sweeps` times until converged quantity changes by less then :code:`atol` from sweep to sweep.

To lower the cost of DMRG you can provide :code:`env` produced by previous run. 

The optimisation of MPS can be performed in restricted subspace where contrubutions of some MPS is projected out. This can be used for searching for excited states which can be done by giving a list of MPS :code:`project=[lower_E_MPS, ...]` we want to project out. 

.. autofunction:: yamps.dmrg

See examples for :ref:`examples/mps/mps:dmrg`.

.. todo:: can I use some other case then the smalles energy, e.g. the highest EV, should I say more on info from returns? 


Time-dependent variational principle (TDVP) algorithm
=======================================================

:ref:`theory/mps/algorithms:TDVP` algorithm is suitable to perform exponentiation of some operator :math:`\hat H` acting on a state :math:`\Psi` producing :math:`\Psi(t)=e^{\hat H} \Psi`. 
The algorithm allows to use MPO operator :code:`H` which can be either hermitian or non-hermitian. The state :code:`psi` is obviously in MPS ansatz. 
In TDVP we use the concept if time-ste :code:`dt` which scale the exponent such that we get  :math:`\Psi(t)=e^{dt \hat H} \Psi`. Notice that for unitary time evolution with Hamiltonian :code:`H` you have imaginary :code:`dt`. 

The TDVP splits the exponentiation to local operation on 1 or 2 tensors depending on :code:`version='1site'` or :code:`'2site'` similar as for :ref:`mps/algorithms:DMRG` folowing Suzuki-Trotter decomposition of given :code:`order`. 
The `2site` is suitable for dynamically expanding the Mps virtual dimensions controlled by `opts_svd` option. 

To lower the cost you can provide :code:`env` produced by previous run. 


.. autofunction:: yamps.tdvp

See examples: :ref:`examples/mps/mps:tdvp`.


Maximize overlap
==================

This is the procedure which performs DMRG sweep such that the initial state has the maximal ovelap with the target state.

.. autofunction:: yamps.variational_sweep_1site

