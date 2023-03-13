Time-dependent variational principle (TDVP) algorithm
=======================================================

TDVP algorithm is suitable to perform exponentiation of some operator :math:`\hat H` acting on a state :math:`\Psi` producing :math:`\Psi(t)=e^{\hat H} \Psi`. 
The algorithm allows to use MPO operator :code:`H` which can be either hermitian or non-hermitian. The state :code:`psi` is obviously in MPS ansatz. 
In TDVP we use the concept if time-ste :code:`dt` which scale the exponent such that we get  :math:`\Psi(t)=e^{dt \hat H} \Psi`. Notice that for unitary time evolution with Hamiltonian :code:`H` you have imaginary :code:`dt`. 

The TDVP splits the exponentiation to local operation on 1 or 2 tensors depending on :code:`version='1site'` or :code:`'2site'` similar as for 
:ref:`DMRG<mps/algorithms:density matrix renormalisation group (dmrg) algorithm>`  folowing Suzuki-Trotter decomposition of given :code:`order`. 
The `2site` is suitable for dynamically expanding the Mps virtual dimensions controlled by `opts_svd` option. 

To lower the cost you can provide :code:`env` produced by previous run. 


.. autofunction:: yast.tn.mps.tdvp_

See examples: :ref:`examples/mps/mps:tdvp`.