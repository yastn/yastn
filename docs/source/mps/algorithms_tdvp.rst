Time-dependent variational principle (TDVP) algorithm
=====================================================

TDVP algorithm is suitable to perform exponentiation of some operator :math:`\hat H` acting on a state :math:`\Psi` producing :math:`\Psi(t)=e^{u\int^t_{t_0} \hat H(w) dw} \Psi`.
The algorithm allows to use MPO operator :code:`H` which can be either hermitian or non-hermitian, possibly time-dependent. The state :code:`psi` is an MPS ansatz.

A high-level function organizing TDVP simulations is :code:`yastn.tn.mps.tdvp_()`. For examples, see :ref:`examples/mps/mps:tdvp`.

The TDVP splits the exponentiation to local operation on 1 or 2 tensors depending on :code:`version='1site'` or :code:`'2site'` similar as for
:ref:`DMRG<mps/algorithms_dmrg:density matrix renormalization group (dmrg) algorithm>`  following Suzuki-Trotter decomposition of given :code:`order`.
The `2site` is suitable for dynamically expanding the MPS virtual dimensions controlled by `opts_svd` option.

.. autofunction:: yastn.tn.mps.tdvp_
