Time-dependent variational principle (TDVP)
===========================================

TDVP algorithm is suitable to perform exponentiation of some operator :math:`\hat H` acting on a state :math:`\Psi(t_0)` producing :math:`\Psi(t)=T e^{-u\int^{t_1}_{t_0} \hat H(w) dw} \Psi(t_0)`,
with :math:`T` indicating time ordering.
The algorithm allows to use MPO operator :code:`H` which can be either hermitian or non-hermitian, possibly time-dependent. The state :code:`psi` is given as MPS or MPO.

A high-level function organizing TDVP simulations is :code:`yastn.tn.mps.tdvp_()`. For examples, see :ref:`examples/mps/algorithms:tdvp`.

The TDVP splits the exponentiation to local operation on 1 or 2 tensors depending on :code:`method='1site'` or :code:`'2site'` similar as for
:ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG)>`  following Suzuki-Trotter decomposition of given :code:`order`.
The :code:`'2site'`, although more expensive, is one of the ways to dynamically expand the MPS virtual dimensions, controlled by :code:`opts_svd`.
We also provide :code:`method='12site'` that performs 2-site operations only for bonds that have the potential to be expanded, based on their bond dimension and Schmidt spectrum.

.. autofunction:: yastn.tn.mps.tdvp_
