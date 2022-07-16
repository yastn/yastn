MPS based algorithms
=====================


Maximize overlap
---------------------------------

This is the procedure which performs DMRG sweep such that the initial state has the maximal ovelap with the target state.

.. automodule:: yamps
	:noindex:
	:members: variational_sweep_1site

.. todo:: I wasn't using it a lot, maybe there is sth more to say.


Density matrix renormalisation group (DMRG) algorithm
---------------------------------------------------------

Density matrix renormalisation group (DMRG) algorithm is a variational technique devised to obtain the state od smallest expentation value for hermitian matrix-product operator `H`. 
The final state can be equivalent to e.g. the ground-state for a Hamiltonian. The algorithm can be run in `1site` or `2site` version. Where only the `2site` is suitable for dynamically expanding the Mps virtual dimensions controlled by `opts_svd` option. 
The convergence of the DMRG is controlled either by the expectational values in iteration of the variational step or by the Schmidt values which is more sensitive parameter.

.. automodule:: yamps
	:noindex:
	:members: dmrg

See examples: :ref:`examples/mps/mps:dmrg`.

.. todo:: why option for eigs is called opts_expmv? can I use some other case then the smalles energy, e.g. the highest EV, should I say more on info from returns? say sth on fing excited states


Time-dependent variational principle (TDVP) algorithm
---------------------------------------------------------

Time-dependent variational principle (TDVP) algorithm is a method to apply the matrix exponent on a state. The matrix exponent is split using TDVP and applied to an initial state `psi`. 
The matrix-product operator `H` can be any but the accuracy of reproducing exponentiation strongly depends on the time step `dt` and `order` for Suzuki-Trotter decompositon. 
The final state can be equivalent to e.g. a state evolved in imaginary or real time under some Hamiltonian. 
The algorithm can be run in `1site` or `2site` version. Where only the `2site` is suitable for dynamically expanding the Mps virtual dimensions controlled by `opts_svd` option. 

.. automodule:: yamps
	:noindex:
	:members: tdvp

See examples: :ref:`examples/mps/mps:tdvp`.

.. todo:: in another brabch I have tdvp modified so that we do not normalise the state