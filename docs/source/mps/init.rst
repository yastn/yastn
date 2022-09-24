Initialization
==============


Creating empty MPS/MPO
----------------------

Both MPS and MPO are represented by the same class :class:`yamps.MpsMpo`, sharing many operations. The only difference between them is the number of their physical dimensions. The class :class:`yamps.MpsMpo` defines MPS/MPO through the set of tensors *A*
which are stored as integer-indexed dictionary :code:`yamps.MpsMpo.A` 
of rank-3/rank-4 :class:`yast.Tensor`'s. 

To create empty MPS/MPO, i.e., without any tensors call

.. autoclass:: yamps.MpsMpo

Short-hand functions for creation of empty MPS/MPO

.. autofunction:: yamps.Mps
.. autofunction:: yamps.Mpo


Setting MPS/MPO tensors
-----------------------

The tensors of MPS/MPO can be set manually, using familiar :code:`dict` access

.. code-block::

	# create empty MPS over three sites
	Y= yamps.Mps(3)

	# create 3x2x3 random dense tensor
	A_1= yast.rand(yast.make_config(), Legs=(yast.Leg(s=1,D=(3,)), 
		yast.Leg(s=1,D=(2,)), yast.Leg(s=-1,D=(3,))))

	# assign tensor to site 1
	Y[1]= A_1

.. note::
	The virtual dimensions/spaces of the neighbouring MPS/MPO tensors have to remain consistent.

To create :class:`yast.Tensor`'s see :ref:`YAST's basic creation operations<tensor/init:basic creation operations>`. 
For more examples, see :ref:`Setting MPS/MPO manually<examples/mps/mps:building yamps object manually>`. 


Automatic creation of MPS
-------------------------


Generate MPO automatically
--------------------------

`YAMPS` provides a tool for automatic MPO generation, which allows to construct Hamiltonian of any form for both bosonic and fermionic systems.

.. autoclass:: yamps.Generator

To initiallize the generator :code:`gen = yamps.GenerateOpEnv(N, config, opts)` we need to provide a length of the MPO :code':`N`, and :ref:`tensor/configuration:YAST configuration` :code:`config` which will also define the commutation rules :code:`config.fermionic`.
The instruction for truncation :code:`opts` is an optional argument.

The generator has to be supplied with the operators we want to use. If you want to use predefined operators suitable with :code:`config` then you should :code:`gen.use_default(basis_type)`. Make sure that you know the definition for the operators by checking out:

..
	=======
	The class :class:`yamps.Mps` defines matrix product states and operators through a set of tensors *A*. It consists of integer-indexed dictionary of :class:`yast.Tensor`. 
	Using rank-3 tensors *A*, :class:`yamps.Mps` can represent states

	.. math::
		
		|\psi\rangle &= \sum_{\{\sigma\}} c_{\{\sigma\}} |\{\sigma\}\rangle,\\
		c_{\{\sigma\}} &= Tr_{aux}[A^{\sigma_0}_{a_0,a_1}A^{\sigma_1}_{a_1,a_2}\ldots
		A^{\sigma_{N-1}}_{a_{N-1},a_N}],

	where each of tensors *A* has three indices, in order: left virtual :math:`a_i`, single physical :math:`\sigma`, and right virtual index :math:`a_{i+1}`. 
	The operators are represented similarily by rank-4 tensors *A*

	.. math::
		
		O &= \sum_{\{\sigma\},\{\sigma'\}} O_{\{\sigma\},\{\sigma'\}} |\{\sigma\}\rangle\langle\{\sigma'\}|,\\
		O_{\{\sigma\},\{\sigma'\}} &= Tr_{aux}[A^{\sigma_0,\sigma'_0}_{a_0,a_1}A^{\sigma_1,\sigma'_1}_{a_1,a_2}\ldots
		A^{\sigma_{N-1}\sigma'_{N-1}}_{a_{N-1},a_N}]

	with convention for index order as follows: left virtual :math:`a_i`, physical :math:`\sigma`, physical :math:`\sigma'`, and right virtual index :math:`a_{i+1}`.
	The indices :math:`\sigma,\sigma'` represent *bra* and *ket* physical indices
	of the operator.

	The `yamps.Mps` defines matrix products with open-boundary condition. Therefore, 
	the bond dimension of virtual indices on the edges, :math:`a_0` and :math:`a_N` is 1 by default.


	Creating `yamps.Mps` matrix product
	-----------------------------------

	.. autoclass:: yamps.Mps
	>>>>>>> Stashed changes

.. autofunction:: yamps.basis.use_default_basis

For custom basis you can use :code:`gen.use_basis(basis_dict)`, where :code:`basics_dict` is new self of single-particle operators.

.. autofunction:: yamps._generator.GenerateOpEnv.use_basis

After providing the basis for the generator you can use it to generate MPO. In order to do that you can use latex-to-yamps converter:

.. autofunction:: yamps._generator.GenerateOpEnv.latex2yamps

You can also generate the MPO by hand. If your MPO is a sum or operator products then define each operator product separately.

.. automodule:: yamps._generator
	:noindex:
	:members: mpo_term

The :code:`mpo_term` gives only a  definition for a product operator with a multiplier. The MPO is an output of :code:`my_mpo = gen.generate(my_mpo_term)`. 

.. autofunction:: yamps._generator.GenerateOpEnv.generate

To sum many MPO entries at ones use :code:`my_mpo = gen.sum(list_of_mpo_terms)`. 

For examples of the MPO generator see the code in :ref:`examples/mps/mps:Automatically generated matrix product`.

