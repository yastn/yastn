Initialisation
===================


General information
--------------------------------

`YAMPS` provides a class object :code:`Mps` dedicated to represent `matrix product state` (MPS) and :code:`Mpo` dedicated to represent `matrix product operator` (MPO). :code:`Mps` and :code:`Mpo` include abtract information on general contruction and properties of an object. The 
composition of MPS/MPO as itself is build with :code:`yast.Tensor` which are assigned to MPS/MPO as ordered list of tensors. The tensors building the body of MPS/MPO has to be matching along virtual dimension. The virtual dimension on the eges of the MPS/MPO should be `D=1`. 

`YAMPS` supports one-dimensional structures with open boundary conditions which can be used to perform :ref:`theory/mps/basics:Algorithms` for e.g. energy optimisation or time evolution, or to do :ref:`theory/mps/basics:Measurements`.

:code:`Mps` and :code:`Mpo` are belong to a common class :code:`yamps.MpsMpo` sharing many operations the only difference between them is that :code:`Mps` reprsents MPS with one phisical dimension for a tensor and :code:`Mpo` repsents MPO with two phisical dimensions. 


.. autoclass:: yamps.MpsMpo
	:noindex:
	:members: __init__
	:exclude-members: __new__


Creating :code:`yamps.Mps` and :code:`yamps.Mpo`
------------------------------------------------

The initialisation of an empty object can be done by calling :code:`yamps.Mps(N)` (:code:`yamps.Mpo(N)`) where :code:`N` is number to tensors building the object. 
The symmetry of :code:`yamps.Mps` is inherited by its building blocks. Its building blocks can be assigned directly by :ref:`examples/mps/mps:Filling matrix product with tensors`. 

`YAMPS` provides a tool for automatic MPO generation. The genetator works under :code:`GenerateOpEnv` class which allows to contruct Hamiltonian of any form for bosonic or fermionic system. 

.. automodule:: yamps._generator.GenerateOpEnv
	:noindex:
	:members: __init__

To initiallize the generator :code:`gen = yamps.GenerateOpEnv(N, config, opts)` we need to provide a length of the MPO :code':`N`, and :ref:`tensor/configuration:YAST configuration` :code:`config` which will also define the commutation rules :code:`config.fermionic`.
The instruction for truncation :code:`opts` is an optional argument.

The generator has to be supplied with the operators we want to use. If you want to use predefined operators suitable with :code:`config` then you should :code:`gen.use_default()`. Make sure that you know the definition for the operators by checking out:

.. automodule:: yamps._generator.GenerateOpEnv
	:noindex:
	:members: use_default

For custom basis you can use :code:`gen.use_basis(basis_dict)`, where :code:`basics_dict` is new self of single-particle operators.

.. automodule:: yamps._generator.GenerateOpEnv
	:noindex:
	:members: use_basis

After providing the basis for the generator you can use it to generate MPO. In order to do that you can use latex-to-yamps converter:

.. automodule:: yamps._generator.GenerateOpEnv
	:noindex:
	:members: latex2yamps

You can also generate the MPO by hand. If your MPO is a sum or operator products then define each operator product separately.

.. automodule:: yamps._generator
	:noindex:
	:members: mpo_term

The :code:`mpo_term` gives only a  definition for a product operator with a multiplier. The MPO is an output of :code:`my_mpo = gen.generate(my_mpo_term)`. 

.. automodule:: yamps._generator.GenerateOpEnv
	:noindex:
	:members: generate

To sum many MPO entries at ones use :code:`my_mpo = gen.sum(list_of_mpo_terms)`. 

For examples of the MPO generator see the code in :ref:`examples/mps/mps:Automatically generated matrix product`.

