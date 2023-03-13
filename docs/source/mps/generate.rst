Creating MPS/MPO tensors with Generator
=======================================

:class:`yast.tn.mps.Generator` provies an environment to automitize setting MPS and MPO. The tool allows to create any `YAMPS` object 
providing the :class:`yast.Tensor`-s used for the construction. The instruction for MPS/MPO can be given and LaTeX-like expression or 
relying on custom templetes as described below. 

.. autoclass:: yast.tn.mps.Generator

The generator can be accessed with 

.. code-block::

	import yast.tn.mps

Initializing the environment you have to provide set of basic operators. This is done by cutom class containing necessary information. 
There are number of predefines operators which can be foing at :code:`yast.operators`, e.g.:

.. code-block::

	# for uniform lattice with spinless fermions
	ops = yast.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device) 

.. autoclass:: yast.operators.SpinlessFermions
	
In order to set your own set of basic operators you can use general class

.. code-block::

    ops = yast.operators.General({'cpc': lambda j: cpc, 'ccp': lambda j: ccp, 'I': lambda j: I})

.. autoclass:: yast.operators.General

.. note::
	`Generator` has to contain operator `I` which is an identity matrix for the model.

.. note::
	Operators have to be defined as :class:`yast.Tensor`-s without virtual legs, i.e., rank-1 for MPS and rank-2 for MPO.

.. note::
	Make sure that physical dimensions and symmetries are consistent.

.. note::
	The set of operators can be listed by :code:`ops.to_dict()`. 

The instruction can be defined using the abstract indicies for each element of the system. The mapping from abstract indicies to a position in 
MPS/MPO is given by :code:`map` which is a dictionary of abstract indicies with values given by a real position. 

Finally, the initialisation for :code:`N` site lattice with indicies given by :code:`map`:

.. code-block::

	gen = yast.tn.mps.Generator(N, operators, map)

These operators can be refered in the intruction by using its name as the keys in :code:`operators` and indicies as keys in :code:`map`.

Below you can find detailed description how to set MPS and MPO using LaTeX-like instruction:

1/ :ref:`Generate MPS from LaTex-like instruction<examples/mps/mps:Generate MPS from LaTex-like instruction>`

2/ :ref:`Generate MPO from LaTex-like instruction<examples/mps/mps:Generate MPO from LaTex-like instruction>`

Alternatively, :class:`yast.tn.mps.Generatot` allows to set the instruction using prefined names for operators, parameters and indicies 
by using :class:`single_term`.

.. autoclass:: yast.tn.mps.single_term

For more on method based on templete see :ref:`here<examples/mps/mps:Create MPS or MPO based on templete object>`.


For consistency you can create also other `YAMPS` objects withing the environment. 
For random MPS and MPO see :ref:`here<examples/mps/mps:create random mps or mpo>`.

