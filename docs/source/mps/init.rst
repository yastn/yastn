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


Generating MPS or MPO with Generator
-------------------------------------

`YAMPS` includes a tool which interprets LaTex-like instruction in order to generate MPO. The tool allows to create bosonic and fermionic operators. 
The `Generator` provides an environment to use interpreter.

.. autoclass:: yamps.Generator

The minimal example to initialize a `Generator` is :code:`gen = yast.tn.mps.Generator(N, operators, map)`. Which creates an environment
to generate MPO of `N`  tensors. The generator uses the `yast.Tensor`-s provided in `operators` which are refered in the intruction based 
on their names in :code:`operators.to_dict()` with indicies refered by names as defined in the map `map` which maps from an abract name to
index in `YAMPS` object.

In order to create a generator we need to provide the basis operators used for the generator. In `yast.operators` there are some most common
classes with bases operators, e.g., an instance of :class:`yast.operators.SpinlessFermions` for uniform lattice with spinless fermions. 
After importing the basis class we can use all operators in :code:`operators.to_dict()`. These operators can be refered in the intruction 
by using its name as in keys :code:`operators.to_dict()`. 

.. autoclass:: yast.operators.SpinlessFermions

There is a possibility to create your own class with basis operators. In order to do that use :class:`yast.operators.General` which is a tamplete. 

The `YAMPS` object is indexed by the :code:`map` which is a dictionary mapping from abstract label to a number according to the enumeration in the 
`YAMPS` object. 

`Generator` allows to:

1/ :ref:`Create random MPS or MPO<examples/mps/mps:create random mps or mpo>`

2/ :ref:`Create MPS or MPO based on templete object<examples/mps/mps:Create MPS or MPO based on templete object>`

3/ :ref:`Generate MPS from LaTex-like instruction<examples/mps/mps:Generate MPS from LaTex-like instruction>`

4/ :ref:`Generate MPO from LaTex-like instruction<examples/mps/mps:Generate MPO from LaTex-like instruction>`
