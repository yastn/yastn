Initialization
==============


Creating empty MPS/MPO
----------------------

In order to initialize MPS :class:`yast.tn.mps.Mps` or MPO :class:`yast.tn.mps.MPO` we have to create an object of that class.

.. autofunction:: yast.tn.mps.Mps
.. autofunction:: yast.tn.mps.Mpo

To initialize an empty object for MPS use :code:`psi = yast.tn.mps.Mps(N)` which creates MPS of `N` sites but without any tensors defined.

Both :class:`yast.tn.mps.Mps` and :class:`yast.tn.mps.MPO` inherit all functions from parent class :class:`yast.tn.mps.MpsMpo` but differ by a 
number of physical legs, i.e., one for MPS and 2 for MPO. :class:`yast.tn.mps.MpsMpo` supports one dimensional structures which means that contains a list
of rank-3 or rank-4 :class:`yast.Tensor`-s for  :class:`yast.tn.mps.Mps` and :class:`yast.tn.mps.Mpo` respectively.

.. autoclass:: yast.tn.mps.MpsMpo


Setting MPS/MPO tensors by hand
-------------------------------

:ref:`An empty MPS/MPO<mps/init:Creating empty MPS/MPO>` can be filled with tensors by setting it one by one into empty `YAMPS` object. 

.. code-block::

    import yast.tn.mps as mps

    # create empty MPS over three sites
    Y= mps.Mps(3)

    # create 3x2x3 random dense tensor
    A_1= yast.rand(yast.make_config(), Legs=(\
            yast.Leg(s=1,D=(3,)), yast.Leg(s=1,D=(2,)), yast.Leg(s=-1,D=(3,))))

    # assign tensor to site 1
    Y[1]= A_1

.. note::
    Tensor should be of the rank expected for initialized object. See :ref:`basic concepts<theory/basics:>` for more.

.. note::
    The virtual dimensions/spaces of the neighbouring MPS/MPO tensors have to remain consistent.

.. note::
    To create :class:`yast.Tensor`'s look :ref:`here<tensor/init:Creating symmetric YAST tensors>`. 

The examples of creating MPS/MPO by hand can be found here:
:ref:`Ground state of Spin-1 AKLT model<examples/mps/mps:Ground state of Spin-1 AKLT model>`,
:ref:`MPO for hopping model with Z2 symmetry<examples/mps/mps:MPO for hopping model with :math:`\mathbb{Z}_2` symmetry>`,
:ref:`MPO for hopping model with U(1) symmetry<examples/mps/mps:MPO for hopping model with U(1) symmetry`.

.. todo:
    Failed to create references to Z2 and U1 above. 

Alternatively, MPS/MPO can be set using :class:`yast.tn.mps.Generator` environment (see  :ref:`here<mps/init:Setting MPS/MPO tensors with Generator>` for more)
or using :class:`yast.tn.mps.Hterm` templete (see :ref:`here<mps/init:Setting MPS/MPO tensors with Hterm>` for more).


Setting MPS/MPO tensors with Hterm
-----------------------------------

:class:`yast.tn.mps.Hterm` is a templete object which includes an information about a product state we would like to generate. 

.. autoclass:: yast.tn.mps.Hterm

.. note::
    The object :code:`hterm` has operators :code:`hterm.operators` without virtual legs, i.e., rank-1 for MPS and rank-2 for MPO.

A list containing such templetes can be used to create a sum of products. In order to generate full MPO use :code:`exMPO = mps.generate_mpo(I, man_input)`, 
where :code:`man_input` is a list of `Hterm`-s and :code:`I` is identity matrix for your basis.

.. autofunction:: yast.tn.mps.generate_mpo

In order to generate MPS use :code:`exMPS = mps.generate_mps(I, man_input)`, where `I` is identity matrix for your basis.

.. autofunction:: yast.tn.mps.generate_mps

.. note::
    To create MPS you need to provide a vector for each site `0` through `N-1`.

An example for MPO using this method can be found :ref:`here<examples/mps/mps:Create MPS or MPO based on templete object>`.


Setting MPS/MPO tensors with Generator
--------------------------------------

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

