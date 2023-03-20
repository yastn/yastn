Creating empty MPS/MPO
======================

In order to initialize MPS :class:`yast.tn.mps.Mps` or MPO :class:`yast.tn.mps.Mpo` we have to create an object of that class.

.. autofunction:: yast.tn.mps.Mps
.. autofunction:: yast.tn.mps.Mpo

To initialize an empty object for MPS use :code:`psi = yast.tn.mps.Mps(N)` which creates MPS of `N` sites but without any tensors defined.

Both :class:`yast.tn.mps.Mps` and :class:`yast.tn.mps.Mpo` inherit all functions from parent class :class:`yast.tn.mps.MpsMpo` but differ by a 
number of physical legs, i.e., one for MPS and 2 for MPO. :class:`yast.tn.mps.MpsMpo` supports one dimensional tensor networks, defined by list 
of rank-3 or rank-4 :class:`yast.Tensor`-s for  :class:`yast.tn.mps.Mps` and :class:`yast.tn.mps.Mpo` respectively.

.. autoclass:: yast.tn.mps.MpsMpo


Setting MPS/MPO tensors by hand
===============================

:ref:`An empty MPS/MPO<mps/init:Creating empty MPS/MPO>` can be filled with tensors by setting them one by one. 

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
:ref:`Ground state of Spin-1 AKLT model<examples/mps/mps:ground state of spin-1 aklt model>`,
:ref:`MPO for hopping model with U(1) symmetry<example hopping with u1 symmetry>`.

.. todo:
	Failed to create references to Z2 and U1 above. 

Alternatively, MPS/MPO can be set using :class:`yast.tn.mps.Generator` environment (see  :ref:`here<mps/init:Setting MPS/MPO tensors with Generator>` for more)
or using :class:`yast.tn.mps.Hterm` (see :ref:`here<mps/init:Setting MPS/MPO tensors with Hterm>` for more).