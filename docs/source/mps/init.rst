Creating empty MPS/MPO
======================

In order to initialize MPS :class:`yastn.tn.mps.Mps` or MPO :class:`yastn.tn.mps.Mpo` we have to create an object of that class.

.. autofunction:: yastn.tn.mps.Mps
.. autofunction:: yastn.tn.mps.Mpo

To initialize an empty object for MPS use :code:`psi = yastn.tn.mps.Mps(N)` which creates MPS of `N` sites but without any tensors defined.

Both :class:`yastn.tn.mps.Mps` and :class:`yastn.tn.mps.Mpo` inherit all functions from parent class :class:`yastn.tn.mps.MpsMpo` but differ by a 
number of physical legs, i.e., one for MPS and 2 for MPO. :class:`yastn.tn.mps.MpsMpo` supports one dimensional tensor networks, defined by list 
of rank-3 or rank-4 :class:`yastn.Tensor`-s for  :class:`yastn.tn.mps.Mps` and :class:`yastn.tn.mps.Mpo` respectively.

.. autoclass:: yastn.tn.mps.MpsMpo


Setting MPS/MPO tensors by hand
===============================

:ref:`An empty MPS/MPO<mps/init:Creating empty MPS/MPO>` can be filled with tensors by setting them one by one. 

.. code-block::

    import yastn.tn.mps as mps

    # create empty MPS over three sites
    Y= mps.Mps(3)

    # create 3x2x3 random dense tensor
    A_1= yastn.rand(yastn.make_config(), Legs=(\
            yastn.Leg(s=1,D=(3,)), yastn.Leg(s=1,D=(2,)), yastn.Leg(s=-1,D=(3,))))

    # assign tensor to site 1
    Y[1]= A_1

.. note::
    Tensor should be of the rank expected for initialized object. See :ref:`basic concepts<theory/basics:>` for more.

.. note::
    The virtual dimensions/spaces of the neighbouring MPS/MPO tensors have to remain consistent.

.. note::
    To create :class:`yastn.Tensor`'s look :ref:`here<tensor/init:Creating symmetric YAST tensors>`. 

The examples of creating MPS/MPO by hand can be found here:
:ref:`Ground state of Spin-1 AKLT model<examples/mps/mps:ground state of spin-1 aklt model>`,
:ref:`MPO for hopping model with U(1) symmetry<example hopping with u1 symmetry>`.

.. todo:
    Failed to create references to Z2 and U1 above. 

Alternatively, MPS/MPO can be set using :class:`yastn.tn.mps.Generator` environment (see  :ref:`here<mps/init:Setting MPS/MPO tensors with Generator>` for more)
or using :class:`yastn.tn.mps.Hterm` templete (see :ref:`here<mps/init:Setting MPS/MPO tensors with Hterm>` for more).
