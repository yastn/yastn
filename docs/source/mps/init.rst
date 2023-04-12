Creating empty MPS/MPO
======================

Create empty, i.e. with no tensors specified, MPS :class:`yastn.tn.mps.Mps` or MPO :class:`yastn.tn.mps.Mpo`.

.. autofunction:: yastn.tn.mps.Mps
.. autofunction:: yastn.tn.mps.Mpo

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
    Tensor should have correct rank and the virtual dimensions/spaces of the neighbouring MPS/MPO tensors have to be consistent, see :ref:`basic concepts<theory/mps/basics:matrix product state (mps)>`. 

.. note::
    To create :class:`yastn.Tensor`'s look :ref:`here<tensor/init:Creating symmetric YASTN tensors>`. 

The examples of creating MPS/MPO by hand can be found here:
:ref:`Ground state of Spin-1 AKLT model<examples/mps/build_mps_manually:ground state of spin-1 aklt model>`,
:ref:`MPO for hopping model with U(1) symmetry<examples/mps/build_mps_manually:hamiltonian for nearest-neighbor hopping/xx model>`.

.. todo:
    Failed to create references to Z2 and U1 above. 

Alternatively, MPS/MPO can be created using 

    * :class:`yastn.tn.mps.Generator` shown :ref:`here<mps/generate:creating mps/mpo tensors with generator>`.
    * :class:`yastn.tn.mps.Hterm` template shown :ref:`here<mps/init_hterm:setting mps/mpo tensors with hterm>`.
