Creating empty MPS/MPO
======================

To initialize MPS or MPO for :code:`N` sites, we have to create an instance of class :class:`yastn.tn.mps.MpsMpo` specifying the number of physical legs, i.e., one for MPS and 2 for MPO.
It supports one-dimensional tensor networks, defined by dict of rank-3 or rank-4 :class:`yastn.Tensor`-s for MPS and MPO, respectively.
The new instance starts without any tensors defined.

.. autoclass:: yastn.tn.mps.MpsMpo

One can directly create a new empty MPS of `N` sites using :code:`psi = yastn.tn.mps.Mps(N)`. Tensors are not initialized at this point.
Similarly, :code:`psi = yastn.tn.mps.Mpo(N)` initializes MPO.

.. autofunction:: yastn.tn.mps.Mps
.. autofunction:: yastn.tn.mps.Mpo


Setting MPS/MPO tensors by hand
===============================

:ref:`An empty MPS/MPO<mps/init:Creating empty MPS/MPO>` can be filled with tensors by setting them one by one. 

.. code-block::

    import yastn.tn.mps as mps

    # create empty MPS over three sites
    Y= mps.Mps(3)

    # create 3x2x3 random dense tensor
    A_1 = yastn.rand(yastn.make_config(), legs=(\
            yastn.Leg(s=1,D=(3,)), yastn.Leg(s=1,D=(2,)), yastn.Leg(s=-1,D=(3,))))

    # assign tensor to site 1
    Y[1] = A_1


Tensor should be of the rank expected for :ref:`MPS<theory/mps/basics:Matrix product state (MPS)>` or :ref:`MPO<theory/mps/basics:Matrix product operator (MPO)>`.
The virtual dimensions/spaces of the neighbouring MPS/MPO tensors should be consistent.

.. note::
    To create :class:`yastn.Tensor`'s look :ref:`here<tensor/init:Creating symmetric YASTN tensors>`. 

The examples of creating MPS/MPO by hand can be found here:
:ref:`Ground state of Spin-1 AKLT model<examples/mps/build_mps_manually:Ground state of spin-1 AKLT model>`,
:ref:`MPO for hopping model with U(1) symmetry<examples/mps/build_mps_manually:Hamiltonian for nearest-neighbor hopping/XX model>`.


Setting MPO tensors with Hterm
==============================

:class:`yastn.tn.mps.Hterm` is a basic building block of operators. Each ``Hterm`` represents
a product of local (on-site) operators. 

.. autoclass:: yastn.tn.mps.Hterm

.. note::
    The :code:`Hterm` has operators :code:`Hterm.operators` without virtual legs, rank-2 for MPO.

A :code:`list(Hterm)` can be used to create a sum of products. In order to generate full MPO use :code:`exMPO = mps.generate_mpo(I, man_input)`, 
where :code:`man_input` is a list of ``Hterm``-s and :code:`I` is the identity MPO.

.. autofunction:: yastn.tn.mps.generate_mpo

An example using this method can be found :ref:`here<examples/mps/build_mps_Hterm:Building MPO using Hterm>`.


Creating MPO tensors with Generator
===================================

:class:`yastn.tn.mps.Generator` automatizes the creation of MPS and MPO. 
Given a set of local (on-site) operators, i.e. :class:`yastn.operators.Spin12` 
or :class:`yastn.operators.SpinlessFermions`,
one can build both the states and operators. The MPS/MPO can be given as a LaTeX-like expression. 

.. autoclass:: yastn.tn.mps.Generator

We can directly output identity MPO build from identity `I` in operator class.

.. autofunction:: yastn.tn.mps.Generator.I

Generator supports latex-like string instructions to help building MPOs. 
For examples, see :ref:`Generate MPO from LaTex<examples/mps/build_mps_latex:Building MPO from LaTex>`.

Generator provides direct link to random number generator in the backend to fix the seed

.. autofunction:: yastn.tn.mps.Generator.random_seed




Creating random MPS or MPO
--------------------------

:ref:`Generator<mps/init:Creating MPO tensors with Generator>` allows initialization of MPS and MPO filled with random tensors

.. autofunction:: yastn.tn.mps.Generator.random_mps

.. autofunction:: yastn.tn.mps.Generator.random_mpo
