Creating MPS and MPO
====================

Initializing empty MPS/MPO
--------------------------

MPS and MPO are instances of class :class:`yastn.tn.mps.MpsMpo`, where :code:`N` specifies number of sites, and :code:`nr_phys` specifies the number of physical legs, i.e., one for MPS and 2 for MPO.
It supports one-dimensional tensor networks, defined by dict of rank-3 or rank-4 :class:`yastn.Tensor`-s for MPS and MPO, respectively.

.. autoclass:: yastn.tn.mps.MpsMpo

One can directly create a new empty MPS of `N` sites using :code:`psi = yastn.tn.mps.Mps(N)`. Tensors are not initialized at this point.
Similarly, :code:`psi = yastn.tn.mps.Mpo(N)` initializes MPO.
The new instance starts without any tensors defined.

.. autofunction:: yastn.tn.mps.Mps
.. autofunction:: yastn.tn.mps.Mpo


Setting MPS/MPO tensors by hand
-------------------------------

An empty MPS/MPO can be filled with tensors by setting them one by one.

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
:ref:`Ground state of Spin-1 AKLT model<examples/mps/build:Ground state of spin-1 AKLT model>`,
:ref:`MPO for hopping model with U(1) symmetry<examples/mps/build:Hamiltonian for nearest-neighbor hopping/XX model>`.


Generating MPO using Hterm
--------------------------

:class:`yastn.tn.mps.Hterm` is a basic building block of operators. Each ``Hterm`` represents
a product of local (on-site) operators.

.. autoclass:: yastn.tn.mps.Hterm

.. note::
    The :code:`Hterm` has operators :code:`Hterm.operators` without virtual legs, rank-2 for MPO.

A :code:`list(Hterm)` can be used to create a sum of products. In order to generate full MPO use :code:`exMPO = mps.generate_mpo(I, man_input)`,
where :code:`man_input` is a list of ``Hterm``-s and :code:`I` is the identity MPO.

.. autofunction:: yastn.tn.mps.generate_mpo

An example using this method can be found :ref:`here<examples/mps/build:Building MPO using Hterm>`.


Generator class for MPO/MPS
---------------------------

:class:`yastn.tn.mps.Generator` automatizes the creation of MPS and MPO.
Given a set of local (on-site) operators, i.e. :class:`yastn.operators.Spin12`
or :class:`yastn.operators.SpinlessFermions`,
one can build both the states and operators. The MPS/MPO can be given as a LaTeX-like expression.

.. autoclass:: yastn.tn.mps.Generator

We can directly output identity MPO build from identity `I` in operator class.

.. autofunction:: yastn.tn.mps.Generator.I

Generator supports latex-like string instructions to help building MPOs.
For examples, see :ref:`Generate MPO from LaTex<examples/mps/build:Generator class for MPO/MPS>`.

Generator provides direct link to random number generator in the backend to fix the seed

.. autofunction:: yastn.tn.mps.Generator.random_seed


Creating random MPS/MPO
-----------------------

:ref:`Generator<mps/init:Generator class for MPO/MPS>` allows initialization of MPS and MPO filled with random tensors, where local Hilbert spaces are read from identity operator in the Generator.

.. autofunction:: yastn.tn.mps.Generator.random_mps

.. autofunction:: yastn.tn.mps.Generator.random_mpo


Making a copy of MPS/MPO
------------------------

To create an independent copy or clone of MPS/MPO :code:`psi` call :code:`psi.copy()`
or :code:`psi.clone()`, respectively.
It is also possible to make a shallow copy with :code:`psi.shallow_copy()`, where explicit copies of data tensors are not created.

.. autofunction:: yastn.tn.mps.MpsMpo.shallow_copy

.. autofunction:: yastn.tn.mps.MpsMpo.copy

.. autofunction:: yastn.tn.mps.MpsMpo.clone


Import and export MPS/MPO from/to different formats
---------------------------------------------------

MPS/MPO can save as Python :code:`dict` or HDF5 file.
The MPS/MPO previously serialized by :meth:`yastn.tn.mps.MpsMpo.save_to_dict` or :meth:`yastn.tn.mps.MpsMpo.save_to_hdf5` can be again deserialized into MPS/MPO.

Examples of exporting and loading MPS/MPO can be found in :ref:`examples/mps/build:save and load mps/mpo`.

.. autofunction:: yastn.tn.mps.MpsMpo.save_to_dict

.. autofunction:: yastn.tn.mps.MpsMpo.save_to_hdf5

.. autofunction:: yastn.tn.mps.load_from_dict

.. autofunction:: yastn.tn.mps.load_from_hdf5
