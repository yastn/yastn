Creating MPS and MPO
====================

Initializing empty MPS/MPO
--------------------------

MPS and MPO are instances of class :class:`yastn.tn.mps.MpsMpo`.

.. autoclass:: yastn.tn.mps.MpsMpo

One can directly create a new empty MPS of `N` sites using :code:`yastn.tn.mps.Mps(N)`.
Similarly, :code:`yastn.tn.mps.Mpo(N)` initializes MPO.
The new instances start without any tensors defined.

.. autofunction:: yastn.tn.mps.Mps
.. autofunction:: yastn.tn.mps.Mpo


Setting MPS/MPO tensors by hand
-------------------------------

An empty MPS/MPO can be filled with tensors by setting them one by one.

.. code-block::

    import yastn.tn.mps as mps

    # create empty MPS over three sites
    psi = mps.Mps(3)

    # create 3x2x3 random dense tensor
    cfg = yastn.make_config()
    legs = [yastn.Leg(config=cgf, s=-1, D=(3,)),
            yastn.Leg(config=cgf, s=1, D=(2,)),
            yastn.Leg(config=cgf, s=1, D=(3,))]
    A_1 = yastn.rand(config=cfg, legs=legs)

    # assign tensor to site 1
    psi[1] = A_1


Tensor should be of the rank expected for :ref:`MPS<theory/mps/basics:Matrix product state (MPS)>` or :ref:`MPO<theory/mps/basics:Matrix product operator (MPO)>`.
The virtual dimensions/spaces of the neighbouring MPS/MPO tensors should be consistent.
The examples of creating MPS/MPO by hand can be found here:
:ref:`Ground state of Spin-1 AKLT model<examples/mps/build:Ground state of spin-1 AKLT model>`, and
:ref:`MPO for hopping model with U(1) symmetry<examples/mps/build:Hamiltonian for nearest-neighbor hopping/XX model>`.

.. note::
    To create :class:`yastn.Tensor`'s look :ref:`here<tensor/init:Creating symmetric YASTN tensors>`.


Generating MPO using Hterm
--------------------------

:class:`yastn.tn.mps.Hterm` is a basic building block to define operators.
Each ``Hterm`` represents a product of local (on-site) operators.

.. autoclass:: yastn.tn.mps.Hterm

.. note::
     :code:`Hterm.operators` should be a list of matrices with signatutes :math:`s=(1, -1)`.

A list(Hterm) defines a broad class of operators of interests.
In order to generate the corresponding MPO use :code:`mps.generate_mpo(I, terms)`,
where :code:`terms` is a list of Hterm-s and :code:`I` is the identity MPO.
The latter can conviniently be created with a :meth:`generator<yastn.tn.mps.Generator.I>` described below.
An example using this method can be found :ref:`here<examples/mps/build:Building MPO using Hterm>`.

.. autofunction:: yastn.tn.mps.generate_mpo

.. autofunction:: yastn.tn.mps.generate_mpo_fast

.. autofunction:: yastn.tn.mps.generate_mpo_template


Generating product MPS
----------------------

.. autofunction:: yastn.tn.mps.generate_product_mps


Generator class for MPO/MPS
---------------------------

:class:`yastn.tn.mps.Generator` automatizes the creation of MPS and MPO.
Given a set of local (on-site) operators, e.g. :class:`yastn.operators.Spin12`
one can build both the states and operators. The MPS/MPO can be given as a LaTeX-like expression.

.. autoclass:: yastn.tn.mps.Generator

We can directly output identity MPO build from identity `I` in operator class.

.. autofunction:: yastn.tn.mps.Generator.I

Generator supports latex-like string instructions to help building MPOs.
For examples, see :ref:`Generate MPO from LaTex<examples/mps/build:Generator class for MPO/MPS>`.


Generator allows initialization of MPS and MPO filled with random tensors, where local Hilbert spaces are read from identity operator in the Generator.
It also provides a direct link to random number generator in the backend to fix the seed.

.. autofunction:: yastn.tn.mps.Generator.random_mps

.. autofunction:: yastn.tn.mps.Generator.random_mpo

.. autofunction:: yastn.tn.mps.Generator.random_seed


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

MPS/MPO can save as Python `dict` or `HDF5`` file.
The MPS/MPO previously serialized by :meth:`yastn.tn.mps.MpsMpo.save_to_dict` or :meth:`yastn.tn.mps.MpsMpo.save_to_hdf5` can be again deserialized into MPS/MPO.

Examples of exporting and loading MPS/MPO can be found in :ref:`examples/mps/build:save and load mps/mpo`.

.. autofunction:: yastn.tn.mps.MpsMpo.save_to_dict

.. autofunction:: yastn.tn.mps.MpsMpo.save_to_hdf5

.. autofunction:: yastn.tn.mps.load_from_dict

.. autofunction:: yastn.tn.mps.load_from_hdf5
