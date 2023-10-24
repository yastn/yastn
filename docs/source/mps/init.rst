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
The virtual dimensions/spaces of the neighboring MPS/MPO tensors should be consistent, which, however, is not tested during direct assigment.
For examples creating MPS/MPO by hand, see :ref:`Ground state of Spin-1 AKLT model<examples/mps/build:Ground state of spin-1 AKLT model>`
and :ref:`MPO for hopping model with U(1) symmetry<examples/mps/build:Hamiltonian for nearest-neighbor hopping/XX model>`.


Initializing product MPS/MPO
-----------------------------

.. autofunction:: yastn.tn.mps.product_mps

.. autofunction:: yastn.tn.mps.product_mpo


Initializing random MPS/MPO
-----------------------------

.. autofunction:: yastn.tn.mps.random_mps

.. autofunction:: yastn.tn.mps.random_mpo


Generating MPO using Hterm
--------------------------

We provide functionality to build MPO representations for a broad class of
operators, e.g., Hamiltonians, given as a sum of products of local (on-site) operators.
They are encoded as a list of :class:`mps.Hterm<yastn.tn.mps.Hterm>`, where each :class:`mps.Hterm<yastn.tn.mps.Hterm>`
represents a product of local operators, including the numerical coefficient (amplitude) in front of that product.

To generate the corresponding MPO use :code:`mps.generate_mpo(I, terms)`,
where `terms` is a list of Hterm-s and `I` is the identity MPO.
The latter can be conveniently created as shown in :class:`mps.product_mpo<yastn.tn.mps.product_mpo>`.
For an example, see :ref:`MPO for spinless fermions with hopping at arbitrary range using Hterms<examples/mps/build:Building MPO using Hterm>`.

.. autoclass:: yastn.tn.mps.Hterm

.. autofunction:: yastn.tn.mps.generate_mpo

.. autofunction:: yastn.tn.mps.generate_mpo_fast

.. autofunction:: yastn.tn.mps.generate_mpo_preprocessing


Generator class for MPO/MPS (beta)
----------------------------------

A class supporting automatizes generation of MPOs from LaTeX-like expressions.

.. autoclass:: yastn.tn.mps.Generator

We can directly output identity MPO built from the identity `I` from the operator generator class.

.. autofunction:: yastn.tn.mps.Generator.I

.. autofunction:: yastn.tn.mps.Generator.mpo_from_latex

Generator supports latex-like string instructions to help building MPOs.
For examples, see :ref:`Generate MPO from LaTex<examples/mps/build:Generator class for MPO/MPS>`.


Generator allows initialization of MPS and MPO filled with random tensors, where local Hilbert spaces are read from the identity operator in the Generator.
It also provides a direct link to a random number generator in the backend to fix the seed.

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

MPS/MPO can be saved as Python `dict` or `HDF5` file.
The MPS/MPO previously serialized by :meth:`yastn.tn.mps.MpsMpo.save_to_dict`
or :meth:`yastn.tn.mps.MpsMpo.save_to_hdf5` can be again deserialized into MPS/MPO.

Examples of exporting and loading MPS/MPO can be found in :ref:`examples/mps/build:save and load mps/mpo`.

.. autofunction:: yastn.tn.mps.MpsMpo.save_to_dict

.. autofunction:: yastn.tn.mps.MpsMpo.save_to_hdf5

.. autofunction:: yastn.tn.mps.load_from_dict

.. autofunction:: yastn.tn.mps.load_from_hdf5
