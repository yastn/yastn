Building MPS/MPO manually
=========================

The tensor making up MPS/MPO can be assigned manually, setting them one by one.

.. note::
        The virtual dimensions/spaces of the neighboring MPS/MPO tensors should be consistent.

Ground state of spin-1 AKLT model
---------------------------------

Here, as an example, we set up a well-known exact MPS: A ground state of
`Affleck-Kennedy-Lieb-Tasaki (AKLT) model <https://en.wikipedia.org/wiki/AKLT_model>`_.

.. literalinclude:: /../../tests/mps/test_initialization.py
        :pyobject: build_spin1_aklt_state


Hamiltonian for nearest-neighbor hopping/XX model
-------------------------------------------------

The same can be done for MPOs. Here, we show a construction of a simple
nearest-neighbour hopping Hamiltonian with hopping amplitude :math:`t`
and on-site energy :math:`\mu` with different 
realizations of explicit symmetry.

.. literalinclude:: /../../tests/mps/test_generator.py
        :pyobject: mpo_nn_hopping_manually
