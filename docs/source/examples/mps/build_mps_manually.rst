Building MPS/MPO manually
=========================

The tensor making up MPS/MPO can be assigned manualy by setting them one by one.

.. note::
        The virtual dimensions/spaces of the neighbouring MPS/MPO tensors have to remain consistent.

Ground state of Spin-1 AKLT model
---------------------------------

Here, we show an example of such code for the most well-known exact MPS: The ground state of  
`Affleck-Kennedy-Lieb-Tasaki (AKLT) model <https://en.wikipedia.org/wiki/AKLT_model>`_.

.. literalinclude:: /../../tests/mps/test_initialization.py
        :pyobject: build_spin1_aklt_state


MPO for hopping model (no symmetry)
-----------------------------------

The same can be done for MPOs. Here, we show construction of a simple 
nearest-neighbour hopping Hamiltonian with hopping amplitude `t` 
and on-site energy :math:`\mu` with different 
realizations of explicit symmetry.

.. literalinclude:: /../../tests/mps/test_generator.py
        :pyobject: mpo_XX_model_dense

.. _example hopping with z2 symmetry:

MPO for hopping model with :math:`\mathbb{Z}_2` symmetry
--------------------------------------------------------

.. literalinclude:: /../../tests/mps/test_generator.py
        :pyobject: mpo_XX_model_Z2

.. _example hopping with u1 symmetry:

MPO for hopping model with U(1) symmetry
----------------------------------------

.. literalinclude:: /../../tests/mps/test_generator.py
        :pyobject: mpo_XX_model_U1