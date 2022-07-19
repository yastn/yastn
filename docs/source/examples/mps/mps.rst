Examples for :meth:`yamps` module
=====================================

Building `YAMPS` object manually
----------------------------------

The content of MPS/MPO can be assigned manualy by setting known tensors one by one.
In order to do that you should prepare :meth:`yast.Tensor` which fits the structure of physical and virtual legs according to :ref:`mps/properties:YAMPS properties`. 
After initialising `YAMPS` object we assign tensor of index :code:`j` with :code:`psi.A[3] = <example tensor>`.
We have to make sure that assined tensor fit along virtual dimension.

The most well known exact MPS construction is the ground state for Affleck-Kennedy-Lieb-Tasaki (AKLT) model.

.. literalinclude:: /../../tests/mps/test_initialization.py
        :pyobject: test_assign_block

The same can be done for MPO. Depending on symmetry of the tensors we will have diffrent definition of them.  
This can be shown for a simple nearest-neightbour hopping Hamiltonian with hopping amplitude `t` and on-site energy `mu`.

.. literalinclude:: /../../tests/mps/generate_by_hand.py
        :pyobject: mpo_XX_model_dense

.. literalinclude:: /../../tests/mps/generate_by_hand.py
        :pyobject: mpo_XX_model_Z2

.. literalinclude:: /../../tests/mps/generate_by_hand.py
        :pyobject: mpo_XX_model_U1


Building `YAMPS` object automatically
-------------------------------------

The MPO can be constructed automatically using dedicated generator supplied with the Hamiltonian. 
This can be shown for a simple nearest-neightbour hopping Hamiltonian 
with hopping amplitude `t` and on-site energy `mu`.

Automatic generator creates MPO with symmetries defies by building operators. 

.. literalinclude:: /../../tests/mps/generate_automatic.py
        :pyobject: mpo_XX_model


.. FOR ALGEBRA 

Copying
------------------------

.. literalinclude:: /../../tests/mps/test_copy.py


Addition
------------------------

.. literalinclude:: /../../tests/mps/test_addition.py


Multiplication
------------------------

.. literalinclude:: /../../tests/mps/test_multiplication.py


.. QR and SVD

Canonical form by QR decomposition
---------------------------------------

.. literalinclude:: /../../tests/mps/test_canonical.py


Canonical form by SVD decomposition
----------------------------------------

.. literalinclude:: /../../tests/mps/test_truncate_svd.py


.. outside world
Save and load
----------------------------------------

.. literalinclude:: /../../tests/mps/test_save_load.py


.. algorithms
DMRG
-----

.. literalinclude:: /../../tests/mps/test_dmrg.py

TDVP
-----

.. literalinclude:: /../../tests/mps/test_tdvp.py
