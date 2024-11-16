Algorithms: DMRG
================

Low energy states of the XX model
---------------------------------

In order to execute :ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG)>`
we need the hermitian operator (typically a Hamiltonian) written as MPO and an initial guess for MPS.
Here is a simple example of DMRG used to obtain the ground state of quadratic Hamiltonian:

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: test_dmrg_XX_model_dense

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: run_dmrg

The same can be done for other symmetries:

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: test_dmrg_XX_model_U1

See as well the examples for :ref:`examples/mps/algebra:Multiplication`,
which contains DMRG and variational MPS compression.
