.. QR and SVD

Algorithms for MPS
==================


DMRG
----

In order to execute :ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG) algorithm>` we need the hermitian operator (typically a Hamiltonian) written as MPO, and an initial guess for MPS.

Here is a simple example for DMRG used to obtain a ground state for a quadratic Hamiltonian:

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: test_dense_dmrg

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: run_dmrg


The same can be done for any symmetry:

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: test_Z2_dmrg

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: test_U1_dmrg

Also see the test in examples for :ref:`examples/mps/algebra:Multiplication`.


TDVP
----

Test :ref:`TDVP<mps/algorithms_tdvp:time-dependent variational principle (tdvp) algorithm>` simulating time evolution after a sudden quench in a free-fermionic model.

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: test_tdvp_hermitian

Slow quench across a quantum critical point in a transverse Ising chain.

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: test_tdvp_time_dependent

