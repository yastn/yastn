Algorithms for MPS
==================


DMRG
----

In order to execute :ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG)>`
we need the hermitian operator (typically a Hamiltonian) written as MPO and an initial guess for MPS.
Here is a simple example of DMRG used to obtain the ground state of quadratic Hamiltonian:

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: dmrg_XX_model_dense

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: run_dmrg


The same can be done for other symmetries:

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: dmrg_XX_model_Z2

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: dmrg_XX_model_U1

See as well the examples for :ref:`examples/mps/algebra:Multiplication`,
which contains DMRG and variational MPS compression.


TDVP
----

Sudden quench in a free-fermionic model
"""""""""""""""""""""""""""""""""""""""

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: tdvp_sudden_quench

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: correlation_matrix_from_mps

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: gs_correlation_matrix_exact

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: evolve_correlation_matrix_exact


Slow quench across a quantum critical point in a transverse Ising chain
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: tdvp_KZ_quench
