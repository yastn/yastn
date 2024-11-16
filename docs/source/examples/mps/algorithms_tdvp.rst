Algorithms: TDVP
================

Sudden quench in a free-fermionic model
---------------------------------------

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: test_tdvp_sudden_quench

The example employs a few auxiliary functions to calculate the exact reference results
using the free-fermionic nature of the illustrated setup.

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: correlation_matrix_from_mps

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: gs_correlation_matrix_exact

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: evolve_correlation_matrix_exact


Quench across a quantum critical point in a transverse Ising chain
-----------------------------------------------------------------------

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: test_tdvp_KZ_quench
