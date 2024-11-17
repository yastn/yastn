Algorithms: TDVP
================

.. code-block:: python

    import numpy as np
    import yastn
    import yastn.tn.mps as mps
    config_kwargs = {"backend": "np"}

This tests employs ``mpo_hopping_Hterm`` function defined in :ref:`examples/mps/build:building mpo using hterm`,
and auxlliary functions defined below.

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
