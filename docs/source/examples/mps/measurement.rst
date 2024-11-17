
Expectation values
==================

.. code-block:: python

    import numpy as np
    import yastn
    import yastn.tn.mps as mps
    config_kwargs = {"backend": "np"}

Expectation values in AKLT state
--------------------------------

.. literalinclude:: /../../tests/mps/test_measurement.py
        :pyobject: test_measure_mps_aklt

Entropy and spectrum in GHZ-like state
--------------------------------------

.. literalinclude:: /../../tests/mps/test_measurement.py
        :pyobject: test_mps_spectrum_ghz
