CTMRG for 2D Ising model
========================

We test the :ref:`theory/fpeps/basics:Corner transfer matrix renormalization group (CTMRG)` algorithm by
setting up a well-known exact PEPS: Thermal state of the 2D classical Ising model.
We match the output of the CTMRG procedure with the exact results coming from
`Onsager solution of the 2D Ising model <https://en.wikipedia.org/wiki/Ising_model>`_.

.. code-block:: python

   import numpy as np
   import pytest
   import yastn
   import yastn.tn.fpeps as fpeps


.. literalinclude:: /../../tests/peps/test_ctmrg.py
        :pyobject: test_ctmrg_Ising


.. code-block:: python

   config_kwargs = {"backend": "np"}
   test_ctmrg_Ising(config_kwargs)