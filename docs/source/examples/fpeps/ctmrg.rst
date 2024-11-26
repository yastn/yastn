CTMRG for 2D Ising model
========================

We test the :ref:`theory/fpeps/basics:Corner transfer matrix renormalization group (CTMRG)` algorithm by
setting up a well-known exact PEPS: Thermal state of the 2D classical Ising model.
We match the output of the CTMRG procedure with the exact results coming from
`Onsager solution of the 2D Ising model <https://en.wikipedia.org/wiki/Ising_model>`_.

We enforce the Z2 symmetry of the model,
which prevents spontaneous symmetry breaking in the ordered phase.
Enforcing this symmetry is important because, in finite systems or numerical simulations,
artificial symmetry breaking can occur due to numerical inaccuracies or finite-size effects.
To that end, we consider Hamiltonian :math:`H = - \sum_{\langle i, j \rangle} X_i X_j`,
where :math:`X_i` is the first Pauli matrix operating at the site :math:`i`, and the sum
is over the nearest neighbor sites :math:`\langle i, j \rangle` on a square lattice.

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