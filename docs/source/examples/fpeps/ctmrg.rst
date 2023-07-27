Test for CTMRG
==============

The tensor making up PEPS can be assigned manually, setting them one by one.

.. note::
        The virtual dimensions/spaces of the neighboring PEPS tensors should be consistent.

CTMRG for 2D Ising model
^^^^^^^^^^^^^^^^^^^^^^^^^
We test the CTMRG :ref:`CTMRG<fpeps/algorithms_CTMRG:Corner transfer matrix renormalization group (CTMRG) algorithm>` by setting 
up a well-known exact PEPS: Thermal State of 2D Ising model amd match the exact solution of magnetization with that of CTMRG 
`Onsager Solution of the 2D Ising model <https://en.wikipedia.org/wiki/Ising_model>`_.

.. literalinclude:: /../../tests/peps/test_ctmrg.py
        :pyobject: test_ctm_loop





