CTMRG for 2D Ising model
========================

We test the Corner Transfer Matrix Renormalization Group Algorithm :ref:`CTMRG<fpeps/environments:Corner transfer matrix renormalization group (CTMRG)>` by setting
up a well-known exact PEPS: Thermal State of 2D Ising model amd match the exact solution of magnetization with that of CTMRG
`Onsager Solution of the 2D Ising model <https://en.wikipedia.org/wiki/Ising_model>`_.

.. literalinclude:: /../../tests/peps/test_ctmrg.py
        :pyobject: run_ctm
