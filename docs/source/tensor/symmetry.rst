Specifying symmetry
===================

YAST specifies symmetry through any object be it plain Python module, 
`types.SimpleNamespace <https://docs.python.org/3/librarytypes.html#types.SimpleNamespace>`_,
or class which defines

    * ``SYM_ID`` string label for the symmetry
    * ``NSYM`` number of elements in the charge vector. For example, `NSYM=1` for U(1) or :math:`Z_2`
      group. For product groups such as U(1)xU(1) instead `NSYM=2`.
    * how to add charges by implementing a `fuse` function

.. automodule:: yast.sym.sym_abelian
    :members: sym_abelian

Example symmetries
------------------

* U(1) symmetry

.. literalinclude:: /../../yast/sym/sym_U1.py

* :math:`Z_2` symmetry

.. literalinclude:: /../../yast/sym/sym_Z2.py

* :math:`Z_2\times U(1)` as plain Python module

.. literalinclude:: /../../tests/tensor/configs/syms/sym_Z2xU1.py
