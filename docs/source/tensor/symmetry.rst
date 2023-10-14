Specifying symmetry
===================

YASTN specifies symmetry through any object be it plain Python module, or class which defines

    #. ``SYM_ID`` string label for the symmetry

    #. ``NSYM`` number of elements in the charge vector. For example, `NSYM=1`
       for U(1) or :math:`Z_2` group. For product groups such
       as U(1)xU(1) instead `NSYM=2`.

    #. how to add up charges by implementing a `fuse` function

.. automodule:: yastn.sym.sym_abelian
    :members: sym_abelian

Example symmetries
------------------

* U(1) symmetry

.. literalinclude:: /../../yastn/sym/sym_U1.py

* :math:`Z_2` symmetry

.. literalinclude:: /../../yastn/sym/sym_Z2.py

* :math:`Z_2\times U(1)`

.. literalinclude:: /../../tests/tensor/configs/syms/sym_Z2xU1.py
