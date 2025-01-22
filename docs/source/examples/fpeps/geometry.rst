Lattice geometry
================

We generate a square lattice geometries and verify the expected output of its methods.
It checks the dimensions, sites, bonds, and nearest neighbor sites of the lattice.
The function is tested on an infinite checkerboard lattice and a finite square lattice with open boundary conditions.

.. code-block:: python

    import pytest
    import yastn
    import yastn.tn.fpeps as fpeps
    from yastn.tn.fpeps import Bond, Site


.. literalinclude:: /../../tests/peps/test_geometry.py
        :pyobject: test_CheckerboardLattice

.. literalinclude:: /../../tests/peps/test_geometry.py
        :pyobject: test_SquareLattice_obc

.. code-block:: python

    test_CheckerboardLattice()
    test_SquareLattice_obc()
