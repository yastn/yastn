Testing Class Lattice and PEPS 
===============================

Test Lattice
------------

The ``test_Lattice()`` function generates a few lattices and verifies the expected output of some functions. It checks the dimensions, sites, bonds, and nearest neighbor sites
of the lattice. The function is tested on both checkerboard and rectangle lattices with infinite and finite boundaries.

.. literalinclude:: /../../tests/peps/test_geometry.py
        :pyobject: test_Lattice


Test NtuEnv
-----------

The ``test_NtuEnv()`` function tests the nearest environmental sites around a bond and creates indices of the NTU environment. It checks the tensors of the NTU environment 
for both finite and infinite lattices.

.. literalinclude:: /../../tests/peps/test_geometry.py
        :pyobject: test_NtuEnv


