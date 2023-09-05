Structure
=========

Geometry
--------

The module :code:`yastn.tn.fpeps._geometry.py` contains classes to represent the lattice geometry of the Projected Entangled Pair States (PEPS).
The classes defined in the module include:

- **Bond**:

A named tuple represents a bond between two neighboring lattice sites.
The sites are arranged in agreement with the fermionic order.
Each bond has a directionality, captured by the `dirn` property, equal to `ver` or `hor`, respectively for vertical and horizontal bonds.

.. autofunction:: yastn.tn.fpeps._geometry.Bond

- **Lattice**:

A class that represents the geometric information about a 2D lattice.
Two supported lattice types are 'checkerboard' or 'rectangle', and different boundary conditions are 'finite' or 'infinite'.
The Lattice class also provides methods to navigate this geometry, for instance, by providing the neighboring sites or bonds. It thus provides the backbone for
the PEPS by defining its spatial structure. In the context of strongly correlated systems, the lattice and its properties can drastically affect the system's behavior.

.. autofunction:: yastn.tn.fpeps._geometry.Lattice

- **Peps**:

The Peps class extends the Lattice class, holding the PEPS data itself, and provides additional functionalities specifically related to PEPS.
The methods included in the Peps class, such as mpo and boundary_mps, allow for efficient manipulation and transformation of the PEPS, which are key tasks in many numerical algorithms used to study these systems.

.. autofunction:: yastn.tn.fpeps._geometry.Peps

.. literalinclude:: /../../tests/peps/test_geometry.py
      :pyobject: test_Lattice

.. literalinclude:: /../../tests/peps/test_geometry.py
      :pyobject: test_Peps_get_set

.. literalinclude:: /../../tests/peps/test_geometry.py
      :pyobject: test_NtuEnv
