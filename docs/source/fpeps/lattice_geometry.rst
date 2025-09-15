Lattice Geometry
================

Geometric information about 2D lattice is captured by a class :class:`yastn.tn.fpeps.SquareLattice`
or its special subclasses :class:`yastn.tn.fpeps.RectangularUnitcell`, :class:`yastn.tn.fpeps.CheckerboardLattice`, and :class:`yastn.tn.fpeps.TriangularLattice`.
They all operate on a square lattice, where :class:`yastn.tn.fpeps.SquareLattice` can be used for a finite system or an infinite system with a rectangular unit cell.
:class:`yastn.tn.fpeps.RectangularUnitcell` works for an infinite system with a pattern of repeating tensors inside the unit cell.
:class:`yastn.tn.fpeps.CheckerboardLattice` is a special case of the latter, with a :math:`2 \times 2` unit cell and checkerboard pattern, and
:class:`yastn.tn.fpeps.TriangularLattice` is a special case with a :math:`3 \times 3` unit cell.

The geometry classes provide information on lattice sites (in particular, unique sites in the unit cell),
unique bonds, and a way to navigate the lattice through the information on the neighborhood of each site.

Basic Elements
--------------

Auxiliary objects in lattice definition are :class:`yastn.tn.fpeps.Bond` representing a pair of nearest-neighbor lattice sites,
and :class:`yastn.tn.fpeps.Site` for sites.

.. autoclass:: yastn.tn.fpeps.Bond
    :exclude-members: __init__, __new__

.. autoclass:: yastn.tn.fpeps.Site
    :exclude-members: __init__, __new__

Core Classes
------------

.. autoclass:: yastn.tn.fpeps.SquareLattice
    :members: sites, bonds, nn_site

.. autoclass:: yastn.tn.fpeps.CheckerboardLattice

.. autoclass:: yastn.tn.fpeps.RectangularUnitcell

.. autoclass:: yastn.tn.fpeps.TriangularLattice
