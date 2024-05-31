Lattice Geometry
================

Geometric information about 2D lattice are captured by a class :class:`yastn.tn.fpeps.SquareLattice`
or its special subclass :class:`yastn.tn.fpeps.CheckerboardLattice`.
They provide information on lattice sites (in particular, unique sites in the unit cell),
unique bonds, and a way to navigate the lattice through the neighborhood of each site.

.. autoclass:: yastn.tn.fpeps.SquareLattice
    :members: sites, bonds, nn_site

.. autoclass:: yastn.tn.fpeps.CheckerboardLattice

Auxiliary objects in lattice definition are :class:`yastn.tn.fpeps.Bond`
representing a pair of nearest-neighbor lattice sites,
and :class:`yastn.tn.fpeps.Site` for sites.

.. autoclass:: yastn.tn.fpeps.Bond
    :exclude-members: __init__, __new__

.. autoclass:: yastn.tn.fpeps.Site
    :exclude-members: __init__, __new__
