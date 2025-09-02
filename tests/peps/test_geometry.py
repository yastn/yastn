# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Test operation of geometry classes for PEPS on families of square lattices. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site, Bond


def test_CheckerboardLattice(net=None):
    """ Test checkerboard lattice. """
    if net is None:
        net = fpeps.CheckerboardLattice()

    assert net.dims == (2, 2)  # size of unit cell
    assert net.sites() == (Site(0, 0), Site(0, 1))  # two unique sites
    assert net.sites(reverse=True) == (Site(0, 1), Site(0, 0))
    assert net._periodic == 'ii'  # lattice is infinite in both directions.

    # unique horizontal bonds
    assert net.bonds(dirn='h') == (Bond((0, 0), (0, 1)), Bond((0, 1), (0, 2)))
    # unique vertical bonds
    assert net.bonds(dirn='v') == (Bond((0, 0), (1, 0)), Bond((0, 1), (1, 1)))
    # all unique bonds; also in reversed order
    assert net.bonds(reverse=True)[::1] == net.bonds()[::-1]

    # map from site to unique index
    # it is used in Peps.__getitem__(site);  EncCTM.__getitem__(site), ...
    assert all(net.site2index(site) == 0
               for site in [(0, 0), (1, 1), (-3, 3), (1, -1), (2, 0)])
    assert all(net.site2index(site) == 1
               for site in [(1, 0), (0, 1), (2, 5), (1, 2), (-1, 0)])

    # nearest-neighbor site to: right, left, top, bottom
    assert net.nn_site(Site(1, 1), d='r') == (1, 2)
    assert net.nn_site(Site(1, 0), d='l') == (1, -1)
    assert net.nn_site(Site(0, 0), d='t') == (-1, 0)
    assert net.nn_site(Site(1, 0), d='b') == (2, 0)

    # whether a nearest-neighbor bond is horizontal or vertical and
    # whether it is ordered as (left, right) or (top, bottom) lattice sites.
    assert net.nn_bond_dirn(Bond((0, 0), (0, 1))) == 'lr'
    assert net.nn_bond_dirn(Bond((0, 3), (0, 2))) == 'rl'
    assert net.nn_bond_dirn(Bond((1, 3), (0, 3))) == 'bt'
    assert net.nn_bond_dirn(Bond((1, 0), (2, 0))) == 'tb'
    with pytest.raises(yastn.YastnError):
        net.nn_bond_dirn(Bond((1, 0), (3, 0)))
        # Bond((1, 0),(3, 0)) is not a nearest-neighbor bond.

    # order of bonds in net.bond()
    assert all(net.nn_bond_dirn(bond) == 'lr'
               for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_dirn(bond) == 'tb'
               for bond in net.bonds(dirn='v'))

    # pairs of sites consistent with the assumed fermionic order.
    assert net.f_ordered((2, 0), (4, 0))
    assert net.f_ordered((4, 0), (0, 4))
    assert all(net.f_ordered(s0, s1)
               for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert all(net.f_ordered(*bond) for bond in net.bonds())


def test_SquareLattice_obc():
    """ Test SquareLattice with open boundary conditions. """
    net = fpeps.SquareLattice(dims=(3, 2), boundary='obc')

    assert net.dims == (3, 2)  # size of unit cell
    # sites in the lattice
    assert net.sites() == ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
    # horizontal bonds
    assert net.bonds(dirn='h') == (Bond((0, 0), (0, 1)),
                                   Bond((1, 0), (1, 1)),
                                   Bond((2, 0), (2, 1)))
    # vertical bonds
    assert net.bonds(dirn='v') == (Bond((0, 0), (1, 0)),
                                   Bond((1, 0), (2, 0)),
                                   Bond((0, 1), (1, 1)),
                                   Bond((1, 1), (2, 1)))
    #
    # Lattice has open boundary conditions in both directions.
    assert net._periodic == 'oo'
    #
    # no nearest-neighbor sites from some edges in some directions
    assert net.nn_site((0, 1), d='r') is None
    assert net.nn_site((0, 1), d='t') is None
    assert net.nn_site((2, 0), d='l') is None
    assert net.nn_site((2, 0), d='b') is None
    assert net.nn_site(None, d='l') is None
    #
    # nn_site can shift by more then 1 site.
    assert net.nn_site((0, 1), d=(2, -1)) == Site(2, 0)
    #
    # trivial map from site to unique index
    assert net.site2index((1, 0)) == (1, 0)
    assert net.site2index(None) == None

    # whether a nearest-neighbor bond is horizontal or vertical and
    # whether it is ordered as (left, right) or (top, bottom) lattice sites.
    assert net.nn_bond_dirn(Bond((0, 0), (0, 1))) == 'lr'
    assert net.nn_bond_dirn(Bond((0, 1), (0, 0))) == 'rl'
    assert net.nn_bond_dirn(Bond((2, 1), (1, 1))) == 'bt'
    assert net.nn_bond_dirn(Bond((1, 0), (2, 0))) == 'tb'
    with pytest.raises(yastn.YastnError):
        net.nn_bond_dirn(Bond((0, 0), (2, 0)))
        # Bond((0, 0), (2, 0)) is not a nearest-neighbor bond.

    # order of bonds in net.bond()
    assert all(net.nn_bond_dirn(bond) == 'lr'
               for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_dirn(bond) == 'tb'
               for bond in net.bonds(dirn='v'))

    # pairs of sites consistent with the assumed fermionic order.
    assert net.f_ordered((0, 0), (1, 0))
    assert net.f_ordered((0, 0), (0, 1))
    assert net.f_ordered((1, 0), (0, 1))
    assert all(net.f_ordered(s0, s1)
               for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert all(net.f_ordered(*bond) for bond in net.bonds())


def test_SquareLattice_cylinder():
    net = fpeps.SquareLattice(dims=(3, 2), boundary='cylinder')

    assert net.dims == (3, 2)
    assert net.sites() == ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
    assert net.bonds('h') == (Bond((0, 0), (0, 1)),
                              Bond((1, 0), (1, 1)),
                              Bond((2, 0), (2, 1)))
    assert net.bonds('v') == (Bond((0, 0), (1, 0)),
                              Bond((1, 0), (2, 0)),
                              Bond((2, 0), (0, 0)),
                              Bond((0, 1), (1, 1)),
                              Bond((1, 1), (2, 1)),
                              Bond((2, 1), (0, 1)))

    assert net.nn_site((0, 1), d='r') is None
    assert net.nn_site((0, 1), d='t') == Site(2, 1)
    assert net.nn_site((2, 0), d='l') is None
    assert net.nn_site((2, 0), d='b') == Site(0, 0)
    assert net.nn_site((0, 0), d=(4, 4)) is None
    assert net.nn_site((2, 0), d=(2, 1)) == Site(1, 1)
    assert net.nn_site((2, 0), d=(-3, 1)) == Site(2, 1)

    assert net.nn_bond_dirn(Bond((0, 0), (0, 1))) == 'lr'
    assert net.nn_bond_dirn(Bond((0, 1), (0, 0))) == 'rl'
    assert net.nn_bond_dirn(Bond((2, 0), (0, 0))) == 'tb'
    assert net.nn_bond_dirn(Bond((0, 1), (2, 1))) == 'bt'
    with pytest.raises(yastn.YastnError):
        net.nn_bond_dirn(Bond((3, 0), (2, 0)))
        # Bond((3, 0), (2, 0)) is not a nearest-neighbor bond.

    assert all(net.nn_bond_dirn(bond) == 'lr'
               for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_dirn(bond) == 'tb'
               for bond in net.bonds(dirn='v'))

    assert net.nn_bond_dirn(Bond((2, 0), (0, 0))) == 'tb'
    assert net.nn_bond_dirn(Bond((0, 0), (2, 0))) == 'bt'
    assert not net.f_ordered((2, 0), (0, 0))
    assert net.f_ordered((0, 0), (2, 0))

    assert all(net.f_ordered(s0, s1) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert not all(net.f_ordered(*bond) for bond in net.bonds())  #  PBC bonds in cylinder are not fermionically alligned


def test_SquareLattice_infinite():
    assert str(Site(0, 0)) == "Site(0, 0)"
    assert str(Bond(Site(0, 0), Site(0, 1))) == "Bond((0, 0), (0, 1))"

    net = fpeps.SquareLattice(dims=(3, 2), boundary='infinite')

    assert net.dims == (3, 2)
    assert net.sites() == ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
    assert net.bonds(dirn='h') == (Bond((0, 0), (0, 1)),
                                   Bond((1, 0), (1, 1)),
                                   Bond((2, 0), (2, 1)),
                                   Bond((0, 1), (0, 2)),
                                   Bond((1, 1), (1, 2)),
                                   Bond((2, 1), (2, 2)))
    assert net.bonds(dirn='v') == (Bond((0, 0), (1, 0)),
                                   Bond((1, 0), (2, 0)),
                                   Bond((2, 0), (3, 0)),
                                   Bond((0, 1), (1, 1)),
                                   Bond((1, 1), (2, 1)),
                                   Bond((2, 1), (3, 1)))

    assert net.nn_site((0, 1), d='r') == (0, 2)
    assert net.nn_site((0, 1), d='t') == (-1, 1)
    assert net.nn_site((2, 0), d='l') == (2, -1)
    assert net.nn_site((2, 0), d='b') == (3, 0)
    assert net.nn_site((0, 1), d=(3, 4)) == Site(3, 5)
    assert net.site2index((1, 0)) == (1, 0)
    assert net.site2index(None) == None

    assert all(net.nn_bond_dirn(bond) == 'lr' for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_dirn(bond) == 'tb' for bond in net.bonds(dirn='v'))

    assert all(net.f_ordered(s0, s1) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert all(net.f_ordered(*bond) for bond in net.bonds())

    ##########

    with pytest.raises(yastn.YastnError):
        fpeps.SquareLattice(dims=(3, 2), boundary='some')
        #  boundary='some' not recognized; should be 'obc', 'infinite', or 'cylinder'


def test_RectangularUnitCell_1x1():
    for pattern in [[[0,],],
                    {(0, 0): 0}]:
        g = fpeps.RectangularUnitcell(pattern=pattern)

        assert g.dims == (1,1)
        assert g.sites() == (Site(0, 0),)

        assert g.bonds(dirn='h') == (Bond((0, 0), (0, 1)),)
        assert g.bonds(dirn='v') == (Bond((0, 0), (1, 0)),)

        assert g.nn_site(Site(0, 0), d='r') == (0, 1)
        assert g.nn_site(Site(0, 0), d='b') == (1, 0)

        assert all(g.site2index(site) == 0 for site in [(0, 0), (1, 1), (-3, 3), (1, -1), (2, 0)])
        assert all(g.f_ordered(*bond) for bond in g.bonds())


def test_RectangularUnitCell_2x2_bipartite():
    for pattern in ([[0, 1], [1, 0]],
                    {(0, 0): 0, (1, 1): 0, (0, 1): 1, (1, 0): 1}):

        g = fpeps.RectangularUnitcell(pattern=pattern)

        test_CheckerboardLattice(net=g)

        assert all(g.site2index(s) == 0 for s in [(0, 0), (1, 1), (-3, 3), (1, -1), (2, 0)])
        assert all(g.sites()[g.site2index(s)] == (0, 0) for s in [(0, 0), (1, 1), (-3, 3), (1, -1), (2, 0)])

        assert all(g.site2index(s) == 1 for s in [(1, 0), (0, 1), (2, 5), (1, 2), (-1, 0)])
        assert all(g.sites()[g.site2index(s)] == (0, 1) for s in [(1, 0), (0, 1), (2, 5), (1, 2), (-1, 0)])


def test_RectangularUnitCell_3x3_Q_1o3_1o3():
    g = fpeps.RectangularUnitcell(pattern=[[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    assert g.dims == (3, 3)
    assert g.sites() == (Site(0, 0), Site(0, 1), Site(0, 2))

    print(g)

    assert all(g.site2index(s) == 0 for s in [(0, 0), (0, 3), (0, -3), (1, -1), (-2, 2)])
    assert all(g.sites()[g.site2index(s)] == (0, 0) for s in [(0, 0), (0, 3), (0, -3), (1, -1), (-2, 2)])

    assert all(g.site2index(s) == 1 for s in [(0, 1), (0, 4), (0, -2), (1, 0), (-2, 3)])
    assert all(g.sites()[g.site2index(s)] == (0, 1) for s in [(0, 1), (0, 4), (0, -2), (1, 0), (-2, 3)])

    assert all(g.site2index(s) == 2 for s in [(0, 2), (0, 5), (0, -1), (1, 1), (-2, 4)])
    assert all(g.sites()[g.site2index(s)] == (0, 2) for s in [(0, 2), (0, 5), (0, -1), (1, 1), (-2, 4)])


def test_geometry_equal():
    gs = [fpeps.TriangularLattice(),
          fpeps.RectangularUnitcell(pattern={(0, 0): 0, (1, 1): 0, (0, 1): 1, (1, 0): 1}),
          fpeps.CheckerboardLattice(),
          fpeps.SquareLattice(dims=(2, 2), boundary='infinite'),
          fpeps.SquareLattice(dims=(2, 2), boundary='obc')]

    assert gs[0] != None
    assert all(g == g for g in gs)
    assert all(g0 != g1 for n, g0 in enumerate(gs) for g1 in gs[n+1:])


def test_RectangularUnitCell_raises():
    with pytest.raises(yastn.YastnError):
        fpeps.RectangularUnitcell(pattern={(-1, -1): 1, (0, 0): 1, (0, -1): 2, (-1, 0): 2})
        # RectangularUnitcell: pattern keys should cover a rectangle index (0, 0) to (Nx - 1, Ny - 1).
    with pytest.raises(yastn.YastnError):
        fpeps.RectangularUnitcell(pattern={(0, 0): 1, (1, 0): 1, (0, 1): 2})
        # RectangularUnitcell: pattern keys should cover a rectangle index (0, 0) to (Nx - 1, Ny - 1).
    with pytest.raises(yastn.YastnError):
        fpeps.RectangularUnitcell(pattern=[[1, 0], [0]])
        # RectangularUnitcell: pattern should form a two-dimensional square matrix of hashable labels.
    with pytest.raises(yastn.YastnError):
        fpeps.RectangularUnitcell(pattern=[1, 0])
        # RectangularUnitcell: pattern should form a two-dimensional square matrix of hashable labels.
    with pytest.raises(yastn.YastnError):
        fpeps.RectangularUnitcell(pattern=[[1, 0], [['a'], 1]])
        # RectangularUnitcell: pattern labels should be hashable.
    with pytest.raises(yastn.YastnError):
        fpeps.RectangularUnitcell(pattern=[[1, 0], [1, 1]])
        # RectangularUnitcell: each unique label should have the same neighbors.


def test_Peps_get_set():
    """ Setitem and getitem in peps allows to acces individual tensors. """
    net = fpeps.CheckerboardLattice()
    #
    psi = fpeps.Peps(net)
    assert all(psi[site] == None for site in psi.sites())  # all tensors initialized as None
    #
    psi[(0, 0)] = "Wannabe tensor"
    # on a checkerboard lattice we have the same tensor at Site(0, 0) and Site(1, 1)
    assert psi[(0, 0)] == psi[(1, 1)] == psi[(5, 1)] == "Wannabe tensor"
    assert psi[(0, 1)] == psi[(1, 0)] == psi[(5, 0)] == None

    ##########

    net = fpeps.SquareLattice(dims=(3, 3), boundary='obc')
    psi = fpeps.Peps(net)

    assert all(psi[site] == None for site in psi.sites())  # all tensors initialized as None
    psi[(0, 0)] = "Wannabe tensor"
    assert psi[(0, 0)] == "Wannabe tensor"
    assert psi[(1, 1)] is None

    with pytest.raises(KeyError):
        psi[None]

    with pytest.raises(KeyError):
        psi[(5, 3)]

    ##########

    net = fpeps.SquareLattice(dims=(3, 3), boundary='infinite')
    psi = fpeps.Peps(net)

    assert all(psi[site] == None for site in psi.sites())  # all tensors initialized as None
    psi[(0, 0)] = "Wannabe tensor"
    assert psi[(0, 0)] == psi[(3, 3)] == psi[(9, 6)] == "Wannabe tensor"
    assert psi[(0, 1)] is None


def test_Peps_inheritance():
    net = fpeps.SquareLattice(dims=(3, 2), boundary='infinite')
    psi = fpeps.Peps(net)

    assert psi.Nx == 3
    assert psi.Ny == 2
    assert psi.dims == (3, 2)
    assert len(psi.bonds()) == 12
    assert len(psi.sites()) == 6
    assert psi.nn_site((0, 0), 'r') == (0, 1)
    assert psi.nn_bond_dirn(Bond((0, 0), (0, 1))) == 'lr'
    assert psi.f_ordered((0, 0), (0, 1))


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
