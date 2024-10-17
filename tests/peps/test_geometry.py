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
    """ Generate a few lattices and verify the expected output of some functions. """

    assert str(Site(0, 0)) == "Site(0, 0)"
    assert str(Bond(Site(0, 0), Site(0, 1))) == "Bond((0, 0), (0, 1))"

    if net is None:
        net = fpeps.CheckerboardLattice()

    assert net.dims == (2, 2)
    assert net.sites() == (Site(0, 0), Site(0, 1))
    assert net.sites(reverse=True) == (Site(0, 1), Site(0, 0))
    assert net._periodic == 'ii'

    assert net.bonds(dirn='h') == (Bond((0, 0), (0, 1)), Bond((0, 1), (0, 2)))
    assert net.bonds(dirn='v') == (Bond((0, 0), (1, 0)), Bond((0, 1), (1, 1)))
    assert net.bonds(reverse=True)[::1] == net.bonds()[::-1]

    assert all(net.site2index(site) == 0 for site in [(0, 0), (1, 1), (-3, 3), (1, -1), (2, 0)])
    assert all(net.site2index(site) == 1 for site in [(1, 0), (0, 1), (2, 5), (1, 2), (-1, 0)])

    assert net.nn_site(Site(1, 1), d='r') == (1, 2)
    assert net.nn_site(Site(1, 0), d='l') == (1, -1)
    assert net.nn_site(Site(0, 0), d='t') == (-1, 0)
    assert net.nn_site(Site(1, 0), d='b') == (2, 0)

    assert net.nn_bond_type(Bond((0, 0), (0, 1))) == ('h', True)  # 'lr'
    assert net.nn_bond_type(Bond((0, 3), (0, 2))) == ('h', False) # 'rl'
    assert net.nn_bond_type(Bond((1, 3), (0, 3))) == ('v', False) # 'bt'
    assert net.nn_bond_type(Bond((1, 0), (2, 0))) == ('v', True)  # 'tb'
    with pytest.raises(yastn.YastnError):
        net.nn_bond_type(Bond((1, 0), (3, 0)))
        # Bond((1,0),(3,0)) is not a nearest-neighboor bond.

    assert all(net.nn_bond_type(bond) == ('h', True) for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_type(bond) == ('v', True) for bond in net.bonds(dirn='v'))

    assert net.f_ordered((2, 0), (4, 0))
    assert net.f_ordered((4, 0), (0, 4))
    assert all(net.f_ordered(s0, s1) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert all(net.f_ordered(*bond) for bond in net.bonds())


def test_SquareLattice():
    net = fpeps.SquareLattice(dims=(3, 2), boundary='obc')

    assert net.dims == (3, 2)
    assert net.sites() == ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
    assert net.bonds(dirn='h') == (Bond((0, 0), (0, 1)),
                                   Bond((1, 0), (1, 1)),
                                   Bond((2, 0), (2, 1)))
    assert net.bonds(dirn='v') == (Bond((0, 0), (1, 0)),
                                   Bond((1, 0), (2, 0)),
                                   Bond((0, 1), (1, 1)),
                                   Bond((1, 1), (2, 1)))

    assert net.nn_site((0, 1), d='r') is None
    assert net.nn_site((0, 1), d='t') is None
    assert net.nn_site((2, 0), d='l') is None
    assert net.nn_site((2, 0), d='b') is None
    assert net.nn_site(None, d='l') is None

    assert net.nn_site((0, 1), d=(2, -1)) == Site(2, 0)
    assert net.site2index((1, 0)) == (1, 0)
    assert net.site2index(None) == None

    assert net.nn_bond_type(Bond((0, 0), (0, 1))) == ('h', True)  # 'lr'
    assert net.nn_bond_type(Bond((0, 1), (0, 0))) == ('h', False) # 'rl'
    assert net.nn_bond_type(Bond((2, 1), (1, 1))) == ('v', False) # 'bt'
    assert net.nn_bond_type(Bond((1, 0), (2, 0))) == ('v', True)  # 'tb'
    with pytest.raises(yastn.YastnError):
        net.nn_bond_type(Bond((0, 0), (2, 0)))
        # Bond((0,0),(2,0)) is not a nearest-neighboor bond.

    assert all(net.nn_bond_type(bond) == ('h', True) for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_type(bond) == ('v', True) for bond in net.bonds(dirn='v'))

    assert net.f_ordered((0, 0), (1, 0))
    assert net.f_ordered((0, 0), (0, 1))
    assert net.f_ordered((1, 0), (0, 1))
    assert all(net.f_ordered(s0, s1) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert all(net.f_ordered(*bond) for bond in net.bonds())

    ##########

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

    assert net.nn_bond_type(Bond((0, 0), (0, 1))) == ('h', True)  # 'lr'
    assert net.nn_bond_type(Bond((0, 1), (0, 0))) == ('h', False) # 'rl'
    assert net.nn_bond_type(Bond((2, 0), (0, 0))) == ('v', True)  # 'tb'
    assert net.nn_bond_type(Bond((0, 1), (2, 1))) == ('v', False) # 'bt'
    with pytest.raises(yastn.YastnError):
        net.nn_bond_type(Bond((3, 0), (2, 0)))
        # Bond((3,0),(2,0)) is not a nearest-neighboor bond.

    assert all(net.nn_bond_type(bond) == ('h', True) for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_type(bond) == ('v', True) for bond in net.bonds(dirn='v'))

    assert net.nn_bond_type(Bond((2, 0), (0, 0))) == ('v', True)
    assert net.nn_bond_type(Bond((0, 0), (2, 0))) == ('v', False)
    assert not net.f_ordered((2, 0), (0, 0))
    assert net.f_ordered((0, 0), (2, 0))

    assert all(net.f_ordered(s0, s1) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert not all(net.f_ordered(*bond) for bond in net.bonds())  #  PBC bonds in cylinder are not fermionically alligned

    ##########

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

    assert all(net.nn_bond_type(bond) == ('h', True) for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_type(bond) == ('v', True) for bond in net.bonds(dirn='v'))

    assert all(net.f_ordered(s0, s1) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert all(net.f_ordered(*bond) for bond in net.bonds())

    ##########

    with pytest.raises(yastn.YastnError):
        fpeps.SquareLattice(dims=(3, 2), boundary='some')
        #  boundary='some' not recognized; should be 'obc', 'infinite', or 'cylinder'


def test_RectangularUnitCell_1x1():
    g = fpeps.RectangularUnitcell(pattern=[[0,],])

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
    g = fpeps.RectangularUnitcell(pattern=[[0,1,2],[1,2,0],[2,0,1]])

    assert g.dims == (3, 3)
    assert g.sites() == (Site(0, 0), Site(0, 1), Site(0, 2))

    print(g)

    assert all(g.site2index(s) == 0 for s in [(0, 0), (0, 3), (0, -3), (1, -1), (-2, 2)])
    assert all(g.sites()[g.site2index(s)] == (0, 0) for s in [(0, 0), (0, 3), (0, -3), (1, -1), (-2, 2)])

    assert all(g.site2index(s) == 1 for s in [(0, 1), (0, 4), (0, -2), (1, 0), (-2, 3)])
    assert all(g.sites()[g.site2index(s)] == (0, 1) for s in [(0, 1), (0, 4), (0, -2), (1, 0), (-2, 3)])

    assert all(g.site2index(s) == 2 for s in [(0, 2), (0, 5), (0, -1), (1, 1), (-2, 4)])
    assert all(g.sites()[g.site2index(s)] == (0, 2) for s in [(0, 2), (0, 5), (0, -1), (1, 1), (-2, 4)])


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
    assert psi.nn_bond_type(Bond((0, 0), (0, 1))) == ('h', True)  # 'lr'
    assert psi.f_ordered((0, 0), (0, 1))


if __name__ == '__main__':
    test_CheckerboardLattice()
    test_SquareLattice()
    test_Peps_get_set()
    test_Peps_inheritance()
    test_RectangularUnitCell_1x1()
    test_RectangularUnitCell_2x2_bipartite()
    test_RectangularUnitCell_3x3_Q_1o3_1o3()
