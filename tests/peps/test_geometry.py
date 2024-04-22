""" Test operation of peps.Lattice and peps.Peps that inherits Lattice"""
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site, Bond

def test_CheckerboardLattice():
    """ Generate a few lattices veryfing expected output of some functions. """

    net = fpeps.CheckerboardLattice()

    assert net.dims == (2, 2)
    assert net.sites() == (Site(0, 0), Site(0, 1))
    assert net.sites(reverse=True) == (Site(0, 1), Site(0, 0))

    assert net.bonds(dirn='h') == (Bond((0, 0), (0, 1)), Bond((0, 1), (0, 2)))
    assert net.bonds(dirn='v') == (Bond((0, 0), (1, 0)), Bond((1, 0), (2, 0)))
    assert net.bonds(reverse=True)[::1] == net.bonds()[::-1]

    assert all(net.site2index(site) == 0 for site in [(0, 0), (1, 1), (-3, 3), (1, -1), (2, 0)])
    assert all(net.site2index(site) == 1 for site in [(1, 0), (0, 1), (2, 5), (1, 2), (-1, 0)])

    assert net.nn_site(Site(1, 1), d='r') == (1, 2)
    assert net.nn_site(Site(1, 0), d='l') == (1, -1)
    assert net.nn_site(Site(0, 0), d='t') == (-1, 0)
    assert net.nn_site(Site(1, 0), d='b') == (2, 0)

    assert net.nn_bond_type(Bond((0, 0), (0, 1))) == ('h', True)  # 'lr'
    assert net.nn_bond_type(Bond((0, 3), (0, 2))) == ('h', False)  # 'rl'
    assert net.nn_bond_type(Bond((1, 3), (0, 3))) == ('v', False)  # 'bt'
    assert net.nn_bond_type(Bond((1, 0), (2, 0))) == ('v', True)  # 'tb'
    with pytest.raises(yastn.YastnError):
        net.nn_bond_type(Bond((1, 0), (3, 0)))
        # Bond((1,0),(3,0)) is not a nearest-neighboor bond.

    assert all(net.nn_bond_type(bond) == ('h', True) for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_type(bond) == ('v', True) for bond in net.bonds(dirn='v'))

    assert net.f_ordered(Bond((2, 0), (4, 0)))
    assert net.f_ordered(Bond((4, 0), (0, 4)))
    assert all(net.f_ordered((s0, s1)) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert all(net.f_ordered(bond) for bond in net.bonds())


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

    assert net.nn_site((0, 1), d=(2, -1)) == Site(2, 0)
    assert net.site2index((1, 0)) == (1, 0)
    assert net.site2index(None) == None

    assert net.nn_bond_type(Bond((0, 0), (0, 1))) == ('h', True)  # 'lr'
    assert net.nn_bond_type(Bond((0, 1), (0, 0))) == ('h', False)  # 'rl'
    assert net.nn_bond_type(Bond((2, 1), (1, 1))) == ('v', False)  # 'bt'
    assert net.nn_bond_type(Bond((1, 0), (2, 0))) == ('v', True)  # 'tb'
    with pytest.raises(yastn.YastnError):
        net.nn_bond_type(Bond((0, 0), (2, 0)))
        # Bond((0,0),(2,0)) is not a nearest-neighboor bond.
    assert all(net.nn_bond_type(bond) == ('h', True) for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_type(bond) == ('v', True) for bond in net.bonds(dirn='v'))

    assert net.f_ordered(Bond((0, 0), (1, 0)))
    assert net.f_ordered(Bond((0, 0), (0, 1)))
    assert net.f_ordered(Bond((1, 0), (0, 1)))
    assert all(net.f_ordered((s0, s1)) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert all(net.f_ordered(bond) for bond in net.bonds())

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
    assert net.nn_bond_type(Bond((0, 1), (0, 0))) == ('h', False)  # 'rl'
    assert net.nn_bond_type(Bond((2, 0), (0, 0))) == ('v', True)  # 'tb'
    assert net.nn_bond_type(Bond((0, 1), (2, 1))) == ('v', False)  # 'bt'
    with pytest.raises(yastn.YastnError):
        net.nn_bond_type(Bond((3, 0), (2, 0)))
        # Bond((3,0),(2,0)) is not a nearest-neighboor bond.
    assert all(net.nn_bond_type(bond) == ('h', True) for bond in net.bonds(dirn='h'))
    assert all(net.nn_bond_type(bond) == ('v', True) for bond in net.bonds(dirn='v'))

    assert not net.f_ordered(Bond((2, 0), (0, 0)))
    assert all(net.f_ordered((s0, s1)) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert not all(net.f_ordered(bond) for bond in net.bonds())  #  PBC bonds in cylinder are not fermionically alligned

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

    assert all(net.f_ordered((s0, s1)) for s0, s1 in zip(net.sites(), net.sites()[1:]))
    assert all(net.f_ordered(bond) for bond in net.bonds())


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
    assert psi.f_ordered(Bond((0, 0), (0, 1)))


if __name__ == '__main__':
    test_CheckerboardLattice()
    test_SquareLattice()
    test_Peps_get_set()
    test_Peps_inheritance()
