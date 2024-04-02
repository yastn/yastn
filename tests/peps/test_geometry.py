""" Test operation of peps.Lattice and peps.Peps that inherits Lattice"""
import pytest
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site

def test_Lattice():
    """ Generate a few lattices veryfing expected output of some functions. """
    #
    ##########
    net = fpeps.CheckerboardLattice()

    assert net.dims == (2, 2)
    assert net.sites() == (Site(0, 0), Site(0, 1))
    assert net.sites(reverse=True) == (Site(0, 1), Site(0, 0))

    assert net.bonds(dirn='h') == (((0, 0), (0, 1)), ((0, 1), (0, 0)))
    assert net.bonds(dirn='v') == (((0, 0), (1, 0)), ((1, 0), (0, 0)))
    assert net.bonds() == (((0, 0), (0, 1)),
                              ((0, 1), (0, 0)),
                              ((0, 0), (1, 0)),
                              ((1, 0), (0, 0)))
    assert net.bonds(reverse=True)[::1] == net.bonds()[::-1]

    assert net.site2index((0, 0)) == net.site2index((1, 1)) == net.site2index((-3, 3)) == 0
    assert net.site2index((1, 0)) == net.site2index((0, 1)) == net.site2index((2, 5)) == 1
    assert net.nn_site(Site(0, 1), d='r') == (0, 0)
    #
    ##########
    net = fpeps.SquareLattice(dims=(3, 2), boundary='obc')

    assert net.dims == (3, 2)
    assert net.sites() == ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
    assert net.bonds(dirn='h') == (((0, 0), (0, 1)),
                                      ((1, 0), (1, 1)),
                                      ((2, 0), (2, 1)))
    assert net.bonds(dirn='v') == (((0, 0), (1, 0)),
                                      ((0, 1), (1, 1)),
                                      ((1, 0), (2, 0)),
                                      ((1, 1), (2, 1)))

    assert net.nn_site((0, 1), d='r') is None
    assert net.nn_site((0, 1), d=(2, -1)) == Site(2, 0)
    assert net.site2index((1, 0)) == (1, 0)
    #
    ##########
    net = fpeps.SquareLattice(dims=(3, 2), boundary='cylinder')

    assert net.dims == (3, 2)
    assert net.sites() == ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
    assert net.bonds('h') == (((0, 0), (0, 1)),
                              ((1, 0), (1, 1)),
                              ((2, 0), (2, 1)))
    assert net.bonds('v') == (((0, 0), (1, 0)),
                              ((0, 1), (1, 1)),
                              ((1, 0), (2, 0)),
                              ((1, 1), (2, 1)),
                              ((2, 0), (0, 0)),
                              ((2, 1), (0, 1)))

    assert net.nn_site((0, 1), d='r') is None
    assert net.nn_site((0, 1), d='t') == Site(2, 1)
    assert net.nn_site((2, 0), d='b') == Site(0, 0)
    assert net.nn_site((0, 0), d=(4, 4)) is None
    assert net.nn_site((2, 0), d=(2, 1)) == Site(1, 1)
    assert net.nn_site((2, 0), d=(-3, 1)) == Site(2, 1)


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
    #
    ##########
    net = fpeps.SquareLattice(dims=(3, 3), boundary='obc')
    psi = fpeps.Peps(net)

    assert all(psi[site] == None for site in psi.sites())  # all tensors initialized as None
    psi[(0, 0)] = "Wannabe tensor"
    assert psi[(0, 0)] == "Wannabe tensor"
    assert psi[(1, 1)] is None


if __name__ == '__main__':
    test_Lattice()
    test_Peps_get_set()
