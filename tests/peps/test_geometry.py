""" Test operation of peps.Lattice and peps.Peps that inherits Lattice"""
import pytest
import yast.tn.peps as peps

def test_Lattice():
    """ Generate a few lattices veryfing expected output of some functions. """
    net = peps.Lattice(lattice='checkerboard', boundary='infinite')
    assert net.dims == (2, 2)
    assert net.sites() == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert net.sites(reverse=True) == ((1, 1), (1, 0), (0, 1), (0, 0))
    
    bonds_hor = tuple((bnd.site_0, bnd.site_1) for bnd in net.bonds(dirn='h'))
    bonds_ver = tuple((bnd.site_0, bnd.site_1) for bnd in net.bonds(dirn='v'))
    assert bonds_hor == (((0, 0), (0, 1)), ((0, 1), (0, 0)), ((1, 0), (1, 1)), ((1, 1), (1, 0)))
    assert bonds_ver == (((0, 0), (1, 0)), ((0, 1), (1, 1)), ((1, 0), (0, 0)), ((1, 1), (0, 1)))

    assert net.site2index((0, 0)) == 0 == net.site2index((1, 1))
    assert net.site2index((1, 0)) == 1 == net.site2index((0, 1))
    assert net.nn_site((0, 1), d='r') == (0, 0)

    net = peps.Lattice(lattice='rectangle', dims=(3, 2), boundary='finite')
    assert net.dims == (3, 2)
    assert net.sites() == ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
    bonds_hor = tuple((bnd.site_0, bnd.site_1) for bnd in net.bonds(dirn='h'))
    bonds_ver = tuple((bnd.site_0, bnd.site_1) for bnd in net.bonds(dirn='v'))
    assert bonds_hor == (((0, 0), (0, 1)), ((1, 0), (1, 1)), ((2, 0), (2, 1)))
    assert bonds_ver == (((0, 0), (1, 0)), ((0, 1), (1, 1)), ((1, 0), (2, 0)), ((1, 1), (2, 1)))
    assert net.nn_site((0, 1), d='r') is None
    assert net.site2index((1, 0)) == (1, 0)


def test_NtuEnv():
    """ nearest environmental sites around a bond;  creates indices of the NTU environment """

    bd00_01_h = peps.Bond(site_0 = (0, 0), site_1=(0, 1), dirn='h')
    bd11_21_v = peps.Bond(site_0 = (1, 1), site_1=(2, 1), dirn='v')

    net_1 = peps.Lattice(lattice='rectangle', dims=(3, 4), boundary='finite')  # dims = (rows, columns) # finite lattice
    assert net_1.tensors_NtuEnv(bd00_01_h) == {'tl': None, 'l': None, 'bl': (1, 0), 'tr': None, 'r': (0, 2), 'br': (1, 1)}
    assert net_1.tensors_NtuEnv(bd11_21_v) == {'tl': (1, 0), 't': (0, 1), 'tr': (1, 2), 'bl': (2, 0), 'b': None, 'br': (2, 2)}
    
    net_2 = peps.Lattice(lattice='rectangle', dims=(3, 4), boundary='infinite')  # dims = (rows, columns) # infinite lattice
    assert net_2.tensors_NtuEnv(bd00_01_h) == {'tl':(2, 0), 'l': (0, 3), 'bl': (1, 0), 'tr': (2, 1), 'r': (0, 2), 'br': (1, 1)}
    assert net_2.tensors_NtuEnv(bd11_21_v) == {'tl': (1, 0), 't': (0, 1), 'tr': (1, 2), 'bl': (2, 0), 'b': (0, 1), 'br': (2, 2)}

def test_Peps_get_set():
    """ Setitem and getitem in peps allows to acces individual tensors. """
    net = peps.Peps(lattice='checkerboard', dims=(2, 2), boundary='infinite')

    assert all(net[site] == None for site in net.sites())  # all tensors initialized as None
    net[(0, 0)] = "Wannabe tensor"
    # on a checkerboard lattice we have the same tensor at (0, 0) and (1, 1
    assert net[(0, 0)] == "Wannabe tensor" == net[(1, 1)]
    assert net[(0, 1)] == None == net[(1, 0)]

    net = peps.Peps(lattice='rectangle', dims=(3, 3), boundary='finite')
    assert all(net[site] == None for site in net.sites())  # all tensors initialized as None
    net[(0, 0)] = "Wannabe tensor"
    assert net[(0, 0)] == "Wannabe tensor" 
    assert net[(1, 1)] is None



if __name__ == '__main__':
    test_Lattice()
    test_Peps_get_set()
    test_NtuEnv()

