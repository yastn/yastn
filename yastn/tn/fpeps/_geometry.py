""" Basic structures forming PEPS network. """
from itertools import product
from typing import NamedTuple


class Bond(NamedTuple):  # Not very convinient to use
    """ A bond between two lattice sites. site_0 should be before site_1 in the fermionic order. """
    site_0 : tuple = None
    site_1 : tuple = None
    dirn : str = ''


class SquareLattice():
    """ Geometric information about 2D lattice. """

    def __init__(self, lattice='checkerboard', dims=(2, 2), boundary='infinite'):
        r"""
        Geometric information about the lattice.

        Parameters
        ----------
            lattice : str
                'square' or 'checkerboard'
            dims : tuple(int)
                Size of elementary cell.
                For 'checkerboard' it is always (2, 2)
            boundary : str
                'obc', 'infinite' or 'cylinder'

        Notes
        -----
            Site (0, 0) corresponds to top-left corner of the lattice
        """
        assert lattice in ('checkerboard', 'square'), "lattice should be 'checkerboard' or 'square' or ''"
        assert boundary in ('obc', 'infinite', 'cylinder'), "boundary should be 'obc', 'infinite' or 'cylinder'"
        self.lattice = lattice
        self.boundary = boundary
        self.Nx, self.Ny = (2, 2) if lattice == 'checkerboard' else dims
        self._sites = tuple(product(*(range(k) for k in self.dims)))
        self._dir = {'t': (-1, 0), 'l': (0, -1), 'b': (1, 0), 'r': (0, 1),
                     'tl': (-1, -1), 'bl': (1, -1), 'br': (1, 1), 'tr': (-1, 1)}

        bonds = []
        if self.lattice == 'checkerboard':
            self._bonds = (Bond(site_0=(0, 0), site_1=(0, 1), dirn='h'), Bond(site_0=(0, 1), site_1=(0, 0), dirn='h'), Bond(site_0=(0, 0), site_1=(0, 1), dirn='v'), Bond(site_0=(0, 1), site_1=(0, 0), dirn='v'))
        else:
            for s in self._sites:
                s_b = self.nn_site(s, d='b')
                if s_b is not None:
                    bonds.append(Bond(s, s_b, 'v'))
                s_r = self.nn_site(s, d='r')
                if s_r is not None:
                    bonds.append(Bond(s, s_r, 'h'))
            self._bonds = tuple(bonds)


    def site2index(self, site):
        """ Tensor index depending on site """
        assert site in self._sites, "wrong index of site"
        return site if self.lattice == 'square' else sum(site) % 2

    @property
    def dims(self):
        """ Size of the unit cell """
        return (self.Nx, self.Ny)

    def sites(self, reverse=False):
        """ Labels of the lattice sites """
        return self._sites[::-1] if reverse else self._sites
  

    def nn_bonds(self, dirn=None, reverse=False):
        """ Labels of the links between sites """

        bnds = self._bonds[::-1] if reverse else self._bonds
        if dirn is None:
            return bnds
        return tuple(bnd for bnd in bnds if bnd.dirn == dirn)

    def nn_site(self, site, d):
        """
        Index of the site in the direction d in ('t', 'b', 'l', 'r', 'tl', 'bl', 'tr', 'br').
        Return None if there is no neighboring site in given direction.
        """
        x, y = site
        dx, dy = self._dir[d]
        x, y = x + dx, y + dy
        if self.boundary == 'obc' and (x < 0 or x >= self.Nx or y < 0 or y >= self.Ny):
            return None
        if self.boundary == 'cylinder' and (y < 0 or y >= self.Ny):
            return None
        return (x % self.Nx, y % self.Ny)

    