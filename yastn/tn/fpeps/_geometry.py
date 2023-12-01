""" Basic structures forming PEPS network. """
from itertools import product
from typing import NamedTuple


class Bond(NamedTuple):
    """ A bond between two lattice sites. site_0 should be before site_1 in the fermionic order. """
    site_0 : tuple = None
    site_1 : tuple = None
    dirn : str = ''

class Lattice():
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
        inds = set(self.site2index(site) for site in self._sites)
        self._data = {ind: None for ind in inds}  # container for site-dependent data

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

    def __getitem__(self, site):
        """ Get data for site. """
        assert site in self._sites, "Site is inconsistent with lattice"
        return self._data[self.site2index(site)]

    def __setitem__(self, site, obj):
        """ Set data at site. """
        assert site in self._sites, "Site is inconsistent with lattice"
        self._data[self.site2index(site)] = obj

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

    def tensors_NtuEnv(self, bds):
        r""" Returns a dictionary containing the neighboring sites of the bond `bds`.
                  The keys of the dictionary are the direction of the neighboring site with respect to
                  the bond: 'tl' (top left), 't' (top), 'tr' (top right), 'l' (left), 'r' (right),
                  'bl' (bottom left), and 'b' (bottom)"""

        neighbors = {}
        site_1, site_2 = bds.site_0, bds.site_1
        if self.lattice == 'checkerboard':
            if bds.dirn == 'h':
                neighbors['tl'], neighbors['l'], neighbors['bl'] = site_2, site_2, site_2
                neighbors['tr'], neighbors['r'], neighbors['br'] = site_1, site_1, site_1
            elif bds.dirn == 'v':
                neighbors['tl'], neighbors['t'], neighbors['tr'] = site_2, site_2, site_2
                neighbors['bl'], neighbors['b'], neighbors['br'] = site_1, site_1, site_1
        else:
            if bds.dirn == 'h':
                neighbors['tl'], neighbors['l'], neighbors['bl'] = self.nn_site(site_1, d='t'), self.nn_site(site_1, d='l'), self.nn_site(site_1, d='b')
                neighbors['tr'], neighbors['r'], neighbors['br'] = self.nn_site(site_2, d='t'), self.nn_site(site_2, d='r'), self.nn_site(site_2, d='b')
            elif bds.dirn == 'v':
                neighbors['tl'], neighbors['t'], neighbors['tr'] = self.nn_site(site_1, d='l'), self.nn_site(site_1, d='t'), self.nn_site(site_1, d='r')
                neighbors['bl'], neighbors['b'], neighbors['br'] = self.nn_site(site_2, d='l'), self.nn_site(site_2, d='b'), self.nn_site(site_2, d='r')

        return neighbors

    def save_to_dict(self):
        d = {'lattice': self.lattice, 'dims': self.dims, 'boundary': self.boundary, 'data': {}}
        for ind in self._data.keys():
            d['data'][ind] = self._data[ind].save_to_dict()
        return d

    def copy(self):
        psi = Lattice(lattice=self.lattice, dims=self.dims, boundary=self.boundary)
        for ind in psi._data.keys():
            psi._data[ind] = self._data[ind]
        return psi
