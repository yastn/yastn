""" Basic structures forming PEPS network. """
from itertools import product
from typing import NamedTuple
from ... import YastnError


class Site(NamedTuple):
    nx : int = 0
    ny : int = 0


class Bond(NamedTuple):  # Not very convinient to use
    """
    A bond between two lattice sites.

    site0 should preceed site1 in the fermionic order.
    """
    site0 : Site = None
    site1 : Site = None

    @property
    def dirn(self):
        """ Bond direction. 'h' when site0.nx == site1.nx; else 'v' (when site0.ny == site1.ny by construction). """
        return 'h' if self.site0[0] == self.site1[0] else 'v'


class SquareLattice():
    """ Geometric information about 2D lattice. """

    def __init__(self, dims=(2, 2), boundary='infinite'):
        r"""
        Geometric information about the lattice.

        Parameters
        ----------
        dims : tuple[int, int]
            Size of elementary cell.
        boundary : str
            'obc', 'infinite' or 'cylinder'

        Notes
        -----
        Site(0, 0) corresponds to top-left corner of the lattice
        """
        if boundary in ('obc', 'infinite', 'cylinder'):
            self.boundary = boundary
        else:
            raise YastnError("boundary should be 'obc', 'infinite' or 'cylinder'")
        self.Nx = dims[0]
        self.Ny = dims[1]
        self._sites = tuple(Site(nx, ny) for nx, ny in product(*(range(k) for k in self.dims)))
        self._dir = {'t': (-1, 0), 'l': (0, -1), 'b': (1, 0), 'r': (0, 1),
                     'tl': (-1, -1), 'bl': (1, -1), 'br': (1, 1), 'tr': (-1, 1)}

        bonds_h, bonds_v = [], []
        for s in self._sites:
            s_r = self.nn_site(s, d='r')
            if s_r is not None:
                bonds_h.append(Bond(s, s_r))
            s_b = self.nn_site(s, d='b')
            if s_b is not None:
                bonds_v.append(Bond(s, s_b))
        self._bonds_h = tuple(bonds_h)
        self._bonds_v = tuple(bonds_v)

    @property
    def dims(self):
        """ Size of the unit cell """
        return (self.Nx, self.Ny)

    def sites(self, reverse=False):
        """ Lattice sites """
        return self._sites[::-1] if reverse else self._sites

    def nn_bonds(self, dirn=None, reverse=False):
        """ Labels of unique nearest neighbor bonds between lattice sites. """
        if dirn == 'v':
            return self._bonds_v[::-1] if reverse else self._bonds_v
        if dirn == 'h':
            return self._bonds_h[::-1] if reverse else self._bonds_h
        return self._bonds_v[::-1] + self._bonds_h[::-1] if reverse else self._bonds_h + self._bonds_v

    def nn_site(self, site, d):
        """
        Index of the site in the direction d in ('t', 'b', 'l', 'r', 'tl', 'bl', 'tr', 'br').
        Return None if there is no neighboring site in a given direction.
        """
        x, y = site
        dx, dy = self._dir[d]
        x, y = x + dx, y + dy
        if self.boundary == 'obc' and (x < 0 or x >= self.Nx or y < 0 or y >= self.Ny):
            return None
        if self.boundary == 'cylinder' and (y < 0 or y >= self.Ny):
            return None
        return Site(x % self.Nx, y % self.Ny)

    def site2index(self, site):
        """ Tensor index depending on site """
        if site not in self._sites:
            raise YastnError(f"Site {site} does not correspond to any lattice site.")
        return site


class CheckerboardLattice(SquareLattice):
    """ Geometric information about 2D lattice. """

    def __init__(self):
        r"""
        Checkerboard lattice is an infinite lattiec with 2x2 unit cell and 2 unique tensors.
        """
        super().__init__(dims=(2, 2), boundary='infinite')
        self._bonds_h = (Bond(Site(0, 0), Site(0, 1)), Bond(Site(0, 1), Site(0, 0)))
        self._bonds_v = (Bond(Site(0, 0), Site(1, 0)), Bond(Site(1, 0), Site(0, 0)))

    def site2index(self, site):
        """ Tensor index depending on site """
        if site not in self._sites:
            raise YastnError(f"Site {site} does not correspond to any lattice site.")
        return sum(site) % 2
