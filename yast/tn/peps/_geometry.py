""" Basic structures forming PEPS network. """
from itertools import product
from typing import NamedTuple
from ...tn.mps import Mps, Mpo
from ._doublePepsTensor import DoublePepsTensor
from ... import tensor, initialize

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
            'finite' or 'infinite'

        Notes
        -----
            Site (0, 0) corresponds to top-left corner of the lattice
        """
        assert lattice in ('checkerboard', 'rectangle'), "lattice should be 'checkerboard' or 'rectangle' or ''"
        assert boundary in ('finite', 'infinite'), "boundary should be 'finite' or 'infinite'"
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
        return site if self.lattice == 'rectangle' else sum(site) % 2

    @property
    def dims(self):
        """ Size of the unit cell """
        return (self.Nx, self.Ny)

    def sites(self, reverse=False):
        """ Labels of the lattice sites """
        return self._sites[::-1] if reverse else self._sites

    def bonds(self, dirn=None, reverse=False):
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
        if self.boundary == 'finite' and (x < 0 or x >= self.Nx or y < 0 or y >= self.Ny):
            return None
        return (x % self.Nx, y % self.Ny)

    def tensors_NtuEnv(self, bds):
        r""" Returns the cluster of sites around the bond to be updated by NTU optimization 

        Parameters
        ----------
        bds: NamedTuple Bond

        Returns
        -------
        neighbors : dict
                  A dictionary containing the neighboring sites of the bond `bds`.
                  The keys of the dictionary are the direction of the neighboring site with respect to
                  the bond: 'tl' (top left), 't' (top), 'tr' (top right), 'l' (left), 'r' (right),
                  'bl' (bottom left), and 'b' (bottom).
        """
        
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


class Peps(Lattice):
    r""" 
    Inherits Lattice Class and manages Peps data with additional functionalities.

    Parameters:
    -----------
    lattice : str, optional
        Name of the lattice ('checkerboard' by default).
    dims : tuple of int, optional
        Dimensions of each PEPS tensor (2,2) by default.
    boundary : str, optional
        Type of boundary ('infinite' by default).

    Methods:
    --------
    mpo(index, index_type, rotation='')
        Converts a specific row or column of PEPS into a matrix product operator (MPO).
    boundary_mps(rotation='')
        Initiates a boundary matrix product state (MPS) at the rightmost column.

    Inherits the methods from the Lattice class.

    """
    
    def __init__(self, lattice='checkerboard', dims=(2, 2), boundary='infinite'):
        super().__init__(lattice=lattice, dims=dims, boundary=boundary)


    def mpo(self, index, index_type, rotation=''):

        """Converts a specific row or column of PEPS into MPO.

        Parameters
        ----------
            index (int): The row or column index to convert.
            index_type (str): The index type to convert, either 'row' or 'column'.
            rotation (str): Optional string indicating the rotation of the PEPS tensor.

        Returns
        -------
            H (Mpo): The resulting MPO.
        """

        if index_type == 'row':
            nx = index
            H = Mpo(N=self.Ny)
            for ny in range(self.Ny):
                site = (nx, ny)
                top = self[site]
                if top.ndim == 3:
                    top = top.unfuse_legs(axes=(0, 1))
                btm = top.swap_gate(axes=(0, 1, 2, 3))
                H.A[ny] = DoublePepsTensor(top=top, btm=btm)
        elif index_type == 'column':
            ny = index
            H = Mpo(N=self.Nx)
            for nx in range(self.Nx):
                site = (nx, ny)
                top = self[site]
                if top.ndim == 3:
                    top = top.unfuse_legs(axes=(0, 1))
                btm = top.swap_gate(axes=(0, 1, 2, 3))
                H.A[nx] = DoublePepsTensor(top=top, btm=btm)

        return H

    def boundary_mps(self, rotation=''):

        r"""Initiates a boundary MPS at the right most column.

        Parameters
        ----------
            rotation (str): Optional string indicating the rotation of the PEPS tensor.

        Returns
        -------
            psi (Mps): The resulting boundary MPS.
        """
        psi = Mps(N=self.Nx)
        cfg = self._data[(0, 0)].config
        n0 = (0,) * cfg.sym.NSYM
        leg0 = tensor.Leg(cfg, s=-1, t=(n0,), D=(1,))
        for nx in range(self.Nx):
            site = (nx, self.Ny-1)
            A = self[site]
            if A.ndim == 3:
                legA = A.get_legs(axis=1)
                _, legA = tensor.leg_undo_product(legA)
            else:
                legA = A.get_legs(axis=3)
            legAAb = tensor.leg_outer_product(legA, legA.conj())
            psi[nx] = initialize.ones(config=cfg, legs=[leg0, legAAb.conj(), leg0.conj()])

        return psi 


 

