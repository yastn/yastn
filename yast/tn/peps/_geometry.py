""" Basic structures forming PEPS network. """
from itertools import product
from typing import NamedTuple
from ...tn.mps import Mps, Mpo
from ._doublePepsTensor import DoublePepsTensor
from ... import tensor, initialize

class Bond(NamedTuple):
    """ site_0 should be before site_1 in the fermionic order. """
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
            'square' or 'checkerboard'  # ADD BRICKWALL
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
        self.Ny, self.Nx = (2, 2) if lattice == 'checkerboard' else dims
        self._sites = tuple(product(*(range(k) for k in self.dims)))
        self._dir = {'t': (-1, 0), 'l': (0, -1), 'b': (1, 0), 'r': (0, 1),
                     'tl': (-1, -1), 'bl': (1, -1), 'br': (1, 1), 'tr': (-1, 1)}
        
        bonds = []
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
        return site if self.lattice == 'rectangle' else sum(site) % 2

    @property
    def dims(self):
        """ Size of the unit cell """
        return (self.Ny, self.Nx)

    def sites(self, reverse=False):
        """ ADD """
        return self._sites[::-1] if reverse else self._sites

    def bonds(self, dirn=None, reverse=False):
        """ ADD """
        bnds = self._bonds[::-1] if reverse else self._bonds
        if dirn is None:
            return bnds
        return tuple(bnd for bnd in bnds if bnd.dirn == dirn)

    def nn_site(self, site, d):
        """ Index of the site to the top. Return None if there is no neighboring site to the top. """
        y, x = site
        dy, dx = self._dir[d]
        y, x = y + dy, x + dx
        if self.boundary == 'finite' and (x < 0 or x >= self.Nx or y < 0 or y >= self.Ny):
            return None
        return (y % self.Ny, x % self.Nx)

    def tensors_NtuEnv(self, bds):
        """ returns the cluster of sites around the bond to be updated """
        neighbors = {}
        site_1, site_2 = bds.site_0, bds.site_1
        if self.lattice == 'checkerboard':
            if bds.dirn == 'h':
                neighbors['tl'], neighbors['l'], neighbors['bl']  = site_2, site_2, site_2
                neighbors['tr'], neighbors['r'], neighbors['br'] = site_1, site_1, site_1
            elif bds.dirn == 'v':
                neighbors['tl'], neighbors['t'], neighbors['tr'] = site_2, site_2, site_2
                neighbors['bl'], neighbors['b'], neighbors['br'] = site_1, site_1, site_1
        else:
            if bds.dirn == 'h':
                neighbors['tl'], neighbors['l'], neighbors['bl']  = self.nn_site(site_1, d='t'), self.nn_site(site_1, d='l'), self.nn_site(site_1, d='b')
                neighbors['tr'], neighbors['r'], neighbors['br'] = self.nn_site(site_2, d='t'), self.nn_site(site_2, d='r'), self.nn_site(site_2, d='b')
            elif bds.dirn == 'v':
                neighbors['tl'], neighbors['t'], neighbors['tr'] = self.nn_site(site_1, d='l'), self.nn_site(site_1, d='t'), self.nn_site(site_1, d='r')
                neighbors['bl'], neighbors['b'], neighbors['br'] = self.nn_site(site_2, d='l'), self.nn_site(site_2, d='b'), self.nn_site(site_2, d='r')

        return neighbors


class Peps(Lattice):
    """ ADD """
    def __init__(self, lattice='checkerboard', dims=(2, 2), boundary='infinite'):
        super().__init__(lattice=lattice, dims=dims, boundary=boundary)
        inds = set(self.site2index(site) for site in self._sites)
        self._data = {ind: None for ind in inds}


    def __getitem__(self, site):
        assert site in self._sites, "Site is inconsistent with lattice"
        return self._data[self.site2index(site)]
    
    def __setitem__(self, site, tensor):
        assert site in self._sites, "Site is inconsistent with lattice"
        self._data[self.site2index(site)] = tensor


    def mpo(self, row_index, rotation=''):

        # converts specific row of PEPS into MPO

        H = Mpo(N=self.Nx)
        ny = row_index
        for nx in range(self.Nx):
            site = (nx, ny)
            top = self[site]
            if top.ndim == 3:
                top = top.unfuse_legs(axes=(0, 1))
            btm = top.swap_gate(axes=(0, 1, 2, 3))
            H.A[nx] = DoublePepsTensor(top=top, btm=btm)
        return H

    def boundary_mps(self, rotation=''):
        # create 
        psi = Mps(N=self.Nx)
        cfg = self._data[(0, 0)].config
        n0 = (0,) * cfg.sym.NSYM
        leg0 = tensor.Leg(cfg, s=-1, t=(n0,), D=(1,))
        for nx in range(self.Nx):
            site = (nx, self.Ny - 1)
            A = self[site]
            if A.ndim == 3:
                legA = A.get_legs(axis=1)
                _, legA = tensor.leg_undo_product(legA)
            else:
                legA = A.get_legs(axis=3)
            legAAb = tensor.leg_outer_product(legA, legA.conj())
            psi[nx] = initialize.ones(config=cfg, legs=[leg0, legAAb.conj(), leg0.conj()])
        return psi


 

