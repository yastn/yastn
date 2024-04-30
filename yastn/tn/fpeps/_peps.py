from __future__ import annotations
from ...tn.mps import Mpo
from ._doublePepsTensor import DoublePepsTensor
from ._geometry import SquareLattice, CheckerboardLattice


class Peps():

    def __init__(self, geometry):
        """
        Initialize empty PEPS on a lattice specified by provided geometry.

        Empty PEPS has no tensors assigned.
        """
        self.geometry = geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "nn_bond_type", "f_ordered"]:
            setattr(self, name, getattr(geometry, name))
        self._data = {self.site2index(site): None for site in self.sites()}

    @property
    def config(self):
        return self[0, 0].config

    def has_physical(self):
        return self[0, 0].ndim in (3, 5)

    def __getitem__(self, site):
        """ Get tensor for site. """
        return self._data[self.site2index(site)]

    def __setitem__(self, site, obj):
        """
        Set tensor at site.
        5-leg tensors are fused to 3-leg: [t l] [b r] sa
        2-leg tensors are unfused to 4-leg: t l b r
        """
        if hasattr(obj, 'ndim') and obj.ndim == 5 :
            obj = obj.fuse_legs(axes=((0, 1), (2, 3), 4))
        if hasattr(obj, 'ndim') and obj.ndim == 2 :
            obj = obj.unfuse_legs(axes=(0, 1))
        self._data[self.site2index(site)] = obj

    def save_to_dict(self):
        """
        Serialize PEPS into a dictionary.
        """
        if isinstance(self.geometry, CheckerboardLattice):
            lattice = "checkerboard"
        elif isinstance(self.geometry, SquareLattice):
            lattice = "square"

        d = {'lattice': lattice,
             'dims': self.dims,
             'boundary': self.boundary,
             'data': {}}
        for ind, tensor in self._data.items():
            d['data'][ind] = tensor.save_to_dict()
        return d

    def copy(self):
        r"""
        Makes a copy of PEPS by :meth:`copying<yastn.Tensor.copy>` all :class:`yastn.Tensor<yastn.Tensor>`'s
        into a new and independent :class:`yastn.tn.fpeps.Peps`.
        """
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind].copy()
        return psi

    def clone(self):
        r"""
        Makes a clone of PEPS by :meth:`copying<yastn.Tensor.clone>` all :class:`yastn.Tensor<yastn.Tensor>`'s
        into a new and independent :class:`yastn.tn.fpeps.Peps`.
        """
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind].copy()
        return psi

    def transfer_mpo(self, n=0, dirn='v') -> yastn.tn.mps.MpsMpo:
        """
        Converts a specific row or column of PEPS into MPO.

        For tensor with physical leg, Mpo consists of DoublePepsTensor (2-layers)
        For tensor without physical leg, directly uses Peps tensors (1-layer)


        Parameters
        ----------
        n: int
            index of row/column.
        dirn: str
            'v' for column, 'h' for row.
        """

        if dirn == 'h':
            op = Mpo(N=self.Ny)
            for ny in range(self.Ny):
                site = (n, ny)
                top = self[site]
                op.A[ny] = top.transpose(axes=(1, 2, 3, 0)) if top.ndim == 4 else \
                           DoublePepsTensor(top=top, btm=top, transpose=(1, 2, 3, 0))
        elif dirn == 'v':
            periodic = (self.boundary == "cylinder")
            op = Mpo(N=self.Nx, periodic=periodic)
            op.tol = self._data['tol'] if 'tol' in self._data else None
            for nx in range(self.Nx):
                site = (nx, n)
                top = self[site]
                op.A[nx] = top if top.ndim == 4 else \
                           DoublePepsTensor(top=top, btm=top)
        return op


class Peps2Layers():

    def __init__(self, bra, ket=None):
        """
        Initialize empty PEPS on a lattice specified by provided geometry.

        Empty PEPS has no tensors assigned.
        """
        self.bra = bra
        self.ket = bra if ket is None else ket
        self.geometry = bra.geometry

        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "nn_bond_type", "f_ordered"]:
            setattr(self, name, getattr(bra.geometry, name))

    @property
    def config(self):
        return self.ket.config

    def has_physical(self) -> bool:
        return False

    def __getitem__(self, site) -> yastn.tn.fpeps.DoublePepsTensor:
        """ Get tensor for site. """
        return DoublePepsTensor(top=self.ket[site], btm=self.bra[site])
