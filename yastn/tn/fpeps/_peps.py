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
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary"]:
            setattr(self, name, getattr(geometry, name))
        self._data = {self.site2index(site): None for site in self.sites()}

    @property
    def config(self):
        return self[0, 0].config

    def __getitem__(self, site):
        """ Get tensor for site. """
        return self._data[self.site2index(site)]

    def __setitem__(self, site, obj):
        """ Set tensor at site. """
        self._data[self.site2index(site)] = obj

    def save_to_dict(self):
        """
        Serialize PEPS into a dictionary.
        """
        if isinstance(self.geometry, SquareLattice):
            lattice = "square"
        elif isinstance(self.geometry, CheckerboardLattice):
            lattice = "checkerboard"

        d = {'lattice': lattice,
             'dims': self.dims,
             'boundary': self.boundary,
             'data': {}}
        for ind, tensor in self._data.items():
            d['data'][tuple(ind)] = tensor.save_to_dict()
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

    def transfer_mpo(self, n=0, dirn='v'):
        """
        Converts a specific row or column of PEPS into MPO.

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
                if top.ndim in (2, 3):
                    top = top.unfuse_legs(axes=(0, 1))
                op.A[ny] = top.transpose(axes=(1, 2, 3, 0)) if top.ndim == 4 else \
                           DoublePepsTensor(top=top, btm=top, transpose=(1, 2, 3, 0))
        elif dirn == 'v':
            periodic = (self.boundary == "cylinder")
            op = Mpo(N=self.Nx, periodic=periodic)
            op.tol = self._data['tol'] if 'tol' in self._data else None
            for nx in range(self.Nx):
                site = (nx, n)
                top = self[site]
                if top.ndim in (2, 3):
                    top = top.unfuse_legs(axes=(0, 1))
                op.A[nx] = top if top.ndim == 4 else \
                           DoublePepsTensor(top=top, btm=top)
        return op
