from ._geometry import SquareLattice, CheckerboardLattice
from ... import YastnError

class Peps():
    r"""
    Has data container and info about the geometry.
    """

    def __init__(self, geometry):
        self.geometry = geometry
        for name in ["dims", "sites", "nn_site", "nn_bonds", "site2index", "Nx", "Ny", "boundary"]:
            setattr(self, name, getattr(geometry, name))
        self._data = {self.site2index(site): None for site in self.sites()}

    def __getitem__(self, site):
        """ Get data for site. """
        if site not in self.sites():
            raise YastnError(f"Site {site} is not a lattice site.")
        return self._data[self.site2index(site)]

    def __setitem__(self, site, obj):
        """ Set data at site. """
        if site not in self.sites():
            raise YastnError(f"Site {site} is not a lattice site.")
        self._data[self.site2index(site)] = obj

    def save_to_dict(self):
        if isinstance(self.geometry, SquareLattice):
            lattice = "square"
        else:  # isinstance(self.g, CheckerboardLattice):
            lattice = "checkerboard"

        d = {'lattice': lattice,
             'dims': self.dims,
             'boundary': self.boundary,
             'data': {}}
        for ind, tensor in self._data.items():
            d['data'][ind] = tensor.save_to_dict()
        return d

    def copy(self):
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind].copy()
        return psi
