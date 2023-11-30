

# g = SquareLattice(lattice='checkerboard', dims=(2, 2), boundary='infinite')

# peps1 = Peps(g)  # empty peps
# peps2 = peps.product_peps(vectors: yastn.Tensor | dict[tuple, Tensor], g)  # product state/operator


class Peps():
    r""" 
    Has data container and info about the geometry.
    """
    
    def __init__(self, geometry):
        self.g = geometry
        self.lattice = self.g.lattice
        self.Nx, self.Ny = self.g.Nx, self.g.Ny
        self.boundary = self.g.boundary

        self._data = {self.site2index(site): None for site in self.sites()}  # Initialize _data

    def site2index(self, site):
        """ Tensor index depending on site """
        return self.g.site2index(site)

    @property
    def dims(self):
        """ Size of the unit cell """
        return self.g.dims

    def sites(self, reverse=False):
        """ Labels of the lattice sites """
        return self.g.sites(reverse)

    def nn_bonds(self, dirn=None, reverse=False):
        """ Labels of the links between sites """
        return self.g.nn_bonds(dirn, reverse)

    def nn_site(self, site, d):
        """
        Index of the site in the direction d in ('t', 'b', 'l', 'r', 'tl', 'bl', 'tr', 'br').
        Return None if there is no neighboring site in given direction.
        """
        return self.g.nn_site(site, d)

    def __getitem__(self, site):
        """ Get data for site. """
        assert site in self.sites(), "Site is inconsistent with lattice"  # Call self.sites() as a method
        return self._data[self.site2index(site)]

    def __setitem__(self, site, obj):
        """ Set data at site. """
        assert site in self.sites(), "Site is inconsistent with lattice"  # Call self.sites() as a method
        self._data[self.site2index(site)] = obj

    def save_to_dict(self):
        d = {'lattice': self.g.lattice, 'dims': self.dims, 'boundary': self.g.boundary, 'data': {}}
        for ind, tensor in self._data.items():
            d['data'][ind] = tensor.save_to_dict()  # Assuming tensors have a save_to_dict method
        return d

    def copy(self):
        psi = Peps(geometry=self.g)  # Use self.g for geometry
        for ind in self._data:
            psi._data[ind] = self._data[ind].copy()  # Assuming tensors have a copy method
        return psi
    
    def tensors_NtuEnv(self, bd):
        return self.g.tensors_NtuEnv(bd)
