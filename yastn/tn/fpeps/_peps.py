

# g = SquareLattice(lattice='checkerboard', dims=(2, 2), boundary='infinite')

# peps1 = Peps(g)  # empty peps
# peps2 = peps.product_peps(vectors: yastn.Tensor | dict[tuple, Tensor], g)  # product state/operator



class Peps():
    def __init__(self, geometry):
        self.g = geometry
        self._data = {}

    def __getitem__(self, key):
        pass
    def __setitem__(self, key, val):
        pass

    # has data container


