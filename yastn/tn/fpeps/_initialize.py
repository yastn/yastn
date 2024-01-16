""" Initialization of peps tensors for real or imaginary time evolution """
from ._geometry import SquareLattice, CheckerboardLattice
from ._peps import Peps
from ...initialize import load_from_dict as load_tensor_from_dict
from ... import YastnError, Tensor

def product_peps(geometry, vectors): #-> fpeps.Peps:  #   (geometry, vectors : yastn.Tensor | Dict[tuple[Int, Int], yastn.Tensor])
    """
    Initializes and returns Projected Entangled Pair States (PEPS) tensors based on provided parameters.

    This function serves as a versatile initializer for PEPS tensors, handling different initialization schemes.
    It can initialize tensors either by repeating a given tensor across a lattice geometry or based on a specific occupation pattern.

    Parameters
    ----------
    geometry : SquareLattice or TriangularLattice
        An instance of a lattice class representing the lattice geometry.
    vectors : yastn.Tensor or Dict[tuple[Int, Int], yastn.Tensor]
        - If a yastn.Tensor is provided, it is repeated across the lattice.
        - If a Dict is provided, it specifies the occupation pattern on the lattice with tensors at each site.

    Returns
    -------
    fpeps.Peps
        The initialized PEPS.

    """
    if not isinstance(geometry, (SquareLattice, CheckerboardLattice)):
        raise YastnError("Geometry should be an instance of SquareLattice or CheckerboardLattice")

    if isinstance(vectors, Tensor):
        vectors = {site: vectors.copy() for site in geometry.sites()}

    psi = Peps(geometry)
    for site, vec in vectors.items():
        for s in (-1, 1, 1, -1):
            vec = vec.add_leg(axis=0, s=s)
        psi[site] = vec.fuse_legs(axes=((0, 1), (2, 3), 4))
    if any(psi[site] is None for site in psi.sites()):
        raise YastnError("product_peps did not initialize some peps tensor")
    return psi


def load_from_dict(config, d):
    if d['lattice'] == "square":
        net = SquareLattice(dims=d['dims'], boundary=d['boundary'])
    elif d['lattice'] == "checkerboard":
        net = CheckerboardLattice()

    psi = Peps(net)
    for ind, tensor in d['data'].items():
        psi._data[ind] = load_tensor_from_dict(config, tensor)
    return psi
