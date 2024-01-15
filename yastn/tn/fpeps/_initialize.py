import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
from ...initialize import ones as ones
from ...initialize import load_from_dict as load_tensor_from_dict
from ._geometry import SquareLattice

r""" Initialization of peps tensors for real or imaginary time evolution """

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

    # Lattice Geometry with Repeated Tensor Initialization
    if not isinstance(geometry, (SquareLattice)):
        raise TypeError("Expected geometry to be an instance of SquareLattice")

    if isinstance(vectors, yastn.Tensor):
        # Lattice Geometry with Repeated Tensor Initialization
        A = vectors.fuse_legs(axes=[(0, 1)])
        for s in (-1, 1, 1, -1):
            A = A.add_leg(axis=0, s=s)
        A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
        gamma = fpeps.Peps(geometry)
        for ms in gamma.sites():
            gamma[ms] = A
    elif isinstance(vectors, dict):
        # Occupation Pattern Initialization
        gamma = fpeps.Peps(geometry)
        for kk in gamma.sites():
            Ga = vectors.get(kk)
            if Ga is None:
                continue  # Skip sites not defined in the vectors dictionary
            for s in (-1, 1, 1, -1):
                Ga = Ga.add_leg(axis=0, s=s)
            gamma[kk] = Ga.fuse_legs(axes=((0, 1), (2, 3), 4))
    else:
        raise TypeError("Invalid type for vectors. Expected yastn.Tensor or Dict.")

    return gamma




def load_from_dict(config, d):
    psi = SquareLattice(lattice=d['lattice'], dims=d['dims'], boundary=d['boundary'])
    psi = fpeps.Peps(psi)
    for ind, dtensor in d['data'].items():
        psi._data[ind] = load_tensor_from_dict(config, dtensor)
    return psi
