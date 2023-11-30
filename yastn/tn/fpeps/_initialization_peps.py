import numpy as np
import yastn.tn.fpeps as fpeps
from ...initialize import ones as ones
from ...initialize import load_from_dict as load_tensor_from_dict
from ._geometry import SquareLattice

r""" Initialization of peps tensors for real or imaginary time evolution """

# def reduce_operators(ops):
#         ds = ops.remove_zero_blocks()
#         lg = ds.get_legs(1).conj()
#         vone = ones(config=ds.config,legs=[lg], n=lg.t[0])
#         W = ds@vone
#         W = W.add_leg(s=-1)
#         return W


def product_peps(*args, **kwargs):  #   (geometry, vectors : yastn.Tensor | Dict[tuple[Int, Int], yastn.Tensor])
    """
    Initializes and returns Projected Entangled Pair States (PEPS) tensors based on provided parameters.

    This function serves as a versatile initializer for PEPS tensors, handling different initialization schemes.
    It can initialize tensors either by repeating a given tensor across a lattice geometry or based on a specific occupation pattern.
    Parameters
    ----------
    *args : variable
        - If two arguments are provided:
            - args[0] (SquareLattice): An instance of the SquareLattice class representing the lattice geometry.
            - args[1] (Tensor): A tensor to be repeated across the lattice.
        - If three arguments are provided:
            - args[0] (SquareLattice or TriangularLattice): Lattice geometry.
            - args[1] (List[Vec]): A list of vectors representing initial states.
            - args[2] (Dict): A dictionary specifying the occupation pattern on the lattice.
    """

    # Lattice Geometry with Repeated Tensor Initialization
    if len(args) == 2 and isinstance(args[0], SquareLattice):
        geometry, A = args

        A = A.fuse_legs(axes=[(0, 1)])
        for s in (-1, 1, 1, -1):
            A = A.add_leg(axis=0, s=s)
        A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
        gamma = fpeps.Peps(geometry)
        for ms in gamma.sites():
            gamma[ms] = A
        return gamma

    # Occupation Pattern Initialization
    elif len(args) == 3:
        pass

    else:
        raise ValueError("Invalid arguments for PEPS initialization")




def initialize_diagonal_basis(projectors, net, out):
    """"
    Return state according to the specified occupation pattern.

    Parameters
    ----------
        projectors: list of operators in the diagonal basis
        net : class Lattice
        out : dict
            A dictionary specifying the occupation pattern. The keys are the lattice sites
            and the values are integers indicating the occupation type
            (For Spinful fermions the convention can be to choose 0 for spin-up, 1 for spin-down,
            2 for double occupancy, and 3 for hole).

    """

    #projectors = [reduce_operators(proj) for proj in projectors]

    gamma = fpeps.Peps(net)
    for kk in gamma.sites():
        Ga = projectors[out[kk]].fuse_legs(axes=[(0, 1)])
        for s in (-1, 1, 1, -1):
            Ga = Ga.add_leg(axis=0, s=s)
        gamma[kk] = Ga.fuse_legs(axes=((0, 1), (2, 3), 4))

    return gamma


def load_from_dict(config, d):
    psi = SquareLattice(lattice=d['lattice'], dims=d['dims'], boundary=d['boundary'])
    psi= fpeps.Peps(psi)
    for ind, dtensor in d['data'].items():
        psi._data[ind] = load_tensor_from_dict(config, dtensor)
    return psi
