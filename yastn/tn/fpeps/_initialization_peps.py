import numpy as np
import yastn.tn.fpeps as fpeps
from ...initialize import ones as ones
from ...initialize import load_from_dict as load_tensor_from_dict
from ._geometry import Lattice

r""" Initialization of peps tensors for real or imaginary time evolution """

# def reduce_operators(ops):
#         ds = ops.remove_zero_blocks()
#         lg = ds.get_legs(1).conj()
#         vone = ones(config=ds.config,legs=[lg], n=lg.t[0])
#         W = ds@vone
#         W = W.add_leg(s=-1)
#         return W


def product_peps():
    pass

def initialize_peps_purification(fid, net):
    """
    Returns PEPS tensors initialized at infinite-temperature state.

    Parameters
    ----------
        fid : Identity operator in local space with desired symmetry.
        net : class Lattice

    """

    A = fid / np.sqrt(fid.get_shape(1))
    A = A.fuse_legs(axes=[(0, 1)])
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)

    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    gamma = fpeps.Lattice(net.lattice, net.dims, net.boundary)

    for ms in net.sites():
        gamma[ms] = A
    return gamma


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

    gamma = fpeps.Lattice(net.lattice, net.dims, net.boundary)
    for kk in gamma.sites():
        Ga = projectors[out[kk]].fuse_legs(axes=[(0, 1)])
        for s in (-1, 1, 1, -1):
            Ga = Ga.add_leg(axis=0, s=s)
        gamma[kk] = Ga.fuse_legs(axes=((0, 1), (2, 3), 4))

    return gamma


def load_from_dict(config, d):
    psi = Lattice(lattice=d['lattice'], dims=d['dims'], boundary=d['boundary'])
    for ind, dtensor in d['data'].items():
        psi._data[ind] = load_tensor_from_dict(config, dtensor)
    return psi
