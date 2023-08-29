import numpy as np
import yastn.tn.fpeps as fpeps
import yastn
import random
from ...initialize import *

r""" Initialization of peps tensors for real or imaginary time evolution """

def reduce_operators(ops):
        ds = ops.remove_zero_blocks()
        lg = ds.get_legs(1).conj()
        vone = ones(config=ds.config,legs=[lg], n=lg.t[0])
        W = ds@vone
        W = W.add_leg(s=-1)
        return W

def initialize_peps_purification(fid, net):
    """
    Initialize PEPS tensors at infinite-temperature state.

    Parameters
    ----------
        fid : Identity operator in local space with desired symmetry.
        net : class Lattice 

    Returns
    -------
        gamma : class Peps
              PEPS tensor network class with data representing the infinite-temperature state.
    """
    
    A = fid / np.sqrt(fid.get_shape(1)) 
    A = A.fuse_legs(axes=[(0, 1)])            
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)
   
    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    gamma = fpeps.Peps(net.lattice, net.dims, net.boundary)

    for ms in net.sites():
        gamma[ms] = A
    return gamma
  

def initialize_diagonal_basis(projectors, net, out):
    """"
    Initializes state according to the specified occupation pattern.

    Parameters
    ----------
    fc_up : Annihilation operator for spin up fermions.
    fc_dn : Annihilation operator for spin down fermions.
    fcdag_up : Creation operator for spin up fermions.
    fcdag_dn : Creation operator for spin down fermions.
    net : class Lattice
    out : dict
        A dictionary specifying the occupation pattern. The keys are the lattice sites
        and the values are integers indicating the occupation type (0 for spin-up, 1 for spin-down,
        2 for double occupancy, and 3 for hole).

    Returns
    -------
    gamma : Peps object
        The post-sampling state tensor network.
    """
   
    projectors = [reduce_operators(proj) for proj in projectors]

    gamma = fpeps.Peps(net.lattice, net.dims, net.boundary)
    for kk in gamma.sites():
        Ga = projectors[out[kk]].fuse_legs(axes=[(0, 1)])
        for s in (-1, 1, 1, -1):
            Ga = Ga.add_leg(axis=0, s=s)
        gamma[kk] = Ga.fuse_legs(axes=((0, 1), (2, 3), 4))
        
    return gamma