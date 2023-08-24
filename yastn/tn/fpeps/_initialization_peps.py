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

def initialize_vacuum(fid, net):
    """
    Initialize PEPS tensors as a vacuum state.

    Parameters
    ----------
        fid : Identity operator in local space with desired symmetry.
        net : class Lattice 

    Returns
    -------
        gamma : class Peps
              PEPS tensor network class with data representing the vacuum state.
    """
    
    A = yastn.Leg(fid.config, t= ((0,0,0),), D=((1),))
    A = yastn.ones(fid.config, legs=[A])
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)

    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    gamma = fpeps.Peps(net.lattice, net.dims, net.boundary)

    for ms in net.sites():
        gamma[ms] = A
    return gamma


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


def initialize_Neel_spinful(fc_up, fc_dn, fcdag_up, fcdag_dn, net):

    """ 
    Initializes a Neel state in a PEPS representation. Useful for finding ground state
    of Hubbard model near half-filling
    
    Parameters
    ----------
    fc_up : Annihilation operator for spin up fermions.
    fc_dn : Annihilation operator for spin down fermions.
    fcdag_up : Creation operator for spin up fermions.
    fcdag_dn : Creation operator for spin down fermions.
    net : class Lattice
        
    Returns
    -------
    gamma : class Peps
        A PEPS object representing the Neel state.
    """

    n_up, n_dn, h_up, h_dn = fcdag_up @ fc_up, fcdag_dn @ fc_dn, fc_up @ fcdag_up, fc_dn @ fcdag_dn
    nn_up, nn_dn, nn_do, nn_hole = n_up @ h_dn, h_up @ n_dn, n_up @ n_dn, h_up @ h_dn  

    nn_up = reduce_operators(nn_up)
    nn_dn = reduce_operators(nn_dn)

    A = nn_up.fuse_legs(axes=[(0, 1)])  # spin up
    B = nn_dn.fuse_legs(axes=[(0, 1)])   # spin down

    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)
        B = B.add_leg(axis=0, s=s)
    
    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    B = B.fuse_legs(axes=((0, 1), (2, 3), 4))

    m = list([A, B])
    i = 0
    gamma = fpeps.Peps(net.lattice, net.dims, net.boundary)
    for x in range(net.Nx):
        for y in range(net.Ny)[::2]:
            gamma[x, y] = m[i]
        for y in range(net.Ny)[1::2]:
            gamma[x, y] = m[(i+1)%2]
        i = (i+1)%2
    
    return gamma


def initialize_n_0p875_pattern_1(fc_up, fc_dn, fcdag_up, fcdag_dn, net):
    """"
    Initializes the post-sampling state to density n=0.875.

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

    n_up, n_dn, h_up, h_dn = fcdag_up @ fc_up, fcdag_dn @ fc_dn, fc_up @ fcdag_up, fc_dn @ fcdag_dn
    nn_up, nn_dn, nn_do, nn_hole = n_up @ h_dn, h_up @ n_dn, n_up @ n_dn, h_up @ h_dn      # up - 0; down - 1; double occupancy - 2; hole - 3
    nn_up = reduce_operators(nn_up)
    nn_dn = reduce_operators(nn_dn)
    nn_do = reduce_operators(nn_do)
    nn_hole = reduce_operators(nn_hole)

    tt = {0: nn_up, 1: nn_dn, 2: nn_do, 3: nn_hole}
    out = {(0, 0): 0, (0, 1): 1, (0, 2): 0, (0, 3): 1,
    (1, 0): 1, (1, 1): 2, (1, 2): 1, (1, 3): 0,
    (2, 0): 0, (2, 1): 3, (2, 2): 3, (2, 3): 1,
    (3, 0): 3, (3, 1): 0, (3, 2): 1, (3, 3): 0}
   
    gamma = fpeps.Peps(net.lattice, net.dims, net.boundary)
    for kk in gamma.sites():
        Ga = tt[out[kk]].fuse_legs(axes=[(0, 1)])
        for s in (-1, 1, 1, -1):
            Ga = Ga.add_leg(axis=0, s=s)
        gamma[kk] = Ga.fuse_legs(axes=((0, 1), (2, 3), 4))
        
    return gamma


def initialize_post_sampling_spinful_sz_basis(fc_up, fc_dn, fcdag_up, fcdag_dn, net, out):
    """"
    Initializes the post-sampling state according to the specified occupation pattern.

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

    n_up, n_dn, h_up, h_dn = fcdag_up @ fc_up, fcdag_dn @ fc_dn, fc_up @ fcdag_up, fc_dn @ fcdag_dn
    nn_up, nn_dn, nn_do, nn_hole = n_up @ h_dn, h_up @ n_dn, n_up @ n_dn, h_up @ h_dn      # up - 0; down - 1; double occupancy - 2; hole - 3
    nn_up = reduce_operators(nn_up)
    nn_dn = reduce_operators(nn_dn)
    nn_do = reduce_operators(nn_do)
    nn_hole = reduce_operators(nn_hole)
    tt = {0: nn_up, 1: nn_dn, 2: nn_do, 3: nn_hole}

    gamma = fpeps.Peps(net.lattice, net.dims, net.boundary)
    for kk in gamma.sites():
        Ga = tt[out[kk]].fuse_legs(axes=[(0, 1)])
        for s in (-1, 1, 1, -1):
            Ga = Ga.add_leg(axis=0, s=s)
        gamma[kk] = Ga.fuse_legs(axes=((0, 1), (2, 3), 4))
        
    return gamma
