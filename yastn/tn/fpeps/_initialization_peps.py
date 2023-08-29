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


def initialize_spinful_random(fc_up, fc_dn, fcdag_up, fcdag_dn, net, n_up, n_down):
    """"
    Randomly initializes a 2D rectangular spinful lattice with a specified
    number of up and down spin electrons.

    Parameters
    ----------
    fc_up : Annihilation operator for spin up fermions.
    fc_dn : Annihilation operator for spin down fermions.
    fcdag_up : Creation operator for spin up fermions.
    fcdag_dn : Creation operator for spin down fermions.
    net : class Lattice
    n_up, n_down: number of up and down-spin electrons to be distributed

    Returns
    -------
    gamma : Peps object
    """

    Nx, Ny = net.Nx, net.Ny
    out = {(x, y): 3 for x in range(Nx) for y in range(Ny)}
    available_sites = [(x, y) for x in range(Nx) for y in range(Ny)]
    for _ in range(n_up):
        site = random.choice(available_sites)
        if out[site] == 3:
            out[site] = 0  # Only up-spin
        available_sites.remove(site)
    available_sites = [(x, y) for x in range(Nx) for y in range(Ny)]
    for _ in range(n_down):
        site = random.choice(available_sites)
        if out[site] == 3:
            out[site] = 1  # Only down-spin
        elif out[site] == 0:
            out[site] = 2  # Both up-spin and down-spin
        available_sites.remove(site)

    n_up, n_dn, h_up, h_dn = fcdag_up @ fc_up, fcdag_dn @ fc_dn, fc_up @ fcdag_up, fc_dn @ fcdag_dn
    nn_up, nn_dn, nn_do, nn_hole = n_up @ h_dn, h_up @ n_dn, n_up @ n_dn, h_up @ h_dn   # up - 0; down - 1; double occupancy - 2; hole - 3
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


