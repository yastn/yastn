import numpy as np
import yastn.tn.fpeps as fpeps
import yastn
import random

r""" Initialization of peps tensors for real or imaginary time evolution """


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

    nu = fcdag_up @ fc_up
    hd = fc_dn @ fcdag_dn
    nd = fcdag_dn @ fc_dn
    hu = fc_up @ fcdag_up
   
    f1 = nu @ hd
    f2 = hu @ nd

    A = f1.fuse_legs(axes=[(0, 1)])  # spin up
    B = f2.fuse_legs(axes=[(0, 1)])   # spin down

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