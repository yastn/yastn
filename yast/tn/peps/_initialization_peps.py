import numpy as np
import yast.tn.peps as peps

r""" Initialization of peps tensors for evolution """

def initialize_peps_purification(fid, net):
    """
    initialize peps tensors at infinite-temperature state, 
    fid is identity operator in local space with desired symmetry
    (ancilla present)
    """
    
    A = fid / np.sqrt(fid.get_shape(1)) 
    A = A.fuse_legs(axes=[(0, 1)])            
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)
   
    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    Gamma = peps.Peps(net.lattice, net.dims, net.boundary)

    for ms in net.sites():
        Gamma[ms] = A
    return Gamma


def initialize_spinless_filled(fid, fc, fcdag, net):
    """ initialize spinless fermi sea with all sites filled """

    n = fcdag @ fc
    f = n
    A = f.fuse_legs(axes=[(0, 1)])  # particle
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)
    
    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    Gamma = peps.Peps(net.lattice, net.dims, net.boundary)
    for ms in net.sites():
        Gamma[ms] = A

    return Gamma


def initialize_Neel_spinfull(fc_up, fc_dn, fcdag_up, fcdag_dn, net):
    """ initializes Neel state """

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
    Gamma = peps.Peps(net.lattice, net.dims, net.boundary)
    for x in range(net.Nx):
        for y in range(net.Ny)[::2]:
            Gamma[x, y] = m[i]
        for y in range(net.Ny)[1::2]:
            Gamma[x, y] = m[(i+1)%2]
        i = (i+1)%2
    
    return Gamma


def initialize_post_sampling(fc_up, fc_dn, fcdag_up, fcdag_dn, net, out):
    """ initializes the post-sampling initialize state according to out """

    n_up, n_dn, h_up, h_dn = fcdag_up @ fc_up, fcdag_dn @ fc_dn, fc_up @ fcdag_up, fc_dn @ fcdag_dn
    nn_up, nn_dn, nn_do, nn_hole = n_up @ h_dn, h_up @ n_dn, n_up @ n_dn, h_up @ h_dn # up - 0; down - 1; double occupancy - 2; hole - 3
    tt = {0: nn_up, 1: nn_dn, 2: nn_do, 3: nn_hole}
   
    Gamma = peps.Peps(net.lattice, net.dims, net.boundary)
    for kk in Gamma.sites():
        Ga = tt[out[kk]].fuse_legs(axes=[(0, 1)])
        for s in (-1, 1, 1, -1):
            Ga = Ga.add_leg(axis=0, s=s)
        Gamma[kk] = Ga.fuse_legs(axes=((0, 1), (2, 3), 4))
        
    return Gamma