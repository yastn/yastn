import numpy as np
import yast.tn.peps as peps

def initialize_peps_purification(fid, net):
    """
    initialize peps tensors into infinite-temeprature state, 
    fid is identity operator in local space with desired symmetry
    """
    
    A = fid / np.sqrt(fid.get_shape(1)) 
    A = A.fuse_legs(axes=[(0, 1)])            
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)
   
    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    Gamma = peps.Peps(net.lattice, net.dims, net.boundary)
    Gamma._data = {ms: A for ms in net.sites()}
    return Gamma


def initialize_Neel_spinfull(fid, fc_up, fc_dn, fcdag_up, fcdag_dn, net):

    Nx = net.Nx
    Ny = net.Ny

    nu = fcdag_up @ fc_up
    nd = fcdag_dn @ fc_dn
   
    f1 = (nu @ (fid-nd)).remove_zero_blocks()
    f2 = ((fid-nu) @ nd).remove_zero_blocks()

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
    for x in range(Nx):
        for y in range(Ny)[::2]:
            Gamma._data.update({(x, y): m[i]})
        for y in range(Ny)[1::2]:
            Gamma._data.update({(x, y): m[(i+1)%2]})
        i = (i+1)%2
    
    return Gamma