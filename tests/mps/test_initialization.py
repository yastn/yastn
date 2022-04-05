""" basic procedures of single mps """
import numpy as np
import pytest
import yast
import yamps
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


def test_assign_block():
    """ Initialise mps with known blocks. Example for AKLT state"""
    AM = - np.sqrt(2.0/3) * np.array([[0, 0],[1, 0]])
    A0 = - np.sqrt(1.0/3) * np.array([[1, 0],[0, -1]])
    AP = np.sqrt(2.0/3) * np.array([[0, 1],[0, 0]])
    
    T = np.array([AM,A0,AP])
    T = np.transpose(T,(1,0,2))
    bL = T[0,:,:].reshape((1,3,2))
    bR = T[:,:,1].reshape((2,3,1))
    
    N = 5
    psi = yamps.Mps(N, nr_phys=1)
    for n in range(N):
        psi.A[n] = yast.Tensor(config=config_dense, s=(1, 1, -1))
        if n == 0:
            tmp = bL
            Ds = (1,3,2)
        elif n == N-1:
            tmp = bR
            Ds = (2,3,1)
        else:
            tmp = T
            Ds = (2,3,2)
        psi.A[n].set_block(val=tmp, Ds=Ds)
    return psi

    
def test_assign_block():
    """ Initialise mps with known blocks. Example for AKLT state"""
    AM = - np.sqrt(2.0/3) * np.array([[0, 0],[1, 0]])
    A0 = - np.sqrt(1.0/3) * np.array([[1, 0],[0, -1]])
    AP = np.sqrt(2.0/3) * np.array([[0, 1],[0, 0]])
    
    T = np.array([AM,A0,AP])
    T = np.transpose(T,(1,0,2))
    bL = T[0,:,:].reshape((1,3,2))
    bR = T[:,:,1].reshape((2,3,1))
    
    N = 5
    psi = yamps.Mps(N, nr_phys=1)
    for n in range(N):
        psi.A[n] = yast.Tensor(config=config_dense, s=(1, 1, -1))
        if n == 0:
            tmp = bL
            Ds = (1,3,2)
        elif n == N-1:
            tmp = bR
            Ds = (2,3,1)
        else:
            tmp = T
            Ds = (2,3,2)
        psi.A[n].set_block(val=tmp, Ds=Ds)
    return psi


if __name__ == "__main__":
    test_assign_block()
    test_set_random()