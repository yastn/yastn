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
    #
    # Prepare the representation of local tensors
    #
    AM = - np.sqrt(2.0/3) * np.array([[0, 0],[1, 0]])
    A0 = - np.sqrt(1.0/3) * np.array([[1, 0],[0, -1]])
    AP = np.sqrt(2.0/3) * np.array([[0, 1],[0, 0]])
    #
    # Prepare local tensor with appropriate virtual and physical dimensions
    #
    T = np.array([AM,A0,AP])
    T = np.transpose(T,(1,0,2))
    #
    # In open boundary condition for MPS we shuold make sure that 
    # terminating virtual dimensions are 1.
    #
    bL = T[0,:,:].reshape((1,3,2))
    bR = T[:,:,1].reshape((2,3,1))
    #
    # Setting up MPS always involves initialisation of YAMPS object with proper length.
    #
    N = 5
    psi = yamps.Mps(N)
    for n in range(N):
        if n == 0:
            tmp = bL
            Ds = (1,3,2)
        elif n == N-1:
            tmp = bR
            Ds = (2,3,1)
        else:
            tmp = T
            Ds = (2,3,2)
        #
        # The loop will assign each tensor in the MPS chain one by one.
        # We create a site_tensor with appropriate legs and we push it to MPS. 
        #
        site_tensor = yast.Tensor(config=config_dense, s=(1, 1, -1))
        site_tensor.set_block(val=tmp, Ds=Ds)
        psi.A[n] = site_tensor
    return psi


def test_set_random():
    """ Initialise mps with random blocks."""
    N, d, Dmax, dtype = 16, 2, 30, 'float64'
    if isinstance(d, int):
        d = [d]
    d *= (N + len(d) - 1) // len(d)

    psi = yamps.Mps(N)
    Dl, Dr = 1, Dmax
    for n in range(N):
        Dr = Dmax if n < N - 1 else 1
        Dl = Dmax if n > 0 else 1
        psi.A[n] = yast.rand(config=config_dense, s=(1, 1, -1), D=[Dl, d[n], Dr], dtype=dtype)
    return psi


if __name__ == "__main__":
    test_assign_block()
    test_set_random()
