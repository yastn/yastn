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
    # Initialize MPS tensor by tensor. Example for Spin-1 AKLT state.
    #
    # Prepare rank-2 blocks (matrices) of on-site tensors
    #
    AM = - np.sqrt(2.0/3) * np.array([[0, 0],[1, 0]])
    A0 = - np.sqrt(1.0/3) * np.array([[1, 0],[0, -1]])
    AP = np.sqrt(2.0/3) * np.array([[0, 1],[0, 0]])
    #
    # Prepare rank-3 on-site tensor with virtual dimensions 2 
    # and physical dimension dim(Spin-1)=3
    #                _ 
    # dim(left)=2 --|T|-- dim(right)=2
    #                |
    #           dim(Spin-1)=3 
    #
    T = np.array([AM,A0,AP])
    T = np.transpose(T,(1,0,2))
    #
    # Due to open boundary conditions for MPS, the first and the last 
    # on-site tensors have left and right virtual indices of dimension 1.
    #
    # 1--|bL|--  and  --|bR|--1
    #     |               |
    #
    bL = T[0,:,:].reshape((1,3,2))
    bR = T[:,:,1].reshape((2,3,1))
    #
    # First, we initialize empty MPS for N=5 sites. Then assign 
    # its on-site tensors one-by-one.
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
        # Create a yast.Tensor with appropriate legs and assign it to MPS.
        # Here, the choice of signatures is as follows
        #
        # (+1) ->--|T|-->- (-1)
        #           ^
        #           |(+1)  
        #
        site_tensor = yast.Tensor(config=config_dense, s=(1, 1, -1))
        site_tensor.set_block(val=tmp, Ds=Ds)
        
        # Finally assign the on-site tensor.
        #
        psi[n] = site_tensor
    return psi


def test_set_random():
    """ Initialize mps with random blocks."""
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
