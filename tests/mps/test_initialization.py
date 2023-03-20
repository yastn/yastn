""" basic procedures of single mps """
import numpy as np
import yast
import yast.tn.mps as mps
try:
    from .configs import config_dense as cfg
    # pytest modifies cfg to inject different backends and devices during tests
except ImportError:
    from configs import config_dense as cfg


def build_spin1_aklt_state(N=5):
    """
    Initialize MPS tensors by hand. Example for Spin-1 AKLT state of N sites.
    """
    # Prepare rank-3 on-site tensor with virtual dimensions 2
    # and physical dimension dim(Spin-1)=3
    #                _
    # dim(left)=2 --|T|-- dim(right)=2
    #                |
    #           dim(Spin-1)=3
    #
    T = np.zeros((2, 3, 2))
    T[:, 0, :] = - np.sqrt(2.0 / 3) * np.array([[0, 0], [1, 0]])
    T[:, 1, :] = - np.sqrt(1.0 / 3) * np.array([[1, 0], [0, -1]])
    T[:, 2, :] = + np.sqrt(2.0 / 3) * np.array([[0, 1], [0, 0]])
    #
    # Due to open boundary conditions for MPS, the first and the last
    # on-site tensors will have left and right virtual indices projected to dimension 1.
    #
    # First, we initialize empty MPS for N sites.
    #
    psi = mps.Mps(N)
    #
    # Then assign its on-site tensors one-by-one.
    #
    for n in range(N):
        # Create a yast.Tensor with appropriate legs and assign it to MPS.
        # Here, the choice of signatures is as follows
        #
        # (+1) ->--|T|-->- (-1)
        #           ^
        #           |(+1)
        psi[n] = yast.Tensor(config=cfg, s=(1, 1, -1))
        if n == 0:
            psi[n].set_block(val=T[0, :, :], Ds=(1, 3, 2))
        elif n == N - 1:
            psi[n].set_block(val=T[:, :, 1], Ds=(2, 3, 1))
        else:
            psi[n].set_block(val=T, Ds=(2, 3, 2))
    return psi


if __name__ == "__main__":
    build_spin1_aklt_state()
