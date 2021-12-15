""" basic procedures of single mps """
import numpy as np
import pytest
import h5py
import os
import yast
try:
    from .configs import config_dense, config_U1, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12

def check_copy(A, B):
    """ Test if two Tensor-s have the same values. """
    assert np.allclose(A.to_numpy(), B.to_numpy())


def test_full_io():
    """ Initialize random Tensor of full tensors and checks copying. """
    a = yast.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    with h5py.File('tmp.h5', 'a') as f:
        a.export_to_hdf5(f, './')
    with h5py.File('tmp.h5', 'r') as f:
        b = yast.import_from_hdf5(a.config, f, './')
    check_copy(a, b)
    os.remove("tmp.h5")


def test_U1_io():
    """ Initialize random Tensor of U1 tensors and checks copying. """
    a = yast.rand(config=config_U1, s=(-1, -1, 1, 1), n=1,
                  t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                  D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    with h5py.File('tmp.h5', 'a') as f:
        a.export_to_hdf5(f, './')
    with h5py.File('tmp.h5', 'r') as f:
        b = yast.import_from_hdf5(a.config, f, './')
    check_copy(a, b)
    os.remove("tmp.h5")


def test_Z2_U1_io():
    """ Initialize random Tensor of Z2xU1 tensors and checks copying. """
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2_U1, s=(-1, -1, 1, 1),
                  t=[t1, t1, t1, t1],
                  D=[(2, 3, 4, 5), (5, 4, 3, 2), (3, 4, 5, 6), (1, 2, 3, 4)])
    with h5py.File('tmp.h5', 'a') as f:
        a.export_to_hdf5(f, './')
    with h5py.File('tmp.h5', 'r') as f:
        b = yast.import_from_hdf5(a.config, f, './')
    check_copy(a, b)
    os.remove("tmp.h5")


if __name__ == "__main__":
    test_full_io()
    test_U1_io()
    test_Z2_U1_io()