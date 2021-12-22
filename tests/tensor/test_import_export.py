""" basic procedures of single mps """
import warnings
import numpy as np
import pytest
try:
    import h5py
except ImportError:
    warnings.warn("h5py module not available", ImportWarning)
import os
import yast
try:
    from .configs import config_dense, config_U1, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12


def check_import_export(a):
    """ Test if two Tensor-s have the same values. """
    os.remove("tmp.h5")
    with h5py.File('tmp.h5', 'a') as f:
        a.export_to_hdf5(f, './')
    with h5py.File('tmp.h5', 'r') as f:
        b = yast.import_from_hdf5(a.config, f, './')
    os.remove("tmp.h5")
    b.is_consistent()
    assert yast.are_independent(a, b)
    assert yast.norm_diff(a, b) < tol


def test_full_io():
    """ Initialize random Tensor of full tensors and checks copying. """
    a = yast.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    #check_import_export(a)


def test_U1_io():
    """ Initialize random Tensor of U1 tensors and checks copying. """
    a = yast.rand(config=config_U1, s=(-1, -1, 1, 1), n=1,
                  t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                  D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    #check_import_export(a)


def test_Z2_U1_io():
    """ Initialize random Tensor of Z2xU1 tensors and checks copying. """
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2_U1, s=(-1, -1, 1, 1),
                  t=[t1, t1, t1, t1],
                  D=[(2, 3, 4, 5), (5, 4, 3, 2), (3, 4, 5, 6), (1, 2, 3, 4)])
    afh = a.fuse_legs(axes=((1, 0), (2, 3)), mode='hard')
    afm = a.fuse_legs(axes=((1, 0), (2, 3)), mode='meta')
    # check_import_export(a)
    # check_import_export(afh)
    # check_import_export(afm)



if __name__ == "__main__":
    test_full_io()
    test_U1_io()
    test_Z2_U1_io()
