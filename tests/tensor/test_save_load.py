
""" yast.save_to_dict() yast.load_from_dict() yast.save_to_hdf5() yast.load_from_hdf5(). """
import warnings
import os
import pytest
import numpy as np
try:
    import h5py
except ImportError:
    warnings.warn("h5py module not available", ImportWarning)
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1, config_Z2, config_Z2_fermionic
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1, config_Z2, config_Z2_fermionic

tol = 1e-12  #pylint: disable=invalid-name


def check_to_numpy(a1, config):
    """ save/load to numpy and tests consistency."""
    d1 = a1.save_to_dict()
    a2 = 2 * a1  # second tensor to be saved
    d2 = yast.save_to_dict(a2)
    data={'tensor1': d1, 'tensor2': d2}  # two tensors to be saved
    np.save('tmp.npy', data)
    ldata = np.load('tmp.npy', allow_pickle=True).item()
    os.remove('tmp.npy')

    b1 = yast.load_from_dict(config=config, d=ldata['tensor1'])
    b2 = yast.load_from_dict(config=config, d=ldata['tensor2'])

    assert all(yast.norm(a - b) < tol for a, b in [(a1, b1), (a2, b2)])
    assert all(b.is_consistent for b in (b1, b2))
    assert all(yast.are_independent(a, b) for a, b in [(a1, b1), (a2, b2)])


def check_to_hdf5(a):
    """ Test if two Tensor-s have the same values. """
    # os.remove("tmp.h5") remove if exists .. perhaps 'w' in the line below
    with h5py.File('tmp.h5', 'a') as f:
        a.save_to_hdf5(f, './')
    with h5py.File('tmp.h5', 'r') as f:
        b = yast.load_from_hdf5(a.config, f, './')
    os.remove("tmp.h5")
    b.is_consistent()
    assert yast.are_independent(a, b)
    assert yast.norm(a - b) < tol


def test_dict():
    """ test exporting tensor to native python data-structure,
        that allows robust saving/loading with np.save/load."""
    a = yast.rand(config=config_dense)  # s=() i.e. scalar
    check_to_numpy(a, config_dense)

    a = yast.rand(config=config_U1, isdiag=False, s=(1, -1, 1),
                  t=((0, 1, 2), (0, 1, 3), (-1, 0, 1)),
                  D=((3, 5, 2), (1, 2, 3), (2, 3, 4)))
    check_to_numpy(a, config_U1)

    a = yast.randC(config=config_U1, isdiag=False, s=(1, -1, 1),
                  t=((0, 1, 2), (0, 1, 3), (-1, 0, 1)),
                  D=((3, 5, 2), (1, 2, 3), (2, 3, 4)))
    check_to_numpy(a, config_U1) # here a is complex

    a = yast.rand(config=config_U1, isdiag=True, t=(0, 1), D=(3, 5))
    check_to_numpy(a, config_U1)

    a = yast.ones(config=config_Z2xU1, s=(-1, 1, 1), n=(0, -2),
                  t=(((0, 0), (0, 2), (1, 0), (1, 2)), ((0, -2), (0, 2)), ((0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2))),
                  D=((1, 2, 3, 4), (2, 1), (2, 3, 5, 4, 1, 6)))
    check_to_numpy(a, config_Z2xU1)

    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    b = a.fuse_legs(axes=((0, 2), 1, (4, 3), 5), mode='hard')
    b = b.fuse_legs(axes=((0, 2), 1, 3), mode='hard')
    b = b.fuse_legs(axes=((0, 2), 1), mode='meta')
    check_to_numpy(b, config_U1)



def test_load_exceptions():
    """ handling exceptions """
    with pytest.raises(yast.YastError):
        _ = yast.load_from_dict(config=config_U1)  # Dictionary d is required

    a = yast.randC(config=config_Z2, isdiag=False, s=(1, -1, 1), n=1,
                  t=((0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6)))
    check_to_numpy(a, config_Z2)  # OK

    with pytest.raises(yast.YastError):
        check_to_numpy(a, config_U1)  # Symmetry rule in config do not match loaded one.
    with pytest.raises(yast.YastError):
        check_to_numpy(a, config_Z2_fermionic)  # Fermionic statistics in config do not match loaded one.


if __name__ == '__main__':
    test_dict()
    test_load_exceptions()
