
""" yastn.save_to_dict() yastn.load_from_dict() yastn.save_to_hdf5() yastn.load_from_hdf5(). """
import warnings
import os
import pytest
import numpy as np
try:
    import h5py
except ImportError:
    warnings.warn("h5py module not available", ImportWarning)
import yastn
try:
    from .configs import config_dense, config_U1, config_Z2xU1, config_Z2, config_Z2_fermionic
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1, config_Z2, config_Z2_fermionic

tol = 1e-12  #pylint: disable=invalid-name


def check_to_numpy(a1, config):
    """ save/load to numpy and tests consistency."""
    d1 = a1.save_to_dict()
    a2 = 2 * a1  # second tensor to be saved
    d2 = yastn.save_to_dict(a2)
    data={'tensor1': d1, 'tensor2': d2}  # two tensors to be saved
    np.save('tmp.npy', data)
    ldata = np.load('tmp.npy', allow_pickle=True).item()
    os.remove('tmp.npy')

    b1 = yastn.load_from_dict(config=config, d=ldata['tensor1'])
    b2 = yastn.load_from_dict(config=config, d=ldata['tensor2'])

    assert all(yastn.norm(a - b) < tol for a, b in [(a1, b1), (a2, b2)])
    assert all(b.is_consistent for b in (b1, b2))
    assert all(yastn.are_independent(a, b) for a, b in [(a1, b1), (a2, b2)])


def check_to_hdf5(a):
    """ Test if two Tensor-s have the same values. """
    # os.remove("tmp.h5") remove if exists .. perhaps 'w' in the line below
    try:
        os.remove("tmp.h5")
    except OSError:
        pass
    with h5py.File('tmp.h5', 'a') as f:
        a.save_to_hdf5(f, './')
    with h5py.File('tmp.h5', 'r') as f:
        b = yastn.load_from_hdf5(a.config, f, './')
    os.remove("tmp.h5")
    b.is_consistent()
    assert yastn.are_independent(a, b)
    assert yastn.norm(a - b) < tol


def test_dict():
    """ test exporting tensor to native python data-structure,
        that allows robust saving/loading with np.save/load."""
    a = yastn.rand(config=config_dense)  # s=() i.e. a scalar
    assert a.size == 1
    check_to_numpy(a, config_dense)
    check_to_hdf5(a)

    legs = [yastn.Leg(config_U1, s=1, t=(0, 1, 2), D= (3, 5, 2)),
            yastn.Leg(config_U1, s=-1, t=(0, 1, 3), D= (1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D= (2, 3, 4))]

    a = yastn.rand(config=config_U1, legs=legs)
    check_to_numpy(a, config_U1)
    check_to_hdf5(a)

    a = yastn.randC(config=config_U1, legs=legs, n=1)
    check_to_numpy(a, config_U1) # here a is complex
    check_to_hdf5(a)

    a = yastn.rand(config=config_U1, isdiag=True, legs=legs[0])
    check_to_numpy(a, config_U1)
    check_to_hdf5(a)


    legs = [yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 4)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, -2), (0, 2)), D=(2, 1)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)), D=(2, 3, 5, 4, 1, 6))]
    a = yastn.ones(config=config_Z2xU1, legs=legs, n=(0, -2))
    check_to_numpy(a, config_Z2xU1)
    check_to_hdf5(a)

    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    a = a.fuse_legs(axes=((0, 2), 1, (4, 3), 5), mode='hard')
    a = a.fuse_legs(axes=((0, 2), 1, 3), mode='hard')
    a = a.fuse_legs(axes=((0, 2), 1), mode='meta')
    check_to_numpy(a, config_U1)
    check_to_hdf5(a)



def test_load_exceptions():
    """ handling exceptions """
    with pytest.raises(yastn.YastnError):
        _ = yastn.load_from_dict(config=config_U1)  # Dictionary d is required

    leg = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(2, 3))
    a = yastn.randC(config=config_Z2, n=1, legs=[leg, leg, leg.conj()])
    check_to_numpy(a, config_Z2)  # OK

    with pytest.raises(yastn.YastnError):
        check_to_numpy(a, config_U1)  # Symmetry rule in config do not match loaded one.
    with pytest.raises(yastn.YastnError):
        check_to_numpy(a, config_Z2_fermionic)  # Fermionic statistics in config do not match loaded one.


if __name__ == '__main__':
    test_dict()
    test_load_exceptions()
