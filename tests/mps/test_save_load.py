""" basic procedures of single mps """
import os
import warnings
import numpy as np
try:
    import h5py
except ImportError:
    warnings.warn("h5py module not available", ImportWarning)

import yamps
try:
    from . import generate_random, generate_by_hand
    from .configs import config_dense, config_dense_fermionic
    from .configs import config_U1, config_U1_fermionic
    from .configs import config_Z2, config_Z2_fermionic
except ImportError:
    import generate_random, generate_by_hand
    from configs import config_dense, config_dense_fermionic
    from configs import config_U1, config_U1_fermionic
    from configs import config_Z2, config_Z2_fermionic



tol = 1e-12

def check_copy(psi1, psi2):
    """ Test if two mps-s have the same tensors (velues). """
    for n in psi1.sweep():
        assert np.allclose(psi1.A[n].to_numpy(), psi2.A[n].to_numpy())


def test_basic_hdf5():
    # Initialize random MPS with dense tensors and checks saving/loading 
    # to and from HDF5 file.
    #
    psi = yamps.random_dense_mps(N=16, D=15, d=2)
    #
    # We delete file if it already exists. 
    # (It is enough to clear the address './state' if the file already exists)
    #
    try:
        os.remove("tmp.h5")
    except OSError:
        pass
    #
    # We save the MPS to file 'tmp.h5' under the address 'state/'
    #
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')

    #
    # To read MPS from HDF5 file, open the file and load the MPS stored 
    # at address 'state/'. 
    # Note: You have to provide valid YAST configuration 
    # TODO: basic properties of the object like nr_phys = 1 (for MPS) or 2 (for MPO).
    #
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(config_dense, 1, f, 'state/')

    # Similarily, one can save and load MPO
    #
    psi = generate_random.mpo_random(config_dense, N=16, Dmax=25, d=[2, 3], d_out=[2, 1])
    with h5py.File('tmp.h5', 'w') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(config_dense, 1, f, 'state/')
    os.remove("tmp.h5")


def test_full_hdf5():
    try:
        os.remove("tmp.h5")
    except OSError:
        pass

    # basic MPS
    psi = generate_random.mps_random(config_dense, N=16, Dmax=15, d=2)
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(config_dense, 1, f, 'state/')
    check_copy(psi, phi)
    os.remove("tmp.h5") 

    # MPS with alternating physical dimension
    psi = generate_random.mps_random(config_dense, N=16, Dmax=19, d=[2, 3])
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(psi.A[0].config, 1, f, 'state/')
    check_copy(psi, phi)
    os.remove("tmp.h5") 

    # MPO with alternating physical dimensions of both bra and ket indices
    psi = generate_random.mpo_random(config_dense, N=16, Dmax=25, d=[2, 3], d_out=[2, 1])
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(psi.A[0].config, 1, f, 'state/')
    check_copy(psi, phi)
    os.remove("tmp.h5") 


def test_Z2_hdf5():
    psi = generate_random.mps_random(config_Z2, N=16, Dblock=25, total_parity=0)
    try:
        os.remove("tmp.h5")
    except OSError:
        pass
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(psi.A[0].config, 1, f, 'state/')
    check_copy(psi, phi)
    os.remove("tmp.h5") 


    psi = generate_random.mps_random(config_Z2, N=16, Dblock=25, total_parity=1)
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(psi.A[0].config, 1, f, 'state/')
    check_copy(psi, phi)
    os.remove("tmp.h5") 


def test_basic_dict():
    #
    # First, we generate random MPS without any symmetry.
    #    
    psi = yamps.random_dense_mps(N=16, D=25, d=3)
    #
    # Next, we serialize MPS into dictionary.
    #
    tmp = psi.save_to_dict()
    #
    # Last, we load the MPS from the dictionary, providing valid YAST configuration
    #
    phi = yamps.load_from_dict(config_dense, 1, tmp)


def test_full_dict():
    psi = generate_random.mpo_random(config_dense, N=16, Dmax=25, d=[2, 3], d_out=[2, 1])
    tmp = psi.save_to_dict()
    phi = yamps.load_from_dict(psi.A[0].config, 1, tmp)
    check_copy(psi, phi)


def test_Z2_dict():
    psi = generate_random.mps_random(config_Z2, N=16, Dblock=25, total_parity=0)
    tmp = psi.save_to_dict()
    phi = yamps.load_from_dict(psi.A[0].config, 1, tmp)
    check_copy(psi, phi)


if __name__ == "__main__":
    test_full_hdf5()
    test_Z2_hdf5()
    test_full_dict()
    test_Z2_dict()