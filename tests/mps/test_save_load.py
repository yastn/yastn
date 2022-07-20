""" basic procedures of single mps """
import numpy as np
import h5py
import os
import yamps
try:
    from . import generate_random, generate_by_hand, generate_automatic
    from .configs import config_dense, config_dense_fermionic
    from .configs import config_U1, config_U1_fermionic
    from .configs import config_Z2, config_Z2_fermionic
except ImportError:
    import generate_random, generate_by_hand, generate_automatic
    from configs import config_dense, config_dense_fermionic
    from configs import config_U1, config_U1_fermionic
    from configs import config_Z2, config_Z2_fermionic



tol = 1e-12

def check_copy(psi1, psi2):
    """ Test if two mps-s have the same tensors (velues). """
    for n in psi1.sweep():
        assert np.allclose(psi1.A[n].to_numpy(), psi2.A[n].to_numpy())


def test_full_hdf5():
    """ Initialize random mps of full tensors and checks copying. """
    #
    # The initialization produces a random MPS with certain symmetry given by config.
    # Here there are no symmetries imposed.
    #
    psi = generate_random.mps_random(config_dense, N=16, Dmax=15, d=2)
    #
    # In order to save the MPS you have to make sure that the place we are saving to is empty
    # Here we delete full file if it exist but it is enough to clear a group './state' inside it.
    #
    try:
        os.remove("tmp.h5")
    except OSError:
        pass
    #
    # We save to a file after opening it in Python. The file can contain other element but you 
    # have to make sure that the address is empty.
    #
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    #
    # To read MPS from HDF5 to yamps we have to open the file and get 
    # saved instructions. You have to specify basic properties of the object 
    # like nr_phys = 1 (for MPS) or 2 (for MPO).
    #
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(psi.A[0].config, 1, f, 'state/')
    #
    # In this test we will also check if the import was successful and delete HDF5 file we produced.
    #
    check_copy(psi, phi)
    os.remove("tmp.h5") 
    
    # Here are some more advanced examples:
    #
    psi = generate_random.mps_random(config_dense, N=16, Dmax=19, d=[2, 3])
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(psi.A[0].config, 1, f, 'state/')
    check_copy(psi, phi)
    os.remove("tmp.h5") 

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


def test_full_dict():
    #
    # The initialization produces a random MPS with certain symmetry given by config.
    # Here there are no symmetries imposed.
    #    
    psi = generate_random.mpo_random(config_dense, N=16, Dmax=25, d=[2, 3], d_out=[2, 1])
    #
    # We write instructions to recover the MPS in a dictionary tmp:
    #
    tmp = psi.save_to_dict()
    #
    # Using these intructions we can load it to yamps and produce the MPS we saved.
    # You have to specify basic properties of the object 
    # like nr_phys = 1 (for MPS) or 2 (for MPO).
    #
    phi = yamps.load_from_dict(psi.A[0].config, 1, tmp)
    #
    # In this test we will also check if the import was successful.
    #
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