""" basic procedures of single mps """
import numpy as np
import h5py
import os
import yast
import yamps
try:
    from . import ops_dense
    from . import ops_Z2
except ImportError:
    import ops_dense
    import ops_Z2

tol = 1e-12

def check_copy(psi1, psi2):
    """ Test if two mps-s have the same tensors (velues). """
    for n in psi1.sweep():
        assert np.allclose(psi1.A[n].to_numpy(), psi2.A[n].to_numpy())


def test_full_hdf5():
    """ Initialize random mps of full tensors and checks copying. """
    psi = ops_dense.mps_random(N=16, Dmax=15, d=2)
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
    
    psi = ops_dense.mps_random(N=16, Dmax=19, d=[2, 3])
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(psi.A[0].config, 1, f, 'state/')
    check_copy(psi, phi)
    os.remove("tmp.h5") 

    psi = ops_dense.mpo_random(N=16, Dmax=25, d=[2, 3], d_out=[2, 1])
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(psi.A[0].config, 1, f, 'state/')
    check_copy(psi, phi)
    os.remove("tmp.h5") 


def test_Z2_hdf5():
    psi = ops_Z2.mps_random(N=16, Dblock=25, total_parity=0)
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


    psi = ops_Z2.mps_random(N=16, Dblock=25, total_parity=1)
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = yamps.load_from_hdf5(psi.A[0].config, 1, f, 'state/')
    check_copy(psi, phi)
    os.remove("tmp.h5") 


def test_full_dict():
    psi = ops_dense.mpo_random(N=16, Dmax=25, d=[2, 3], d_out=[2, 1])
    tmp = psi.save_to_dict()
    phi = yamps.load_from_dict(psi.A[0].config, 1, tmp)
    check_copy(psi, phi)


def test_Z2_dict():
    psi = ops_Z2.mps_random(N=16, Dblock=25, total_parity=0)
    tmp = psi.save_to_dict()
    phi = yamps.load_from_dict(psi.A[0].config, 1, tmp)
    check_copy(psi, phi)


if __name__ == "__main__":
    test_full_hdf5()
    test_Z2_hdf5()
    test_full_dict()
    test_Z2_dict()