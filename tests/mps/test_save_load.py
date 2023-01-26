""" basic procedures of single mps """
import os
import warnings
import numpy as np
import yast
import yast.tn.mps as mps

try:
    import h5py
except ImportError:
    warnings.warn("h5py module not available", ImportWarning)
try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense as cfg




tol = 1e-12


def check_copy(psi1, psi2):
    """ Test if two mps-s have the same tensors (velues). """
    for n in psi1.sweep():
        assert np.allclose(psi1.A[n].to_numpy(), psi2.A[n].to_numpy())
    assert psi1.A is not psi2.A
    assert psi1 is not psi2


def test_basic_hdf5():
    # Initialize random MPS with dense tensors and checks saving/loading 
    # to and from HDF5 file.
    #
    psi = mps.random_dense_mps(N=16, D=15, d=2, backend=cfg.backend, default_device=cfg.default_device)
    config_dense = psi.config
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
    #
    with h5py.File('tmp.h5', 'r') as f:
        phi = mps.load_from_hdf5(config_dense, f, './state/')
    #
    # Test psi == phi
    #
    check_copy(psi, phi)
    #
    # Similarily, one can save and load MPO
    #
    psi = mps.random_dense_mpo(N=16, D=8, d=3, backend=cfg.backend, default_device=cfg.default_device)
    with h5py.File('tmp.h5', 'w') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = mps.load_from_hdf5(config_dense, f, './state/')
    os.remove("tmp.h5")
    check_copy(psi, phi)


def test_Z2_hdf5():
    operators = yast.operators.SpinlessFermions(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=16, operators=operators)
    #
    psi = generate.random_mps(D_total=25, n=0)
    try:
        os.remove("tmp.h5")
    except OSError:
        pass
    with h5py.File('tmp.h5', 'a') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = mps.load_from_hdf5(psi.A[0].config, f, './state/')
    os.remove("tmp.h5") 
    check_copy(psi, phi)


def test_basic_dict():
    #
    # First, we generate random MPS without any symmetry.
    #    
    psi = mps.random_dense_mps(N=16, D=25, d=3, backend=cfg.backend, default_device=cfg.default_device)
    config_dense = psi.config
    #
    # Next, we serialize MPS into dictionary.
    #
    tmp = psi.save_to_dict()
    #
    # Last, we load the MPS from the dictionary, providing valid YAST configuration
    #
    phi = mps.load_from_dict(config_dense, tmp)
    #
    # Test psi == phi
    #
    check_copy(psi, phi)



def test_Z2_dict():
    operators = yast.operators.SpinlessFermions(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=16, operators=operators)

    psi = generate.random_mps(D_total=15, n=0)
    tmp = psi.save_to_dict()
    phi = mps.load_from_dict(generate.config, tmp)
    check_copy(psi, phi)


if __name__ == "__main__":
    test_basic_hdf5()
    test_Z2_hdf5()
    test_basic_dict()
    test_Z2_dict()
