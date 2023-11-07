""" basic procedures of single mps """
import os
import warnings
import pytest
import yastn
import yastn.tn.mps as mps

try:
    import h5py
except ImportError:
    warnings.warn("h5py module not available", ImportWarning)
try:
    from .configs import config_dense as config
    # pytest modifies cfg to inject different backends and devices during tests
except ImportError:
    from configs import config_dense as config


@pytest.mark.parametrize('kwargs', [{'sym': 'dense', 'config': config},
                                    {'sym': 'Z3', 'config': config},
                                    {'sym': 'U1', 'config': config}])
def test_save_load_mps_hdf5(kwargs):
    save_load_mps_hdf5(**kwargs)

def save_load_mps_hdf5(sym='dense', config=None, tol=1e-12):
    """
    Initialize random MPS and checks saving/loading to/from HDF5 file.
    """
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject backend and device for testing
    ops = yastn.operators.Spin1(sym=sym, **opts_config)
    I = mps.product_mpo(ops.I(), N=16)
    psi = 2 * mps.random_mps(I, D_total=25)  # adding extra factor
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
    # Note: You have to provide valid YASTN configuration
    #
    config = ops.config
    with h5py.File('tmp.h5', 'r') as f:
        phi = mps.load_from_hdf5(config, f, './state/')
    #
    # Test psi == phi
    #
    #assert (psi - phi).norm() < tol and psi is not phi
    assert all((psi[n] - phi[n]).norm() < tol for n in psi.sweep())
    #
    # Similarily, one can save and load MPO
    #
    psi = mps.random_mpo(I, D_total=8)
    with h5py.File('tmp.h5', 'w') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = mps.load_from_hdf5(config, f, './state/')
    os.remove("tmp.h5")
    assert len(psi) == len(phi) and psi is not phi
    assert all((psi[n] - phi[n]).norm() < tol for n in psi.sweep())
    # assert (psi - phi).norm() < tol


@pytest.mark.parametrize('kwargs', [{'sym': 'dense', 'config': config},
                                    {'sym': 'Z3', 'config': config},
                                    {'sym': 'U1', 'config': config}])
def test_save_load_mps_dict(kwargs):
    save_load_mps_dict(**kwargs)

def save_load_mps_dict(sym='dense', config=None, tol=1e-12):
    """
    Initialize random MPS and checks saving/loading to/from npy file.
    """
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject backend and device for testing
    ops = yastn.operators.Spin1(sym=sym, **opts_config)
    I = mps.product_mpo(ops.I(), N=16)
    psi = -0.5 * mps.random_mps(I, D_total=25)  # adding extra factor
    #
    # Next, we serialize MPS into dictionary.
    #
    tmp = psi.save_to_dict()
    #
    # Last, we load the MPS from the dictionary,
    # providing valid YASTN configuration
    #
    config = ops.config
    phi = mps.load_from_dict(config, tmp)
    #
    # Test psi == phi
    #
    assert len(psi) == len(phi) and psi is not phi
    assert all((psi[n] - phi[n]).norm() < tol for n in psi.sweep())
    # assert (psi - phi).norm() < tol


if __name__ == "__main__":
    for sym in ['dense', 'Z3', 'U1']:
        save_load_mps_hdf5(sym=sym)
        save_load_mps_dict(sym=sym)
