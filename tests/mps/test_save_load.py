# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" basic procedures of single Mps """
import os
import pytest
import yastn
import yastn.tn.mps as mps


@pytest.mark.parametrize('sym', ['dense', 'Z3', 'U1'])
def test_save_load_mps_hdf5(config_kwargs, sym, tol=1e-12):
    """
    Initialize random MPS and checks saving/loading to/from HDF5 file.
    """
    h5py = pytest.importorskip('h5py')
    #
    # generate random mps with 3-dimensional local spaces
    #
    ops = yastn.operators.Spin1(sym=sym, **config_kwargs)
    I = mps.product_mpo(ops.I(), N=31)
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
    assert (psi - phi).norm() < tol * psi.norm()
    #
    # Similarily, one can save and load MPO
    #
    psi = -1j * mps.random_mpo(I, D_total=8, dtype='complex128')
    psi.canonize_(to='last', normalize=False)  # extra cannonization
    psi.canonize_(to='first', normalize=False)  # retaining the norm
    with h5py.File('tmp.h5', 'w') as f:
        psi.save_to_hdf5(f, 'state/')
    with h5py.File('tmp.h5', 'r') as f:
        phi = mps.load_from_hdf5(config, f, './state/')
    os.remove("tmp.h5")
    assert (psi - phi).norm() < tol * psi.norm()


@pytest.mark.parametrize('sym', ['dense', 'Z3', 'U1'])
def test_save_load_mps_dict(config_kwargs, sym, tol=1e-12):
    """
    Initialize random MPS and checks saving/loading to/from npy file.
    """
    #
    # generate random mps with 3-dimensional local spaces
    #
    ops = yastn.operators.Spin1(sym=sym, **config_kwargs)
    I = mps.product_mpo(ops.I(), N=31)
    psi = -0.5 * mps.random_mps(I, D_total=25, dtype='complex128')
    #
    # Next, we serialize MPS into dictionary.
    #
    tmp = psi.save_to_dict()
    tmp2 = psi.to_dict(level=2)
    #
    # Last, we load the MPS from the dictionary,
    # providing valid YASTN configuration
    #
    config = ops.config
    phi = mps.load_from_dict(config, tmp)
    phi2 = mps.MpsMpoOBC.from_dict(tmp2)
    #
    # Test psi == phi
    #
    assert (psi - phi).norm() < tol * psi.norm()
    assert (psi - phi2).norm() < tol * psi.norm()
    #
    # Similarly for MPO
    #
    psi = -1j * mps.random_mpo(I, D_total=11)  # adding extra complex factor
    psi.canonize_(to='last', normalize=False)  # extra cannonization
    psi.canonize_(to='first', normalize=False)  # retaining the norm
    tmp = psi.save_to_dict()
    phi = mps.load_from_dict(config, tmp)
    tmp2 = psi.to_dict(level=2)
    phi2 = mps.MpsMpoOBC.from_dict(tmp2)
    assert (psi - phi).norm() < tol * psi.norm()
    assert (psi - phi2).norm() < tol * psi.norm()


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
