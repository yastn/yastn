# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
""" basic procedures of single Mps """
import pytest
import yastn
import yastn.tn.mps as mps


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
    tmp = psi.to_dict(level=2)
    #
    # Last, we load the MPS from the dictionary,
    # providing valid YASTN configuration
    #
    phi1 = mps.MpsMpoOBC.from_dict(tmp)
    phi2 = yastn.from_dict(tmp)
    #
    # Test psi == phi
    #
    assert (psi - phi1).norm() < tol * psi.norm()
    assert (psi - phi2).norm() < tol * psi.norm()
    #
    # Similarly for MPO
    #
    psi = -1j * mps.random_mpo(I, D_total=11)  # adding extra complex factor
    psi.canonize_(to='last', normalize=False)  # extra cannonization
    psi.canonize_(to='first', normalize=False)  # retaining the norm
    tmp = psi.to_dict(level=2)
    phi1 = mps.MpsMpoOBC.from_dict(tmp)
    phi2 = yastn.from_dict(tmp)
    assert (psi - phi1).norm() < tol * psi.norm()
    assert (psi - phi2).norm() < tol * psi.norm()

    psi_pbc = mps.MpoPBC(N=len(psi))
    for n in psi_pbc.sweep():
        psi_pbc[n] = psi[n]
    tmp3 = psi_pbc.to_dict()
    phi_pbc1 = mps.MpoPBC.from_dict(tmp3)
    phi_pbc2 = yastn.from_dict(tmp3)
    assert type(phi_pbc1) is mps.MpoPBC
    assert type(phi_pbc2) is mps.MpoPBC

    with pytest.raises(yastn.YastnError, match="does not match"):
        mps.MpoPBC.from_dict(tmp)
        # MpoPBC does not match d['type'] == MpsMpoOBC
    with pytest.raises(yastn.YastnError, match="unsupported dict_ver"):
        mps.MpsMpoOBC.from_dict({**tmp, 'dict_ver': -1})
        # MpsMpoOBC.from_dict: unsupported dict_ver == -1.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
