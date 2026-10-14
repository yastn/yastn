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
""" Predefined dense qdit operator. """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_qdit(config_kwargs, d=5):
    """ Standard operators and some vectors in two-dimensional Hilbert space for various symmetries. """
    ops_dense = yastn.operators.Qdit(d=d, **config_kwargs)

    I = ops_dense.I()
    leg = ops_dense.space()

    assert leg == I.get_legs(axes=0)
    assert np.allclose(I.to_numpy(), np.eye(d))
    assert config_kwargs["default_device"] in I.device  # accept 'cuda' in 'cuda:0'

    assert ops_dense.key() is None  # Qdit has no symmetry, so key is None

    # used in mps Generator
    dictionary = ops_dense.to_dict()
    (dictionary["I"](3) - I).norm() < tol  # here 3 is a posible position in the mps
    assert all(k in dictionary for k in ('I',))

    # override default d
    local_d = d + 4
    I = ops_dense.I(d=local_d)
    leg = ops_dense.space(d=local_d)
    assert leg == I.get_legs(axes=0)
    assert np.allclose(I.to_numpy(), np.eye(local_d))

    # wrong init with config
    with pytest.raises(yastn.YastnError):
        config_U1 = yastn.make_config(sym="U1", **config_kwargs)
        yastn.operators.Qdit(d=5, **config_U1._asdict())
        # For Qdit sym should be 'dense'.
    with pytest.raises(yastn.YastnError):
        config_fer = yastn.make_config(fermionic=True, **config_kwargs)
        yastn.operators.Qdit(d=5, **config_fer._asdict())
        # For Qdit config.fermionic should be False.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
