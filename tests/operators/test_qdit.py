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
