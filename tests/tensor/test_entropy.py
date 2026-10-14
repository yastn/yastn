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
""" yastn.linalg.entropy() """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_entropy(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    P = yastn.eye(config=config_U1, legs=yastn.Leg(config_U1, s=1, t=(-2, -1, 0), D=(4, 12, 25)))

    # here P does not sum to 1.
    # This gets normalized during calculation of entropy
    entropy = yastn.entropy(P)
    assert pytest.approx(entropy.item(), rel=tol) == np.log2(41)

    entropy = yastn.entropy(P, alpha=2)
    assert pytest.approx(entropy.item(), rel=tol) == np.log2(41)

    # zero tensor
    assert yastn.entropy(P * 0) == 0

    #  empty tensor
    b = yastn.Tensor(config=config_U1, s=(1, -1), isdiag=True)
    assert yastn.entropy(b) == 0


    with pytest.raises(yastn.YastnError):
        Pnondiag = P.diag()
        entropy = yastn.entropy(Pnondiag)
        # yastn.linalg.entropy requires diagonal tensor.

    with pytest.raises(yastn.YastnError):
        entropy = yastn.entropy(P, alpha=-2)
        # yastn.linalg.entropy requires positive order alpha.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
