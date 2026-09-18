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
""" yastn.linalg.entropy() """
import numpy as np
import pytest
import yastn
from ._nonabelian_utils import matrix_tensor

tol = 1e-12  #pylint: disable=invalid-name


def _run_entropy_nonabelian(config_kwargs, sym):
    config = yastn.make_config(sym=sym, **config_kwargs)
    _, S, _ = matrix_tensor(config).svd(axes=(0, 1))
    assert yastn.entropy(S) >= 0
    assert yastn.entropy(S, alpha=2) >= 0


def test_entropy_SU2(config_kwargs):
    _run_entropy_nonabelian(config_kwargs, 'SU2')


def test_entropy_SU2xU1(config_kwargs):
    _run_entropy_nonabelian(config_kwargs, 'SU2xU1')


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
