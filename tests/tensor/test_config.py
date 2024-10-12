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
import pytest
import yastn
try:
    from .configs import config_U1, config_Z2, config_Z2_fermionic
except ImportError:
    from configs import config_U1, config_Z2, config_Z2_fermionic


def test_config_exceptions():
    """ handling mismatches of tensor configs when combining two tensors"""
    leg_U1 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(1, 2))
    leg_Z2 = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(1, 2))
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg_U1, leg_U1, leg_U1.conj()])
        b = yastn.rand(config=config_Z2, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        _ = a + b
        # Two tensors have different symmetry rules.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_Z2, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        # leg do not depend on fermionic statistics so the next line is fine
        b = yastn.rand(config=config_Z2_fermionic, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        _ = a + b
        # Two tensors have different assignment of fermionic statistics.


@pytest.mark.skipif(config_U1.backend.BACKEND_ID=="numpy", reason="requires different backends or devices")
def test_config_exceptions_torch():
    """ mismatches requiring different backends or devices"""
    config_np = yastn.make_config(sym='U1', backend='np')
    config_torch = yastn.make_config(sym='U1', backend='torch')
    a = yastn.rand(config=config_torch, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
    with pytest.raises(yastn.YastnError):
        b = yastn.rand(config=config_np, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        _ = a + b
        # Two tensors have different backends.
    if config_U1.backend.torch.cuda.is_available():
        with pytest.raises(yastn.YastnError):
            a = a.to(device='cpu')
            b = yastn.rand(config=config_torch, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
            b = b.to(device='cuda')
            _ = a + b
            # Devices of the two tensors do not match.


def test_make_config():
    cfg_U1 = yastn.make_config(sym="U1", backend=config_U1.backend)
    assert cfg_U1.sym == config_U1.sym

    with pytest.raises(yastn.YastnError):
        yastn.make_config(sym="random_name")
        #  sym encoded as string only supports: 'dense', 'Z2', 'Z3', 'U1', 'U1xU1', 'U1xU1xZ2'

    with pytest.raises(yastn.YastnError):
        yastn.make_config(backend="random_name")
        # backend encoded as string only supports: 'np', 'torch'

if __name__ == '__main__':
    # test_config_exceptions()
    # test_config_exceptions_torch()
    test_make_config()
