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
""" yastn.make_config(); catching some config mismatches between tensors. """
import pytest
import yastn


def test_config_exceptions(config_kwargs):
    """ Handling mismatches of tensor configs when combining two tensors. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    config_Z2 = yastn.make_config(sym='Z2', **config_kwargs)
    config_Z2_f = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)

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
        b = yastn.rand(config=config_Z2_f, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        _ = a + b
        # Two tensors have different assignment of fermionic statistics.


def test_config_exceptions_torch(config_kwargs):
    """ Mismatches requiring different backends or devices. """
    config = yastn.make_config(sym='U1', **config_kwargs)
    config_np = yastn.make_config(sym='U1', backend='np')

    if config.backend != config_np.backend:
        a = yastn.rand(config=config, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        b = yastn.rand(config=config_np, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        with pytest.raises(yastn.YastnError):
            _ = a + b
            # Two tensors have different backends.


def test_make_config(config_kwargs):
    """ Parameters in yastn.make_config(). """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    assert config_U1.sym.SYM_ID == 'U1'

    with pytest.raises(yastn.YastnError):
        yastn.make_config(sym="random_name")
        #  sym encoded as string only supports: 'dense', 'Z2', 'Z3', 'U1', 'U1xU1', 'U1xU1xZ2'

    with pytest.raises(yastn.YastnError):
        yastn.make_config(backend="random_name")
        # backend encoded as string only supports: 'np', 'torch'


if __name__ == '__main__':
    # pytest.main([__file__, "-vs", "--durations=0"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--device", "cuda"])
