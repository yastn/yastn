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
""" change device/dtype with yastn.to()"""
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_to(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(1, 2, 3))
    ta = yastn.rand(config=config_U1, legs=[leg, leg, leg.conj()], dtype='float64', device='cpu')

    tb = ta.to(dtype='complex128')
    assert tb.yastn_dtype == 'complex128'
    assert tb.dtype == config_U1.backend.DTYPE['complex128']
    assert tb.is_consistent()

    if config_U1.backend.cuda_is_available():
        tc = ta.to(device='cuda:0')
        assert tc.device == 'cuda:0'
        assert tc.yastn_dtype == 'float64'
        assert tc.dtype == config_U1.backend.DTYPE['float64']
        assert tc.is_consistent()

        td = ta.to(device='cuda:0', dtype='complex128')
        assert td.device == 'cuda:0'
        assert td.yastn_dtype == 'complex128'
        assert td.dtype == config_U1.backend.DTYPE['complex128']
        assert td.is_consistent()


if __name__ == '__main__':
    # pytest.main([__file__, "-vs", "--durations=0"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
