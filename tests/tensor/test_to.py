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
