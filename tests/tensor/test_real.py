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
""" yastn.real yastn.imag() """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_real_basic(config_kwargs):
    """ real and imag parts of tensor"""
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config_U1, s=1, t=(0, 1), D=(1, 2))

    a = yastn.rand(config=config_U1, legs=[leg, leg.conj()], dtype='complex128')

    assert a.is_complex()
    assert np.iscomplexobj(a.to_numpy())

    for b in (a.real(), a.imag()):
        assert np.isrealobj(b.to_numpy())  # to_dense, to_numpy use config.dtype
        assert not b.is_complex()

    c = yastn.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))  # ndim = 0
    num_c = c.to_number()  # dtype same as that of data in tensor c
    num_r1 = c.real().to_number()  # dtype same as that of data in tensor c.real()
    num_r2 = c.to_number('real')  # takes real part of to_number (backend ascetic)

    assert isinstance(num_c.item(), complex)
    assert isinstance(num_r1.item(), float)
    assert isinstance(num_r2.item(), float)
    assert abs(num_c.real - num_r1) < tol
    assert abs(num_c.real - num_r2) < tol

    ar = a.real()
    ai = a.imag()
    assert not yastn.are_independent(ar, a)  # may share data
    assert not yastn.are_independent(ai, a)  # may share data
    assert yastn.norm(ar + 1j * ai - a) < tol


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
