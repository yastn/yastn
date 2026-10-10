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
""" Test tensor operations on an empty tensor. """
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_empty_tensor(config_kwargs):
    """ Test some tensor operations on an empty tensor. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.Tensor(config=config_U1, s=(1, 1, -1, -1))

    assert a.norm() < tol

    b = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    c = b.unfuse_legs(axes=(0, 1))
    assert (a - c).norm() < tol

    d = yastn.tensordot(a, a, axes=((2, 3), (0, 1)))
    assert (a - d).norm() < tol

    assert a.item() == 0.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
