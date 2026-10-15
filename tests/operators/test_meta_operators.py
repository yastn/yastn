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
import pytest
import yastn


def test_meta_operators(config_kwargs):
    """ Parent class for defining operator classes """

    # allows initializing config
    ops = yastn.operators.meta_operators(**config_kwargs)

    # desired signature of matrix operators
    assert ops.s == (1, -1)

    # default _sym follows from make_config()
    assert ops._sym == 'dense'

    # provides a short-cut to set the seed of random number generator in the backend
    ops.random_seed(seed=0)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
