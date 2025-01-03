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
