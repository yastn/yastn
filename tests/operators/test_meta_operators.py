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
import yastn
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


def test_meta_operators():
    """ Parent class for defining operator classes """
    # pytest switches backends and default_device in config files for testing
    backend = config_dense.backend
    default_device = config_dense.default_device

    # allows initializing config
    ops = yastn.operators.meta_operators(backend=backend, default_device=default_device)

    # desired signature of matrix operators
    assert ops.s == (1, -1)

    # provides a short-cut to set the seed of random number generator in the backend
    ops.random_seed(seed=0)


if __name__ == '__main__':
    test_meta_operators()
