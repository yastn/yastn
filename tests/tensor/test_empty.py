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
""" Test tensor operations on an empty tensor """
import yastn
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_empty_tensor():
    """ Test some tensor operations on an empty tensor. """
    a = yastn.Tensor(config=config_U1, s=(1, 1, -1, -1))

    assert a.norm() < tol

    b = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    c = b.unfuse_legs(axes=(0, 1))
    assert (a - c).norm() < tol

    d = yastn.tensordot(a, a, axes=((2, 3), (0, 1)))
    assert (a - d).norm() < tol

    assert a.item() == 0.


if __name__ == '__main__':
    test_empty_tensor()
