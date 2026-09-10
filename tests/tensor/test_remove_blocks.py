# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
""" yastn.remove_random_blocks()  yastn.remove_zero_blocks() """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_remove_random_blocks(config_kwargs):
    """ testig remove_random_blocks """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    #
    legs = [yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(1, 2, 3))]
    a = yastn.rand(config=config_U1, legs=legs)
    assert a.nblocks == 13
    assert a.size == 220
    b = a.remove_random_blocks(number=5, keep_legs = True)
    assert b.nblocks < a.nblocks
    assert b.size < a.size
    assert a.get_legs() == b.get_legs()
    c = b.remove_random_blocks(number=5, keep_legs = True)
    assert c.nblocks <= b.nblocks
    assert c.size <= b.size
    assert a.get_legs() == c.get_legs()
    assert np.array_equal(b.struct.mask.array * c.struct.mask.array, c.struct.mask.array)


def test_remove_zero_blocks(config_kwargs):
    """ testig remove_zero_blocks """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    #
    legs = [yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(1, 2, 3))]
    a = yastn.rand(config=config_U1, legs=legs)
    assert a.nblocks == 13
    assert a.size == 220

    target_blocks = [(2, 1, 0, 1),
                     (0, 0, 0, 0)]

    ts = target_blocks[0]
    a[ts] = a[ts] * 0
    b = a.remove_zero_blocks()
    assert b.nblocks == 12
    assert b.size == 220 - 54
    assert (b - a).norm() < tol

    ts = target_blocks[1]
    b[ts] = b[ts] * 0
    c = b.remove_zero_blocks()
    assert c.nblocks == 11
    assert c.size == 220 - 54 - 16
    assert (c - b).norm() < tol

    assert all(ts not in c for ts in target_blocks)

    assert a.struct.mask.array is None
    assert b.struct.mask.array is not None
    assert c.struct.mask.array is not None


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
