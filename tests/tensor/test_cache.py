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
""" changing tests controls and size of lru_cache in some auxiliary functions """
import numpy as np
import pytest
import yastn
from yastn.tensor._auxiliary import get_blocks, hash_blocks

tol = 1e-12  #pylint: disable=invalid-name


def test_cache(config_kwargs):
    config = yastn.make_config(sym='Z2', **config_kwargs)
    a = yastn.rand(config=config, s=(-1, 1, 1, -1),
                  t=((0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (2, 3), (3, 4), (4, 5)))
    for _ in range(100):
        a.svd(axes=((0, 1), (2, 3)))
        a.svd(axes=((0, 2), (1, 3)))
        a.svd(axes=((1, 3), (2, 0)))

    yastn.set_cache_maxsize(maxsize=10)
    cache_info = yastn.get_cache_info()

    for _ in range(100):
        a.svd(axes=((0, 1), (2, 3)))
        a.svd(axes=((0, 2), (1, 3)))
        a.svd(axes=((1, 3), (2, 0)))

    b = yastn.eye(config=config, t=(0, 1), D=(4, 5))
    for _ in range(100):
        b.broadcast(a, axes=3)

    cache_info = yastn.get_cache_info()
    assert cache_info["broadcast"] == (99, 1, 10, 1)
    yastn.clear_cache()
    cache_info = yastn.get_cache_info()
    assert cache_info["broadcast"] == (0, 0, 10, 0)


def test_hash_blocks(config_kwargs):
    config = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config, s=(1, -1),
                   t=((0, 1, 2), (0, 1, 2)),
                   D=((2, 3, 4), (2, 3, 4)))
    bl = get_blocks(a.config.sym, a.struct)

    h = hash_blocks(bl)
    # blake2b hexdigest: 128 hex chars, deterministic across processes
    assert isinstance(h, str) and len(h) == 128

    # stable: recomputing the same blocks gives the same digest
    assert hash_blocks(get_blocks(a.config.sym, a.struct)) == h

    # contiguity is canonicalized: an F-ordered copy yields the same digest
    bl_f = bl._replace(t=np.asfortranarray(bl.t))
    assert hash_blocks(bl_f) == h

    # a change in any field changes the digest
    assert hash_blocks(bl._replace(size=bl.size + 1)) != h
    assert hash_blocks(bl._replace(t=bl.t + 1)) != h

    # distinct tensor structure -> distinct digest
    b = yastn.rand(config=config, s=(1, -1),
                   t=((0, 1), (0, 1)),
                   D=((2, 3), (2, 3)))
    assert hash_blocks(get_blocks(b.config.sym, b.struct)) != h


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
