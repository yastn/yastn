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
import pathlib
import subprocess
import sys
import numpy as np
import pytest
import yastn
from yastn.tensor._auxiliary import get_blocks, hash_blocks
from ._nonabelian_utils import four_leg_tensor

tol = 1e-12  #pylint: disable=invalid-name


def _run_cache_nonabelian(config_kwargs, sym):
    a = four_leg_tensor(yastn.make_config(sym=sym, **config_kwargs))
    yastn.clear_cache()
    for _ in range(3):
        a.svd(axes=((0, 1), (2, 3)))
    assert any(info.hits > 0 for info in yastn.get_cache_info().values())


def test_cache_SU2(config_kwargs):
    _run_cache_nonabelian(config_kwargs, 'SU2')


def test_cache_SU2xU1(config_kwargs):
    _run_cache_nonabelian(config_kwargs, 'SU2xU1')


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


def test_fusion_weights_cache(config_kwargs):
    """The fusion-weights cache keys on an encoded meta: no aliasing, same values."""
    from array import array
    from yastn.tensor import _nonabelian

    def _freeze(x):
        if isinstance(x, slice):
            return (x.start, x.stop)
        return tuple(_freeze(y) for y in x) if isinstance(x, (tuple, list)) else x

    def _decode(key):
        flat = array('q')
        flat.frombytes(key)
        pos = 0

        def read():
            nonlocal pos
            tag = flat[pos]
            pos += 1
            if tag == _nonabelian._TAG_NONE:
                return None
            if tag == _nonabelian._TAG_SLICE:
                start, stop = flat[pos], flat[pos + 1]
                pos += 2
                return slice(start, stop)
            if tag == _nonabelian._TAG_SEQ:
                count = flat[pos]
                pos += 1
                return tuple(read() for _ in range(count))
            assert tag == _nonabelian._TAG_INT
            value = flat[pos]
            pos += 1
            return value

        value = _freeze(read())
        assert pos == len(flat), "encoded key has trailing fields"
        return value

    # collect the (sym, struct, fused, axes, meta) of every hard fuse/unfuse step
    calls, scale = [], _nonabelian.fusion_scale

    def probe(config, data, struct, fused, axes, meta, inverse=False):
        calls.append((config.sym, struct, fused, axes, meta))
        return scale(config, data, struct, fused, axes, meta, inverse=inverse)

    _nonabelian.fusion_scale = probe
    try:
        a = four_leg_tensor(yastn.make_config(sym='SU2xU1', **config_kwargs))
        b = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        b.unfuse_legs(axes=(0, 1))
    finally:
        _nonabelian.fusion_scale = scale
    assert calls

    keys = []
    for sym, struct, fused, axes, meta in calls:
        key = _nonabelian.meta_key(meta)
        assert isinstance(key, bytes), "real metas must encode, not fall back to the tuple key"
        assert _decode(key) == _freeze(meta), "encoding must be lossless"
        # the cached value must equal the uncached computation
        cached = _nonabelian.fusion_weights(sym, struct, fused, axes, meta)
        direct = _nonabelian._compute_fusion_weights(sym, struct, fused, axes, meta)
        assert all(wc == wd for wc, wd in zip(cached[0], direct[0]))
        assert all(lc == ld for lc, ld in zip(cached[1], direct[1]))
        keys.append(key)

    # distinct metas must not share a key
    assert len(set(keys)) == len({_freeze(meta) for *_, meta in calls})
    assert _nonabelian._fusion_weights_cache.cache_info().hits > 0

    # structural equality: list/tuple and int/np.integer spellings encode identically
    meta = calls[0][4]
    assert _nonabelian.meta_key(meta) == _nonabelian.meta_key(tuple(list(rec) for rec in meta))
    assert _nonabelian.meta_key((np.int64(5),)) == _nonabelian.meta_key((5,))

    # a field that cannot be encoded falls back to the exact tuple, which never aliases
    plain = ((slice(0, 2), (1, 2), slice(0, 1), (), (slice(0, 1),), ()),)
    assert isinstance(_nonabelian.meta_key(plain), bytes)
    assert _nonabelian.meta_key(plain) == _nonabelian.meta_key(list(plain))
    assert _decode(_nonabelian.meta_key(plain)) == _freeze(plain)
    out_of_range = ((slice(0, 2), (1 << 70,)),)
    assert _nonabelian.meta_key(out_of_range) == _freeze(out_of_range)  # frozen fallback
    assert isinstance(hash(_nonabelian.meta_key(out_of_range)), int)
    assert _decode(_nonabelian.meta_key(((None,),))) == ((None,),)  # None has its own tag

    variants = []
    for shift in range(6):
        record = list(plain[0])
        record[shift] = (shift + 7,)
        variants.append((tuple(record),))
    assert len({_nonabelian.meta_key(v) for v in variants}) == len(variants)


#: Child program for test_import_and_cache_api_without_torch. Runs in a subprocess so the
#: torch blocker cannot pollute the sys.modules of the test session itself.
_NO_TORCH_PROGRAM = """
import sys, importlib.abc
sys.path.insert(0, %r)

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ImportError("No module named %%r (simulated)" %% name)
        return None

sys.meta_path.insert(0, BlockTorch())

import yastn
config = yastn.make_config(sym='Z2')   # no backend= kwarg -> numpy default
a = yastn.rand(config=config, s=(-1, 1, 1, -1), t=((0, 1),) * 4,
               D=((1, 2), (2, 3), (3, 4), (4, 5)))
a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
yastn.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))

yastn.set_cache_maxsize(maxsize=10)
info = yastn.get_cache_info()
yastn.clear_cache()

assert info["broadcast"].maxsize == 10, info["broadcast"]
# backend-side caches register themselves; the torch one must be absent here
assert "pack_transpose_and_merge_params" not in info, sorted(info)
assert 'torch' not in sys.modules, "the cache API pulled in torch"
"""


def test_import_and_cache_api_without_torch():
    """
    ``import yastn`` and the whole cache-control API must work with no torch installed.

    Regression guard: ``_control_lru`` used to import ``_backend_torch_backwards`` at module
    level, which made torch a hard dependency of ``import yastn``. Backend-side caches now
    register themselves through :mod:`yastn._cache_registry` when their backend is imported.
    """
    root = str(pathlib.Path(yastn.__file__).parents[1])
    out = subprocess.run([sys.executable, '-c', _NO_TORCH_PROGRAM % root],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


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
