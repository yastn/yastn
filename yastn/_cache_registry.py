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
"""
Registry of ``lru_cache``-d functions that live in optional backend modules.

:mod:`yastn.tensor._control_lru` drives the cache-control API from an explicit list of
the pure-Python tensor modules, which are always importable. Backend modules are not.
Instead, they register their own caches when *they* are imported, which is exactly when
that backend is in use. ``_control_lru`` then drives the registry without knowing which
backends exist.

Entries hold ``(module_name, attr)`` strings rather than function objects, because
:func:`set_registered_maxsize` rebinds the module attribute to a fresh wrapper.
"""
from __future__ import annotations

import sys
from functools import lru_cache

__all__ = ['register_cache']

#: Registered caches as ``(module_name, attr, info_key)``.
_REGISTRY: list[tuple[str, str, str]] = []

#: ``maxsize`` of the most recent :func:`set_registered_maxsize`, or ``None`` if the user
#: has not called ``set_cache_maxsize``. Re-applied to backends that register later, so a
#: backend loaded *after* ``set_cache_maxsize`` still gets the requested size.
_MAXSIZE = None


def register_cache(func=None, *, key=None):
    """
    Mark an ``lru_cache``-d backend function for the yastn cache-control API
    (:func:`yastn.set_cache_maxsize`, :func:`yastn.clear_cache`, :func:`yastn.get_cache_info`).

    Apply outermost, above ``@lru_cache``::

        @register_cache
        @lru_cache(maxsize=1024)
        def my_backend_meta(...):
            ...

    ``key`` names the entry in :func:`yastn.get_cache_info`, defaulting to the function name.
    """
    def _register(f):
        _REGISTRY.append((f.__module__, f.__name__, key if key is not None else f.__name__))
        if _MAXSIZE is not None:
            # Registration runs before the ``def`` binds the name, so we cannot rebind the
            # module attribute here -- return the correctly sized wrapper and let the
            # ``def`` bind that instead.
            f = lru_cache(_MAXSIZE)(f.__wrapped__)
        return f
    return _register(func) if func is not None else _register


def _loaded():
    """Yield ``(module, attr, info_key)`` for registered caches whose module is still loaded."""
    for mod_name, attr, info_key in _REGISTRY:
        mod = sys.modules.get(mod_name)  # absent if the module body raised after registering
        if mod is not None:
            yield mod, attr, info_key


def set_registered_maxsize(maxsize):
    """Rebind every registered cache with a fresh ``lru_cache(maxsize)``."""
    global _MAXSIZE
    _MAXSIZE = maxsize
    for mod, attr, _ in _loaded():
        setattr(mod, attr, lru_cache(maxsize)(getattr(mod, attr).__wrapped__))


def clear_registered():
    """Clear every registered cache."""
    for mod, attr, _ in _loaded():
        getattr(mod, attr).cache_clear()


def registered_info():
    """``{info_key: CacheInfo}`` for registered caches; only backends actually loaded appear."""
    return {info_key: getattr(mod, attr).cache_info() for mod, attr, info_key in _loaded()}
