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
Registry of ``lru_cache``-d functions and symmetry class methods.

:mod:`yastn.tensor._control_lru` drives the cache-control API from an explicit list of
the pure-Python tensor modules, which are always importable. Backend modules are not.
Instead, they register their own caches when *they* are imported, which is exactly when
that backend is in use. ``_control_lru`` then drives the registry without knowing which
backends exist.

Entries hold ``(module_name, qualified_name)`` strings rather than function objects, because
:func:`set_registered_maxsize` rebinds the module attribute to a fresh wrapper.
"""
from __future__ import annotations

import sys
from collections import namedtuple
from functools import lru_cache

__all__ = ['register_cache', 'register_cache_group']

#: Registered caches as ``(module_name, qualified_name, info_key)``.
_REGISTRY: list[tuple[str, str, str]] = []

# Non-LRU caches owned by optional process-pool modules. Hooks inspect only
# current-process metadata; they must not create pools or contact workers.
_CACHE_GROUPS = []
CacheInfo = namedtuple('CacheInfo', 'hits misses maxsize currsize')


def register_cache_group(clear, info):
    """Register clear/info hooks for caches with their own capacity policy."""
    _CACHE_GROUPS.append((clear.__module__, clear.__name__, info.__name__))


def clear_cache_groups():
    for module_name, clear, _ in _CACHE_GROUPS:
        module = sys.modules.get(module_name)
        if module is not None:
            getattr(module, clear)()


def cache_groups_info():
    result = {}
    for module_name, _, info in _CACHE_GROUPS:
        module = sys.modules.get(module_name)
        if module is not None:
            result.update(getattr(module, info)())
    return result


#: ``maxsize`` of the most recent :func:`set_registered_maxsize`, or ``_UNSET`` if the user
#: has not called ``set_cache_maxsize``. Re-applied to backends that register later, so a
#: backend loaded *after* ``set_cache_maxsize`` still gets the requested size.
_UNSET = object()
_MAXSIZE = _UNSET


def register_cache(func=None, *, key=None):
    """
    Mark an ``lru_cache``-d function for the yastn cache-control API
    (:func:`yastn.set_cache_maxsize`, :func:`yastn.clear_cache`, :func:`yastn.get_cache_info`).

    Apply outermost, above ``@lru_cache``::

        @register_cache
        @lru_cache(maxsize=1024)
        def my_backend_meta(...):
            ...

    ``key`` names the entry in :func:`yastn.get_cache_info`, defaulting to the function name.
    For class methods, place ``@classmethod`` above ``@register_cache``.
    """
    def _register(f):
        _REGISTRY.append((f.__module__, f.__qualname__, key if key is not None else f.__name__))
        if _MAXSIZE is not _UNSET:
            # Registration runs before the ``def`` binds the name, so we cannot rebind the
            # module attribute here -- return the correctly sized wrapper and let the
            # ``def`` bind that instead.
            f = lru_cache(_MAXSIZE)(f.__wrapped__)
        return f
    return _register(func) if func is not None else _register


def _loaded():
    """Yield ``(owner, attr, info_key)`` for loaded module functions or class methods."""
    for mod_name, qualified_name, info_key in _REGISTRY:
        owner = sys.modules.get(mod_name)
        *parents, attr = qualified_name.split('.')
        for parent in parents:
            owner = getattr(owner, parent, None)
        # A module or class body may still be importing, or may have failed.
        if owner is not None and hasattr(owner, attr):
            yield owner, attr, info_key


def resize_cache(owner, attr, maxsize):
    """Replace an LRU wrapper without losing a classmethod descriptor."""
    descriptor = vars(owner)[attr]
    function = descriptor.__func__ if isinstance(descriptor, classmethod) else descriptor
    wrapper = lru_cache(maxsize)(function.__wrapped__)
    function.cache_clear()
    setattr(owner, attr, classmethod(wrapper) if isinstance(descriptor, classmethod) else wrapper)
    # Internal `from ... import ...` aliases must use the new wrapper too;
    # otherwise they refill an orphan cache that clear/info no longer see.
    for name, module in tuple(sys.modules.items()):
        if name.startswith('yastn.') and module is not None:
            for alias, value in tuple(vars(module).items()):
                if value is function:
                    setattr(module, alias, wrapper)


def set_registered_maxsize(maxsize):
    """Rebind every registered cache with a fresh ``lru_cache(maxsize)``."""
    global _MAXSIZE
    _MAXSIZE = maxsize
    for mod, attr, _ in _loaded():
        resize_cache(mod, attr, maxsize)


def clear_registered():
    """Clear every registered cache."""
    for mod, attr, _ in _loaded():
        getattr(mod, attr).cache_clear()


def registered_info():
    """``{info_key: CacheInfo}`` for registered caches; only backends actually loaded appear."""
    return {info_key: getattr(mod, attr).cache_info() for mod, attr, info_key in _loaded()}
