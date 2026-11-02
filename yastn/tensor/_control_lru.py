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
""" Dynamical changing of lru_cache maxsize. """

from . import _algebra, _merging, _contractions, _contractions_cutensor, _einsum, _auxiliary
from ._yastnerror import YastnError
from .._cache_registry import _loaded, resize_cache, set_registered_maxsize
from .._cache_registry import clear_cache_groups, cache_groups_info

__all__ = ['clear_cache', 'get_cache_info', 'set_cache_maxsize']


def _builtin_caches():
    """Core metadata caches; optional and symmetry caches register themselves."""
    return [
        ("fuse_hard", _merging, "_meta_fuse_hard"),
        ("adjacent_swap_coefficient", _merging, "_adjacent_swap_coefficient_cached"),
        ("unfuse_hard", _merging, "_meta_unfuse_hard"),
        ("intersect_hfs", _merging, "_masks_hfs_intersection"),
        ("combine_leg_structure", _merging, "_leg_structure_combine_charges_prod"),
        ("tensordot_f2m", _contractions, "_meta_tensordot_f2m"),
        ("tensordot_fc", _contractions, "_meta_tensordot_fc"),
        ("tensordot_nf", _contractions, "_meta_tensordot_nf"),
        ("tensordot_cutensor_cpu", _contractions, "_meta_tensordot_cutensor_cpu"),
        ("tensordot_cutensor_gpu", _contractions_cutensor, "_meta_tensordot_cutensor_gpu"),
        ("get_trimmed_struct_engine_gpu", _contractions_cutensor, "_get_trimmed_struct_engine_gpu"),
        ("broadcast", _contractions, "_meta_broadcast"),
        ("mask", _merging, "_meta_mask"),
        ("trace", _contractions, "_meta_trace"),
        ("vdot", _contractions, "_meta_vdot"),
        ("swap_gate", _contractions, "_meta_swap_gate"),
        ("swap_gate_charge", _contractions, "_meta_swap_gate_charge"),
        ("ncon", _einsum, "_meta_ncon"),
        ("addition", _algebra, "_meta_addition"),
        ("get_blocks", _auxiliary, "get_blocks"),
        ("get_blocks_charges_all", _auxiliary, "get_blocks_charges_all"),
        ("get_trimmed_struct_engine", _auxiliary, "get_trimmed_struct_engine"),
    ]


def _named_caches():
    """Yield the same LRU caches for clearing, reporting, and resizing."""
    return _builtin_caches() + [(info_key, owner, attr) for owner, attr, info_key in _loaded()]


def set_cache_maxsize(maxsize=0, only=None):
    """
    Rebind yastn metadata caches with a new ``lru_cache`` ``maxsize``, discarding what they hold.

    Parameters
    ----------
    maxsize : int | None | dict
        An ``int`` resizes all managed LRU caches; ``None`` means unbounded. The default is ``0``:
        a bare ``set_cache_maxsize()`` *disables* caching, since ``lru_cache(0)`` stores
        nothing.  A ``dict`` maps cache names -- as reported by :func:`get_cache_info` -- to
        their own ``maxsize``, so one hot cache can be enlarged without paying the memory of
        the others; names that are not listed keep their current size.
    only : iterable of str, optional
        Restrict the plain ``int``/``None`` form to these cache names.

    Default capacities vary by cache. Caches of modules that are not
    loaded yet are covered by the plain form -- the requested size is remembered and applied
    when such a backend registers (see :mod:`yastn._cache_registry`).
    Process-pool dictionary caches keep their own capacity policy and are not
    resized by this function.
    """
    from . import _nonabelian
    if isinstance(maxsize, dict) or only is not None:
        named = {name: (module, attr) for name, module, attr in _named_caches()}
        wanted = dict(maxsize) if isinstance(maxsize, dict) else {name: maxsize for name in only}
        fixed = sorted(set(wanted) & set(cache_groups_info()))
        if fixed:
            raise YastnError(f"set_cache_maxsize: cache capacities managed by process pools: {fixed}")
        unknown = sorted(set(wanted) - set(named) - {'fusion_weights'})
        if unknown:
            raise YastnError(f"set_cache_maxsize: unknown cache name(s) {unknown}; "
                             f"known names are {sorted(named) + ['fusion_weights']}")
        for name, size in wanted.items():
            if name == 'fusion_weights':
                _nonabelian._fusion_weights_cache.set_maxsize(size)
            else:
                module, attr = named[name]
                resize_cache(module, attr, size)
        return
    for _, owner, attr in _builtin_caches():
        resize_cache(owner, attr, maxsize)
    _nonabelian._fusion_weights_cache.set_maxsize(maxsize)
    set_registered_maxsize(maxsize)  # caches of whichever backends are loaded


def clear_cache():
    """
    Drop the contents of every yastn metadata cache, leaving their ``maxsize`` unchanged.

    Includes symmetry coefficients, fusion paths, contraction plans, backend metadata,
    and metadata dictionaries of already-loaded process pools in this process.
    It does not contact workers, drop live autograd graphs, or flush allocators.
    Cached device tensors are released only when no other references remain.
    """
    from . import _nonabelian
    _nonabelian._fusion_weights_cache.cache_clear()
    for _, owner, attr in _named_caches():
        getattr(owner, attr).cache_clear()
    clear_cache_groups()


def get_cache_info():
    """
    Return current-process cache statistics used in yastn.

    Symmetry and optional-module entries (registered via :func:`yastn._cache_registry.register_cache`)
    are present only for modules that are actually loaded, so the key set depends on
    which modules are in use.
    Process-pool dictionary entries report ``None`` for hits/misses, which are
    not tracked. Sizes count entries, not bytes. Allocator memory is not included.
    """
    from . import _nonabelian
    return {"fusion_weights": _nonabelian._fusion_weights_cache.cache_info(),
            **{name: getattr(owner, attr).cache_info() for name, owner, attr in _named_caches()},
            **cache_groups_info()}
