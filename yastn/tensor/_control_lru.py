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
from functools import lru_cache

from . import _algebra, _merging, _contractions, _contractions_cutensor, _einsum, _auxiliary
from ._yastnerror import YastnError
from .._cache_registry import _loaded, clear_registered, registered_info, set_registered_maxsize

__all__ = ['clear_cache', 'get_cache_info', 'set_cache_maxsize']


def _named_caches():
    """ ``(info_key, module, attr)`` of every lru cache that :func:`get_cache_info` reports. """
    pure = [
        ("fuse_hard", _merging, "_meta_fuse_hard"),
        ("adjacent_swap_coefficient", _merging, "_adjacent_swap_coefficient"),
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
        ("mask", _contractions, "_meta_mask"),
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
    return pure + [(info_key, module, attr) for module, attr, info_key in _loaded()]


def set_cache_maxsize(maxsize=0, only=None):
    """
    Rebind yastn metadata caches with a new ``lru_cache`` ``maxsize``, discarding what they hold.

    Parameters
    ----------
    maxsize : int | None | dict
        An ``int`` rebinds every cache; ``None`` means unbounded.  Note the default is ``0``:
        a bare ``set_cache_maxsize()`` *disables* caching, since ``lru_cache(0)`` stores
        nothing.  A ``dict`` maps cache names -- as reported by :func:`get_cache_info` -- to
        their own ``maxsize``, so one hot cache can be enlarged without paying the memory of
        the others; names that are not listed keep their current size.
    only : iterable of str, optional
        Restrict the plain ``int``/``None`` form to these cache names.

    Each builder is created with ``maxsize=1024``.  Caches of backend modules that are not
    loaded yet are covered by the plain form -- the requested size is remembered and applied
    when such a backend registers (see :mod:`yastn._cache_registry`).
    """
    from . import _nonabelian
    if isinstance(maxsize, dict) or only is not None:
        named = {name: (module, attr) for name, module, attr in _named_caches()}
        wanted = dict(maxsize) if isinstance(maxsize, dict) else {name: maxsize for name in only}
        unknown = sorted(set(wanted) - set(named) - {'fusion_weights'})
        if unknown:
            raise YastnError(f"set_cache_maxsize: unknown cache name(s) {unknown}; "
                             f"known names are {sorted(named) + ['fusion_weights']}")
        for name, size in wanted.items():
            if name == 'fusion_weights':
                _nonabelian._fusion_weights_cache.set_maxsize(size)
            else:
                module, attr = named[name]
                setattr(module, attr, lru_cache(size)(getattr(module, attr).__wrapped__))
        return
    _contractions._meta_broadcast = lru_cache(maxsize)(_contractions._meta_broadcast.__wrapped__)
    _contractions._meta_tensordot_f2m = lru_cache(maxsize)(_contractions._meta_tensordot_f2m.__wrapped__)
    _contractions._meta_tensordot_fc = lru_cache(maxsize)(_contractions._meta_tensordot_fc.__wrapped__)
    _contractions._meta_tensordot_nf = lru_cache(maxsize)(_contractions._meta_tensordot_nf.__wrapped__)
    _contractions._meta_tensordot_cutensor_cpu = lru_cache(maxsize)(_contractions._meta_tensordot_cutensor_cpu.__wrapped__)
    _contractions_cutensor._meta_tensordot_cutensor_gpu = lru_cache(maxsize)(_contractions_cutensor._meta_tensordot_cutensor_gpu.__wrapped__)
    _contractions_cutensor._get_trimmed_struct_engine_gpu = lru_cache(maxsize)(_contractions_cutensor._get_trimmed_struct_engine_gpu.__wrapped__)
    _contractions._meta_mask = lru_cache(maxsize)(_contractions._meta_mask.__wrapped__)
    _contractions._meta_swap_gate = lru_cache(maxsize)(_contractions._meta_swap_gate.__wrapped__)
    _contractions._meta_swap_gate_charge = lru_cache(maxsize)(_contractions._meta_swap_gate_charge.__wrapped__)
    _contractions._meta_trace = lru_cache(maxsize)(_contractions._meta_trace.__wrapped__)
    _contractions._meta_vdot = lru_cache(maxsize)(_contractions._meta_vdot.__wrapped__)
    _einsum._meta_ncon = lru_cache(maxsize)(_einsum._meta_ncon.__wrapped__)
    _merging._masks_hfs_intersection = lru_cache(maxsize)(_merging._masks_hfs_intersection.__wrapped__)
    _merging._leg_structure_combine_charges_prod = lru_cache(maxsize)(_merging._leg_structure_combine_charges_prod.__wrapped__)
    _merging._meta_fuse_hard = lru_cache(maxsize)(_merging._meta_fuse_hard.__wrapped__)
    _merging._meta_unfuse_hard = lru_cache(maxsize)(_merging._meta_unfuse_hard.__wrapped__)
    _merging._adjacent_swap_coefficient = lru_cache(maxsize)(_merging._adjacent_swap_coefficient.__wrapped__)
    _algebra._meta_addition = lru_cache(maxsize)(_algebra._meta_addition.__wrapped__)
    _auxiliary.get_blocks = lru_cache(maxsize)(_auxiliary.get_blocks.__wrapped__)
    _auxiliary.get_blocks_charges_all = lru_cache(maxsize)(_auxiliary.get_blocks_charges_all.__wrapped__)
    _auxiliary.get_trimmed_struct_engine = lru_cache(maxsize)(_auxiliary.get_trimmed_struct_engine.__wrapped__)
    _nonabelian._fusion_weights_cache.set_maxsize(maxsize)
    set_registered_maxsize(maxsize)  # caches of whichever backends are loaded


def clear_cache():
    """
    Drop the contents of every yastn metadata cache, leaving their ``maxsize`` unchanged.

    Besides freeing host memory, this releases the device-resident index tensors that the
    torch backend's fusion cache pins after its first GPU use.
    """
    from . import _nonabelian
    for function in (_nonabelian.contraction_coefficient,
                     _nonabelian.intertwiner, _nonabelian.fusion_coefficient,
                     _contractions._fusion_intertwiner,
                     _contractions._contraction_recoupling):
        function.cache_clear()
    _nonabelian._fusion_weights_cache.cache_clear()
    _contractions._meta_broadcast.cache_clear()
    _contractions._meta_tensordot_f2m.cache_clear()
    _contractions._meta_tensordot_fc.cache_clear()
    _contractions._meta_tensordot_nf.cache_clear()
    _contractions._meta_tensordot_cutensor_cpu.cache_clear()
    _contractions_cutensor._meta_tensordot_cutensor_gpu.cache_clear()
    _contractions_cutensor._get_trimmed_struct_engine_gpu.cache_clear()
    _contractions._meta_mask.cache_clear()
    _contractions._meta_swap_gate.cache_clear()
    _contractions._meta_swap_gate_charge.cache_clear()
    _contractions._meta_trace.cache_clear()
    _contractions._meta_vdot.cache_clear()
    _einsum._meta_ncon.cache_clear()
    _merging._masks_hfs_intersection.cache_clear()
    _merging._leg_structure_combine_charges_prod.cache_clear()
    _merging._meta_fuse_hard.cache_clear()
    _merging._meta_unfuse_hard.cache_clear()
    _merging._adjacent_swap_coefficient.cache_clear()
    _algebra._meta_addition.cache_clear()
    _auxiliary.get_blocks.cache_clear()
    _auxiliary.get_blocks_charges_all.cache_clear()
    _auxiliary.get_trimmed_struct_engine.cache_clear()
    clear_registered()  # caches of whichever backends are loaded


def get_cache_info():
    """
    Return statistics of lru_caches used in yastn.

    Backend-side entries (registered via :func:`yastn._cache_registry.register_cache`)
    are present only for backends that are actually loaded, so the key set depends on
    which backend is in use.
    """
    from . import _nonabelian
    return {"fusion_weights": _nonabelian._fusion_weights_cache.cache_info(),
            "fuse_hard": _merging._meta_fuse_hard.cache_info(),
            "adjacent_swap_coefficient": _merging._adjacent_swap_coefficient.cache_info(),
            "unfuse_hard": _merging._meta_unfuse_hard.cache_info(),
            "intersect_hfs": _merging._masks_hfs_intersection.cache_info(),
            "combine_leg_structure": _merging._leg_structure_combine_charges_prod.cache_info(),
            "tensordot_f2m": _contractions._meta_tensordot_f2m.cache_info(),
            "tensordot_fc": _contractions._meta_tensordot_fc.cache_info(),
            "tensordot_nf": _contractions._meta_tensordot_nf.cache_info(),
            "tensordot_cutensor_cpu": _contractions._meta_tensordot_cutensor_cpu.cache_info(),
            "tensordot_cutensor_gpu": _contractions_cutensor._meta_tensordot_cutensor_gpu.cache_info(),
            "get_trimmed_struct_engine_gpu": _contractions_cutensor._get_trimmed_struct_engine_gpu.cache_info(),
            "broadcast": _contractions._meta_broadcast.cache_info(),
            "mask": _contractions._meta_mask.cache_info(),
            "trace": _contractions._meta_trace.cache_info(),
            "vdot": _contractions._meta_vdot.cache_info(),
            "swap_gate": _contractions._meta_swap_gate.cache_info(),
            "swap_gate_charge": _contractions._meta_swap_gate_charge.cache_info(),
            "ncon": _einsum._meta_ncon.cache_info(),
            "addition": _algebra._meta_addition.cache_info(),
            "get_blocks": _auxiliary.get_blocks.cache_info(),
            "get_blocks_charges_all": _auxiliary.get_blocks_charges_all.cache_info(),
            "get_trimmed_struct_engine": _auxiliary.get_trimmed_struct_engine.cache_info(),
            **registered_info(),
            }
