from . import _C, ops
from ._C import clear_plan_cache, plan_cache_size, plan_cache_keys, plan_cache_stats

__all__ = [
    "ops",
    "clear_plan_cache",
    "plan_cache_size",
    "plan_cache_keys",
    "plan_cache_stats",
]