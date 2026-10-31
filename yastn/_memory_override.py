"""Global memory-guard override for the CUDA backends.

``YASTN_MEMORY_OVERRIDE=1`` disables the memory-saving caps introduced for the
CUDA contraction paths: the cuTENSOR packing-budget divisor, adaptive fusion
scatter tiles, NNN append chunking, and cache-release calls.  It trades peak
GPU memory for speed and should be used only when the device has enough
headroom.  The default (unset/0) keeps every guard unchanged.
"""
import os


def memory_override():
    """Return True when ``YASTN_MEMORY_OVERRIDE`` requests the old, larger-memory path."""
    return os.environ.get('YASTN_MEMORY_OVERRIDE', '0').lower() in ('1', 'true', 'yes', 'on')
