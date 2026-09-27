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
""" Optional NVTX instrumentation, controlled by the YASTN_PROFILE env variable. """
from __future__ import annotations

import contextvars
import os
import warnings
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace

__all__ = ['nsys_profile', 'nvtx', 'profiling_enabled', 'trace_flops', 'active_flop_tracer']


def profiling_enabled() -> bool:
    """ Whether NVTX instrumentation is requested through ``YASTN_PROFILE``. """
    return os.getenv("YASTN_PROFILE", "0").lower() in ("1", "true", "on")


def _get_nvtx():
    """
    Namespace with ``range_push``, ``range_pop`` and ``mark``, backed by torch
    or the standalone nvtx package. ``None`` if neither is importable.
    """
    try:
        import torch.cuda.nvtx as _nvtx
        return SimpleNamespace(range_push=_nvtx.range_push,
                               range_pop=_nvtx.range_pop,
                               mark=_nvtx.mark)
    except ImportError:
        pass
    try:
        import nvtx as _nvtx
        return SimpleNamespace(range_push=lambda msg: _nvtx.push_range(msg),
                               range_pop=_nvtx.pop_range,
                               mark=lambda msg: _nvtx.mark(message=msg))
    except ImportError:
        return None


def _noop(msg=None):
    pass


PROFILE = profiling_enabled()
_NVTX = _get_nvtx() if PROFILE else None

if PROFILE and _NVTX is None:
    warnings.warn("YASTN_PROFILE is set, but NVTX is unavailable "
                  "(neither torch.cuda.nvtx nor the nvtx package can be imported); "
                  "profiling annotations are disabled.", RuntimeWarning)

#: Always-usable NVTX namespace: the real backend when profiling is on, no-ops otherwise.
nvtx = _NVTX if _NVTX is not None else SimpleNamespace(range_push=_noop, range_pop=_noop, mark=_noop)
nvtx.enabled = _NVTX is not None


@contextmanager
def nvtx_range(msg):
    """ Context-manager form of :data:`nvtx`, e.g. ``with nvtx_range("step"): ...``. """
    nvtx.range_push(msg)
    try:
        yield
    finally:
        nvtx.range_pop()


nvtx.range = nvtx_range


def nsys_profile(arg=None):
    """
    Decorator wrapping a function in an NVTX range when ``YASTN_PROFILE`` is set
    to one of ``1``, ``true``, ``on``; otherwise the function is returned unchanged.

    Usable bare, ``@nsys_profile``, where the range is named after the function,
    or with an explicit range name, ``@nsys_profile("my_range")``.
    """
    def decorator(f, name=None):
        if not nvtx.enabled:
            return f
        msg = name if name is not None else f.__qualname__

        @wraps(f)
        def wrapper(*args, **kwargs):
            nvtx.range_push(msg)
            try:
                return f(*args, **kwargs)
            finally:
                nvtx.range_pop()
        return wrapper

    if callable(arg):  # used as @nsys_profile
        return decorator(arg)
    return lambda f: decorator(f, arg)  # used as @nsys_profile("name")


#####################################################
#     metadata-only FLOP tracing of contractions     #
#####################################################


class _FlopTracer:
    """Accumulator for the FLOPs of the tensordots executed in a tracing context.

    ``gemm`` counts the block GEMM multiply-adds (``2 M N K``, ``8`` for complex);
    ``sum`` counts the ``+=`` accumulation adds (``M N``, ``2 M N`` for complex).
    """
    __slots__ = ('gemm', 'sum')

    def __init__(self):
        self.gemm = 0
        self.sum = 0

    @property
    def total(self):
        return self.gemm + self.sum

    def __repr__(self):
        return f"_FlopTracer(gemm={self.gemm}, sum={self.sum}, total={self.total})"


#: Context-local slot holding the active :class:`_FlopTracer`, or ``None`` when not tracing.
_flop_tracer: contextvars.ContextVar = contextvars.ContextVar("yastn_flop_tracer", default=None)


def active_flop_tracer():
    """ The :class:`_FlopTracer` for the current dynamic context, or ``None`` when not tracing. """
    return _flop_tracer.get()


@contextmanager
def trace_flops():
    """
    Run contractions in metadata-only *tracing* mode: accumulate the FLOPs of tensordots
    while skipping all operations on tensor data. Tensors produced inside the context are
    empty (length-0 data) but carry correct structs, so struct propagation through a network
    (e.g. via :func:`yastn.ncon`) is exact.

    Yields the :class:`_FlopTracer`, whose ``gemm``, ``sum`` and ``total`` attributes hold the
    accumulated counts and remain readable after the context exits::

        with yastn.trace_flops() as tr:
            out = yastn.ncon(tensors, inds)
        print(tr.total, tr.gemm, tr.sum)  # out.data is empty; out.struct is correct

    Nested ``with trace_flops()`` blocks each get an independent fresh tracer.
    """
    tracer = _FlopTracer()  # created here, discarded on exit -- not a global
    token = _flop_tracer.set(tracer)
    try:
        yield tracer
    finally:
        _flop_tracer.reset(token)  # slot returns to its previous value (None / outer tracer)
