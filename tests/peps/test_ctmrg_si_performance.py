# Copyright 2026 The YASTN Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in performance acceptance benchmark for SI projector construction.

Run explicitly with, for example::

    pytest -q tests/peps/test_ctmrg_si_performance.py --long_tests \
        --backend torch --device cuda -s

It is excluded from ordinary CI because timing and accelerator-memory
assertions require an otherwise idle machine.
"""

from collections import defaultdict
from time import perf_counter
import tracemalloc

import numpy as np
import pytest

import yastn
import yastn.tn.fpeps.envs._env_ctm as env_ctm_module
from yastn.tn.fpeps.envs._env_ctm import initialize_si_bases, proj_corners


pytestmark = pytest.mark.skipif(
    "not config.getoption('long_tests')",
    reason="SI performance acceptance is an opt-in benchmark")


def _dense_benchmark_corners(config, dimension):
    """Random dense corners with an unfusable projector-output CTM leg."""
    r0 = yastn.rand(config, s=(1, 1, -1, 1),
                    D=(1, dimension, dimension, 1))
    r1 = yastn.rand(config, s=(-1, -1, 1, -1),
                    D=(1, dimension, dimension, 1))
    return (r0.fuse_legs(axes=((0, 1), (2, 3))),
            r1.fuse_legs(axes=((0, 1), (2, 3))))


def _synchronize(config):
    if ('torch' in config.backend.BACKEND_ID
            and 'cuda' in config.default_device):
        config.backend.torch.cuda.synchronize()


def _profile_call(config, call):
    """Return value, wall time, Python peak, and CUDA-native peak bytes."""
    torch = getattr(config.backend, 'torch', None)
    on_cuda = torch is not None and 'cuda' in config.default_device
    if on_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    _synchronize(config)
    start = perf_counter()
    value = call()
    _synchronize(config)
    elapsed = perf_counter() - start
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_peak = torch.cuda.max_memory_allocated() if on_cuda else 0
    return value, elapsed, python_peak, cuda_peak


def _benchmark_time(config, call, repeats=3):
    """Best repeated wall time without probes or allocation tracing."""
    # One untimed call removes lazy backend/setup effects from the comparison.
    call()
    _synchronize(config)
    samples = []
    for _ in range(repeats):
        _synchronize(config)
        start = perf_counter()
        call()
        _synchronize(config)
        samples.append(perf_counter() - start)
    return min(samples)


def _install_shape_and_region_probes(monkeypatch, config, records):
    """Instrument effective matrices, decomposition sizes, and GPU regions."""
    tensor_type = yastn.Tensor
    original_tensordot = env_ctm_module.tensordot
    original_qr = env_ctm_module.qr
    original_svd = tensor_type.svd
    original_svd_truncated = tensor_type.svd_with_truncation

    def timed(region, function, *args, **kwargs):
        _synchronize(config)
        start = perf_counter()
        result = function(*args, **kwargs)
        _synchronize(config)
        records['region_seconds'][region] += perf_counter() - start
        return result

    def tensordot_probe(*args, **kwargs):
        result = timed('contraction', original_tensordot, *args, **kwargs)
        if result.ndim == 2:
            shape = result.get_shape()
            records['matrix_shapes'].append(tuple(shape))
            records['max_matrix_elements'] = max(
                records['max_matrix_elements'], int(np.prod(shape)))
        return result

    def qr_probe(*args, **kwargs):
        return timed('decomposition', original_qr, *args, **kwargs)

    def svd_probe(self, *args, **kwargs):
        records['svd_shapes'].append(tuple(self.get_shape()))
        return timed('decomposition', original_svd, self, *args, **kwargs)

    def svd_truncated_probe(self, *args, **kwargs):
        records['svd_shapes'].append(tuple(self.get_shape()))
        return timed('decomposition', original_svd_truncated,
                     self, *args, **kwargs)

    monkeypatch.setattr(env_ctm_module, 'tensordot', tensordot_probe)
    monkeypatch.setattr(env_ctm_module, 'qr', qr_probe)
    monkeypatch.setattr(tensor_type, 'svd', svd_probe)
    monkeypatch.setattr(tensor_type, 'svd_with_truncation',
                        svd_truncated_probe)


def _new_records():
    return {'matrix_shapes': [], 'svd_shapes': [],
            'max_matrix_elements': 0,
            'region_seconds': defaultdict(float)}


def test_si_projector_performance_acceptance(config_kwargs, monkeypatch):
    """Validate SI's allocation, decomposition, scaling, and GPU profile."""
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=91)
    on_cuda = 'cuda' in config.default_device
    chi, oversampling = 32, 4
    bond_dimensions = (3, 4, 5) if on_cuda else (2, 3, 4)
    results = []

    for bond_dimension in bond_dimensions:
        dimension = chi * bond_dimension ** 2
        r0, r1 = _dense_benchmark_corners(config, dimension)
        opts_svd = {'D_total': chi, 'tol': 0, 'fix_signs': True}
        opts_si = {'enabled': True, 'oversampling': oversampling,
                   'niter': 1, 'tol': 0}
        X, Y = initialize_si_bases(r0, r1, chi + oversampling)

        # Record full SVD and SI independently; probes are reset between paths.
        full_records = _new_records()
        _install_shape_and_region_probes(
            monkeypatch, config, full_records)
        _, full_time, full_python_peak, full_cuda_peak = _profile_call(
            config, lambda: proj_corners(r0, r1, opts_svd))
        monkeypatch.undo()

        si_records = _new_records()
        _install_shape_and_region_probes(monkeypatch, config, si_records)
        (_, _, X, Y), si_time, si_python_peak, si_cuda_peak = _profile_call(
            config, lambda: proj_corners(
                r0, r1, opts_svd, opts_si=opts_si, X=X, Y=Y,
                return_si_state=True))
        monkeypatch.undo()

        # Shape/memory probes above intentionally synchronize and trace every
        # primitive.  Do clean repeated timings after removing all probes.
        full_time = _benchmark_time(
            config, lambda: proj_corners(r0, r1, opts_svd))
        si_time = _benchmark_time(
            config, lambda: proj_corners(
                r0, r1, opts_svd, opts_si=opts_si, X=X, Y=Y,
                return_si_state=True))

        rank = chi + oversampling
        assert (dimension, dimension) in full_records['matrix_shapes']
        assert (dimension, dimension) not in si_records['matrix_shapes']
        assert any(max(shape) == dimension
                   for shape in full_records['svd_shapes'])
        assert si_records['svd_shapes']
        assert all(max(shape) <= rank for shape in si_records['svd_shapes'])
        assert si_records['max_matrix_elements'] <= dimension * rank

        results.append({
            'D': bond_dimension, 'N': dimension,
            'full_time': full_time, 'si_time': si_time,
            'full_python_peak': full_python_peak,
            'si_python_peak': si_python_peak,
            'full_cuda_peak': full_cuda_peak,
            'si_cuda_peak': si_cuda_peak,
            'full_matrix_elements': full_records['max_matrix_elements'],
            'si_matrix_elements': si_records['max_matrix_elements'],
            'si_contraction_time': si_records['region_seconds']['contraction'],
            'si_decomposition_time': si_records['region_seconds']['decomposition'],
        })

    first, last = results[0], results[-1]
    full_growth = last['full_time'] / first['full_time']
    si_growth = last['si_time'] / first['si_time']
    assert last['si_time'] < last['full_time']
    assert full_growth > 1.25 * si_growth
    # Allocation proxy is exact for the dominant tensor payload and remains
    # meaningful on CPU, where tracemalloc cannot see NumPy/PyTorch buffers.
    assert 4 * last['si_matrix_elements'] < last['full_matrix_elements']

    if on_cuda:
        assert last['si_cuda_peak'] < last['full_cuda_peak']
        assert last['si_contraction_time'] > last['si_decomposition_time']

    print("\nSI-CTMRG projector performance acceptance")
    for result in results:
        print(result)
