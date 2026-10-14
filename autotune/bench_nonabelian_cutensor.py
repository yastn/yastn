"""Compare ordinary PyTorch and channel-aware cuTENSOR on the same reduced data.

Run from the repository root, e.g.:
    python autotune/bench_nonabelian_cutensor.py --dimensions 2 4 8 --iterations 20

Reports end-to-end latency (including packing) and PyTorch peak allocated memory.
External cuTENSOR workspace allocations are not included in PyTorch's counters.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yastn
from yastn.backend import backend_torch


def measure(operation, iterations, warmup, device):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial = torch.cuda.memory_allocated(device)
    start = perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize(device)
    elapsed = (perf_counter() - start) * 1000 / iterations
    peak = (torch.cuda.max_memory_allocated(device) - initial) / 2**20
    return {'ms': elapsed, 'extra_torch_peak_MiB': peak}


def benchmark(sym, dimension, args):
    from yastn.backend import backend_torch_cutensor
    config = yastn.make_config(sym=sym, backend=backend_torch, default_device=args.device)
    charges = tuple((j,) if sym == 'SU2' else (j, 0) for j in range(3))
    leg = yastn.Leg(config, s=1, t=charges, D=(dimension,) * 3)
    a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()), dtype=args.dtype)
    b = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()), dtype=args.dtype)
    a.requires_grad_()
    b.requires_grad_()
    cu_config = config._replace(backend=backend_torch_cutensor)
    ca, cb = a._replace(config=cu_config), b._replace(config=cu_config)
    axes = ((2, 3), (0, 1))

    def forward(x, y):
        return yastn.tensordot(x, y, axes=axes).data

    reference, actual = forward(a, b), forward(ca, cb)
    tolerance = 1e-10 if args.dtype in ('float64', 'complex128') else 1e-4
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    weight = torch.randn_like(reference)
    grad_ref = torch.autograd.grad((reference.conj() * weight).real.sum(), (a.data, b.data))
    grad_cu = torch.autograd.grad((actual.conj() * weight).real.sum(), (a.data, b.data))
    for found, wanted in zip(grad_cu, grad_ref):
        torch.testing.assert_close(found, wanted, atol=tolerance, rtol=tolerance)
    del actual, reference, grad_cu, grad_ref

    def operation(x, y, backward):
        if backward:
            out = forward(x, y)
            return torch.autograd.grad((out.conj() * weight).real.sum(), (x.data, y.data))
        with torch.no_grad():
            return forward(x, y)

    for backward in (False, True):
        ordinary = measure(lambda: operation(a, b, backward), args.iterations, args.warmup, args.device)
        accelerated = measure(lambda: operation(ca, cb, backward), args.iterations, args.warmup, args.device)
        print(json.dumps({'sym': sym, 'degeneracy': dimension, 'input_blocks': a.nblocks,
                          'dtype': args.dtype, 'backward': backward,
                          'torch': ordinary, 'torch_cutensor': accelerated,
                          'speedup': ordinary['ms'] / accelerated['ms']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimensions', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--sym', choices=['SU2', 'SU2xU1'], nargs='+', default=['SU2', 'SU2xU1'])
    parser.add_argument('--dtype', choices=['float32', 'float64', 'complex64', 'complex128'], default='float64')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or any(d < 1 for d in args.dimensions):
        parser.error('dimensions and iterations must be positive; warmup must be nonnegative')
    if not torch.cuda.is_available():
        parser.error('a working CUDA device and tapp_torch are required')
    torch.cuda.set_device(args.device)
    torch.manual_seed(123)
    print(json.dumps({'gpu': torch.cuda.get_device_name(args.device), 'torch': torch.__version__,
                      'cuda': torch.version.cuda}), flush=True)
    for sym in args.sym:
        for dimension in args.dimensions:
            yastn.clear_cache()
            benchmark(sym, dimension, args)


if __name__ == '__main__':
    main()
