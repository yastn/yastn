"""Isolate U1 cache duplication with equal spin content and dense bond dimension.

Run from the repository root with a single BLAS thread:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python autotune/bench_su2_u1_cache.py

Each tensor has the same spins and multiplicities. SU2xU1 tensors differ only
by an additive U1 label, so this measures charge-dependent metadata overhead,
not the block fragmentation of a particular PEPS. Cold sweeps clear caches;
warm sweeps immediately repeat the same operations. JSON includes actual cache
sizes and counts, and optional NPZ output permits before/after value checks.
"""
import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yastn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', type=int, default=24,
                        help='full bond dimension; reduced dimension is half this value')
    parser.add_argument('--sectors', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--save-results', type=Path)
    args = parser.parse_args()
    if args.dimension < 24 or args.dimension % 6 or args.sectors < 2 or args.rounds < 1:
        parser.error('dimension must be a multiple of 6 >= 24; sectors >= 2; rounds >= 1')
    expected_source = Path(__file__).resolve().parents[1] / 'yastn' / '__init__.py'
    assert Path(yastn.__file__).resolve() == expected_source, yastn.__file__
    print(json.dumps({'source': yastn.__file__, 'dimension': args.dimension,
                      'reduced_dimension': args.dimension // 2,
                      'sectors': args.sectors, 'rounds': args.rounds}), flush=True)
    saved = {}
    operations = {
        'transpose': lambda a: a.transpose((0, 2, 1, 3)).consume_transpose(),
        'fuse': lambda a: a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard'),
        'contract': lambda a: yastn.tensordot(a, a, axes=((2, 3), (0, 1))),
    }
    names = ('adjacent_swap_coefficient', 'nonabelian_fusion_coefficient',
             'nonabelian_contraction_coefficient', 'adjacent_swap_u1',
             'nonabelian_fusion_u1')
    names = tuple(name for name in names if name in yastn.get_cache_info())
    for case in ('SU2', 'SU2xU1_zero', 'SU2xU1_varied'):
        sym = 'SU2' if case == 'SU2' else 'SU2xU1'
        config = yastn.make_config(sym=sym, backend='np')
        tensors = []
        for i in range(args.sectors):
            q = i - args.sectors // 2 if case == 'SU2xU1_varied' else 0
            charges = tuple((j,) if sym == 'SU2' else (j, q) for j in range(3))
            leg = yastn.Leg(config, s=1, t=charges, D=(args.dimension // 6,) * 3)
            config.backend.random_seed(123)
            a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()))
            assert a.nblocks > 0 and a.size > 0
            tensors.append(a)
        for operation_name, operation in operations.items():
            times = {'cold': [], 'warm': []}
            counts = {}
            for repeat in range(args.rounds):
                yastn.clear_cache()
                for phase in times:
                    before = yastn.get_cache_info()
                    start = perf_counter()
                    outputs = [operation(a) for a in tensors]
                    times[phase].append(perf_counter() - start)
                    after = yastn.get_cache_info()
                    counts[phase] = {
                        name: {'hits': after[name].hits - before[name].hits,
                               'misses': after[name].misses - before[name].misses,
                               'currsize': after[name].currsize, 'maxsize': after[name].maxsize}
                        for name in names}
                    for i, output in enumerate(outputs):
                        values = np.asarray(output.data)
                        assert values.size > 0 and np.isfinite(values).all()
                        assert np.linalg.norm(values) > 0
                        key = f'{case}_{operation_name}_{i}'
                        if repeat == 0 and phase == 'cold':
                            saved[key] = values.copy()
                        else:
                            np.testing.assert_allclose(values, saved[key], atol=1e-12, rtol=1e-12)
            print(json.dumps({'case': case, 'operation': operation_name,
                              'input_blocks': tensors[0].nblocks, 'input_size': tensors[0].size,
                              'seconds': {phase: median(t) for phase, t in times.items()},
                              'samples': times, 'caches': counts}), flush=True)
    if args.save_results:
        np.savez(args.save_results, **saved)
    print(json.dumps({'status': 'completed', 'checked_arrays': len(saved)}), flush=True)


if __name__ == '__main__':
    main()
