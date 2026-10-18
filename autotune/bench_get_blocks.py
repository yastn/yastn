"""Measure block-table construction as D, masks or U1 sectors change.

By default only metadata is allocated. Every leg has reduced D >= 12. --source selects
a baseline checkout; output includes block fingerprints and actual cache counts.
Use --fuse to instead measure complete rank-four hard-fusion operations.
"""
import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter


def benchmark_fuse(args, yastn):
    import numpy as np

    saved = {}
    checked = 0
    for sym in ('SU2', 'SU2xU1'):
        config = yastn.make_config(sym=sym, backend='np')
        charges = ((0,), (1,), (2,)) if sym == 'SU2' else ((0, -1), (0, 1), (1, 0), (2, 0))
        dimension = 12 // len(charges)
        tensors = []
        for step in range(1, args.steps + 1):
            leg = yastn.Leg(config, s=1, t=charges,
                            D=(dimension + step,) + (dimension,) * (len(charges) - 1))
            config.backend.random_seed(123)
            tensors.append(yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj())))
        samples = {'cold': [], 'warm': []}
        counts = {}
        for repeat in range(args.rounds):
            yastn.clear_cache()
            for phase in samples:
                before = yastn.get_cache_info()
                start = perf_counter()
                outputs = [a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard') for a in tensors]
                samples[phase].append(perf_counter() - start)
                after = yastn.get_cache_info()
                counts[phase] = {name: {'hits': after[name].hits - before[name].hits,
                                       'misses': after[name].misses - before[name].misses}
                                 for name in ('get_blocks', 'get_blocks_topology') if name in after}
                for step, output in enumerate(outputs):
                    values = np.asarray(output.data)
                    assert values.size > 0 and np.isfinite(values).all() and np.linalg.norm(values) > 0
                    key = f'{sym}_{step}'
                    if repeat == 0 and phase == 'cold':
                        saved[key] = values.copy()
                    else:
                        np.testing.assert_array_equal(saved[key], values)
                    checked += 1
        print(json.dumps({'sym': sym, 'operation': 'fuse', 'steps': args.steps,
                          'input_blocks': tensors[0].nblocks,
                          'seconds': {phase: median(t) for phase, t in samples.items()},
                          'samples': samples, 'caches': counts}), flush=True)
    if args.save_results:
        np.savez(args.save_results, **saved)
    print(json.dumps({'status': 'completed', 'checked_outputs': checked}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--fuse', action='store_true')
    parser.add_argument('--save-results', type=Path, help='save fused arrays (with --fuse)')
    args = parser.parse_args()
    if args.steps < 2 or args.steps > 32 or args.rounds < 1:
        parser.error('steps must be 2..32 and rounds must be positive')
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import numpy as np
    import yastn
    from yastn.sym import sym_SU2, sym_SU2xU1
    from yastn.tensor import _auxiliary
    from yastn.tensor._auxiliary import _struct, hash_blocks
    from yastn.tensor._legbasic import LegBasic
    assert Path(yastn.__file__).resolve() == source / 'yastn/__init__.py'
    print(json.dumps({'source': yastn.__file__, 'steps': args.steps, 'rounds': args.rounds}), flush=True)
    if args.fuse:
        benchmark_fuse(args, yastn)
        return
    checked = 0
    for sym in (sym_SU2, sym_SU2xU1):
        charges = ((0,), (1,), (2,)) if sym is sym_SU2 else ((0, -1), (0, 1), (1, 0), (2, 0))
        leg = LegBasic(s=1, t=charges, D=(12 // len(charges),) * len(charges))
        for rank in (4, 6):
            base = _struct(legs=(leg, leg.conj()) * (rank // 2), n=sym.zero())
            count = _auxiliary.get_blocks(sym, base).nblocks
            assert count > args.steps
            workloads = {
                'dimensions': [base.replace(legs=tuple(
                    item._replace(D=(item.D[0] + step,) + item.D[1:]) for item in base.legs))
                    for step in range(1, args.steps + 1)],
                'masks': [base.mask_from_ind(count, np.flatnonzero(np.arange(count) != step))
                          for step in range(args.steps)],
            }
            if sym is sym_SU2xU1:
                workloads['charges'] = [base.replace(legs=tuple(
                    item._replace(t=tuple((j, q + step) for j, q in item.t)) for item in base.legs))
                    for step in range(1, args.steps + 1)]
            for mode, structs in workloads.items():
                samples = {'seed': [], 'first_sweep': [], 'repeat': []}
                fingerprints = None
                counts = {}
                for _ in range(args.rounds):
                    yastn.clear_cache()
                    start = perf_counter()
                    _auxiliary.get_blocks(sym, base)
                    samples['seed'].append(perf_counter() - start)
                    for phase in ('first_sweep', 'repeat'):
                        before = yastn.get_cache_info()
                        start = perf_counter()
                        outputs = [_auxiliary.get_blocks(sym, struct) for struct in structs]
                        samples[phase].append(perf_counter() - start)
                        after = yastn.get_cache_info()
                        counts[phase] = {name: {
                            'hits': after[name].hits - before[name].hits,
                            'misses': after[name].misses - before[name].misses,
                            'currsize': after[name].currsize, 'maxsize': after[name].maxsize}
                            for name in ('get_blocks', 'get_blocks_topology') if name in after}
                        assert all(out.nblocks > 0 and out.size > 0 for out in outputs)
                        found = [hash_blocks(out) for out in outputs]
                        if fingerprints is None:
                            fingerprints = found
                        assert fingerprints == found
                        checked += len(outputs)
                print(json.dumps({'sym': sym.SYM_ID, 'rank': rank, 'mode': mode,
                                  'base_blocks': count, 'fingerprints': fingerprints,
                                  'seconds': {phase: median(t) for phase, t in samples.items()},
                                  'samples': samples, 'caches': counts}), flush=True)
    print(json.dumps({'status': 'completed', 'checked_outputs': checked}), flush=True)


if __name__ == '__main__':
    main()
