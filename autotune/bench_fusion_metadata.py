"""Measure hard-fusion and adjacent-swap metadata builders on nonempty tensors.

The reduced bond dimension is 12. No tensor data is allocated: this benchmark
isolates metadata construction, both with cold managed caches and with warm
dependencies while bypassing only the target builder's LRU. --source permits
the same workload to run against a separate baseline checkout.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
from statistics import median
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('rounds must be positive')
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import yastn
    from yastn.sym import sym_SU2, sym_SU2xU1
    from yastn.tensor import _merging
    from yastn.tensor._auxiliary import _struct, get_blocks, hash_blocks
    from yastn.tensor._legbasic import LegBasic
    assert Path(yastn.__file__).resolve() == source / 'yastn/__init__.py'
    print(json.dumps({'source': yastn.__file__, 'reduced_dimension': 12,
                      'rounds': args.rounds}), flush=True)
    checked = 0
    profiler = cProfile.Profile()
    for sym in (sym_SU2, sym_SU2xU1):
        charges = ((0,), (1,), (2,)) if sym is sym_SU2 else ((0, -1), (0, 1), (1, 0), (2, 0))
        leg = LegBasic(s=1, t=charges, D=(12 // len(charges),) * len(charges))
        for rank in (4, 6):
            struct = _struct(legs=(leg, leg.conj()) * (rank // 2), n=sym.zero())
            input_blocks = get_blocks(sym, struct).nblocks
            assert input_blocks > 0
            axes = (tuple(range(rank // 2)),) + tuple((i,) for i in range(rank // 2, rank))
            cases = [('fuse_hard', _merging._meta_fuse_hard.__wrapped__, (sym, struct, axes))]
            cases.extend((f'swap_{p}', _merging._meta_swap_adjacent_fusion_leaves.__wrapped__,
                          (sym, struct, p)) for p in (0, 1, rank - 2))
            for name, function, inputs in cases:
                samples = {'cold': [], 'dependencies_warm': []}
                reference = None
                for phase in samples:
                    for _ in range(args.rounds):
                        if phase == 'cold':
                            yastn.clear_cache()
                        start = perf_counter()
                        result = function(*inputs)
                        samples[phase].append(perf_counter() - start)
                        if name == 'fuse_hard':
                            meta, size, output_struct, _ = result
                        else:
                            output_struct, _, meta, size = result
                        assert len(meta) > 0 and size > 0
                        fingerprint = (hash_blocks(get_blocks(sym, output_struct)),
                                       hashlib.sha256(repr(meta).encode('ascii')).hexdigest(), size)
                        if reference is None:
                            reference = fingerprint
                        assert fingerprint == reference
                        checked += 1
                if args.profile:
                    profiler.runcall(function, *inputs)
                print(json.dumps({'sym': sym.SYM_ID, 'rank': rank, 'operation': name,
                                  'input_blocks': input_blocks, 'meta_records': len(meta),
                                  'fingerprint': reference,
                                  'seconds': {phase: median(t) for phase, t in samples.items()},
                                  'samples': samples}), flush=True)
    if args.profile:
        profiler.dump_stats(str(args.profile))
        pstats.Stats(profiler, stream=sys.stderr).strip_dirs().sort_stats('cumulative').print_stats(30)
    print(json.dumps({'status': 'completed', 'checked_results': checked}), flush=True)


if __name__ == '__main__':
    main()
