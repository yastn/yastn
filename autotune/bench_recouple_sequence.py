"""Exercise the actual sparse-composition budget fallback at the CTMRG order."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--variant', choices=('baseline', 'candidate'), required=True)
parser.add_argument('--D', type=int, default=24)
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.D < 12 or args.D % 12 or args.repeats < 1:
    parser.error('Require D >= 12 divisible by 12, and repeats >= 1')
root = args.source.resolve()
sys.path.insert(0, str(root))
import torch
import yastn
from yastn.tensor import _merging as merging

assert Path(yastn.__file__).resolve() == root / 'yastn/__init__.py', yastn.__file__
torch.set_num_threads(1)
torch.manual_seed(42)
config = yastn.make_config(sym='SU2', backend='torch', default_device=args.device)
scale = args.D // 12
leg = yastn.Leg(config, s=1, t=(0, 2, 4, 6), D=(9 * scale, scale, scale, scale))
a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj(), leg, leg.conj()), dtype='float64')
order = (0, 2, 5, 1, 3, 4)
plans = merging._meta_fusion_tree_transpose(config.sym, a.struct, order)
assert len(plans) > 1 and sum(len(plan[2]) for plan in plans) > 65536
assert all(plan[3] == a.size for plan in plans)


def sequential():
    result = a
    for plan in plans:
        result = merging._apply_recoupling(result, *plan)
    return result


with torch.no_grad():
    reference = sequential()
assert reference.size > 0 and reference.norm().item() > 0


def synchronize():
    if args.device == 'cuda':
        torch.cuda.synchronize()


def sample():
    with torch.no_grad():
        gc.collect()
        synchronize()
        before = torch.cuda.memory_allocated() if args.device == 'cuda' else None
        if before is not None:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = merging._apply_fusion_tree_transpose(a, order)
        synchronize()
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() if before is not None else None
        assert result.struct == reference.struct and result.hfs == reference.hfs
        torch.testing.assert_close(result.data, reference.data, rtol=1e-10, atol=1e-10)
        return dict(seconds=seconds, checked_elements=result.size,
                    baseline_allocated_bytes=before, peak_allocated_bytes=peak,
                    extra_peak_allocated_bytes=peak - before if before is not None else None)


cold = sample()
warm = [sample() for _ in range(args.repeats)]
print(json.dumps(dict(status='complete', source=str(root), variant=args.variant,
                      device=args.device, D=args.D, order=order, elements=a.size,
                      blocks=a.nblocks, raw_plans=len(plans),
                      raw_terms=[len(plan[2]) for plan in plans],
                      cold=cold, warm=warm,
                      warm_median_seconds=statistics.median(x['seconds'] for x in warm))), flush=True)
