"""Compare QR, truncated SVD and one CTMRG sweep across explicit source trees."""
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
parser.add_argument('--operation', choices=('qr', 'svd', 'ctm'), required=True)
parser.add_argument('--variant', choices=('baseline', 'candidate'), required=True)
parser.add_argument('--D', type=int, default=12)
parser.add_argument('--chi', type=int, default=12)
parser.add_argument('--sym', choices=('SU2', 'SU2xU1'), default='SU2')
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--reference', type=Path)
parser.add_argument('--write-reference', action='store_true')
args = parser.parse_args()
if args.D < 12 or args.chi < 1 or args.repeats < 1:
    parser.error('Require D >= 12, chi >= 1 and repeats >= 1')
if args.operation == 'ctm' and args.reference is None:
    parser.error('CTMRG comparisons require a reference output file')
if args.write_reference and (args.operation != 'ctm' or args.variant != 'baseline'):
    parser.error('Only the CTMRG baseline may write the reference output')
root = args.source.resolve()
sys.path.insert(0, str(root))
import numpy as np
import torch
import yastn
from yastn.tn import fpeps

assert Path(yastn.__file__).resolve() == root / 'yastn/__init__.py', yastn.__file__
torch.set_num_threads(1)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
if args.operation == 'ctm':
    ops = yastn.operators.SpinfulFermions_tJ(sym=args.sym, backend='torch', default_device=args.device)
    config = ops.config
    config.backend.random_seed(seed=42)
    if args.sym == 'SU2':
        charges, dims, boundary_charges = (0, 1), (args.D // 2, args.D - args.D // 2), (0, 2)
    else:
        charges = ((0, 0), (1, 1), (1, -1))
        dims = (args.D // 3, args.D // 3, args.D - 2 * (args.D // 3))
        boundary_charges = ((0, 0), (2, 0))
    leg = yastn.Leg(config, s=1, t=charges, D=dims)
    boundary = yastn.Leg(config, s=1, t=boundary_charges,
                         D=(2 * args.chi // 3, args.chi - 2 * args.chi // 3))
    tensor = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj(), ops.I().get_legs(0)))
    psi = fpeps.Peps(fpeps.CheckerboardLattice(), tensors=tensor)
    initial = fpeps.EnvCTM(psi, init='rand', leg=boundary)
    expected = None
    if args.reference and not args.write_reference:
        with np.load(args.reference) as saved:
            expected = {k: saved[k].copy() for k in saved.files}
else:
    config = yastn.make_config(sym='U1', backend='torch', default_device=args.device)
    left = yastn.Leg(config, s=1, t=(0, 1, 2, 3), D=(12 * args.D ** 2,) * 4)
    right = yastn.Leg(config, s=-1, t=(0, 1, 2, 3), D=(4 * args.D ** 2,) * 4)
    a = yastn.rand(config, legs=(left, right))

    def legacy_svd():
        u, s, v = a.svd(fix_signs=True)
        return yastn.truncation_mask(s, D_total=args.chi).apply_mask(u, s, v, axes=(1, 0, 0))

    if args.operation == 'qr':
        run = lambda: (a.qr()[1] if args.variant == 'baseline' else a.qr(compute_q=False),)
    else:
        run = legacy_svd if args.variant == 'baseline' else lambda: a.svd_with_truncation(
            D_total=args.chi, fix_signs=True)
    with torch.no_grad():
        expected = tuple(x.data.detach().clone() for x in ((a.qr()[1],) if args.operation == 'qr' else legacy_svd()))


def synchronize():
    if args.device == 'cuda':
        torch.cuda.synchronize()


def sample():
    with torch.no_grad():
        env = initial.copy() if args.operation == 'ctm' else None
        gc.collect()
        synchronize()
        before = torch.cuda.memory_allocated() if args.device == 'cuda' else None
        if before is not None:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if env is not None:
            env.update_(opts_svd={'D_total': args.chi}, moves='hv', use_qr=True)
        else:
            result = run()
        synchronize()
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() if before is not None else None
        if env is not None:
            outputs = {f'{i}_{name}': getattr(env[site], name).data.detach().cpu().numpy()
                       for i, site in enumerate(env.sites())
                       for name in ('tl', 'tr', 'br', 'bl', 't', 'l', 'b', 'r')}
            assert len(outputs) == 16 and all(x.size and np.linalg.norm(x) > 0 for x in outputs.values())
            if expected is not None:
                assert outputs.keys() == expected.keys()
                for key in outputs:
                    np.testing.assert_allclose(outputs[key], expected[key], rtol=1e-8, atol=1e-10)
            if args.write_reference:
                np.savez(args.reference, **outputs)
            count = sum(x.size for x in outputs.values())
        else:
            assert len(result) == len(expected) > 0
            for found, wanted in zip(result, expected):
                assert found.size > 0
                torch.testing.assert_close(found.data, wanted, rtol=1e-9, atol=1e-10)
            count = sum(x.size for x in result)
        return dict(seconds=seconds, checked_elements=count,
                    baseline_allocated_bytes=before, peak_allocated_bytes=peak,
                    extra_peak_allocated_bytes=peak - before if peak is not None else None)


cold = sample()
warm = [sample() for _ in range(args.repeats)]
print(json.dumps(dict(status='complete', source=str(root), device=args.device,
                      operation=args.operation, variant=args.variant, D=args.D, chi=args.chi,
                      sym=args.sym if args.operation == 'ctm' else 'U1',
                      grad_enabled=False, cold=cold, warm=warm,
                      warm_median_seconds=statistics.median(x['seconds'] for x in warm))), flush=True)
