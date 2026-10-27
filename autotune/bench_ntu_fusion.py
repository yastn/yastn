"""Compare the NNN environment's single-group fusion and full bond metric."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--variant', choices=('baseline', 'candidate'), required=True)
parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
parser.add_argument('--operation', choices=('fuse', 'metric'), required=True)
parser.add_argument('--direction', choices=('h', 'v'), default='h')
parser.add_argument('--D', type=int, default=12)
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--write-reference', action='store_true')
args = parser.parse_args()
if args.D < 6 or args.repeats < 0:
    parser.error('Require D >= 6 and repeats >= 0')
if args.write_reference and args.variant != 'baseline':
    parser.error('Only baseline may write the reference')
root = args.source.resolve()
sys.path.insert(0, str(root))

import numpy as np
import torch
import yastn
from yastn.backend import backend_torch
from yastn.tn import fpeps
from yastn.tn.fpeps.envs._env_contractions import edge_r, cor_br, edge_b

assert Path(yastn.__file__).resolve() == root / 'yastn/__init__.py', yastn.__file__
torch.set_num_threads(1)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
if args.variant == 'baseline':
    backend_torch.recouple_and_merge = None
ops = yastn.operators.Spin12(sym='SU2', backend='torch', default_device=args.device,
                             default_dtype='float64')
# D counts magnetic states, as in non-Abelian truncation's D_total.
triplets, doublets = max(1, args.D // 12), max(1, args.D // 4)
singlets = args.D - 3 * triplets - 2 * doublets
leg = yastn.Leg(ops.config, s=1, t=(0, 1, 2), D=(singlets, doublets, triplets))
assert sum((t[0] + 1) * d for t, d in zip(leg.t, leg.D)) == args.D
physical = ops.I().fuse_legs(((0, 1),)).get_legs(0)
a = yastn.rand(ops.config, legs=(leg.conj(), leg, leg, leg.conj(), physical), dtype='float64')
a = a / a.norm()
psi = fpeps.Peps(fpeps.CheckerboardLattice(), tensors=a)
print(json.dumps(dict(status='started', source=str(root), variant=args.variant,
                      operation=args.operation, D=args.D, multiplicities=leg.D,
                      device=args.device, gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None)),
      flush=True)

with torch.no_grad():
    if args.operation == 'fuse':
        vector = edge_r(a) @ (cor_br(a) @ edge_b(a))
        run = lambda: vector.fuse_legs((2, (0, 3), 1), mode='hard')
        input_elements = vector.size
    else:
        vertical = args.direction == 'v'
        s0, s1 = (0, 0), ((1, 0) if vertical else (0, 1))
        if vertical:
            Q0, _ = a.qr(axes=((0, 1, 3, 4), 2), sQ=1, Qaxis=2)
            Q1, _ = a.qr(axes=((1, 2, 3, 4), 0), sQ=-1, Qaxis=0, Raxis=-1)
        else:
            Q0, _ = a.qr(axes=((0, 1, 2, 4), 3), sQ=-1, Qaxis=3)
            Q1, _ = a.qr(axes=((0, 2, 3, 4), 1), sQ=1, Qaxis=1, Raxis=-1)
        env = fpeps.EnvNTU(psi, which='NNN')
        run = lambda: env.bond_metric(Q0, Q1, s0, s1, args.direction).g
        input_elements = a.size + Q0.size + Q1.size


def synchronize():
    if args.device == 'cuda':
        torch.cuda.synchronize()


samples = []
for repeat in range(args.repeats + 1):
    gc.collect()
    synchronize()
    before = torch.cuda.memory_allocated() if args.device == 'cuda' else None
    if before is not None:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        result = run()
    synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() if before is not None else None
    output = result.data.detach().cpu().numpy().copy()
    assert output.size > 0 and result.nblocks > 0
    assert np.isfinite(output).all() and np.linalg.norm(output) > 0
    if args.write_reference and repeat == 0:
        np.savez(args.reference, data=output)
    with np.load(args.reference) as saved:
        expected = saved['data']
        assert expected.size == output.size
        np.testing.assert_allclose(output, expected, rtol=1e-10, atol=1e-12)
        relative_error = float(np.linalg.norm(output - expected) / np.linalg.norm(expected))
    sample = dict(repeat=repeat, seconds=seconds, input_elements=input_elements,
                  checked_elements=output.size, blocks=result.nblocks, relative_error=relative_error,
                  baseline_allocated_bytes=before, peak_allocated_bytes=peak,
                  extra_peak_allocated_bytes=peak - before if peak is not None else None)
    samples.append(sample)
    print(json.dumps(dict(status='sample_complete', **sample)), flush=True)
    del result, output, expected
print(json.dumps(dict(status='complete', variant=args.variant, operation=args.operation,
                      direction=args.direction, D=args.D, dtype='float64', sym='SU2',
                      samples=samples)), flush=True)
