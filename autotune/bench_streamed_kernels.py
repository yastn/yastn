"""Compare streamed kernels or 2x2 measurements across explicit source trees.

Run variants serially on a free host. D is at least 12; timings exclude input
construction, reference contractions, host conversion, and correctness checks.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--backend', choices=('torch', 'torch_cutensor'), default='torch')
parser.add_argument('--operation', choices=('mask', 'merge', 'unmerge', 'measure'), required=True)
parser.add_argument('--D', type=int, default=12)
parser.add_argument('--blocks', type=int, default=512)
parser.add_argument('--chi', type=int, default=24)
parser.add_argument('--chunk-size', type=int, default=6)
parser.add_argument('--sym', choices=('SU2', 'SU2xU1'), default='SU2')
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--mask-tile', type=int, default=None)
parser.add_argument('--fuse-tile', type=int, default=None)
args = parser.parse_args()
if args.D < 12 or args.blocks < 2 or args.repeats < 1:
    parser.error('Require D >= 12, blocks >= 2, and repeats >= 1')
root = args.source.resolve()
sys.path.insert(0, str(root))
import numpy as np
import torch
import yastn
from yastn.backend import backend_torch as backend

assert Path(yastn.__file__).resolve() == root / 'yastn/__init__.py', yastn.__file__
torch.set_num_threads(1)
torch.manual_seed(42)
if args.device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
if args.mask_tile is not None:
    if args.mask_tile <= 0:
        parser.error('--mask-tile must be positive')
    backend._MASK_TILE = args.mask_tile
if args.fuse_tile is not None:
    os.environ['YASTN_FUSE_SCATTER_CHUNK'] = str(args.fuse_tile)
tile_decisions = []
if args.operation in ('merge', 'unmerge'):
    from yastn.backend import _backend_torch_backwards as kernels
    if hasattr(kernels, 'adaptive_index_tile'):
        choose_tile = kernels.adaptive_index_tile

        def record_tile(*arguments, **keywords):
            tile = choose_tile(*arguments, **keywords)
            tile_decisions.append(tile)
            return tile

        kernels.adaptive_index_tile = record_tile
D = args.D
if args.operation == 'mask':
    shape = (D,) * 4
    small = (D, D // 2, D, D)
    count, selected_count = D ** 4, int(np.prod(small))
    dtype = [('sln', np.int64, (2,)), ('Dn', np.int64, (4,)),
             ('sla', np.int64, (2,)), ('Da', np.int64, (4,)), ('tm', np.int64, (1,))]
    meta = np.array([((i * selected_count, (i + 1) * selected_count), small,
                      (i * count, (i + 1) * count), shape, (0,))
                     for i in range(args.blocks)], dtype=dtype)
    data = torch.randn(args.blocks * count, dtype=torch.float64, device=args.device)
    indices = torch.arange(0, 2 * (D // 2), 2, device=args.device)
    run = lambda: backend.apply_mask(data, {(0,): indices}, meta, args.blocks * selected_count, 1, 4)
    reference = data.view(args.blocks, *shape).index_select(2, indices).reshape(-1)
elif args.operation in ('merge', 'unmerge'):
    order = (1, 0, 2)
    meta, size = [], 0
    # Many small blocks plus one large block exercise both halves of hybrid.
    for i in range(args.blocks + 1):
        d = 4 * D if i == args.blocks else D
        shape, fused, count = (d, d, d), (d, d * d), d ** 3
        meta.append((slice(size, size + count), fused, slice(size, size + count), shape,
                     (slice(0, d), slice(0, d * d)), fused))
        size += count
    meta = tuple(meta)
    data = torch.randn(size, dtype=torch.float64, device=args.device)
    if args.operation == 'merge':
        run = lambda: backend.transpose_and_merge(data, order, meta, size)
        reference = torch.cat([data[slo].view(Do).permute(order).reshape(-1)
                               for _, _, slo, Do, _, _ in meta])
    else:
        unmeta = tuple((slo, Dns, sln, Dn, ssln) for sln, Dn, slo, Do, ssln, Dns in meta)
        run = lambda: backend.unmerge(data, unmeta, size)
        reference = data
else:
    from yastn.tn import fpeps
    ops = yastn.operators.SpinfulFermions_tJ(sym=args.sym, backend=args.backend, default_device=args.device)
    config = ops.config
    config.backend.random_seed(seed=42)
    if args.sym == 'SU2':
        charges, dimensions = (0, 1), (D // 2, D - D // 2)
        boundary_charges = (0, 2)
    else:
        charges, dimensions = ((0, 0), (1, 1), (1, -1)), (D // 3, D // 3, D - 2 * (D // 3))
        boundary_charges = ((0, 0), (2, 0))
    leg = yastn.Leg(config, s=1, t=charges, D=dimensions)
    boundary = yastn.Leg(config, s=1, t=boundary_charges,
                         D=(2 * args.chi // 3, args.chi - 2 * args.chi // 3))
    tensor = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj(), ops.I().get_legs(0)))
    psi = fpeps.Peps(fpeps.SquareLattice((2, 2), boundary='infinite'), tensors=tensor)
    env = fpeps.EnvCTM(psi, init='rand', leg=boundary)
    local = ops.n_total()
    run = lambda: env.measure_2x2(local, local, sites=((0, 0), (1, 1)), chunk_size=args.chunk_size)
    with torch.no_grad():
        reference = env.measure_2x2(local, local, sites=((0, 0), (1, 1)))


def synchronize():
    if args.device == 'cuda':
        torch.cuda.synchronize()


def sample():
    gc.collect()
    synchronize()
    if args.device == 'cuda':
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        result = run()
    synchronize()
    seconds = time.perf_counter() - start
    extra_peak = torch.cuda.max_memory_allocated() - allocated if args.device == 'cuda' else None
    assert result.numel() == reference.numel() > 0
    torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-11)
    assert result.abs().max().item() > 0
    return dict(seconds=seconds, extra_peak_allocated_bytes=extra_peak,
                checked_elements=result.numel(), checksum=result.sum().item())


cold = sample()
warm = [sample() for _ in range(args.repeats)]
print(json.dumps(dict(status='complete', source=str(root), device=args.device,
                      backend=args.backend, torch=torch.__version__, operation=args.operation,
                      sym=args.sym if args.operation == 'measure' else None,
                      D_reduced=D, blocks=args.blocks, chi=args.chi, chunk_size=args.chunk_size,
                      mask_tile=getattr(backend, "_MASK_TILE", None), fuse_tile=args.fuse_tile,
                      adaptive_tiles=sorted(set(tile_decisions)),
                      cold=cold, warm=warm,
                      warm_median_seconds=statistics.median(x['seconds'] for x in warm))), flush=True)
