"""Ablate three non-Abelian metadata changes on CPU or CUDA.

The baseline tree must be the unmodified parent of the candidate. Reference
functions are loaded from that tree without importing a second yastn package.
Run --variants native in the baseline tree for an independent baseline check.
D in the primitive cases is reduced D=12; NTU uses full D=16 checkpoints.
"""
import argparse
import ast
from contextlib import contextmanager
import json
import math
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yastn
from yastn.tensor import _contractions, _merging, _nonabelian, linalg
from yastn.tn import fpeps, mps


def emit(**record):
    print(json.dumps(record), flush=True)


def reference_function(root, module, name):
    path = root / 'yastn' / 'tensor' / (module.__name__.split('.')[-1] + '.py')
    nodes = ast.parse(path.read_text()).body
    node, = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name]
    node.decorator_list = []
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@contextmanager
def variant(name, root):
    if name == 'native':
        yield
        return
    replacements = []

    def replace(module, key, value):
        replacements.append((module, key, getattr(module, key)))
        setattr(module, key, value)

    if name not in ('coefficient', 'all'):
        replace(_nonabelian, '_prefix_fusion_coefficient', lambda *args: None)
    if name not in ('lookup', 'all'):
        replace(_nonabelian, '_compute_fusion_weights',
                reference_function(root, _nonabelian, '_compute_fusion_weights'))
    if name not in ('transpose', 'all'):
        old = reference_function(root, _merging, '_apply_fusion_tree_transpose')
        for module in (_merging, _contractions, linalg):
            replace(module, '_apply_fusion_tree_transpose', old)
    try:
        yield
    finally:
        for module, key, old in reversed(replacements):
            setattr(module, key, old)


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def primitives(args):
    config = yastn.make_config(sym=args.sym, backend=args.backend, default_device=args.device)
    config.backend.random_seed(123)
    charges = ((0,), (1,), (2,)) if args.sym == 'SU2' else (
        (0, 0), (0, 2), (1, -1), (1, 1), (2, 0), (2, 2))
    leg = yastn.Leg(config, s=1, t=charges, D=(12 // len(charges),) * len(charges))
    a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()), dtype='complex128')
    assert sum(leg.D) == 12 and a.nblocks > 0
    emit(kind='tensor', sym=args.sym, reduced_D=sum(leg.D),
         full_D=sum(d * config.sym.irrep_dimension(t) for t, d in zip(leg.t, leg.D)),
         blocks=a.nblocks, elements=a.size)
    operations = {
        'prefix_fuse': lambda: a.fuse_legs(axes=((0, 1, 2), 3), mode='hard'),
        'two_pair_fuse': lambda: a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard'),
        'transpose': lambda: _merging._apply_fusion_tree_transpose(a, (3, 2, 1, 0)),
    }
    checked = 0
    for operation, fn in operations.items():
        expected = None
        for name in args.variants:
            with variant(name, args.baseline_tree):
                cold, warm = [], []
                for _ in range(args.rounds):
                    yastn.clear_cache()
                    synchronize(args.device)
                    start = perf_counter()
                    result = fn()
                    synchronize(args.device)
                    cold.append(perf_counter() - start)
                    start = perf_counter()
                    for _ in range(10):
                        result = fn()
                    synchronize(args.device)
                    warm.append((perf_counter() - start) / 10)
                data = config.backend.to_numpy(result.data)
                assert data.size > 0 and np.isfinite(data).all() and np.linalg.norm(data) > 0
                if expected is None:
                    expected = (result.struct, data.copy())
                assert result.struct == expected[0]
                np.testing.assert_allclose(data, expected[1], atol=1e-11, rtol=1e-11)
                checked += 1
                emit(kind='timing', sym=args.sym, variant=name, operation=operation,
                     cold_seconds=median(cold), warm_seconds=median(warm),
                     cold_samples=cold, warm_samples=warm,
                     max_abs_diff=float(np.max(np.abs(data - expected[1]))))
    emit(status='completed', checked_cases=checked)


def weights(args):
    """Measure weight-cache misses while coefficient/topology dependencies are warm."""
    config = yastn.make_config(sym=args.sym, backend='np')
    charges = tuple((j,) if args.sym == 'SU2' else (j, q)
                    for j in range(6) for q in ((0,) if args.sym == 'SU2' else (-1, 1)))
    leg = yastn.Leg(config, s=1, t=charges, D=(12 // len(charges),) * len(charges))
    a = yastn.ones(config, legs=(leg, leg, leg.conj(), leg.conj()))
    calls = []
    original = _nonabelian._compute_fusion_weights

    def capture(*values):
        calls.append(values)
        return original(*values)

    yastn.clear_cache()
    _nonabelian._compute_fusion_weights = capture
    try:
        a.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
    finally:
        _nonabelian._compute_fusion_weights = original
    assert calls and a.size > 0 and sum(leg.D) == 12
    expected = None
    for name in ('baseline', 'lookup'):
        with variant(name, args.baseline_tree):
            fn = _nonabelian._compute_fusion_weights
            result = [fn(*values) for values in calls]
            samples = []
            for _ in range(args.rounds):
                start = perf_counter()
                for _ in range(10):
                    result = [fn(*values) for values in calls]
                samples.append((perf_counter() - start) / 10)
            if expected is None:
                expected = result
            for (found, lengths), (wanted, ref_lengths) in zip(result, expected):
                assert len(found) > 0
                np.testing.assert_array_equal(lengths, ref_lengths)
                np.testing.assert_allclose(found, wanted, atol=1e-12, rtol=1e-12)
            emit(kind='timing', sym=args.sym, variant=name, operation='weight_cache_miss',
                 seconds=median(samples), samples=samples, blocks=a.nblocks,
                 reduced_D=sum(leg.D), calls=len(calls), weights=sum(len(item[0]) for item in result))
    emit(status='completed', checked_cases=2)


def ntu(args):
    ops = yastn.operators.SpinfulFermions_tJ(sym=args.sym, backend=args.backend,
                                          default_device=args.device)
    identity = ops.I()
    cp, c, spin, number = ops.cp_spinor(), ops.c_spinor(), ops.S(), ops.n_total()
    hamiltonian = mps.generate_mpo(identity, [
        mps.Hterm(-1., (0, 1), (cp, c)), mps.Hterm(-1., (1, 0), (cp, c)),
        mps.Hterm(0.3, (0, 1), (spin, spin)), mps.Hterm(-0.3 / 4, (0, 1), (number, number)),
        mps.Hterm(-0.2 / 4, (0,), (number,)), mps.Hterm(-0.2 / 4, (1,), (number,)),
    ], N=2)
    gate = fpeps.gates.gate_nn_exp(0.02, identity, hamiltonian)
    geometry = fpeps.CheckerboardLattice()
    gates = fpeps.gates.distribute(geometry, gates_nn=gate, symmetrize=False)
    if args.prepare_checkpoint:
        psi = fpeps.product_peps(geometry, identity)
        bp = fpeps.EnvBP(psi)
        for step in range(1, 6):
            result = fpeps.evolution_step_(bp, gates, opts_svd={'D_total': 16, 'tol': 1e-12},
                                           max_iter=1, pinv_cutoffs=(1e-10,))
            minimum = min(sum(d * ops.config.sym.irrep_dimension(t) for t, d in zip(leg.t, leg.D))
                          for site in psi.sites() for leg in psi[site].get_legs(range(4)))
            emit(kind='prepare', step=step, updates=len(result), min_full_D=minimum)
            if minimum >= 12:
                break
        args.prepare_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(psi.to_dict(level=2), args.prepare_checkpoint)
        args.checkpoint = args.prepare_checkpoint
    else:
        psi = fpeps.Peps.from_dict(torch.load(args.checkpoint, weights_only=False), config=ops.config)
    dimensions = [sum(d * ops.config.sym.irrep_dimension(t) for t, d in zip(leg.t, leg.D))
                  for site in psi.sites() for leg in psi[site].get_legs(range(4))]
    assert dimensions and min(dimensions) >= 12
    emit(kind='checkpoint', path=str(args.checkpoint), full_dimensions=dimensions,
         blocks=[psi[s].nblocks for s in psi.sites()])
    reference_error, checked = None, 0
    for name in args.variants:
        with variant(name, args.baseline_tree), torch.no_grad():
            yastn.clear_cache()
            samples = []
            for repeat in range(args.rounds + 1):
                current = psi.clone()
                env = fpeps.EnvNTU(current, which='NN')
                synchronize(args.device)
                emit(kind='start', variant=name, repeat=repeat)
                start = perf_counter()
                result = fpeps.evolution_step_(env, [gates[0]],
                    opts_svd={'D_total': 16, 'tol': 1e-12}, initialization='SVD',
                    max_iter=1, pinv_cutoffs=(1e-10,), fix_metric=None)
                synchronize(args.device)
                seconds = perf_counter() - start
                assert len(result) == 1 and math.isfinite(result[0].truncation_error)
                error = float(result[0].truncation_error)
                if reference_error is None:
                    reference_error = error
                assert abs(error - reference_error) < 1e-7, (name, error, reference_error)
                for site in current.sites():
                    data = ops.config.backend.to_numpy(current[site].data)
                    assert data.size > 0 and np.isfinite(data).all() and np.linalg.norm(data) > 0
                checked += 1
                samples.append(seconds)
                emit(kind='sample', variant=name, repeat=repeat, cold=repeat == 0,
                     seconds=seconds, truncation_error=error,
                     caches={key: value._asdict() for key, value in yastn.get_cache_info().items()
                             if key in ('fusion_tree_transpose', 'fusion_tree_adjacent_swap',
                                        'torch_recoupling_plan', 'fusion_weights')})
            emit(kind='timing', sym=args.sym, variant=name, operation='NTU_one_bond',
                 cold_seconds=samples[0], warm_seconds=median(samples[1:]), warm_samples=samples[1:])
    emit(status='completed', checked_updates=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sym', required=True, choices=['SU2', 'SU2xU1'])
    parser.add_argument('--backend', choices=['np', 'torch'], default='torch')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--baseline-tree', type=Path, required=True)
    parser.add_argument('--variants', nargs='+',
                        choices=['baseline', 'coefficient', 'transpose', 'lookup', 'all', 'native'],
                        default=['baseline', 'coefficient', 'transpose', 'lookup', 'all'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--prepare-checkpoint', type=Path)
    parser.add_argument('--weights', action='store_true')
    parser.add_argument('--plan-cache-size', type=int)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.weights:
        assert not (args.checkpoint or args.prepare_checkpoint)
        args.backend, args.device = 'np', 'cpu'
    assert args.rounds > 0 and (args.device == 'cpu' or args.backend == 'torch')
    expected = Path(__file__).resolve().parents[1] / 'yastn' / '__init__.py'
    assert Path(yastn.__file__).resolve() == expected, yastn.__file__
    torch.set_num_threads(1)
    if args.plan_cache_size is not None:
        assert args.plan_cache_size >= 0
        yastn.set_cache_maxsize({'fusion_tree_transpose': args.plan_cache_size})
    emit(kind='environment', source=yastn.__file__, torch=torch.__version__,
         backend=args.backend, device=args.device, threads=torch.get_num_threads())
    assert not (args.checkpoint and args.prepare_checkpoint)
    operation = ntu if args.checkpoint or args.prepare_checkpoint else weights if args.weights else primitives
    operation(args)


if __name__ == '__main__':
    main()
