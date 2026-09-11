#!/usr/bin/env python3
# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SPMD (torch.distributed) correctness test for
``contract_with_unroll(distributed=True)`` — the multi-node
``_oe_blocksparse_dist`` path.

Every rank builds a structurally identical network, runs the distributed
contraction, and compares forward + backward against the *serial*
(``devices=None``) path on the same rank. Covers single-key (contracted-only
unroll) and multi-key (output-unrolled) contractions.

Launch (single node, N ranks):

    torchrun --nproc_per_node=2 tests/tensor/test_oe_blocksparse_dist.py
    torchrun --nproc_per_node=4 tests/tensor/test_oe_blocksparse_dist.py

Backend/device is auto-selected: ``nccl``/CUDA when GPUs are present (one rank
per GPU via ``LOCAL_RANK``), else ``gloo``/CPU (validates SPMD logic,
collectives and autograd without GPUs). Exit code 0 iff every rank passes.

A pytest wrapper (``test_distributed_via_torchrun``) launches the 2- and 4-rank
runs as subprocesses; it is skipped unless the torch backend is selected.
"""
import os
import sys
import subprocess

import pytest

# Under torchrun, sys.path[0] is this script's directory, so the source-tree
# ``yastn`` package (repo root) is not importable. Put the repo root first.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TOL = 1e-9
TOL_AD = 1e-7


# --------------------------------------------------------------------------- #
# Network + per-case builders (identical on every rank given a fixed seed).
# --------------------------------------------------------------------------- #
def _build(cfg, seed=1234):
    import torch
    import yastn
    torch.manual_seed(seed)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 3))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 2))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 3))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)
    return (A, B, C), (leg_i, leg_j, leg_k, leg_l)


# unroll specs: label -> list[SlicedLeg]. 'j'/'k' are contracted (single-key),
# 'i' is an output index (multi-key: assembled with block()).
def _unroll_contracted(legs):
    import yastn
    _, leg_j, _, _ = legs
    return {'j': yastn.make_sliced_legs(leg_j)}


def _unroll_output(legs):
    import yastn
    leg_i, _, _, _ = legs
    return {'i': yastn.make_sliced_legs(leg_i)}


def _unroll_mixed(legs):
    import yastn
    leg_i, leg_j, _, _ = legs
    return {'i': yastn.make_sliced_legs(leg_i), 'j': yastn.make_sliced_legs(leg_j)}


_CASES = [
    ("single-key (contracted 'j')", _unroll_contracted),
    ("multi-key (output 'i')", _unroll_output),
    ("mixed (output 'i' + contracted 'j')", _unroll_mixed),
]


def _contract(A, B, C, unroll, path, **kw):
    import yastn
    return yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll=unroll, optimize=path, **kw)


def _forward_diff(cfg, spec):
    import yastn
    (A, B, C), legs = _build(cfg)
    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'))
    unroll = spec(legs)
    serial = _contract(A, B, C, unroll, path, devices=None)
    distrib = _contract(A, B, C, unroll, path, distributed=True)
    return float(yastn.norm(distrib - serial))


def _backward_diff(cfg, spec):
    import yastn
    # serial reference
    (A, B, C), legs = _build(cfg)
    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'))
    for t in (A, B, C):
        t.requires_grad_(True)
    _contract(A, B, C, spec(legs), path, devices=None).norm().backward()
    ref_grads = [A.grad(), B.grad(), C.grad()]

    # distributed
    (A2, B2, C2), legs2 = _build(cfg)
    for t in (A2, B2, C2):
        t.requires_grad_(True)
    _contract(A2, B2, C2, spec(legs2), path, distributed=True).norm().backward()
    dist_grads = [A2.grad(), B2.grad(), C2.grad()]

    return max(float(yastn.norm(g - r)) for g, r in zip(dist_grads, ref_grads))


# --------------------------------------------------------------------------- #
# SPMD entry point (run under torchrun).
# --------------------------------------------------------------------------- #
def _run_spmd():
    import torch
    import torch.distributed as dist
    import yastn

    world = int(os.environ.get("WORLD_SIZE", 1))
    # Use CUDA/NCCL only when each rank can own a distinct GPU; otherwise fall
    # back to gloo/CPU (a single-GPU box can't give N>1 ranks their own device).
    use_cuda = (torch.cuda.is_available()
                and torch.cuda.device_count() >= world
                and os.environ.get("YASTN_DIST_TEST_CPU", "0") != "1")
    backend = 'nccl' if use_cuda else 'gloo'
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if use_cuda:
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        reduce_dev = device
    else:
        device = 'cpu'
        reduce_dev = 'cpu'

    cfg = yastn.make_config(sym='U1', backend='torch',
                            default_device=device, default_dtype='float64')

    failures = []
    for label, spec in _CASES:
        try:
            fdiff = _forward_diff(cfg, spec)
            if not (fdiff < TOL):
                failures.append(f"[{label}] forward diff {fdiff:.3e} >= {TOL:.0e}")
            bdiff = _backward_diff(cfg, spec)
            if not (bdiff < TOL_AD):
                failures.append(f"[{label}] backward diff {bdiff:.3e} >= {TOL_AD:.0e}")
            if rank == 0:
                print(f"rank0 {label}: fwd {fdiff:.2e} bwd {bdiff:.2e}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            failures.append(f"[{label}] EXCEPTION: {e}\n{traceback.format_exc()}")

    n_fail = torch.tensor([len(failures)], dtype=torch.int64, device=reduce_dev)
    dist.all_reduce(n_fail, op=dist.ReduceOp.SUM)
    if failures:
        print(f"rank {rank}/{world} FAILURES:\n  " + "\n  ".join(failures), flush=True)
    dist.barrier()
    if rank == 0:
        total = int(n_fail.item())
        print(f"\n=== distributed test ({backend}, world={world}): "
              f"{'PASS' if total == 0 else f'FAIL ({total} failures)'} ===", flush=True)
    dist.destroy_process_group()
    return 0 if int(n_fail.item()) == 0 else 1


# --------------------------------------------------------------------------- #
# pytest wrapper: launch the SPMD script under torchrun.
# --------------------------------------------------------------------------- #
def _torch_selected(request):
    try:
        return 'torch' in request.config.getoption('--backend')
    except Exception:
        return False


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_via_torchrun(request, nproc):
    if not _torch_selected(request):
        pytest.skip("Requires --backend torch")
    # No GPU-count gate: the SPMD script uses nccl/CUDA when each rank can own a
    # GPU and otherwise falls back to gloo/CPU, so it runs on any box.
    env = dict(os.environ)
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29555")
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={nproc}", "--nnodes=1", os.path.abspath(__file__)]
    proc = subprocess.run(cmd, env=env, cwd=os.getcwd(),
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError(
            f"torchrun (nproc={nproc}) failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


if __name__ == "__main__":
    sys.exit(_run_spmd())
