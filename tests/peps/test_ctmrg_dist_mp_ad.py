# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Small CPU checks for the AD-aware distributed CTM implementation."""

import gc

import pytest

import yastn
import yastn.tn.fpeps as fpeps


ENV_FIELDS = ("tl", "t", "tr", "r", "br", "b", "bl", "l")
PROJ_FIELDS = ("hlt", "hlb", "hrt", "hrb", "vtl", "vtr", "vbl", "vbr")


def _make_env(n_sites=1):
    """Return a deterministic, nontrivial 1-by-n environment on CPU."""
    config = yastn.make_config(
        backend="torch", sym="none", default_device="cpu",
        default_dtype="float64")
    config.backend.random_seed(seed=20261007)

    leg = yastn.Leg(config, s=1, D=(2,))
    geometry = fpeps.SquareLattice(dims=(1, n_sites), boundary="infinite")
    tensors = {
        site: yastn.rand(
            config, legs=(leg, leg, leg.conj(), leg.conj()))
        for site in geometry.sites()
    }
    psi = fpeps.Peps(geometry, tensors=tensors)
    return fpeps.EnvCTM(psi, init="rand", leg=leg)


def _input_leaves(env):
    """Make all tensors read by the CTM move independent AD leaves."""
    tensors = [env.psi[site] for site in env.sites()]
    tensors += [getattr(env[site], name)
                for site in env.sites() for name in ENV_FIELDS]
    leaves = [tensor._data for tensor in tensors]
    for leaf in leaves:
        leaf.requires_grad_(True)
    return leaves


def _output_data(env):
    """Collect the environment and nontrivial projectors in fixed order."""
    tensors = [getattr(env[site], name)
               for site in env.sites() for name in ENV_FIELDS]
    tensors += [getattr(env.proj[site], name)
                for site in env.sites() for name in PROJ_FIELDS
                if getattr(env.proj[site], name) is not None]
    return [tensor._data for tensor in tensors]


def _weighted_loss(outputs):
    """A deterministic cotangent that probes every element of every output."""
    torch = pytest.importorskip("torch")
    loss = outputs[0].new_zeros(())
    for tensor_index, tensor in enumerate(outputs, start=1):
        weights = torch.arange(
            1, tensor.numel() + 1, dtype=tensor.dtype,
            device=tensor.device)
        loss = loss + tensor_index * (tensor.reshape(-1) * weights).sum()
    return loss


def _run_update(base_env, *, move, intra_site_parallel,
                devices=("cpu", "cpu")):
    from yastn.tn.fpeps.envs._env_ctm_dist_mp_AD import (
        get_or_create_pool, update_AD_)

    env = base_env.clone()
    leaves = _input_leaves(env)
    devices = list(devices)
    pool = get_or_create_pool(devices, env.config)
    submitted_forward_jobs = {"stage12": [], "stage3": []}
    original_submit = pool.submit

    def recording_submit(rank, payload):
        if payload[0].endswith("_fwd"):
            stage = payload[0].removesuffix("_fwd")
            extra = payload[2]
            job = tuple(extra["pairs"]) if stage == "stage12" else extra["move"]
            submitted_forward_jobs[stage].append((rank, job))
        return original_submit(rank, payload)

    pool.submit = recording_submit
    try:
        update_AD_(
            env,
            opts_svd={"policy": "fullrank", "D_total": 2, "tol": 1e-12,
                      "fix_signs": True},
            moves=move,
            method="2x2",
            devices=devices,
            use_qr=False,
            intra_site_parallel=intra_site_parallel,
        )
        outputs = _output_data(env)
        _weighted_loss(outputs).backward()
    finally:
        pool.submit = original_submit

    detached_outputs = [tensor.detach().clone() for tensor in outputs]
    assert all(leaf.grad is not None for leaf in leaves)
    gradients = [leaf.grad.detach().clone() for leaf in leaves]
    return detached_outputs, gradients, submitted_forward_jobs


@pytest.mark.parametrize("move", ["h", "v"])
def test_one_site_intra_site_parallel_cpu_forward_and_backward(move):
    """Intra-site dispatch preserves one-site CTM outputs and gradients."""
    torch = pytest.importorskip("torch")
    base_env = _make_env()

    # Finish backward before enabling the alternative dispatch.  Dropping the
    # first graph also lets the persistent workers release their saved graphs.
    outputs_ref, gradients_ref, jobs_ref = _run_update(
        base_env, move=move, intra_site_parallel=False)
    gc.collect()

    outputs_parallel, gradients_parallel, jobs_parallel = _run_update(
        base_env, move=move, intra_site_parallel=True)

    assert {rank for jobs in jobs_ref.values() for rank, _ in jobs} == {0}
    assert {rank for jobs in jobs_parallel.values() for rank, _ in jobs} == {0, 1}

    assert len(outputs_parallel) == len(outputs_ref)
    for actual, expected in zip(outputs_parallel, outputs_ref):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)

    assert len(gradients_parallel) == len(gradients_ref)
    for actual, expected in zip(gradients_parallel, gradients_ref):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("move", ["h", "v"])
def test_two_site_directional_parallel_cpu_forward_and_backward(move):
    """Directional Stage-3 jobs use spare ranks without changing the map."""
    torch = pytest.importorskip("torch")
    base_env = _make_env(n_sites=2)
    devices = ("cpu", "cpu", "cpu")

    outputs_ref, gradients_ref, jobs_ref = _run_update(
        base_env, move=move, intra_site_parallel=False, devices=devices)
    gc.collect()

    outputs_parallel, gradients_parallel, jobs_parallel = _run_update(
        base_env, move=move, intra_site_parallel=True, devices=devices)

    # Stage 1/2 has one job per site. The baseline pins each site's two
    # directional Stage-3 jobs to that site's rank. Load-aware directional
    # scheduling assigns the third job to the otherwise idle rank 2, then
    # keeps the fourth job on its equally-loaded projector-producing rank 1.
    pairs = ("rh", "lh") if move == "h" else ("tv", "bv")
    submoves = ("l", "r") if move == "h" else ("t", "b")
    assert jobs_ref["stage12"] == [(0, pairs), (1, pairs)]
    assert jobs_parallel["stage12"] == [(0, pairs), (1, pairs)]
    assert jobs_ref["stage3"] == [
        (0, submoves[0]), (0, submoves[1]),
        (1, submoves[0]), (1, submoves[1]),
    ]
    assert jobs_parallel["stage3"] == [
        (0, submoves[0]), (1, submoves[1]),
        (2, submoves[0]), (1, submoves[1]),
    ]

    assert len(outputs_parallel) == len(outputs_ref)
    for actual, expected in zip(outputs_parallel, outputs_ref):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)

    assert len(gradients_parallel) == len(gradients_ref)
    for actual, expected in zip(gradients_parallel, gradients_ref):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
