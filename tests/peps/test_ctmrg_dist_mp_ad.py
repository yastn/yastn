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


def _make_env(n_sites=1, sym='none'):
    """Return a deterministic, nontrivial 1-by-n environment on CPU."""
    config = yastn.make_config(
        backend="torch", sym=sym, default_device="cpu",
        default_dtype="float64")
    config.backend.random_seed(seed=20261007)

    leg = (yastn.Leg(config, s=1, D=(2,)) if sym == 'none'
           else yastn.Leg(config, s=1, t=(0, 1), D=(1, 1)))
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


def _run_update(base_env, *, move, parallelism, devices=("cpu", "cpu")):
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
            parallelism=parallelism,
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
@pytest.mark.parametrize("parallelism", ["auto", "intra_site"])
def test_one_site_intra_site_parallel_cpu_forward_and_backward(
        move, parallelism):
    """Intra-site dispatch preserves one-site CTM outputs and gradients."""
    torch = pytest.importorskip("torch")
    base_env = _make_env()

    # Finish backward before enabling the alternative dispatch.  Dropping the
    # first graph also lets the persistent workers release their saved graphs.
    outputs_ref, gradients_ref, jobs_ref = _run_update(
        base_env, move=move, parallelism="sites")
    gc.collect()

    outputs_parallel, gradients_parallel, jobs_parallel = _run_update(
        base_env, move=move, parallelism=parallelism)

    assert {rank for jobs in jobs_ref.values() for rank, _ in jobs} == {0}
    assert {rank for jobs in jobs_parallel.values() for rank, _ in jobs} == {0, 1}

    assert len(outputs_parallel) == len(outputs_ref)
    for actual, expected in zip(outputs_parallel, outputs_ref):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)

    assert len(gradients_parallel) == len(gradients_ref)
    for actual, expected in zip(gradients_parallel, gradients_ref):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_intra_site_policy_degrades_to_sites_with_one_worker():
    """Do not duplicate the fused halves when no second worker can help."""
    base_env = _make_env()
    _, _, jobs = _run_update(
        base_env, move="h", parallelism="intra_site", devices=("cpu",))

    assert jobs["stage12"] == [(0, ("rh", "lh"))]
    assert jobs["stage3"] == [(0, "l"), (0, "r")]


@pytest.mark.parametrize("move", ["h", "v"])
@pytest.mark.parametrize("parallelism", ["auto", "intra_site"])
def test_two_site_directional_parallel_cpu_forward_and_backward(
        move, parallelism):
    """Directional Stage-3 jobs use spare ranks without changing the map."""
    torch = pytest.importorskip("torch")
    base_env = _make_env(n_sites=2)
    devices = ("cpu", "cpu", "cpu")

    outputs_ref, gradients_ref, jobs_ref = _run_update(
        base_env, move=move, parallelism="sites", devices=devices)
    gc.collect()

    outputs_parallel, gradients_parallel, jobs_parallel = _run_update(
        base_env, move=move, parallelism=parallelism, devices=devices)

    # Site scheduling has one Stage-12 job per site and pins both directional
    # Stage-3 jobs to that rank. Auto splits only enough projector jobs to use
    # the spare worker; intra_site splits both sites into independent pairs.
    pairs = ("rh", "lh") if move == "h" else ("tv", "bv")
    submoves = ("l", "r") if move == "h" else ("t", "b")
    assert jobs_ref["stage12"] == [(0, pairs), (1, pairs)]
    if parallelism == "auto":
        assert jobs_parallel["stage12"] == [
            (0, (pairs[0],)), (1, (pairs[1],)), (2, pairs)]
    else:
        assert jobs_parallel["stage12"] == [
            (0, (pairs[0],)), (1, (pairs[1],)),
            (2, (pairs[0],)), (0, (pairs[1],)),
        ]
    assert jobs_ref["stage3"] == [
        (0, submoves[0]), (0, submoves[1]),
        (1, submoves[0]), (1, submoves[1]),
    ]
    if move == "h":
        expected_stage3 = [
            (1, "l"), (0, "r"), (2, "l"), (2, "r")]
    elif parallelism == "auto":
        expected_stage3 = [
            (0, "t"), (1, "b"), (2, "t"), (2, "b")]
    else:
        expected_stage3 = [
            (0, "t"), (1, "b"), (2, "t"), (0, "b")]
    assert jobs_parallel["stage3"] == expected_stage3

    assert len(outputs_parallel) == len(outputs_ref)
    for actual, expected in zip(outputs_parallel, outputs_ref):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)

    assert len(gradients_parallel) == len(gradients_ref)
    for actual, expected in zip(gradients_parallel, gradients_ref):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"moves": "l"}, "simultaneous 'h' and 'v' moves"),
        ({"method": "1x2"}, "2x2 projector method"),
        ({"checkpoint_move": "nonreentrant"}, "checkpoint_move"),
    ],
)
def test_distributed_ad_rejects_unsupported_options_before_dispatch(
        kwargs, message):
    """The AD path must not silently ignore serial-only CTM features."""
    from yastn.tn.fpeps.envs._env_ctm_dist_mp_AD import update_AD_

    env = _make_env()
    with pytest.raises(yastn.YastnError, match=message):
        update_AD_(env, devices=["cpu", "cpu"], **kwargs)


def _basis_data(env):
    return [getattr(lattice[site], name)._data
            for lattice in (env.si_X, env.si_Y)
            for site in env.sites() for name in PROJ_FIELDS
            if getattr(lattice[site], name) is not None]


def test_metadata_cache_roundtrips_symmetries_and_statistics():
    """Same-shaped tensors retain their own configuration in worker payloads."""
    from yastn.tn.fpeps.envs._env_ctm_dist_mp_AD import _split_tensor

    for sym, fermionic in (('U1', False), ('Z2', False), ('Z2', True)):
        config = yastn.make_config(backend='torch', sym=sym, fermionic=fermionic)
        leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
        tensor = yastn.eye(config, legs=leg, isdiag=False)
        data, meta = _split_tensor(tensor)
        restored = yastn.Tensor.from_dict({**meta, 'data': data}, config=config)
        assert (restored - tensor).norm().item() == 0


@pytest.mark.parametrize('use_qr', [False, True])
@pytest.mark.parametrize('recycle_grad', [False, True])
@pytest.mark.parametrize('parallelism', ['sites', 'intra_site', 'auto'])
def test_si_recycled_sweeps_match_serial_outputs_and_gradients(
        use_qr, recycle_grad, parallelism):
    """Recycled bases and their gradients survive worker/rank boundaries."""
    from yastn.tn.fpeps.envs import make_ctm_opts
    from yastn.tn.fpeps.envs._env_ctm_dist_mp_AD import update_AD_

    torch = pytest.importorskip('torch')
    base = _make_env(n_sites=2)
    opts = make_ctm_opts(
        opts_svd={'D_total': 2, 'tol': 1e-12, 'fix_signs': True},
        moves='hv', method='2x2', use_qr=use_qr,
        opts_si={'enabled': True, 'oversampling': 1, 'niter': 2,
                 'tol': 0., 'recycle_grad': recycle_grad})
    # Give both paths identical initial subspaces, independent of worker RNG.
    with torch.no_grad():
        base.update_(opts=opts)

    results = []
    for distributed in (False, True):
        env = base.clone()
        leaves = _input_leaves(env)
        for _ in range(2):
            if distributed:
                update_AD_(env, opts=opts, devices=['cpu'] * 3,
                           parallelism=parallelism)
            else:
                env.update_(opts=opts)
        bases = _basis_data(env)
        assert all(x.requires_grad == recycle_grad for x in bases)
        outputs = _output_data(env)
        loss = _weighted_loss(outputs + (bases if recycle_grad else []))
        # FP Neumann iterations reuse the same saved worker graphs.
        gradients = torch.autograd.grad(loss, leaves, retain_graph=True)
        repeated = torch.autograd.grad(loss, leaves)
        for actual, expected in zip(repeated, gradients):
            torch.testing.assert_close(actual, expected)
        results.append(([x.detach().clone() for x in outputs + bases],
                        gradients, env._si_age.copy()))

    actual, expected = results[1], results[0]
    for a, e in zip(actual[0], expected[0]):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
    for a, e in zip(actual[1], expected[1]):
        torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-9)
    assert actual[2].keys() == expected[2].keys()
    for key, state in actual[2].items():
        reference = expected[2][key]
        assert state.age == reference.age == 3
        assert state.niter == reference.niter
        assert state.rank == reference.rank
        assert state.bases == reference.bases
        assert state.error == pytest.approx(reference.error, abs=1e-12)


def test_si_fresh_bases_and_skip_schedule_in_distributed_iteration():
    """No-grad convergence initializes X/Y and returns diagnostics each sweep."""
    from yastn.tn.fpeps.envs._env_ctm_dist_mp_AD import iterate_AD_, fp_update_
    from yastn.tn.fpeps.envs import make_ctm_opts

    env = _make_env()
    opts = make_ctm_opts(
        opts_svd={'D_total': 2}, devices=['cpu', 'cpu'], max_sweeps=2,
        use_qr=False, opts_si={'enabled': True, 'oversampling': 1,
                              'niter': 2, 'warmup': 0, 'tol': 1e100,
                              'skip_SI_update': True})
    iterator = iterate_AD_(env, opts=opts)
    assert next(iterator).sweeps == 1
    assert len(env._si_age) == 4
    first = env._si_age.copy()
    assert all(s.age == 1 and s.niter > 0 for s in first.values())
    assert all(not x.requires_grad for x in _basis_data(env))
    assert next(iterator).sweeps == 2
    assert all(s.age == 2 and s.niter == 0 for s in env._si_age.values())
    assert all(s.error == first[key].error for key, s in env._si_age.items())
    # The FP dispatcher must honor SI with multiple devices, too.
    fp_update_(env, opts)
    assert all(s.age == 3 for s in env._si_age.values())


@pytest.mark.parametrize('sym', ['Z2', 'U1'])
@pytest.mark.parametrize('use_qr', [False, True])
def test_si_symmetric_redistribution_matches_serial(sym, use_qr):
    """Worker serialization preserves charge sectors and redistribution state."""
    from yastn.tn.fpeps.envs import make_ctm_opts
    from yastn.tn.fpeps.envs._env_ctm_dist_mp_AD import update_AD_

    torch = pytest.importorskip('torch')
    base = _make_env(sym=sym)
    opts = make_ctm_opts(
        opts_svd={'D_total': 2, 'tol': 0}, use_qr=use_qr,
        opts_si={'enabled': True, 'oversampling': 4, 'niter': 3,
                 'tol': 0, 'redistribute_sectors': True,
                 'refinement': 'sector_dimensions',
                 'warmup': 0, 'redistribute_frequency': 1})
    with torch.no_grad():
        base.update_(opts=opts)
        serial, distributed = base.clone(), base.clone()
        for _ in range(2):
            serial.update_(opts=opts)
            update_AD_(distributed, opts=opts, devices=['cpu', 'cpu'])
    assert distributed.is_consistent()
    for actual, expected in zip(_output_data(distributed), _output_data(serial)):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    for key, state in distributed._si_age.items():
        assert state.age == serial._si_age[key].age == 3
        assert state.rank == serial._si_age[key].rank


def test_fp_si_convergence_uses_distributed_workers(monkeypatch):
    """SI forward convergence must honor an explicit multi-device selection."""
    from yastn.tn.fpeps.envs import make_ctm_opts
    from yastn.tn.fpeps.envs.fixed_pt import FixedPoint

    env = _make_env()
    opts = make_ctm_opts(
        opts_svd={'D_total': 2}, max_sweeps=2, corner_tol=None,
        use_qr=False, opts_si={'enabled': True, 'niter': 2})

    def unexpected_serial(*args, **kwargs):
        pytest.fail('SI convergence fell back to serial CTM')

    monkeypatch.setattr(env, 'ctmrg_', unexpected_serial)
    result, _, _, _, _ = FixedPoint.get_converged_env(
        env, opts, devices=['cpu', 'cpu'])
    assert result is env
    assert all(state.age == 2 for state in env._si_age.values())
