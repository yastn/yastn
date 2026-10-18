# Copyright 2024 The YASTN Authors. All Rights Reserved.
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
""" Multiprocess CTMRG (:mod:`_env_ctm_dist_mp`) against the serial path.

The distributed path does not need several GPUs to be exercised: the only
device-specific step is pinning the CUDA context, guarded by
``device.startswith("cuda")``, so ``devices=['cpu', 'cpu']`` spawns the real
worker processes and drives the full dispatch. It does need the torch backend,
which is where ``config.backend.mp`` lives.
"""
import pytest

import yastn
import yastn.tn.fpeps as fpeps


def _needs_torch_mp(config_kwargs):
    if 'torch' not in config_kwargs.get('backend', 'np'):
        pytest.skip("distributed CTM needs the torch backend (config.backend.mp)")


def _random_peps(config_kwargs, sym, D, seed=7):
    cfg = yastn.make_config(sym=sym, **config_kwargs)
    if hasattr(cfg.backend, 'random_seed'):
        cfg.backend.random_seed(seed=seed)
    geometry = fpeps.CheckerboardLattice()
    if sym == 'U1':
        leg = yastn.Leg(cfg, s=1, t=(0, 1), D=(D // 2, D - D // 2))
        pleg = yastn.Leg(cfg, s=1, t=(0, 1), D=(1, 1))
    else:
        leg = yastn.Leg(cfg, s=1, D=(D,))
        pleg = yastn.Leg(cfg, s=1, D=(2,))
    tensors = {site: yastn.rand(cfg, legs=[leg, leg, leg.conj(), leg.conj(), pleg])
               for site in geometry.sites()}
    return fpeps.Peps(geometry, tensors=tensors)


def _max_env_diff(env_a, env_b):
    """Largest elementwise difference over the environment tensors of two envs."""
    worst = 0.0
    for site in env_a.sites():
        for dirn in ('tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r'):
            a, b = getattr(env_a[site], dirn), getattr(env_b[site], dirn)
            if a is None or b is None:
                continue
            assert a.get_legs() == b.get_legs(), f"leg mismatch at {site} {dirn}"
            worst = max(worst, (a - b).norm().item())
    return worst


@pytest.mark.parametrize('sym', ['none', 'U1'])
def test_dist_mp_matches_serial(config_kwargs, sym):
    """ The multiprocess sweep reproduces the serial one exactly. """
    _needs_torch_mp(config_kwargs)
    from yastn.tn.fpeps.envs._env_ctm_dist_mp import iterate_D_

    psi = _random_peps(config_kwargs, sym, D=2)
    opts_svd = {'D_total': 8, 'tol': 1e-10}

    serial = fpeps.EnvCTM(psi, init='eye')
    serial.ctmrg_(opts_svd=dict(opts_svd), max_sweeps=3)

    dist = fpeps.EnvCTM(psi, init='eye')
    info = iterate_D_(dist, opts_svd=dict(opts_svd), max_sweeps=3,
                      devices=['cpu', 'cpu'])

    assert info.sweeps == 3
    assert dist.max_D() == serial.max_D()
    assert _max_env_diff(serial, dist) < 1e-12


def test_dist_mp_partial_svd_policy(config_kwargs):
    """ Partial-SVD policies read k_block, which the h-moves never supplied.

    Before both branches predicted into k_block, this raised
    'block_arnoldi policy in svd requires passing argument D_block or k_block'
    inside the workers, and the main loop then blocked on a queue that never
    received results.
    """
    _needs_torch_mp(config_kwargs)
    from yastn.tn.fpeps.envs._env_ctm_dist_mp import iterate_D_

    psi = _random_peps(config_kwargs, 'U1', D=2)
    opts_svd = {'D_total': 8, 'tol': 1e-10, 'policy': 'block_arnoldi'}

    serial = fpeps.EnvCTM(psi, init='eye')
    serial.ctmrg_(opts_svd=dict(opts_svd), max_sweeps=3)

    dist = fpeps.EnvCTM(psi, init='eye')
    iterate_D_(dist, opts_svd=dict(opts_svd), max_sweeps=3, devices=['cpu', 'cpu'])

    assert _max_env_diff(serial, dist) < 1e-12


def test_dist_mp_does_not_mutate_caller_opts(config_kwargs):
    """ The per-sector prediction must not leak into the caller's dict.

    It used to be written into opts_svd between two Queue.put calls, which also
    made it a race: mp.Queue pickles on a background feeder thread.
    """
    _needs_torch_mp(config_kwargs)
    from yastn.tn.fpeps.envs._env_ctm_dist_mp import iterate_D_

    psi = _random_peps(config_kwargs, 'U1', D=2)
    opts_svd = {'D_total': 8, 'tol': 1e-10}

    env = fpeps.EnvCTM(psi, init='eye')
    iterate_D_(env, opts_svd=opts_svd, max_sweeps=2, devices=['cpu', 'cpu'])

    assert set(opts_svd) == {'D_total', 'tol'}, f"caller's opts_svd was mutated: {opts_svd}"


def test_dist_mp_requires_at_least_two_devices(config_kwargs):
    _needs_torch_mp(config_kwargs)
    from yastn.tn.fpeps.envs._env_ctm_dist_mp import iterate_D_

    psi = _random_peps(config_kwargs, 'none', D=2)
    env = fpeps.EnvCTM(psi, init='eye')
    with pytest.raises(yastn.YastnError, match='At least two devices'):
        iterate_D_(env, opts_svd={'D_total': 4}, max_sweeps=1, devices=['cpu'])
    with pytest.raises(yastn.YastnError, match='Devices list must be provided'):
        iterate_D_(env, opts_svd={'D_total': 4}, max_sweeps=1)



def test_dist_mp_corner_tol_and_iterator(config_kwargs):
    """ The distributed loop now shares the serial convergence check.

    It previously re-implemented it inline, which reported max_dsv on a
    different scale and did not accept a callable.
    """
    _needs_torch_mp(config_kwargs)
    from yastn.tn.fpeps.envs._env_ctm_dist_mp import iterate_D_

    psi = _random_peps(config_kwargs, 'U1', D=2)
    opts_svd = {'D_total': 8, 'tol': 1e-10}

    # numeric corner_tol: max_dsv is reported, and is NaN on the first sweep
    # (no history to compare against) exactly as on the serial path
    env = fpeps.EnvCTM(psi, init='eye')
    info = iterate_D_(env, opts_svd=dict(opts_svd), max_sweeps=4,
                      corner_tol=1e-12, devices=['cpu', 'cpu'])
    assert info.max_dsv is not None

    # generator form yields one CTMRG_out per step
    env = fpeps.EnvCTM(psi, init='eye')
    outs = list(iterate_D_(env, opts_svd=dict(opts_svd), max_sweeps=3,
                           iterator=True, devices=['cpu', 'cpu']))
    assert len(outs) == 3 and outs[-1].sweeps == 3

    # a callable convergence check is accepted, as on the serial path
    seen = []
    def check(e, history):
        seen.append(e.max_D())
        return len(seen) >= 2, history

    env = fpeps.EnvCTM(psi, init='eye')
    info = iterate_D_(env, opts_svd=dict(opts_svd), max_sweeps=5,
                      corner_tol=check, devices=['cpu', 'cpu'])
    assert len(seen) == 2 and info.converged and info.sweeps == 2


def test_dist_mp_rejects_unknown_option(config_kwargs):
    """ A typo no longer rides kwargs into the workers and on into svd. """
    _needs_torch_mp(config_kwargs)
    from yastn.tn.fpeps.envs._env_ctm_dist_mp import iterate_D_

    psi = _random_peps(config_kwargs, 'none', D=2)
    env = fpeps.EnvCTM(psi, init='eye')
    with pytest.raises(yastn.YastnError, match='not recognized'):
        iterate_D_(env, opts_svd={'D_total': 4}, max_sweep=2,
                   devices=['cpu', 'cpu'])


if __name__ == '__main__':
    pytest.main([__file__, '-vs', '--backend', 'torch'])


def test_use_qr_reaches_the_multi_device_forward(config_kwargs):
    """ get_converged_env used to force use_qr=False on the multi-device branch,
        silently overriding the caller. Measured before the fix: serial
        use_qr=True differed from use_qr=False, while both multi-device runs
        were bit-identical -- the argument never arrived.

        Asserts delivery, not a numerical value: the two settings must differ.
    """
    _needs_torch_mp(config_kwargs)
    from yastn.tn.fpeps.envs._env_ctm_dist_mp_AD import iterate_AD_

    psi = _random_peps(config_kwargs, 'U1', D=2)
    opts_svd = {'D_total': 8, 'tol': 1e-10}

    envs = {}
    for use_qr in (False, True):
        env = fpeps.EnvCTM(psi, init='eye')
        it = iterate_AD_(env, opts_svd=dict(opts_svd), moves='hv', method='2x2',
                         max_sweeps=2, use_qr=use_qr, devices=['cpu', 'cpu'])
        for _ in range(2):
            next(it)
        envs[use_qr] = env

    assert _max_env_diff(envs[False], envs[True]) > 1e-12, \
        "use_qr did not reach the multi-device projectors"


def test_fp_ctmrg_multi_device(config_kwargs):
    """ The distributed fixed point, end to end, on two cpu 'devices'.

    Exercises iterate_AD_ for the forward, the distributed FP CTM step, and the
    Neumann backward -- a path that has never had coverage.
    """
    _needs_torch_mp(config_kwargs)
    import torch
    from yastn.tn.fpeps.envs.fixed_pt import fp_ctmrg

    # A weakly-entangled state: close to a product state, so CTM converges in a
    # few sweeps. The point is to drive the distributed plumbing, not physics.
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    cfg.backend.random_seed(seed=3)
    geometry = fpeps.CheckerboardLattice()
    leg = yastn.Leg(cfg, s=1, t=(0, 1), D=(1, 1))
    pleg = yastn.Leg(cfg, s=1, t=(0, 1), D=(1, 1))
    tensors = {}
    for site in geometry.sites():
        base = yastn.ones(cfg, legs=[leg, leg, leg.conj(), leg.conj(), pleg], n=0)
        noise = yastn.rand(cfg, legs=[leg, leg, leg.conj(), leg.conj(), pleg], n=0)
        tensors[site] = base + 0.05 * noise
    psi = fpeps.Peps(geometry, tensors=tensors)
    for s in psi.sites():
        psi[s]._data.requires_grad_(True)

    env = fpeps.EnvCTM(psi, init='eye')
    # corner_tol is loose on purpose: this test drives plumbing, and the serial
    # path needs ~57 sweeps to reach 1e-8 on this state but only ~30 for 1e-5.
    fwd = {'opts_svd': {'D_total': 6, 'tol': 1e-10}, 'max_sweeps': 40,
           'corner_tol': 1e-5, 'method': '2x2', 'use_qr': False}
    out = fp_ctmrg(env, ctm_opts_fwd=fwd,
                   ctm_opts_fp={'opts_svd': {'policy': 'fullrank'}},
                   devices=['cpu', 'cpu'])
    assert out is not None

    # the gradient must flow back to the PEPS tensors through the Neumann loop
    loss = sum((out[s].tl.norm() for s in out.sites()))
    loss.backward()
    assert any(psi[s]._data.grad is not None and psi[s]._data.grad.abs().sum() > 0
               for s in psi.sites()), "no gradient reached the PEPS tensors"
