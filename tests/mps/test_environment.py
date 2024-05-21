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
""" basic methods of single Mps """
import numpy as np
import pytest
import yastn
import yastn.tn.mps as mps
try:
    from .configs import config_dense as cfg
except ImportError:
    from configs import config_dense as cfg
# pytest modifies cfg to inject different backends and divices during tests


def test_env2_update(config=cfg, tol=1e-12):
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    opts_config = {} if config is None else \
            {'backend': config.backend, 'default_device': config.default_device}
    N = 12
    for sym, n in [('U1', 0), ('Z2', 1), ('Z2', 0)]:
        ops = yastn.operators.Spin12(sym=sym, **opts_config)
        ops.random_seed(seed=0)
        I = mps.product_mpo(ops.I(), N)
        psi1 = mps.random_mps(I, D_total=15, n=n)
        psi2 = mps.random_mps(I, D_total=7, n=n)
        H1 = mps.random_mpo(I, D_total=10)
        H2 = mps.random_mpo(I, D_total=8)
        #
        check_env2_measure(psi1, psi2, tol)
        check_env2_measure(H1, H2, tol)


def check_env2_measure(psi1, psi2, tol):
    """ Test if different overlaps of psi1 and psi2 give consistent results. """
    N = psi1.N
    env = mps.Env(psi1, psi2)
    env.setup_(to='first')
    env.setup_(to='last')

    results = [env.measure()]
    for n in range(N - 1):
        results.append(env.measure(bd=(n, n + 1)))
    results.append(env.measure(bd=(N - 1, N)))
    results.append(env.measure(bd=(N, N - 1)))
    for n in range(N - 1, 0, -1):
        results.append(env.measure(bd=(n, n - 1)))
    results.append(env.measure(bd=(0, -1)))

    env2 = mps.Env(psi2, psi1)
    env2.setup_(to='last')
    results.append(env2.measure(bd=(N, N - 1)).conj())

    results.append(mps.measure_overlap(bra=psi1, ket=psi2))
    results.append(mps.measure_overlap(bra=psi2, ket=psi1).conj())
    results = [x.item() for x in results]  # added for cuda
    assert np.std(results) / abs(np.mean(results)) < tol


def test_env3_update(config=cfg, tol=1e-12):
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    opts_config = {} if config is None else \
            {'backend': config.backend, 'default_device': config.default_device}
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **opts_config)
    ops.random_seed(seed=0)
    N = 7
    I = mps.product_mpo(ops.I(), N=N)
    psi1 = mps.random_mps(I, D_total=11, n=(4, 2))
    psi2 = mps.random_mps(I, D_total=15, n=(4, 2))
    op = mps.random_mpo(I, D_total=16)
    #
    check_env3_measure(psi1, op, psi2, tol)
    check_env3_measure(op, op, op, tol)
    #
    # also for MpoPBC
    op_pbc = mps.Mpo(N, periodic=True)
    for n in op_pbc.sweep():
        op_pbc[n] = op[(n + 5) % N].copy()
    check_env3_measure(psi1, op_pbc, psi2, tol)


def check_env3_measure(psi1, op, psi2, tol):
    """ Test if different overlaps of psi1 and psi2 give consistent results. """
    N = psi1.N
    env = mps.Env(psi1, [op, psi2])
    env.setup_(to='first')
    env.setup_(to='last')

    results = [env.measure()]
    for n in range(N - 1):
        results.append(env.measure(bd=(n, n + 1)))
    results.append(env.measure(bd=(N - 1, N)))
    results.append(env.measure(bd=(N, N - 1)))
    for n in range(N - 1, 0, -1):
        results.append(env.measure(bd=(n, n - 1)))
    results.append(env.measure(bd=(0, -1)))
    results.append(mps.measure_mpo(bra=psi1, op=op, ket=psi2))
    results = [x.item() for x in results]  # added for cuda
    assert np.std(results) / abs(np.mean(results)) < tol


def test_env_raise(config=cfg):
    opts_config = {} if config is None else \
            {'backend': config.backend, 'default_device': config.default_device}
    ops = yastn.operators.SpinfulFermions(sym='Z2', **opts_config)
    I12 = mps.product_mpo(ops.I(), 12)
    H12 = mps.random_mpo(I12, D_total=10)
    psi12 = mps.random_mps(I12, D_total=15)

    I13 = mps.product_mpo(ops.I(), 13)
    psi13 = mps.random_mps(I13, D_total=4)

    # miscalcullus
    env = mps.Env(psi12, [[H12, H12], psi12])  # Env_sum
    assert env.factor() == 1

    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [psi12, psi12])
        # Env: MPO operator should have 2 physical legs.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [H12, H12])
        # Env: bra and ket should have the same number of physical legs.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [H12, H12, psi12])
        # Env: Input cannot be parsed.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [1, psi12])
        # Env: Input cannot be parsed.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, psi13)
        # Env: bra and ket should have the same number of physical legs.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi13, [H12, psi13])
        # Env: MPO operator, bra and ket should have the same number of sites.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, H12)
        # Env: bra and ket should have the same number of physical legs.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [[H12, psi12], psi12, [psi12]])
        # Env: Input cannot be parsed.
    with pytest.raises(yastn.YastnError):
        H12pbc = mps.Mpo(12, periodic=True)
        mps.Env(H12, [H12pbc, H12])
        # Env: Application of MpoPBC on Mpo is not supported. Contact developers to add this functionality.


if __name__ == "__main__":
    test_env2_update()
    test_env3_update()
    test_env_raise()
