# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
""" test tdvp_Kraus_ """
import numpy as np
import pytest
import yastn
import yastn.tn.mps as mps


def dephasing_channel(dt, gamma, ops):
    """
    Kraus operators of dephasing, rho -> (1 - p) rho + p Z rho Z, during time dt,
    with p = (1 - exp(-2 gamma dt)) / 2, so that coherences decay as exp(-2 gamma dt).
    Returns rank-3 tensor with the last leg enumerating Kraus operators.
    """
    p = (1 - np.exp(-2 * gamma * dt)) / 2
    I3 = ops.I().add_leg(axis=2, s=1)
    Z3 = ops.z().add_leg(axis=2, s=1)
    return yastn.block({0: np.sqrt(1 - p) * I3, 1: np.sqrt(p) * Z3}, common_legs=(0, 1))


def purification_from_mps(psi):
    """ Add an auxiliary leg of dimension 1 to each tensor of MPS. """
    psi._nr_phys = 2
    for n in psi.sweep():
        psi[n] = psi[n].add_leg(axis=3, s=-1)
    return psi


def test_tdvp_Kraus_dephasing(config_kwargs, tol=1e-10):
    """
    Spins in a time-dependent field H(t) = -h(t) sum_i Z_i with dephasing at rate gamma,
    starting from all spins along x. Each spin evolves independently, with
    <X_i>(t) = exp(-2 gamma t) cos(2 Phi(t)), where Phi(t) = int_0^t h(s) ds.
    Here, H and dephasing commute, and the midpoint rule used for H(t) is exact for linear h(t),
    so the result of tdvp_Kraus_ is exact.
    """
    N = 4
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    I = mps.product_mpo(ops.I(), N)
    HZ = mps.generate_mpo(I, [mps.Hterm(-1, [i], [ops.z()]) for i in range(N)])
    h = lambda t: 1 + 2 * t
    Phi = lambda t: t + t ** 2
    H = lambda t: h(t) * HZ
    #
    gamma = 0.3
    K = lambda n, t, dt: dephasing_channel(dt, gamma, ops)
    #
    psi = purification_from_mps(mps.product_mps(ops.vec_x(val=1), N))
    opts = {'opts_expmv': {'hermitian': True, 'tol': 1e-13},
            'opts_svd': {'D_total': 4}, 'opts_svdK': {'D_total': 4}}  # no truncation needed
    #
    times = (0, 0.25, 0.5)
    for step, t in zip(mps.tdvp_Kraus_(psi, H, K, times=times, dt=0.05, **opts), times[1:]):
        assert abs(step.tf - t) < 1e-12 and step.steps == 5 and not step.time_independent
        EX = mps.measure_1site(psi, ops.x(), psi)
        EZ = mps.measure_1site(psi, ops.z(), psi)
        ref = np.exp(-2 * gamma * t) * np.cos(2 * Phi(t))
        assert all(abs(EX[n] - ref) < tol for n in range(N))
        assert all(abs(EZ[n]) < tol for n in range(N))
        assert abs(mps.vdot(psi, psi) - 1) < tol  # Kraus operators preserve the trace


def test_tdvp_Kraus_raise(config_kwargs):
    """ Input validation of tdvp_Kraus_; scalar times and yield_initial. """
    N = 4
    ops = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    I = mps.product_mpo(ops.I(), N)
    H = mps.generate_mpo(I, [mps.Hterm(-1, [i], [ops.z()]) for i in range(N)])
    psi = purification_from_mps(mps.random_mps(I, D_total=4))
    K = lambda n, t, dt: dephasing_channel(dt, 0.1, ops)
    opts = {'opts_svd': {'D_total': 4}, 'opts_svdK': {'D_total': 2}}

    with pytest.raises(yastn.YastnError, match="provide opts_svd and opts_svdK"):
        next(mps.tdvp_Kraus_(psi, H, K, opts_svd={'D_total': 4}))
    with pytest.raises(yastn.YastnError, match="dt should be positive"):
        next(mps.tdvp_Kraus_(psi, H, K, dt=0, **opts))
    with pytest.raises(yastn.YastnError, match="times should be an ascending tuple"):
        next(mps.tdvp_Kraus_(psi, H, K, times=(0, 0.2, 0.1), **opts))

    steps = list(mps.tdvp_Kraus_(psi, H, K, times=0.2, dt=0.1, yield_initial=True, **opts))
    assert len(steps) == 2
    assert steps[0].ti == steps[0].tf == 0 and steps[0].steps == 0
    assert abs(steps[1].tf - 0.2) < 1e-12 and steps[1].steps == 2


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
