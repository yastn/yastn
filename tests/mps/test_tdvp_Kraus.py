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
""" test tdvp_Kraus_ """
import numpy as np
import pytest
import yastn
import yastn.tn.mps as mps


def dephasing_channel(dt, gamma, ops):
    p = np.exp(-dt * gamma)
    I3 = ops.I().add_leg(axis=2, s=1)
    X3 = ops.x().add_leg(axis=2, s=1)

    tens = {0: np.sqrt(1 - p) * I3,
            1: np.sqrt(p) * X3}
    out = yastn.block(tens, common_legs=(0, 1))
    return out


@pytest.mark.parametrize('sym', ['Z2'])
def test_tdvp_KZ_quench(config_kwargs, sym):
    """
    Simulate a slow quench across a quantum critical point in
    a small transverse field Ising chain with periodic boundary conditions.
    Compare with exact reference results.
    """
    #
    N = 8  # Consider a system of 8 sites
    #
    # Load spin-1/2 operators
    #
    ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
    ops.random_seed(seed=0)
    #
    # Hterm-s to generate H = -sum_i X_i X_{i+1} - g * Z_i
    #
    I = mps.product_mpo(ops.I(), N)  # identity MPO
    termsXX = [mps.Hterm(-1, [i, i + 1], [ops.x(), ops.x()]) for i in range(N - 1)]
    HXX = mps.generate_mpo(I, termsXX)
    termsZ = [mps.Hterm(-1, i, ops.z()) for i in range(N)]
    HZ = mps.generate_mpo(I, termsZ)
    #
    # Kibble-Zurek quench across a critical point at gc = 1
    # tauQ is the quench time
    #
    tauQ, gc = 1, 1
    ti, tf = -tauQ, tauQ  # evolve from gi = 2 to gf = 0
    H = lambda t: [HXX, (gc - t / tauQ) * HZ]  # linear quench
    #
    #
    Dmax = 10
    psi = mps.random_mps(I, D_total=Dmax)
    # we will add auxiliary leg of dim=1
    psi._nr_phys = 2
    for n in psi.sweep():
        psi[n] = psi[n].add_leg(axis=3, s=-1)

    out = mps.dmrg_(psi, H(ti), method='2site', max_sweeps=2,
                    opts_svd={'tol': 1e-6, 'D_total': Dmax})
    out = mps.dmrg_(psi, H(ti), method='1site', max_sweeps=10,
                    energy_tol=1e-12, Schmidt_tol=1e-12)
    print(out)
    #
    Dmax = 8
    DmaxK = 4
    opts_expmv = {'hermitian': True, 'tol': 1e-12}
    opts_svd = {'tol': 1e-6, 'D_total': Dmax}
    opts_svdK = {'tol': 1e-6, 'D_total': DmaxK}
    #
    gamma = 0.01
    K = lambda n, t, dt: dephasing_channel(dt, gamma, ops)
    #
    for step in mps.tdvp_Kraus_(psi, H, K, times=(ti, 0, tf), dt=0.04,
                          opts_svd=opts_svd, opts_expmv=opts_expmv, opts_svdK=opts_svdK):
        #
        EZ = mps.measure_1site(psi, ops.z(), psi)
        EXX = mps.measure_2site(psi, ops.x(), ops.x(), psi, bonds='r1')
        #
        print(EZ)
        print(EXX)
        print(psi.norm())



if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
