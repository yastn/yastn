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
"""
Test ctmrg on 2D Classical Ising model.
Calculate expectation values using ctm for analytical dense peps tensors
of 2D Ising model with zero transverse field (Onsager solution)
"""
import numpy as np
import pytest

import yastn
import yastn.tn.fpeps as fpeps

@pytest.mark.skipif("not config.getoption('ray')", reason="requires ray library")
def test_para_ctmrg_Ising(config_kwargs):
    r"""
    Use CTMRG to calculate some expectation values in classical 2D Ising model.
    Compare with analytical results.
    """
    from yastn.tn.fpeps.envs.para_ctmrg import PARActmrg_
    #
    # We start by representing the partition function
    # of the model at temperature beta as a PEPS network.
    beta = 0.5
    #
    config = yastn.make_config(sym='Z2', **config_kwargs)
    #
    leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    T = yastn.ones(config, legs=[leg, leg, leg.conj(), leg.conj()], n=0)
    X = yastn.ones(config, legs=[leg, leg, leg.conj(), leg.conj()], n=1)
    B = yastn.zeros(config, legs=[leg, leg.conj()])
    B.set_block(ts=(0, 0), val=np.cosh(beta))
    B.set_block(ts=(1, 1), val=np.sinh(beta))
    #
    TB = yastn.ncon([T, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])
    XB = yastn.ncon([X, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])
    #
    # We use infinite translationally invariant square lattice with 1x1 unit cell.
    #
    geometry = fpeps.SquareLattice(dims=(2, 2), boundary='infinite')
    psi = fpeps.Peps(geometry=geometry, tensors={(0, 0): TB, (1, 0): TB, (0, 1): TB, (1, 1): TB})
    #
    # Execute CTMRG algorithm to contract the network.
    # chi is the bond dimension of CTM environment tensors.
    # CTMRG updates are terminated based on the convergence of corner spectra.
    #
    chi = 24
    env = fpeps.EnvCTM(psi, init='rand')
    opts_svd = {"D_total": chi}
    # info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=200, corner_tol=1e-12)
    for info in PARActmrg_(env, opts_svd_ctm=opts_svd, max_sweeps=200, corner_tol=1e-12, cpus_per_task=1, ctm_jobs_hv=[[psi.sites()], [psi.sites()]]):
        pass
    assert info.max_dsv < 1e-12
    assert info.converged == True
    #

    # Local magnetization is obtained by contracting
    # the CTM environment with tensor XB (normalized by the partition function).
    # It is zero by symmetry.
    #
    ev_X = env.measure_1site(XB)  # at all unique lattice sites: [(0, 0)]
    assert abs(ev_X[(0, 0)]) < 1e-10
    #
    # Nearest-neighbor XX correlator
    ev_XXnn = env.measure_nn(XB, XB)
    if beta == 0.5:
        assert abs(ev_XXnn[(0, 0), (0, 1)] - 0.872783) < 1e-6  # horizontal bond
        assert abs(ev_XXnn[(0, 0), (1, 0)] - 0.872783) < 1e-6  # vertical bond
    #
    # Exact magnetization squared in the ordered phase for beta > beta_c = 0.440686...
    #
    MX2 = (1 - np.sinh(2 * beta) ** -4) ** (1 / 4)
    #
    # We calculate the square of the spontaneous magnetization
    # from the large-distance limit of XX correlator
    #
    pairs = [((0, 0), (0, 100)),  # horizontal
             ((0, 0), (100, 0))]  # vertical
    for pair in pairs:
        ev_XXlong = env.measure_line(XB, XB, sites=pair)
        assert abs(MX2 - ev_XXlong) < 1e-10


@pytest.mark.skipif("not config.getoption('ray')", reason="requires ray library")
def test_ctmrg_hexagonal(config_kwargs):
    r"""
    Test hexagonal lattice
    """

    from yastn.tn.fpeps.envs.para_ctmrg import PARActmrg_

    config = yastn.make_config(sym='Z2', **config_kwargs)
    leg1 = yastn.Leg(config, s=1, t=(0, ), D=(1, ))
    leg2 = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    T1a = yastn.ones(config, legs=[leg2, leg2, leg1.conj(), leg2.conj()])
    T2a = yastn.ones(config, legs=[leg1, leg2, leg2.conj(), leg2.conj()])

    T1b = yastn.ones(config, legs=[leg2, leg2, leg2.conj(), leg1.conj()])
    T2b = yastn.ones(config, legs=[leg2, leg1, leg2.conj(), leg2.conj()])

    for T1, T2 in [(T1a, T2a), (T1b, T2b)]:
        geometry = fpeps.CheckerboardLattice()
        psi = fpeps.Peps(geometry=geometry, tensors=[[T1, T2]])

        chi = 2
        env = fpeps.EnvCTM(psi, init='eye')
        opts_svd = {"D_total": chi}
        info = PARActmrg_(env, opts_svd_ctm=opts_svd, moves='hv', max_sweeps=100, corner_tol=1e-5, ctm_jobs_hv=[[psi.sites()], [psi.sites()]])
        for info_ in info:
            pass
        assert env.is_consistent()
        assert info_.max_D == chi


if __name__ == '__main__':
    pytest.main([__file__, "-vv", "--durations=0", "--backend", "np", "--ray"])
