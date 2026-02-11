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
""" Test bond_metric for various environments. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps


def test_spinless_infinite_approx(config_kwargs):
    """ Simulate purification of free fermions in an infinite system.s """
    geometry = fpeps.SquareLattice(dims=(2, 3), boundary='infinite')

    t, beta = 1, 0.6  # chemical potential
    D = 4
    dbeta = 0.1

    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    I, c, cdag = ops.I(), ops.c(), ops.cp()
    g_hop = fpeps.gates.gate_nn_hopping(t, dbeta / 2, I, c, cdag)  # nn gate for 2D fermi sea
    gates = fpeps.gates.distribute(geometry, gates_nn=g_hop)

    psi = fpeps.product_peps(geometry, I) # initialized at infinite temperature
    env = fpeps.EnvBP(psi, which='BP')

    opts_svd = {"D_total": D , 'tol_block': 1e-15}
    steps = round((beta / 2) / dbeta)
    dbeta = (beta / 2) / steps

    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta:0.3f}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd)

    opts_svd = {"D_total": 2 * D, 'tol_block': 1e-15}

    env_NTU = fpeps.EnvNTU(psi, which='NN')

    env_MPS = fpeps.EnvApproximate(psi,
                                   which='43+',
                                   opts_svd=opts_svd,
                                   update_sweeps=1)

    env_CTM = fpeps.EnvCTM(psi, init='eye')
    info = env_CTM.iterate_(opts_svd=opts_svd, max_sweeps=20, corner_tol=1e-8)
    print(info)

    env_BP = fpeps.EnvBP(psi, which='NN+BP')
    info = env_BP.iterate_(max_sweeps=10, diff_tol=1e-10)
    print(info)
    #
    for s0, s1, dirn in [[(0, 0), (0, 1), 'h'], [(0, 1), (1, 1), 'v']]:
        QA, QB = psi[s0], psi[s1]

        Gs = {'FU': env_CTM.bond_metric(QA, QB, s0, s1, dirn).g}

        for k in ['NN', 'NN+', 'NN++', 'NNN', 'NNN+', 'NNN++', 'Ladder']:
            env_NTU.which = k
            assert env_NTU.which == k
            Gs[k] = env_NTU.bond_metric(QA, QB, s0, s1, dirn).g

        for k in ['43', '43+', '65', '65+', '87', '87+']:
            env_MPS.which = k
            assert env_MPS.which == k
            Gs[k] = env_MPS.bond_metric(QA, QB, s0, s1, dirn).g

        for k in ['NN+BP', 'NNN+BP', 'Ladder+BP']:
            env_BP.which = k
            assert env_BP.which == k
            Gs[k] = env_BP.bond_metric(QA, QB, s0, s1, dirn).g

        Gs = {k: v / v.norm() for k, v in Gs.items()}

        assert (Gs['NN'] - Gs['NN+']).norm() < 1e-2
        assert (Gs['NN+'] - Gs['NN++']).norm() < 1e-3
        assert (Gs['NN++'] - Gs['NNN++']).norm() < 1e-3
        assert (Gs['NNN'] - Gs['43']).norm() < 1e-5
        assert (Gs['NNN+'] - Gs['43+']).norm() < 1e-5
        assert (Gs['43'] - Gs['43+']).norm() < 1e-2
        assert (Gs['43+'] - Gs['65']).norm() < 1e-3
        assert (Gs['65'] - Gs['65+']).norm() < 1e-4
        assert (Gs['65+'] - Gs['87']).norm() < 1e-5
        assert (Gs['87'] - Gs['87+']).norm() < 1e-5
        assert (Gs['87+'] - Gs['FU']).norm() < 1e-5
        assert (Gs['NN+BP'] - Gs['FU']).norm() < 1e-3
        assert (Gs['NN+BP'] - Gs['NN+']).norm() < 1e-2
        assert (Gs['NNN+BP'] - Gs['NNN+']).norm() < 1e-3
        assert (Gs['NN+BP'] - Gs['Ladder+BP']).norm() < 1e-4
        assert (Gs['NN'] - Gs['Ladder']).norm() < 1e-3


    with pytest.raises(yastn.YastnError):
        fpeps.EnvNTU(psi, which="some")
        #  Type of EnvNTU which='some' not recognized.

    with pytest.raises(yastn.YastnError):
        fpeps.EnvApproximate(psi, which="some")
        # Type of EnvApprox which='some' not recognized.

    with pytest.raises(yastn.YastnError):
        fpeps.EnvBP(psi, which="some")
        # Type of EnvBP bond_metric which='some' not recognized.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
