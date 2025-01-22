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
import pytest
import yastn
import yastn.tn.fpeps as fpeps


def test_evolution(config_kwargs):
    """ Simulate purification of a small finite system to test evolution options and outputed info. """
    #
    Nx, Ny = 2, 2
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary='obc')
    #
    # prepare evolution gates
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    I = ops.I()
    t, dbeta = 1.0, 0.1
    g_hop = fpeps.gates.gate_nn_hopping(t, dbeta / 2, I, ops.c(), ops.cp())
    gates = fpeps.gates.distribute(geometry, gates_nn=g_hop)
    #
    #
    # time-evolve initial state
    #
    opts_svd = {"D_total": 4, 'tol': 1e-14}
    steps = 4
    old_Delta = 1
    for initialization, le, lp in [("SVD", 2, 1), ("EAT", 2, 2), ("EAT_SVD", 4, 3)]:
        # initialized product state
        psi = fpeps.product_peps(geometry, ops.I())
        env = fpeps.EnvNTU(psi, which='NN')
        infoss = []
        for _ in range(steps):
            infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization=initialization)
            infoss.append(infos)

        for infos in infoss:
            assert len(infos) == 2 * ((Nx - 1) * Ny + (Ny - 1) * Nx)
            for info, bond in zip(infos, psi.bonds() + psi.bonds(reverse=True)):
                assert len(info.truncation_errors) == le and len(info.pinv_cutoffs) == lp
                assert info.bond == bond
                assert 0. <= info.truncation_error
                assert -1. <= info.min_eigenvalue <= 1.
                assert 0. <= info.wrong_eigenvalues <= 1.
                assert 0. <= info.nonhermitian_part
                assert info.truncation_error == info.truncation_errors[info.best_method]
                if 'EAT' in initialization:
                    assert (0 <= info.eat_metric_error)
                else:
                    assert info.eat_metric_error is None

        Delta = fpeps.accumulated_truncation_error(infoss)
        # accumulated truncation error should be <= between 'SVD', 'EAT', 'EAT_SVD'
        assert Delta < old_Delta + 1e-8
        old_Delta = Delta

        infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization=initialization, fix_metric=None)
        for info, bond in zip(infos, psi.bonds() + psi.bonds(reverse=True)):
            assert info.min_eigenvalue is None
            assert info.wrong_eigenvalues is None
            assert 0. <= info.nonhermitian_part

    with pytest.raises(yastn.YastnError):
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization='none')
        # initialization='none' not recognized. Should contain 'SVD' or 'EAT'.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
