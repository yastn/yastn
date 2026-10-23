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
import pickle

import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps._evolution import Evolution_out, symmetrized_svd


def test_evolution_out_compatibility():
    """Old positional/keyword construction and the custom norm_S reader still work."""
    old_values = (None, 0.25, 'svd', 0, None, None, None, (), (), (), 0, ())
    positional = Evolution_out(*old_values)
    keyword = Evolution_out(**dict(zip(Evolution_out._fields[:-1], old_values)))
    assert positional == keyword
    assert positional[:-1] == old_values
    assert positional.normalization_factor == 1.0
    info = positional._replace(normalization_factor=2.5)
    assert info.norm_S == info.normalization_factor == 2.5
    assert pickle.loads(pickle.dumps(info)) == info


@pytest.mark.parametrize('normalize', [False, True])
def test_symmetrized_svd_normalization(config_kwargs, normalize):
    """The factor restores the truncated matrix, not its Frobenius norm."""
    config = yastn.make_config(sym='dense', **config_kwargs)
    leg = yastn.Leg(config, s=1, D=(3,))
    R0 = yastn.eye(config, legs=(leg, leg.conj())).diag()
    R1 = yastn.Tensor(config, s=(1, -1), isdiag=True)
    R1.set_block(Ds=(3, 3), val=[5., 3., 1.])
    R1 = R1.diag()
    opts = {'D_total': 2}
    (N0, N1), factor = symmetrized_svd(R0, R1, opts, normalize=normalize)
    assert isinstance(factor, float)
    assert factor == pytest.approx(5.0 if normalize else 1.0)
    expected = yastn.Tensor(config, s=(1, -1), isdiag=True)
    expected.set_block(Ds=(3, 3), val=[5., 3., 0.])
    assert ((N0 @ N1) * factor - expected.diag()).norm().item() < 1e-12


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
    t, mu, dbeta = 1.0, 0.2, 0.1
    g_hop = fpeps.gates.gate_nn_hopping(t, dbeta / 2, I, ops.c(), ops.cp())
    g_loc = fpeps.gates.gate_local_occupation(mu, dbeta / 2, I, ops.n())
    gates = fpeps.gates.distribute(geometry, gates_nn=g_hop, gates_local=g_loc)
    #
    # time-evolve initial state
    #
    D = 3
    opts_svd = {"D_total": D, 'tol': 1e-10}
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
                assert isinstance(info.normalization_factor, float)
                assert info.normalization_factor > 0
                assert info.norm_S == info.normalization_factor
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
        Delta_max = fpeps.accumulated_truncation_error(infoss, statistics='max')
        assert Delta_max >= Delta
        # accumulated truncation error should be <= between 'SVD', 'EAT', 'EAT_SVD'
        assert Delta < old_Delta + 1e-8
        old_Delta = Delta

        infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization=initialization, fix_metric=None)
        for info, bond in zip(infos, psi.bonds() + psi.bonds(reverse=True)):
            assert info.min_eigenvalue is None
            assert info.wrong_eigenvalues is None
            assert 0. <= info.nonhermitian_part

        bd0 = psi.get_bond_dimensions()
        assert all(v in (1, D) for v in bd0.values())
        #
        D2 = 2
        opts_svd_truncate = {"D_total": D2, 'tol': 1e-14}
        infos = fpeps.truncate_(env, opts_svd=opts_svd_truncate, initialization=initialization)
        bd1 = psi.get_bond_dimensions()
        for k, v in bd1.items():
            assert v == min(D2, bd0[k])
    #
    with pytest.raises(yastn.YastnError,
                       match="initialization='none' not recognized. Should contain 'SVD' or 'EAT'."):
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization='none')
    with pytest.raises(yastn.YastnError,
                       match="statistics='any' in accumulated_truncation_error not recognized; Should be 'max' or 'mean'."):
        fpeps.accumulated_truncation_error(infoss, statistics='any')
    #
    # for bipartite environment
    #
    for which in ['BP', 'NN+BP']:
        psi = fpeps.product_peps(geometry, ops.I())
        env = fpeps.EnvBP(psi, which=which)
        infoss = []
        for _ in range(steps):
            infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization=initialization)
            infoss.append(infos)

        for infos in infoss:
            assert len(infos) == 2 * ((Nx - 1) * Ny + (Ny - 1) * Nx)
            for info, bond in zip(infos, psi.bonds() + psi.bonds(reverse=True)):
                assert info.bond == bond
                assert 0. <= info.truncation_error
                if which == 'BP':
                    assert info.normalization_factor == 1.0

        bd0 = psi.get_bond_dimensions()
        assert all(v in (1, D) for v in bd0.values())
        opts_svd_truncate = {"D_total": D2, 'tol': 1e-14}
        infos = fpeps.truncate_(env, opts_svd=opts_svd_truncate, initialization=initialization)
        bd1 = psi.get_bond_dimensions()
        for k, v in bd1.items():
            assert v == min(D2, bd0[k])


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
