# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
Test EnvCTM_c4v on 2D Classical Ising model.
Calculate expectation values using ctm for analytical dense peps tensors
of 2D Ising model with zero transverse field (Onsager solution)
"""
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps


torch_test = pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                                   reason="requires autograd support")

def MX(beta):
    """
    Analytical formula for spontanous magnetization in 2D Ising model
    in the ordered phase with beta > beta_c = 0.4406868...
    """
    return (1 - np.sinh(2 * beta) ** -4) ** (1 / 8) if beta > 0.4406868 else 0.

def dMX(beta):
    """
    Analytical formula for magnetic susceptibility in 2D Ising model
    in the ordered phase with beta > beta_c = 0.4406868...
    """
    return np.cosh(2 * beta) / np.sinh(2 * beta) ** 5 / (1 - 1 / np.sinh(2 * beta) ** 4) ** 0.875 if beta > 0.4406868 else 0.


def state_Ising(beta, config, layers):
    """
    c4v-symemtric T tensor representing 2D Ising model as a single of double-layer PEPS network.
    """
    back = config.backend.torch if hasattr(config.backend, 'torch') else np
    leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    if layers == 1:
        T = yastn.ones(config, legs=[leg, leg, leg, leg], n=0)
        X = yastn.ones(config, legs=[leg, leg, leg, leg], n=1)
        B2 = yastn.zeros(config, legs=[leg.conj(), leg])
        B2.set_block(ts=(0, 0), val=back.sqrt(back.cosh(beta)))
        B2.set_block(ts=(1, 1), val=back.sqrt(back.sinh(beta)))
        T = yastn.ncon([T, B2, B2, B2, B2], [(1, 2, 3, 4), [1, -1], [2, -2], [3, -3], [4, -4]])
        X = yastn.ncon([X, B2, B2, B2, B2], [(1, 2, 3, 4), [1, -1], [2, -2], [3, -3], [4, -4]])
    else:
        T = yastn.ones(config, legs=[leg, leg, leg, leg, leg], n=0)
        B2 = yastn.zeros(config, legs=[leg.conj(), leg])
        B2.set_block(ts=(0, 0), val=back.sqrt(back.cosh(beta / 2)))
        B2.set_block(ts=(1, 1), val=back.sqrt(back.sinh(beta / 2)))
        T = yastn.ncon([T, B2, B2, B2, B2], [(1, 2, 3, 4, -5), [1, -1], [2, -2], [3, -3], [4, -4]])
        X = yastn.ones(config, legs=[leg, leg.conj()], n=1)
    return T, X


def check_env_c4v_signature_convention(env):
    """ expected signature of PEPS and environment tensors"""
    assert env.psi[0, 0].get_signature() == (1,) * env.psi[0, 0].ndim
    assert env[0, 0].t.get_signature() == (-1, -1, -1)
    assert env[0, 0].tl.get_signature() == (1, 1)
    assert env.psi[0, 1].get_signature() == (-1,) * env.psi[0, 0].ndim
    assert env[0, 1].t.get_signature() == (1, 1, 1)
    assert env[0, 1].tl.get_signature() == (-1, -1)


def ctmrg_c4v_Ising(config, beta, layers, init, method, checkpoint_move):
    """ Perform ctmrg using EnvCTM_c4v and check spontanious magnetization"""
    T, X = state_Ising(beta, config, layers=layers)
    X = fpeps.Lattice(fpeps.CheckerboardLattice(), objects={(0, 0): X, (1, 0): X.flip_signature()})

    net = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(net, tensors=T)
    #
    env = fpeps.EnvCTM_c4v(psi, init=init)
    check_env_c4v_signature_convention(env)
    assert env.is_consistent()
    #
    D = 16
    #
    if '1x2' in method or '2x1' in method:
        opts_svd = {"D_total": D, 'eps_multiplet': 1e-10}
        max_sweeps = int(np.ceil(np.log2(D)))
        info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=max_sweeps, use_qr=False, corner_tol=1e-8, checkpoint_move=checkpoint_move, method='2x2')
    #
    opts_svd = {"D_total": D, 'eps_multiplet': 1e-10}
    info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=200, use_qr=False, corner_tol=1e-8, checkpoint_move=checkpoint_move, method=method)
    assert env.is_consistent()
    check_env_c4v_signature_convention(env)
    #
    assert info.max_dsv < 1e-8
    assert info.converged == True
    #
    # test that spontaneus magnetization is not broken -- per Z2 symemtry
    assert abs(env.measure_1site(X)[0, 0].item()) < 1e-10
    assert abs(env.measure_1site(X, site=(1, 0)).item()) < 1e-10
    #
    nn1 = env.measure_nn(X, X)
    nn2 = env.measure_nn(X, X, bond=[(1, 0), (2, 0)])
    nn3 = env.measure_2x2(X, X, sites=[(1, 0), (2, 0)])
    assert abs(nn1[(0, 0), (0, 1)].item() - nn2.item()) < 1e-10
    assert abs(nn1[(0, 0), (1, 0)].item() - nn2.item()) < 1e-10
    assert abs(nn2.item() - nn3.item()) < 1e-10
    #
    dd1 = env.measure_2x2(X, X, sites=[(0, 0), (1, 1)])
    dd2 = env.measure_2site(X, X, xrange=(0, 2), yrange=(0, 2), pairs="corner <", dirn='v')
    dd3 = env.measure_2site(X, X, xrange=(0, 2), yrange=(0, 2), pairs="corner <", dirn='h')
    assert abs(nn2.item() - dd2[(0, 0), (0, 1)].item()) < 1e-10
    assert abs(nn2.item() - dd2[(0, 0), (1, 0)].item()) < 1e-10
    assert abs(nn2.item() - dd3[(0, 0), (0, 1)].item()) < 1e-10
    assert abs(nn2.item() - dd3[(0, 0), (1, 0)].item()) < 1e-10
    assert abs(dd1.item() - dd2[(0, 0), (1, 1)].item()) < 1e-10
    assert abs(dd1.item() - dd3[(0, 0), (1, 1)].item()) < 1e-10
    #
    # calculate spontanious magnetization from long range correlator; vertical or horizontal
    #
    eXv = env.measure_line(X, X, sites=[(0, 0), (99, 0)]) ** 0.5
    eXh = env.measure_line(X, X, sites=[(0, 0), (0, 99)]) ** 0.5
    #
    assert abs(eXv.item() - MX(beta.item())) < 1e-8
    assert abs(eXh.item() - MX(beta.item())) < 1e-8
    #
    return (eXv + eXh) / 2


@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("init", ['dl', 'eye'])
@pytest.mark.parametrize("method", ['2x2', '2x1 svd', '2x1 qr'])  #, 'block_arnoldi', 'block_propack', 'symeig', ])  'randomized' not supported by backend_np
def test_ctmrg_c4v_Ising(config_kwargs, beta, layers, init, method):
    r"""
    Use CTMRG to calculate some expectation values in classical 2D Ising model.
    Compare with analytical results.
    """
    config = yastn.make_config(sym='Z2', **config_kwargs)
    beta = config.backend.to_tensor(beta)
    ctmrg_c4v_Ising(config, beta, layers, init, method, checkpoint_move=False)


@torch_test
@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("layers", [1])
@pytest.mark.parametrize("init", ['eye'])
@pytest.mark.parametrize("method", ['2x2', '2x1 svd']) # '2x1 qr' is least precise, giving biggest error in AD test
@pytest.mark.parametrize("checkpoint_move", [False, ])  # 'nonreentrant'   TODO: break the tests
def test_ctmrg_c4v_Ising_AD(config_kwargs, beta, layers, init, method, checkpoint_move):
    r"""
    Use CTMRG to calculate some expectation values in classical 2D Ising model.
    Calculate magnetic susceptibility using AD, and compare with the analytical results.
    """
    config = yastn.make_config(sym='Z2', **config_kwargs)
    beta = config.backend.to_tensor(beta)
    beta.requires_grad_()
    eX = ctmrg_c4v_Ising(config, beta, layers, init, method, checkpoint_move=checkpoint_move)

    # calculate gradient to get  dX / dbeta
    eX.backward()
    edX = beta.grad
    # Compare with the the analytical result
    print(abs(edX.item() - dMX(beta.item())))
    assert abs(edX.item() - dMX(beta.item())) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0" , "--backend", "np"])
