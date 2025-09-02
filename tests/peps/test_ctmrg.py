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
import os, json
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.envs.rdm import rdm1x1


def test_ctmrg_Ising(config_kwargs):
    r"""
    Use CTMRG to calculate some expectation values in classical 2D Ising model.
    Compare with analytical results.
    """
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
    # The partition function of the system in the thermodynamic limit
    # follows from contracting infinite network:
    #
    #      |     |     |
    #    ──T──B──T──B──T──
    #      |     |     |
    #      B     B     B
    #      |     |     |
    #    ──T──B──T──B──T──
    #      |     |     |
    #      B     B     B
    #      |     |     |
    #    ──T──B──T──B──T──
    #      |     |     |
    #
    # The expectation values of spin correlation functions are obtained
    # replacing tensors T with X at sites of interest.
    #
    # We absorb bond tensors B into site tensors.
    #
    TB = yastn.ncon([T, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])
    XB = yastn.ncon([X, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])
    #
    # We use infinite translationally invariant square lattice with 1x1 unit cell.
    #
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(geometry=geometry, tensors={(0, 0): TB})
    #
    # Execute CTMRG algorithm to contract the network.
    # chi is the bond dimension of CTM environment tensors.
    # CTMRG updates are terminated based on the convergence of corner spectra.
    #
    chi = 24
    env = fpeps.EnvCTM(psi, init='eye')
    opts_svd = {"D_total": chi}
    info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=200, corner_tol=1e-12)
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


def test_ctmrg_Ising_4x5(config_kwargs):
    r"""
    Test if convergence is reached using various moves in a finite system.
    Also tests bond_update_ after enlarging some bonds in Peps.
    """
    config = yastn.make_config(sym='Z2', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    leg2 = yastn.Leg(config, s=1, t=(0, 1), D=(2, 2))

    T = yastn.ones(config, legs=[leg, leg, leg.conj(), leg.conj()], n=0)
    XL = yastn.ones(config, legs=[leg, leg, leg.conj(), leg2.conj()], n=0)
    XR = yastn.ones(config, legs=[leg, leg2, leg.conj(), leg.conj()], n=0)
    XT = yastn.ones(config, legs=[leg, leg, leg2.conj(), leg.conj()], n=0)
    XB = yastn.ones(config, legs=[leg2, leg, leg.conj(), leg.conj()], n=0)
    #
    geometry = fpeps.SquareLattice(dims=(4, 5), boundary='obc')
    psi = fpeps.Peps(geometry=geometry, tensors=[[T, T, T, T, T],
                                                 [T, T, T, T, T],
                                                 [T, T, T, T, T],
                                                 [T, T, T, T, T]])
    #
    chi = 4
    env = fpeps.EnvCTM(psi, init='eye')
    opts_svd = {"D_total": chi}
    info = env.ctmrg_(opts_svd=opts_svd, moves='lrtb', max_sweeps=10, corner_tol=1e-12)
    assert info.converged
    #
    # enlarge some bonds
    psi[(1, 1)] = XL
    psi[(1, 2)] = XR
    env.update_bond_(((1, 1), (1, 2)), opts_svd=opts_svd)
    psi[(1, 3)] = XT
    psi[(2, 3)] = XB
    env.opts_svd = opts_svd
    env.update_bond_(((1, 3), (2, 3)))
    #
    # now full sweep of ctmrg can be executed
    info = env.ctmrg_(opts_svd=opts_svd, moves='hv', max_sweeps=10, corner_tol=1e-12)
    assert info.converged
    #
    # checks if properly updated on a boundary
    bonds = [((0, 0), (0, 1)), ((3, 4), (3, 3)),
             ((0, 0), (1, 0)), ((3, 4), (2, 4))]
    for bond in bonds:
        env.update_bond_(bond, opts_svd=opts_svd)


@pytest.mark.parametrize("checkpoint_move", ['reentrant', 'nonreentrant', False])
def test_ctmrg_Ising_dense(checkpoint_move, config_kwargs):
    r"""
    Calculate magnetization for classical 2D Ising model and compares with the analytical result.
    """
    beta = 0.5
    chi = 8
    nn_exact = {0.3: 0.352250, 0.5: 0.872783, 0.6: 0.954543, 0.75: 0.988338}
    local_exact  = {0.3: 0.000000, 0.5: 0.911319, 0.6: 0.973609, 0.75: 0.993785}

    config = yastn.make_config(sym='none', **config_kwargs)
    if config.backend.BACKEND_ID != 'torch' and checkpoint_move != False:
        pytest.skip("Checkpoint_move is not supported for this backend.")
    config.backend.random_seed(seed=0)

    leg = yastn.Leg(config, s=1, D=[2])
    TI = yastn.zeros(config, legs=[leg, leg, leg.conj(), leg.conj()])
    TI[()][0, 0, 0, 0] = 1
    TI[()][1, 1, 1, 1] = 1

    TX = yastn.zeros(config, legs=[leg, leg, leg.conj(), leg.conj()])
    TX[()][0, 0, 0, 0] = 1
    TX[()][1, 1, 1, 1] = -1

    B = yastn.zeros(config, legs=[leg, leg.conj()])
    B.set_block(ts=(), val=[[np.exp(beta), np.exp(-beta)],
                            [np.exp(-beta), np.exp(beta)]])

    TI = yastn.ncon([TI, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])
    TX = yastn.ncon([TX, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])

    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(geometry=geometry, tensors={(0, 0): TI})

    env = fpeps.EnvCTM(psi, init='rand')
    opts_svd = {"D_total": chi}
    info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=200, corner_tol=1e-8, checkpoint_move=checkpoint_move)
    print(info)

    ev_XX = env.measure_nn(TX, TX)
    for val in ev_XX.values():
        assert abs(val - nn_exact[beta]) < 1e-5
        print(val)

    ev_X = env.measure_1site(TX)
    assert abs(abs(ev_X[(0 ,0)]) - local_exact[beta]) < 1e-5


@pytest.mark.skipif("not config.getoption('long_tests')", reason="long duration tests are skipped")
@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("fix_signs", [False, True])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("checkpoint_move", ['reentrant', 'nonreentrant', False])
def test_1x1_D1_Z2_spinlessf_conv(ctm_init, fix_signs, truncate_multiplets_mode, checkpoint_move, config_kwargs):
    yastn_cfg_Z2= yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    if yastn_cfg_Z2.backend.BACKEND_ID != 'torch' and checkpoint_move != False:
        pytest.skip("checkpoint_move is not supported for this backend")
    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 'D1_1x1_Z2_spinlessf_honeycomb_35gradsteps.json')
    with open(json_file_path,'r') as f:
        d = json.load(f)

    g= fpeps.RectangularUnitcell(**d['geometry'])
    A= { tuple(d['parameters_key_to_id'][coord]): yastn.load_from_dict(yastn_cfg_Z2, d_ten)
                                 for coord,d_ten in d['parameters'].items() }

    psi = fpeps.Peps(g, tensors=A)
    chi= 20

    if truncate_multiplets_mode == 'expand':
        truncation_f= None
    elif truncate_multiplets_mode == 'truncate':
        def truncation_f(S):
            return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=chi,\
                tol=1.0e-8, tol_block=0.0, eps_multiplet=1.0e-8)

    env_leg = yastn.Leg(yastn_cfg_Z2, s=1, t=(0, 1), D=(chi//2, chi//2))
    env = fpeps.EnvCTM(psi, init=ctm_init, leg=env_leg)

    info = env.ctmrg_(opts_svd = {"D_total": chi, 'fix_signs': fix_signs}, max_sweeps=35,
                        corner_tol=1.0e-8, truncation_f=truncation_f, use_qr=False, checkpoint_move=checkpoint_move)
    print(f"CTM {info}")

    # sum of traces of even sectors across 1x1 RDMs
    loss= sum( rdm1x1( c, psi, env)[0][(0,0)].trace() for c in psi.sites() )
    assert np.allclose([0.22923524,], [loss,], rtol=1e-06, atol=1e-06)


if __name__ == '__main__':

    # test_ctmrg_Ising({"backend": "np"})
    # test_ctmrg_Ising_3x4({"backend": "np"})

    # test_ctmrg_Ising_dense({"backend": "torch", "device": "cuda"})
    pytest.main([__file__, "-vs", "--durations=0", "--long_tests"])
