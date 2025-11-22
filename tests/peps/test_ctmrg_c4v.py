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
    T = yastn.ones(config, legs=[leg, leg, leg, leg], n=0)
    X = yastn.ones(config, legs=[leg, leg, leg, leg], n=1)
    B2 = yastn.zeros(config, legs=[leg.conj(), leg])
    B2.set_block(ts=(0, 0), val=np.sqrt(np.cosh(beta)))
    B2.set_block(ts=(1, 1), val=np.sqrt(np.sinh(beta)))
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
    TB = yastn.ncon([T, B2, B2, B2, B2], [(1, 2, 3, 4), [1, -1], [2, -2], [3, -3], [4, -4]])
    XB = yastn.ncon([X, B2, B2, B2, B2], [(1, 2, 3, 4), [1, -1], [2, -2], [3, -3], [4, -4]])
    #
    # We use infinite translationally invariant square lattice with 1x1 unit cell.
    #
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(geometry=geometry, tensors=[[TB]])
    #
    # Execute CTMRG algorithm to contract the network.
    # chi is the bond dimension of CTM environment tensors.
    # CTMRG updates are terminated based on the convergence of corner spectra.
    #
    chi = 24
    env = fpeps.EnvCTM_c4v(psi, init='eye')
    opts_svd = {"D_total": chi, 'policy': 'fullrank', 'eps_multiplet':1e-10}
    info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=100, use_qr=False, )
    # assert info.max_dsv < 1e-12
    # assert info.converged == True
    #
    # Local magnetization is obtained by contracting
    # the CTM environment with tensor XB (normalized by the partition function).
    # It is zero by symmetry.
    #
    # XB = fpeps.Lattice(fpeps.CheckerboardLattice(), objects={(0, 0): XB, (0, 1): XB.flip_signature()})

    ev_X = env.measure_1site(XB)  # at all unique lattice sites: [(0, 0)]
    print(ev_X)

    assert abs(ev_X[(0, 0)]) < 1e-10
    #
    # Nearest-neighbor XX correlator
    ev_XXnn = env.measure_nn(XB, XB.flip_signature())
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



def test_ctmrg_Ising2(config_kwargs):
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
    T = yastn.ones(config, legs=[leg, leg, leg, leg, leg], n=0)
    X = yastn.ones(config, legs=[leg, leg.conj()], n=1)
    B2 = yastn.zeros(config, legs=[leg.conj(), leg])
    B2.set_block(ts=(0, 0), val=np.sqrt(np.cosh(beta / 2)))
    B2.set_block(ts=(1, 1), val=np.sqrt(np.sinh(beta / 2)))
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
    TB = yastn.ncon([T, B2, B2, B2, B2], [(1, 2, 3, 4, -5), [1, -1], [2, -2], [3, -3], [4, -4]])
    # XB = yastn.ncon([X, B2, B2, B2, B2], [(1, 2, 3, 4, -5), [1, -1], [2, -2], [3, -3], [4, -4]])
    #
    # We use infinite translationally invariant square lattice with 1x1 unit cell.
    #
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(geometry=geometry, tensors=[[TB]])
    #
    # Execute CTMRG algorithm to contract the network.
    # chi is the bond dimension of CTM environment tensors.
    # CTMRG updates are terminated based on the convergence of corner spectra.
    #
    chi = 24
    env = fpeps.EnvCTM_c4v(psi, init='eye')
    opts_svd = {"D_total": chi, 'policy': 'fullrank', 'eps_multiplet':1e-10}
    info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=100, use_qr=False, )
    # assert info.max_dsv < 1e-12
    # assert info.converged == True
    #
    # Local magnetization is obtained by contracting
    # the CTM environment with tensor XB (normalized by the partition function).
    # It is zero by symmetry.
    #
    # XB = fpeps.Lattice(fpeps.CheckerboardLattice(), objects={(0, 0): XB, (0, 1): XB.flip_signature()})

    ev_X = env.measure_1site(X)  # at all unique lattice sites: [(0, 0)]
    print(ev_X)

    assert abs(ev_X[(0, 0)]) < 1e-10
    #
    # Nearest-neighbor XX correlator
    ev_XXnn = env.measure_nn(X, X.flip_signature())
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
        ev_XXlong = env.measure_line(X, X, sites=pair)
        assert abs(MX2 - ev_XXlong) < 1e-10


# @pytest.mark.skipif("not config.getoption('long_tests')", reason="long duration tests are skipped")
# @pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
# @pytest.mark.parametrize("fix_signs", [False, True])
# @pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
# @pytest.mark.parametrize("checkpoint_move", ['reentrant', 'nonreentrant', False])
# def test_1x1_D1_Z2_spinlessf_conv(ctm_init, fix_signs, truncate_multiplets_mode, checkpoint_move, config_kwargs):
#     yastn_cfg_Z2= yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
#     if yastn_cfg_Z2.backend.BACKEND_ID != 'torch' and checkpoint_move != False:
#         pytest.skip("checkpoint_move is not supported for this backend")
#     json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 'D1_1x1_Z2_spinlessf_honeycomb_35gradsteps.json')
#     with open(json_file_path,'r') as f:
#         d = json.load(f)

#     g= fpeps.RectangularUnitcell(**d['geometry'])
#     A= {tuple(d['parameters_key_to_id'][coord]): yastn.Tensor.from_dict(d_ten, config=yastn_cfg_Z2)
#                                  for coord,d_ten in d['parameters'].items() }

#     psi = fpeps.Peps(g, tensors=A)
#     chi= 20

#     if truncate_multiplets_mode == 'expand':
#         truncation_f= None
#     elif truncate_multiplets_mode == 'truncate':
#         def truncation_f(S):
#             return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=chi,\
#                 tol=1.0e-8, tol_block=0.0, eps_multiplet=1.0e-8)

#     env_leg = yastn.Leg(yastn_cfg_Z2, s=1, t=(0, 1), D=(chi//2, chi//2))
#     env = fpeps.EnvCTM(psi, init=ctm_init, leg=env_leg)

#     info = env.ctmrg_(opts_svd = {"D_total": chi, 'fix_signs': fix_signs}, max_sweeps=35,
#                         corner_tol=1.0e-8, truncation_f=truncation_f, use_qr=False, checkpoint_move=checkpoint_move)
#     print(f"CTM {info}")

#     # sum of traces of even sectors across 1x1 RDMs
#     loss= sum( rdm1x1( c, psi, env)[0][(0,0)].trace() for c in psi.sites() )
#     assert np.allclose([0.22923524,], [loss,], rtol=1e-06, atol=1e-06)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"]) #,  "--long_tests", ])
