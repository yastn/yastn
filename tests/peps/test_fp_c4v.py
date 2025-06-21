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
""" Test for AD of environments of selected states """
import os
import json
from typing import Sequence
import pytest
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.envs.rdm import rdm1x1
import yastn.tn.mps as mps
import logging

from yastn.tn.fpeps.envs._env_ctm_c4v import leg_charge_conv_check
from yastn.tn.fpeps.envs.fixed_pt_c4v import fp_ctmrg_c4v, refill_env_c4v
from yastn.tn.fpeps.envs.fixed_pt import fp_ctmrg, refill_env
log= logging.getLogger(__name__)

@pytest.fixture
def additional_imports(config_kwargs):
    if config_kwargs["backend"] != "torch":
        pytest.skip("torch backend is required")
        return config_kwargs, None, None
    else:
        import torch
        from torch.autograd import gradcheck
    return config_kwargs, torch, gradcheck


def _symmetrize_normalize(A):
    A= A+A.transpose(axes=(1,2,3,0,4)) # rot pi/2
    A= A+A.transpose(axes=(3,0,1,2,4)) # rot -pi/2
    A= A+A.transpose(axes=(0,3,2,1,4)) # reflection along y (vertical)
    A= A+A.transpose(axes=(2,1,0,3,4)) # reflection along x (horizontal)
    return A/A.norm(p='inf') # normalize


def cost_U1_c4v_2x2(additional_imports, yastn_cfg, g, A, elems, slices : dict[tuple[int],tuple[slice,slice]], max_sweeps, 
                    ctm_init='dl', fix_signs=False, 
                    truncate_multiplets_mode='truncate', projector_svd_method='fullrank', checkpoint_move=False):
        _, torch, _= additional_imports
        tensors_loc= { k:v.clone() for k,v in A.items() }
        for k in tensors_loc.keys():
            tensors_loc[k]._data[slices[k][1]]= elems[slices[k][0]]

        # 2. convert to 2-site bipartite YASTN's iPEPS
        A0= _symmetrize_normalize(tensors_loc[(0,0)])
        A0= A0.flip_charges(axes=(0,1))

        # 1.2 create phase operator to be applied on B-sublattice
        phase_op= yastn.Tensor(config=A0.config, s=[-1,1], n=0,
            t=((1,-1),(1,-1)), D=((1,1),(1,1)) )
        phase_op.set_block((1,1), (1,1), val=[[-1.],] )
        phase_op.set_block((-1,-1), (1,1), val=[[1.],] )

        # flip_signature() is equivalent to conj().conj_blocks(), which changes the total charge from +n to -n
        # flip_charges(axes) is equivalent to switch_signature(axes), which leaves the total charge unchanged
        #
        # 1.2 create B-tensor
        # A1= state.site((0,0)).flip_signature() # 1.2.1 map into B-sublattice with [phys,u,l,d,r]= [-1,-1,-1,-1,-1]
        # A1= rot_op.tensordot(A1,([1],[0])).switch_signature(axes=(0,3,4))

        A1= A0.flip_signature().switch_signature(axes='all')
        A1= A1 @ phase_op #phase_op @ A1

        psi = fpeps.Peps(fpeps.RectangularUnitcell(pattern=[[0,1],[1,0]]), tensors={(0,0): A0, (1,0): A1})

        # 3. proceed with YASTN's CTMRG implementation
        # 3.1 possibly re-initialize the environment
        if truncate_multiplets_mode == 'expand':
            truncation_f= None
        elif truncate_multiplets_mode == 'truncate':
            def truncation_f(S):
                return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=CHI,\
                    tol=1.0e-8, tol_block=0.0, eps_multiplet=1.0e-8)

        with torch.no_grad():
            env_leg = yastn.Leg(psi.config, s=1, t=(0,), D=(1,))
            env = fpeps.EnvCTM(psi, init=ctm_init, leg=env_leg)

        # 3.2 setup and run CTMRG
        options_svd={
            "policy": projector_svd_method,
            "D_total": CHI, 'D_block': CHI, "tol": 1.0e-8,
            'fix_signs': fix_signs,
            "eps_multiplet": 1.0e-8,
            "svds_thresh": 0.1,
            "verbosity": 3
        }

        info = env.ctmrg_(opts_svd = options_svd, max_sweeps=max_sweeps, 
                        corner_tol=1.0e-8, truncation_f=truncation_f, use_qr=False, checkpoint_move=checkpoint_move)
        log.info(f"WARM-UP: Number of ctm steps: {info}")

        r1x1,norm= rdm1x1( (0,0), psi, env)
        for c in r1x1.get_blocks_charge():
            print(f"{norm} {c} {r1x1[c]}")
        loss= r1x1[(1,1)].trace()
        return loss


def cost_U1_c4v_2x2_fp(additional_imports, yastn_cfg, g, A, elems, slices : dict[tuple[int],tuple[slice,slice]], max_sweeps, 
                    ctm_init='dl', fix_signs=False, 
                    truncate_multiplets_mode='truncate', projector_svd_method='fullrank', checkpoint_move=False):
    _, torch, _= additional_imports
    tensors_loc= { k:v.clone() for k,v in A.items() }
    for k in tensors_loc.keys():
        tensors_loc[k]._data[slices[k][1]]= elems[slices[k][0]]

    # 2. convert to 2-site bipartite YASTN's iPEPS
    A0= _symmetrize_normalize(tensors_loc[(0,0)])
    A0= A0.flip_charges(axes=(0,1))

    # 1.2 create phase operator to be applied on B-sublattice
    phase_op= yastn.Tensor(config=A0.config, s=[-1,1], n=0,
        t=((1,-1),(1,-1)), D=((1,1),(1,1)) )
    phase_op.set_block((1,1), (1,1), val=[[-1.],] )
    phase_op.set_block((-1,-1), (1,1), val=[[1.],] )

    # flip_signature() is equivalent to conj().conj_blocks(), which changes the total charge from +n to -n
    # flip_charges(axes) is equivalent to switch_signature(axes), which leaves the total charge unchanged
    #
    # 1.2 create B-tensor
    # A1= state.site((0,0)).flip_signature() # 1.2.1 map into B-sublattice with [phys,u,l,d,r]= [-1,-1,-1,-1,-1]
    # A1= rot_op.tensordot(A1,([1],[0])).switch_signature(axes=(0,3,4))

    A1= A0.flip_signature().switch_signature(axes='all')
    A1= A1 @ phase_op #phase_op @ A1

    psi = fpeps.Peps(fpeps.RectangularUnitcell(pattern=[[0,1],[1,0]]), tensors={(0,0): A0, (1,0): A1})

    # 3. proceed with YASTN's CTMRG implementation
    # 3.1 possibly re-initialize the environment
    if truncate_multiplets_mode == 'expand':
        truncation_f= None
    elif truncate_multiplets_mode == 'truncate':
        def truncation_f(S):
            return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=CHI,\
                tol=1.0e-8, tol_block=0.0, eps_multiplet=1.0e-8)

    with torch.no_grad():
        env_leg = yastn.Leg(psi.config, s=1, t=(0,), D=(1,))
        env = fpeps.EnvCTM(psi, init=ctm_init, leg=env_leg)

    # 3.2 setup and run CTMRG
    options_svd={
        'policy': projector_svd_method,
        "D_total": CHI, "D_block": CHI,
        "tol": 1.0e-8, "eps_multiplet": 1.0e-8,
        "svds_thresh": 0.1
    }
    env, env_ts_slices, env_ts = fp_ctmrg(env, \
        ctm_opts_fwd= {'opts_svd': options_svd, 'corner_tol': 1.0e-8, 'max_sweeps': max_sweeps, \
            'method': "2site", 'use_qr': False, }, \
        ctm_opts_fp= {'opts_svd': {'policy': 'fullrank'}})
    refill_env(env, env_ts, env_ts_slices)

    # 3.4 evaluate loss
    r1x1,norm= rdm1x1( (0,0), psi, env)
    for c in r1x1.get_blocks_charge():
        print(f"{norm} {c} {r1x1[c]}")
    loss= r1x1[(1,1)].trace()
    return loss


def cost_U1_c4v_1x1_fp(additional_imports, yastn_cfg, g, A, elems, slices : dict[tuple[int],tuple[slice,slice]], max_sweeps, 
                    ctm_init='dl', fix_signs=False, 
                    truncate_multiplets_mode='truncate', projector_svd_method='fullrank', checkpoint_move=False):
    _, torch, _= additional_imports
    tensors_loc= { k:v.clone() for k,v in A.items() }
    for k in tensors_loc.keys():
        tensors_loc[k]._data[slices[k][1]]= elems[slices[k][0]]

    A0= _symmetrize_normalize(tensors_loc[(0,0)])
    psi = fpeps.Peps(g, tensors={(0,0): A0})

    if truncate_multiplets_mode == 'expand':
        truncation_f= None
    elif truncate_multiplets_mode == 'truncate':
        def truncation_f(S):
            return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=CHI,\
                tol=1.0e-8, tol_block=0.0, eps_multiplet=1.0e-8)

    with torch.no_grad():
        env_leg = yastn.Leg(psi.config, s=1, t=(0,), D=(1,))
        envc4v = fpeps.EnvCTM_c4v(psi, init=ctm_init, leg=env_leg)
       
    # 3.1.1 post-init CTM steps (allow expansion of the environment in case of qr policy)
    if projector_svd_method=='qr':
        options_svd_pre_init= {
            "policy": "block_arnoldi",
            "D_total": CHI, 'D_block': CHI, "tol": 1.0e-8,
            'fix_signs': fix_signs
        }
        with torch.no_grad():
            info = envc4v.ctmrg_(opts_svd = options_svd_pre_init, max_sweeps=max_sweeps, 
                        corner_tol=leg_charge_conv_check, truncation_f=truncation_f, use_qr=False, checkpoint_move=checkpoint_move)
            log.info(f"WARM-UP: Number of ctm steps: {info}")

    # 3.2 setup and run CTMRG
    options_svd={
        "policy": projector_svd_method,
        "D_total": CHI, 'D_block': CHI, "tol": 1.0e-8,
        "eps_multiplet": 1.0e-8, "truncation_f": truncation_f, "svds_thresh": 0.1 
        }
    envc4v, env_ts_slices, env_ts, t_ctm = fp_ctmrg_c4v(envc4v, \
        ctm_opts_fwd= {'opts_svd': options_svd, 'corner_tol': 1.0e-8, 'max_sweeps': max_sweeps, \
            'method': "default", 'use_qr': False}, \
        ctm_opts_fp= { 'opts_svd': {'policy': 'fullrank'}})
    refill_env_c4v(envc4v, env_ts, env_ts_slices)
    
    # 3.4 evaluate loss
    # sum of traces of even sectors across 1x1 RDMs
    env= envc4v.get_env_bipartite()

    r1x1,norm= rdm1x1( (0,0), psi, env)
    for c in r1x1.get_blocks_charge():
        print(f"{norm} {c} {r1x1[c]}")
    loss= r1x1[(1,1)].trace()
    return loss


def prepare_1x1(additional_imports, cost_f):
    config_kwargs, _, _ = additional_imports
    yastn_cfg_U1= yastn.make_config(sym='U1', **config_kwargs)
    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 'D3_1x1_c4v_U1_spin-half.json')
    with open(json_file_path,'r') as f:
        d = json.load(f)

    g= fpeps.RectangularUnitcell(**d['geometry'])
    A= { tuple(d['parameters_key_to_id'][coord]): yastn.load_from_dict(yastn_cfg_U1, d_ten) 
                                 for coord,d_ten in d['parameters'].items() }   

    cost_function_1x1= lambda *args, **kwargs : cost_f(additional_imports, yastn_cfg_U1, g,A, *args, **kwargs)

    return A, None, cost_function_1x1 

CHI= 47
REF_D3_U1_c4v_2x2_grad=[
.23087385162275018,
0.11677319328717085,
0.1167731932639163,
0.04498240676206066,
0.1167731932871714,
0.04530136605315259,
0.044982406762061014,
0.001816910648229041,
0.11677319326391573,
0.04498240676206139,
0.04530136605684913,
0.0018169106632112636,
0.04498240676206072,
0.0018169106482296988,
0.0018169106632108987,
-0.022717432774250257,
-0.06375239660966917,
-0.061729995145950504,
-0.056286383006252334,
-0.05394747201413554,
-0.06172999514595074,
-0.058040079687778856,
-0.0539474720141358,
-0.05043516255238746,
-0.06375239660571597,
-0.061729995142423805,
-0.06172999514242368,
-0.05804007969633242,
-0.05628638299120915,
-0.05394747200020278,
-0.053947472000202584,
-0.05043516254851273,
-0.06375239660966861,
-0.05628638300625197,
-0.06172999514595072,
-0.0539474720141358,
-0.061729995145950387,
-0.05394747201413551,
-0.05804007968777912,
-0.050435162552387745,
-0.06375239660571588,
-0.06172999514242386,
-0.0562863829912091,
-0.05394747200020284,
-0.06172999514242326,
-0.05804007969633217,
-0.05394747200020229,
-0.05043516254851257]

##### U(1) 1x1 c4v #####
# @pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("projector_svd_method", ["fullrank",])	
def test_D3_U1_c4v_2x2(ctm_init, truncate_multiplets_mode, projector_svd_method, additional_imports):
    config_kwargs, torch, gradcheck = additional_imports
    A, A_grad_expected, cost_f= prepare_1x1(additional_imports, cost_U1_c4v_2x2)
    test_elems= A[(0,0)]._data.clone()
    slices= { (0,0): (slice(0,len(test_elems)), slice(0,len(test_elems))) }
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_f(x, slices, 50, ctm_init=ctm_init, \
        fix_signs=True, truncate_multiplets_mode=truncate_multiplets_mode, projector_svd_method=projector_svd_method)
    
    l0= loc_cost_f(test_elems)
    l0.backward()

    assert np.allclose(np.asarray(REF_D3_U1_c4v_2x2_grad), test_elems.grad.numpy(), rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("projector_svd_method", ["fullrank", "block_arnoldi", "block_propack"]) #"qr",
def test_D3_U1_c4v_2x2_fp(ctm_init, truncate_multiplets_mode, projector_svd_method, additional_imports):
    """
    Test fp gradients for explicit U(1) and C4v symmetric single-site ansatz 
    for environments evaluated 2x2 U(1) iPEPS, generated from single-site ansatz. C4v symmetry is not explicit.
    """
    config_kwargs, torch, gradcheck = additional_imports
    A, A_grad_expected, cost_f= prepare_1x1(additional_imports, cost_U1_c4v_2x2_fp)
    test_elems= A[(0,0)]._data.clone()
    slices= { (0,0): (slice(0,len(test_elems)), slice(0,len(test_elems))) }
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_f(x, slices, 50, ctm_init=ctm_init, \
        fix_signs=True, truncate_multiplets_mode=truncate_multiplets_mode, projector_svd_method=projector_svd_method)
    
    l0= loc_cost_f(test_elems)
    l0.backward()

    assert np.allclose(np.asarray(REF_D3_U1_c4v_2x2_grad), test_elems.grad.numpy(), rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("projector_svd_method", ["fullrank", "block_arnoldi", "block_propack"]) #"qr",
def test_D3_U1_c4v_1x1_fp(ctm_init, truncate_multiplets_mode, projector_svd_method, additional_imports):
    """
    Test fp gradients for explicit U(1) and C4v symmetric single-site ansatz 
    for environments evaluated from C4v-symmetric CTM.
    """
    config_kwargs, torch, gradcheck = additional_imports
    A, A_grad_expected, cost_f= prepare_1x1(additional_imports, cost_U1_c4v_1x1_fp)
    test_elems= A[(0,0)]._data.clone()
    slices= { (0,0): (slice(0,len(test_elems)), slice(0,len(test_elems))) }
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_f(x, slices, 50, ctm_init=ctm_init, \
        fix_signs=True, truncate_multiplets_mode=truncate_multiplets_mode, projector_svd_method=projector_svd_method)
    
    l0= loc_cost_f(test_elems)
    l0.backward()

    assert np.allclose(np.asarray(REF_D3_U1_c4v_2x2_grad), test_elems.grad.numpy(), rtol=1e-03, atol=1e-05)