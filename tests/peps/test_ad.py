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
import pytest
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.envs.rdm import rdm1x1
import yastn.tn.mps as mps


@pytest.fixture
def additional_imports(config_kwargs):
    if not config_kwargs["backend"] in ["torch", "torch_cpp"]:
        pytest.skip("Backend with AD support is required: [torch, torch_cpp]")
        return config_kwargs, None, None
    else:
        import torch
        from torch.autograd import gradcheck
    return config_kwargs, torch, gradcheck


def prepare_RVB(additional_imports):
    config_kwargs, torch, gradcheck = additional_imports
    # peps-torch/examples/kagome/abelian/optim_kagome_spin_half_u1.py::TestOptim_RVB_r1x1
    #
    test_state_Kagome_RVB_D3_U1_sym= {'_d': np.array(
           [0.782507  -0.17348611j, -0.75472251+0.04994474j,
            0.        +0.j        ,  0.        +0.j        ,
            0.78466866-0.07709138j, -0.67889185+0.13238603j,
            0.        +0.j        ,  0.86422445-0.1961466j ,
            0.        +0.j        ,  0.71731066-0.08005124j,
           -0.78646317+0.19488577j,  0.        +0.j        ,
            0.        +0.j        , -0.78234444+0.14142687j,
            0.73399422-0.09187571j,  0.        +0.j        ,
           -0.93637825+0.1464273j ,  0.        +0.j        ,
           -0.77117993+0.03324111j,  0.85331667-0.15073783j,
            0.83582335-0.15307495j,  0.        +0.j        ,
            0.        +0.j        , -0.74223979+0.15621356j,
            0.84910085-0.0447961j ,  0.        +0.j        ,
            0.        +0.j        ,  0.75132078-0.14778056j,
            0.        +0.j        ,  0.77981359-0.03264839j,
            0.        +0.j        , -0.99737753+0.07237452j,
            0.        +0.j        , -0.81380557-0.03224727j,
            0.91041246-0.08414513j, -0.80153731+0.02328223j,
            0.        +0.j        ,  0.        +0.j        ,
            0.71469189-0.03959038j, -0.79844087-0.07961802j,
            0.        +0.j        ,  0.        +0.j        ,
           -0.72195685+0.03042574j,  0.        +0.j        ,
            0.        +0.j        ,  0.83432898-0.05086651j,
            0.        +0.j        ,  0.        +0.j        ,
           -0.74330549+0.06497449j,  0.83451895+0.0564455j ,
            0.        +0.j        ,  0.        +0.j        ,
            0.75117891-0.05566726j,  0.76308403-0.05664679j,
           -0.84085481+0.1755911j ,  0.        +0.j        ,
           -0.6514202 +0.11089874j,  0.        +0.j        ,
            0.82988544-0.16755248j,  0.        +0.j        ,
           -0.83439248+0.11911678j,  0.        +0.j        ,
            0.        +0.j        ,  0.74176008-0.12587796j,
           -0.8433288 +0.0112789j ,  0.        +0.j        ,
            0.        +0.j        , -0.75043171+0.11716542j,
           -0.81831702+0.00493975j,  0.90994821-0.1261947j ,
            0.        +0.j        ,  0.70311962-0.07094725j,
            0.        +0.j        , -0.89765941+0.11841547j,
            0.80177449-0.12706608j,  0.        +0.j        ,
            0.        +0.j        , -0.86090484-0.06611417j,
            0.9678228 -0.05340271j,  0.        +0.j        ,
            0.74552471-0.01333218j,  0.        +0.j        ,
           -0.95422304+0.04629357j, -0.76605802+0.00388958j,
            0.        +0.j        ,  0.        +0.j        ,
            0.79800778-0.02948396j, -0.79963062+0.09466636j]),
            's': (-1, -1, -1, 1, 1),
            'n': (0,),
            't': ((-3, 0, 1, -1, -1), (-3, 1, 0, -1, -1), (-1, -1, 0, -1, -1), (-1, -1, 1, -1, 0), (-1, -1, 1, 0, -1), (-1, 0, -1, -1, -1), (-1, 0, 0, -1, 0), (-1, 0, 0, 0, -1), (-1, 0, 1, -1, 1), (-1, 0, 1, 0, 0), (-1, 0, 1, 1, -1), (-1, 1, -1, -1, 0), (-1, 1, -1, 0, -1), (-1, 1, 0, -1, 1), (-1, 1, 0, 0, 0), (-1, 1, 0, 1, -1), (1, -1, 0, -1, 1), (1, -1, 0, 0, 0), (1, -1, 0, 1, -1), (1, -1, 1, 0, 1), (1, -1, 1, 1, 0), (1, 0, -1, -1, 1), (1, 0, -1, 0, 0), (1, 0, -1, 1, -1), (1, 0, 0, 0, 1), (1, 0, 0, 1, 0), (1, 0, 1, 1, 1), (1, 1, -1, 0, 1), (1, 1, -1, 1, 0), (1, 1, 0, 1, 1), (3, -1, 0, 1, 1), (3, 0, -1, 1, 1)), 'D': ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (3, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
            'isdiag': False,
            'mfs': ((1,), (1,), (1,), (1,), (1,)),
            'hfs': [
                    {'tree': (3, 1, 1, 1), 'op': 'pooo', 's': (-1, -1, -1, -1), 't': (((-1,), (1,)), ((-1,), (1,)), ((-1,), (1,))), 'D': ((1, 1), (1, 1), (1, 1))},
                    {'tree': (1,), 'op': 'o', 's': (-1,), 't': (), 'D': ()},
                    {'tree': (1,), 'op': 'o', 's': (-1,), 't': (), 'D': ()},
                    {'tree': (1,), 'op': 'o', 's': (1,), 't': (), 'D': ()},
                    {'tree': (1,), 'op': 'o', 's': (1,), 't': (), 'D': ()}
                    ],
            'SYM_ID': 'U1', 'fermionic': False}

    config_kwargs['default_dtype'] = 'complex128'
    yastn_config = yastn.make_config(sym='U1', **config_kwargs)
    # load on-site tensor stored in above dict
    #
    # physical, top, left, bottom, right -> t,l,b,r,p
    A= yastn.Tensor.from_dict(test_state_Kagome_RVB_D3_U1_sym, config=yastn_config).transpose(axes=(1, 2, 3, 4, 0))
    A = A.drop_leg_history(axes=4)

    grad_expected= np.asarray(
       [-5.2009e-02+1.1485e-02j,  4.2988e-02-2.8922e-03j,
        -1.6123e-03+7.3344e-05j, -4.9013e-04+1.7433e-05j,
         8.6896e-03-7.9282e-04j, -5.4564e-03+1.0721e-03j,
        -5.5240e-05+8.3027e-05j,  5.9241e-03-1.3965e-03j,
         1.7918e-03-2.1236e-04j,  3.5818e-03-6.1957e-04j,
        -6.4851e-03+1.5382e-03j,  5.8052e-04-6.2971e-05j,
         1.8185e-03-4.5212e-05j, -8.4812e-03+1.5127e-03j,
         1.5118e-02-1.7300e-03j, -6.3680e-04-3.5287e-04j,
        -1.9483e-02+3.2735e-03j,  8.2122e-04-3.4191e-04j,
        -1.8321e-02+8.8223e-04j,  2.0160e-02-3.3910e-03j,
         7.0923e-03-1.2392e-03j, -7.6992e-04-2.3402e-05j,
        -1.3048e-03+2.5187e-04j, -1.4070e-02+3.0826e-03j,
         1.3739e-02-5.0156e-04j,  2.6678e-03-9.4412e-04j,
        -1.0178e-03+4.0107e-04j,  7.3888e-03-1.3112e-03j,
        -6.1032e-04+5.4541e-06j,  3.3533e-03-1.2939e-04j,
         1.7051e-03+1.2627e-04j, -6.6254e-03+5.4265e-04j,
        -6.2112e-05-4.8911e-05j, -6.2609e-03-3.4679e-04j,
         6.2405e-03-5.1551e-04j, -5.7879e-03+2.2646e-04j,
         8.2752e-04+1.3215e-04j,  3.8558e-04-1.2873e-04j,
         1.0381e-02-6.2434e-04j, -1.4514e-02-1.4015e-03j,
         3.3288e-03-2.1488e-04j,  7.0925e-04-1.6411e-04j,
        -7.3389e-03+4.6318e-04j,  1.5220e-03-4.1660e-04j,
         1.6242e-04+6.3690e-05j, -5.5126e-03+2.9385e-04j,
         2.4697e-04+8.8568e-06j,  1.1373e-04-4.9837e-05j,
         9.1344e-03-6.9762e-04j, -1.0363e-02-7.6070e-04j,
         1.1855e-04+2.9367e-05j,  1.4450e-04-3.0744e-06j,
        -5.0517e-03+2.9935e-04j, -7.7812e-03+6.0841e-04j,
         9.1443e-03-1.9702e-03j, -4.1990e-04+1.5568e-06j,
         6.0149e-03-9.5801e-04j, -1.6259e-04-5.6293e-05j,
        -7.7382e-03+1.6328e-03j, -1.8095e-04-1.4103e-05j,
         5.0505e-03-7.2588e-04j, -1.6424e-04+2.9148e-05j,
        -1.2478e-04+1.6634e-04j, -8.4391e-03+1.3735e-03j,
         9.6367e-03-2.4905e-04j, -2.3484e-04+1.1483e-04j,
        -2.1350e-04+5.6454e-05j,  5.0885e-03-8.9022e-04j,
         7.3152e-03-4.3441e-05j, -7.7464e-03+1.1446e-03j,
         2.8258e-05-8.5489e-05j, -5.4709e-03+5.5554e-04j,
        -7.8343e-05-7.9629e-05j,  6.5104e-03-8.0517e-04j,
        -6.0203e-03+9.8534e-04j,  5.3730e-04-2.0025e-04j,
        -2.1176e-04+4.5296e-05j,  6.5954e-03+4.1093e-04j,
        -7.3355e-03+4.4063e-04j, -2.7989e-04-2.3993e-05j,
        -5.4461e-03+7.1475e-05j, -3.8541e-04+9.8988e-05j,
        7.7498e-03-2.9497e-04j,  4.5305e-03+6.8886e-05j,
         2.4093e-04-7.7295e-05j, -5.8018e-04-4.7404e-05j,
         1.6676e-02-5.2631e-04j, -1.5570e-02+1.9329e-03j])
    A_grad_expected= A.clone().transpose(axes=(4,0,1,2,3))
    A_grad_expected._data= A_grad_expected.config.backend.to_tensor(grad_expected, dtype='complex128',
                                                                    device=config_kwargs["default_device"])
    A_grad_expected= A_grad_expected.transpose(axes=(1,2,3,4,0))


    def cost_function_RVB(elems, slice, max_sweeps, ctm_init='dl', fix_signs=False, truncate_multiplets_mode='truncate',
                          checkpoint_move=False):
        A0= A.clone()
        A0._data[slice]= elems

        g= fpeps.SquareLattice(dims=(1,1), boundary='infinite')
        psi = fpeps.Peps(g, tensors=dict(zip(g.sites(), [A0, ])))

        if truncate_multiplets_mode == 'expand':
            truncation_f= None
        elif truncate_multiplets_mode == 'truncate':
            def truncation_f(S):
                return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=64,\
                    tol=1.0e-8, tol_block=0.0, eps_multiplet=1.0e-8)

        env = fpeps.EnvCTM(psi, init=ctm_init)
        info = env.ctmrg_(opts_svd = {"D_total": 64, 'fix_signs': fix_signs}, max_sweeps=max_sweeps,
                            truncation_f=truncation_f, use_qr=False, checkpoint_move=checkpoint_move)
        r1x1, r1x1_norm= rdm1x1( (0,0), psi, env)
        return r1x1[(-1,-1)].trace().real

    return A, A_grad_expected, cost_function_RVB


def cost_function_f(yastn_cfg, g, A, elems, slices : dict[tuple[int],tuple[slice,slice]], max_sweeps, ctm_init='dl', fix_signs=False,
                    truncate_multiplets_mode='truncate', checkpoint_move=False):
    # For each on-site tensor, corresponding element in slices is a pair,
    # where first entry specified slice in elems 1D-array while second entry specifies slice in target on-site tensor
    #
    tensors_loc= { k:v.clone() for k,v in A.items() }
    for k in tensors_loc.keys():
        tensors_loc[k]._data[slices[k][1]]= elems[slices[k][0]]

    psi = fpeps.Peps(g, tensors=tensors_loc)
    chi= 20

    if truncate_multiplets_mode == 'expand':
        truncation_f= None
    elif truncate_multiplets_mode == 'truncate':
        def truncation_f(S):
            return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=chi,\
                tol=1.0e-8, tol_block=0.0, eps_multiplet=1.0e-8)

    env_leg = yastn.Leg(yastn_cfg, s=1, t=(0, 1), D=(chi//2, chi//2))
    env = fpeps.EnvCTM(psi, init=ctm_init, leg=env_leg)

    info = env.ctmrg_(opts_svd = {"D_total": chi, 'fix_signs': fix_signs}, max_sweeps=max_sweeps,
                        corner_tol=1.0e-8, truncation_f=truncation_f, use_qr=False, checkpoint_move=checkpoint_move)
    print(f"CTM {info}")

    # sum of traces of even sectors across 1x1 RDMs
    loss= sum( rdm1x1( c, psi, env)[0][(0,0)].trace() for c in psi.sites() )

    return loss


def prepare_1x1(additional_imports):
    config_kwargs, _, _ = additional_imports
    yastn_cfg_Z2= yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 'D1_1x1_Z2_spinlessf_honeycomb_35gradsteps.json')
    with open(json_file_path,'r') as f:
        d = json.load(f)

    g= fpeps.RectangularUnitcell(**d['geometry'])
    A= { tuple(d['parameters_key_to_id'][coord]): yastn.Tensor.from_dict(d_ten, config=yastn_cfg_Z2)
                                 for coord,d_ten in d['parameters'].items() }

    cost_function_1x1= lambda *args, **kwargs : cost_function_f(yastn_cfg_Z2,g,A, *args, **kwargs)

    return A, None, cost_function_1x1


def prepare_3x3(additional_imports):
    config_kwargs, _, _ = additional_imports
    yastn_cfg_Z2= yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 'D1_3x3_Z2_spinlessf_honeycomb.json')
    with open(json_file_path,'r') as f:
        d = json.load(f)

    g= fpeps.RectangularUnitcell(**d['geometry'])
    A= {tuple(d['parameters_key_to_id'][coord]): yastn.Tensor.from_dict(d_ten, config=yastn_cfg_Z2)
                                 for coord,d_ten in d['parameters'].items() }

    cost_function_3x3 = lambda *args, **kwargs: cost_function_f(yastn_cfg_Z2, g, A, *args, **kwargs)

    return A, None, cost_function_3x3


##### U(1) RVB Kagome #####

@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("fix_signs", [False, True])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("checkpoint_move", ['nonreentrant',False])
def test_Kagome_RVB_D3_U1_sym_ctmsteps1(ctm_init, fix_signs, truncate_multiplets_mode, checkpoint_move, additional_imports):
    config_kwargs, torch, gradcheck = additional_imports
    if truncate_multiplets_mode == "expand":
        pytest.xfail(f"Expected failure when truncate_multiplets_mode='{truncate_multiplets_mode}'")
    A, A_grad_expected, cost_function_RVB= prepare_RVB(additional_imports)
    test_elems= A._data[36:51].clone()
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_function_RVB(x, slice(36,51), 1, ctm_init=ctm_init, fix_signs=fix_signs,
                                         truncate_multiplets_mode=truncate_multiplets_mode, checkpoint_move=checkpoint_move)

    gradcheck(loc_cost_f, test_elems, eps=1e-06, atol=1e-05, rtol=0.001,
        raise_exception=True, nondet_tol=0.0, check_undefined_grad=True, check_grad_dtypes=False,
        check_batched_grad=False, check_batched_forward_grad=False, check_forward_ad=False,
        check_backward_ad=True, fast_mode=False, masked=None)


@pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("checkpoint_move", ['reentrant','nonreentrant',False])
def test_Kagome_RVB_D3_U1_sym_vs_pepstorch(ctm_init, truncate_multiplets_mode, checkpoint_move, additional_imports):
    config_kwargs, torch, gradcheck = additional_imports
    A, A_grad_expected, cost_function_RVB= prepare_RVB(additional_imports)
    test_elems= A._data.clone()
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_function_RVB(x, slice(0,len(test_elems)), 20, ctm_init=ctm_init, \
            fix_signs=False, truncate_multiplets_mode=truncate_multiplets_mode, checkpoint_move=checkpoint_move)

    R= loc_cost_f(test_elems)
    R.backward()

    assert np.allclose(A_grad_expected._data.numpy(force=True), test_elems.grad.numpy(force=True), rtol=1e-03, atol=1e-05)


@pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
def test_Kagome_RVB_D3_U1_sym_conv(ctm_init, truncate_multiplets_mode, additional_imports):
    config_kwargs, torch, gradcheck = additional_imports
    A, A_grad_expected, cost_function_RVB= prepare_RVB(additional_imports)
    test_elems= A._data[36:36+5].clone()
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_function_RVB(x, slice(36,36+5), 20, ctm_init=ctm_init, \
        fix_signs=True, truncate_multiplets_mode=truncate_multiplets_mode)

    gradcheck(loc_cost_f, test_elems, eps=1e-06, atol=1e-05, rtol=0.001,
        raise_exception=True, nondet_tol=0.0, check_undefined_grad=True, check_grad_dtypes=False,
        check_batched_grad=False, check_batched_forward_grad=False, check_forward_ad=False,
        check_backward_ad=True, fast_mode=False, masked=None)

##### Z_2 1x1 Spinless fermions honeycomb #####

@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("fix_signs", [False, True])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("checkpoint_move", ['nonreentrant',False])
def test_1x1_D1_Z2_spinlessf_ctmsteps1(ctm_init, fix_signs, truncate_multiplets_mode, checkpoint_move, additional_imports):
    if truncate_multiplets_mode == "expand":
        pytest.xfail(f"Expected failure when truncate_multiplets_mode='{truncate_multiplets_mode}'")
    config_kwargs, torch, gradcheck = additional_imports
    A0, _, cost_function= prepare_1x1(additional_imports)
    slices= { k: (slice(9*i,9*(i+1)), slice(0,9)) for i,k in enumerate(A0.keys()) }
    test_elems= torch.cat( [A0[k]._data[slices[k][1]].clone() for i,k in enumerate(A0.keys())] )
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_function(x, slices, 1, ctm_init=ctm_init, fix_signs=fix_signs,
                                         truncate_multiplets_mode=truncate_multiplets_mode, checkpoint_move=checkpoint_move)

    gradcheck(loc_cost_f, test_elems, eps=1e-06, atol=1e-05, rtol=0.001,
        raise_exception=True, nondet_tol=0.0, check_undefined_grad=True, check_grad_dtypes=False,
        check_batched_grad=False, check_batched_forward_grad=False, check_forward_ad=False,
        check_backward_ad=True, fast_mode=False, masked=None)


@pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("tol", [1e-3, 1e-4, 1e-5])
@pytest.mark.parametrize("checkpoint_move", ['nonreentrant',False])
def test_1x1_D1_Z2_spinlessf_conv(ctm_init, truncate_multiplets_mode, tol, checkpoint_move, additional_imports):
    if tol == 1e-5:
        pytest.xfail(f"Expected failure when tol='{tol}'")
    config_kwargs, torch, gradcheck = additional_imports
    A0, _, cost_function= prepare_1x1(additional_imports)
    slices= { k: (slice(6*i,6*(i+1)), slice(0,6)) for i,k in enumerate(A0.keys()) }
    test_elems= torch.cat( [A0[k]._data[slices[k][1]].clone() for i,k in enumerate(A0.keys())] )
    test_elems.requires_grad_()

    # It should take 35 steps to converge
    loc_cost_f= lambda x : cost_function(x, slices, 35, ctm_init=ctm_init, fix_signs=True,
                                         truncate_multiplets_mode=truncate_multiplets_mode,
                                         checkpoint_move=checkpoint_move)

    gradcheck(loc_cost_f, test_elems, eps=1e-06, atol=tol*1e-2, rtol=tol,
        raise_exception=True, nondet_tol=0.0, check_undefined_grad=True, check_grad_dtypes=False,
        check_batched_grad=False, check_batched_forward_grad=False, check_forward_ad=False,
        check_backward_ad=True, fast_mode=False, masked=None)

# TODO: check against known gradient for 1x1 unit cell case using reentrant checkpointing

##### Z_2 3x3 Spinless fermions honeycomb #####

@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("fix_signs", [False, True])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("checkpoint_move", ['nonreentrant',False])
def test_3x3_D1_Z2_spinlessf_ctmsteps1(ctm_init, fix_signs, truncate_multiplets_mode, checkpoint_move, additional_imports):
    if truncate_multiplets_mode == "expand":
        pytest.xfail(f"Expected failure when truncate_multiplets_mode='{truncate_multiplets_mode}'")
    config_kwargs, torch, gradcheck= additional_imports
    A0, _, cost_function= prepare_3x3(additional_imports)
    slices= { k: (slice(3*i,3*(i+1)), slice(0,3)) for i,k in enumerate(A0.keys()) }
    test_elems= torch.cat( [A0[k]._data[slices[k][1]].clone() for i,k in enumerate(A0.keys())] )
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_function(x, slices, 1, ctm_init=ctm_init, fix_signs=fix_signs,
                                         truncate_multiplets_mode=truncate_multiplets_mode, checkpoint_move=checkpoint_move)

    gradcheck(loc_cost_f, test_elems, eps=1e-06, atol=1e-05, rtol=0.001,
        raise_exception=True, nondet_tol=0.0, check_undefined_grad=True, check_grad_dtypes=False,
        check_batched_grad=False, check_batched_forward_grad=False, check_forward_ad=False,
        check_backward_ad=True, fast_mode=False, masked=None)


@pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("checkpoint_move", ['nonreentrant',False])
def test_3x3_D1_Z2_spinlessf_conv(ctm_init, truncate_multiplets_mode, checkpoint_move, additional_imports):
    config_kwargs, torch, gradcheck= additional_imports
    A0, _, cost_function= prepare_3x3(additional_imports)
    slices= { k: (slice(2*i,2*(i+1)), slice(0,2)) for i,k in enumerate(A0.keys()) }
    test_elems= torch.cat( [A0[k]._data[slices[k][1]].clone() for i,k in enumerate(A0.keys())] )
    test_elems.requires_grad_()

    # It should take 35 steps to converge
    loc_cost_f= lambda x : cost_function(x, slices, 35, ctm_init=ctm_init, fix_signs=True,
                                         truncate_multiplets_mode=truncate_multiplets_mode, checkpoint_move=checkpoint_move)

    gradcheck(loc_cost_f, test_elems, eps=1e-06, atol=1e-05, rtol=0.001,
        raise_exception=True, nondet_tol=0.0, check_undefined_grad=True, check_grad_dtypes=False,
        check_batched_grad=False, check_batched_forward_grad=False, check_forward_ad=False,
        check_backward_ad=True, fast_mode=False, masked=None)

# TODO: test against known result for 3x3 unit cell
#
# @pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
# @pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
# @pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
# def test_3x3_D1_Z2_spinlessf_expected(ctm_init, truncate_multiplets_mode, additional_imports):
#     if truncate_multiplets_mode == "expand":
#         pytest.xfail(f"Expected failure when truncate_multiplets_mode='{truncate_multiplets_mode}'")
#     config_kwargs, torch, gradcheck = additional_imports
#     A0, _, cost_function= prepare_3x3(additional_imports)
#     slices= { k: (slice(i*A0[k]._data.shape[0], (i+1)*A0[k]._data.shape[0]), slice(0,A0[k]._data.shape[0])) for i,k in enumerate(A0.keys()) }
#     test_elems= torch.cat( [A0[k]._data[slices[k][1]].clone() for i,k in enumerate(A0.keys())] )
#     test_elems.requires_grad_()

#     # It should take 35 steps to converge
#     loc_cost_f= lambda x : cost_function(x, slices, 35, ctm_init=ctm_init, fix_signs=True,
#                                          truncate_multiplets_mode=truncate_multiplets_mode)

#     R= loc_cost_f(test_elems)
#     R.backward()

#     print(test_elems.grad)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--long_tests"])
