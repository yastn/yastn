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
""" Test for AD of environments of selected states with complex values"""
import os
import json
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.envs.rdm import rdm1x1


import numpy as np
import warnings
try:
    from yastn.tn.fpeps.envs.fixed_pt import fp_ctmrg
except ImportError:
    warnings.warn("This test requires torch")

@pytest.fixture
def additional_imports(config_kwargs):
    if config_kwargs["backend"] != "torch":
        pytest.skip("torch backend is required")
        return config_kwargs, None, None
    else:
        import torch
        from torch.autograd import gradcheck
    return config_kwargs, torch, gradcheck



def cost_function_f(additional_imports, yastn_cfg, g, A, elems, slices : dict[tuple[int],tuple[slice,slice]], max_sweeps, ctm_init='dl', fix_signs=False,
                    truncate_multiplets_mode='truncate', checkpoint_move=False):
    # For each on-site tensor, corresponding element in slices is a pair,
    # where first entry specified slice in elems 1D-array while second entry specifies slice in target on-site tensor
    #
    tensors_loc= { k:v.clone() for k,v in A.items() }
    for k in tensors_loc.keys():
        if k in slices:
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

    return loss.real

def cost_function_fp(additional_imports, yastn_cfg, g, A, elems, slices : dict[tuple[int],tuple[slice,slice]], max_sweeps,\
                    ctm_init='dl', fix_signs=False,
                    truncate_multiplets_mode='truncate', projector_svd_method='fullrank', checkpoint_move=False):
    # For each on-site tensor, corresponding element in slices is a pair,
    # where first entry specified slice in elems 1D-array while second entry specifies slice in target on-site tensor
    #
    tensors_loc= { k:v.clone() for k,v in A.items() }
    for k in tensors_loc.keys():
        if k in slices:
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

    options_svd={
        'policy': projector_svd_method,
        "D_total": chi, "D_block": chi,
        "tol": 1.0e-8, "eps_multiplet": 1.0e-8,
        "svds_thresh": 0.1
    }
    env = fp_ctmrg(env, \
        ctm_opts_fwd= {'opts_svd': options_svd, 'corner_tol': 1.0e-8, 'max_sweeps': max_sweeps, \
            'method': "2site", 'use_qr': False, }, \
        ctm_opts_fp= {'opts_svd': {'policy': 'fullrank'}})

    # sum of traces of even sectors across 1x1 RDMs
    loss= sum( rdm1x1( c, psi, env)[0][(0,0)].trace() for c in psi.sites() )

    return loss.real


def prepare_1x1(additional_imports, cost_f):
    config_kwargs, _, _ = additional_imports
    yastn_cfg_Z2= yastn.make_config(sym='Z2', fermionic=True, default_dtype="complex128", **config_kwargs)
    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 'D1_1x1_Z2_spinlessf_honeycomb_complex.json')
    def complex_decoder(dct):
        if "real" in dct and "imag" in dct:
            return complex(dct["real"], dct["imag"])
        return dct

    with open(json_file_path,'r') as f:
        d = json.load(f, object_hook=complex_decoder)

    g= fpeps.RectangularUnitcell(**d['geometry'])
    A= { tuple(d['parameters_key_to_id'][coord]): yastn.from_dict(d_ten, config=yastn_cfg_Z2)
                                 for coord,d_ten in d['parameters'].items() }

    cost_function_1x1= lambda *args, **kwargs : cost_f(additional_imports, yastn_cfg_Z2,g,A, *args, **kwargs)

    return A, None, cost_function_1x1


def prepare_3x3(additional_imports, cost_f):
    config_kwargs, _, _ = additional_imports
    yastn_cfg_Z2= yastn.make_config(sym='Z2', fermionic=True, default_dtype='complex128', **config_kwargs)
    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 'D1_3x3_Z2_spinlessf_honeycomb_complex.json')
    def complex_decoder(dct):
        if "real" in dct and "imag" in dct:
            return complex(dct["real"], dct["imag"])
        return dct

    with open(json_file_path,'r') as f:
        d = json.load(f, object_hook=complex_decoder)

    g= fpeps.RectangularUnitcell(**d['geometry'])
    A= { tuple(d['parameters_key_to_id'][coord]): yastn.from_dict(d_ten, config=yastn_cfg_Z2)
                                 for coord,d_ten in d['parameters'].items() }

    cost_function_3x3= lambda *args, **kwargs : cost_f(additional_imports, yastn_cfg_Z2,g,A, *args, **kwargs)

    return A, None, cost_function_3x3


##### Z_2 1x1 Spinless fermions honeycomb #####

REF_1x1_D1_Z2_spinlessf_complex_grad=[
  0.00610336-0.00854325j, 0.01105802+0.02787735j,  0.00826395-0.01104103j,
 -0.00944019+0.03006079j, 0.00825207-0.05782752j, -0.02822436-0.06865689j,
 -0.0248587 -0.02942831j,-0.03489552-0.02569632j,  0.00112347+0.02795297j,
 -0.00651136+0.04365782j, 0.01882109-0.0020474j ,  0.01057738+0.00274091j,
  0.00399519+0.02784747j, 0.02313572-0.00322274j, -0.0061573 +0.02352271j,
  0.00665062-0.02051055j, 0.0122168 +0.02189764j, -0.04652818+0.02831279j,
 -0.00701484-0.03090412j,-0.00195805+0.04623103j,  0.00160502+0.04280227j,
  0.0280597 -0.05964317j, 0.01273339+0.0637789j ,  0.01798187-0.0089139j ,
  0.04239365-0.02897299j,-0.02579006+0.0429731j ,  0.03942446-0.0070921j ,
 -0.02887279-0.04164401j,-0.04346859+0.02065776j, -0.01097624-0.0333841j ,
  0.01196544+0.0084767j ,-0.00350575-0.0409068j]

@pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("tol", [1e-3, 1e-4, 1e-5])
@pytest.mark.parametrize("checkpoint_move", ['nonreentrant',False])
def test_1x1_D1_Z2_spinlessf_conv(ctm_init, truncate_multiplets_mode, tol, checkpoint_move, additional_imports):
    config_kwargs, torch, gradcheck = additional_imports
    A0, _, cost_function= prepare_1x1(additional_imports, cost_function_f)
    test_elems= A0[(0,0)]._data.clone()
    test_elems.requires_grad_()
    slices= { (0,0): (slice(0,len(test_elems)), slice(0,len(test_elems))) }

    # It should take 35 steps to converge
    loc_cost_f= lambda x : cost_function(x, slices, 50, ctm_init=ctm_init, fix_signs=True,
                                         truncate_multiplets_mode=truncate_multiplets_mode,
                                         checkpoint_move=checkpoint_move)

    l0= loc_cost_f(test_elems)
    l0.backward()

    assert np.allclose(np.asarray(REF_1x1_D1_Z2_spinlessf_complex_grad), test_elems.grad.numpy(), rtol=1e-03, atol=1e-05)

@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("projector_svd_method", ["fullrank", "block_arnoldi", "block_propack"])
def test_1x1_D1_Z2_spinlessf_fp(ctm_init, truncate_multiplets_mode, projector_svd_method, additional_imports):
    config_kwargs, torch, gradcheck = additional_imports
    A, A_grad_expected, cost_f= prepare_1x1(additional_imports, cost_function_fp)
    test_elems= A[(0,0)]._data.clone()
    slices= { (0,0): (slice(0,len(test_elems)), slice(0,len(test_elems))) }
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_f(x, slices, max_sweeps=50, ctm_init=ctm_init, \
        fix_signs=True, truncate_multiplets_mode=truncate_multiplets_mode, projector_svd_method=projector_svd_method)

    l0= loc_cost_f(test_elems)
    l0.backward()

    assert np.allclose(np.asarray(REF_1x1_D1_Z2_spinlessf_complex_grad), test_elems.grad.numpy(), rtol=1e-03, atol=1e-05)

##### Z_2 3x3 Spinless fermions honeycomb #####
REF_3x3_D1_Z2_spinlessf_complex_grad=[
  0.00188026-0.00492923j,  0.01456902+0.02354232j,  0.01456581-0.01457292j,
 -0.01272293+0.04053106j,  0.00182143-0.03181953j, -0.01484817-0.04201507j,
 -0.03374139-0.04466542j, -0.04566883-0.02616905j,  0.00540566+0.03314856j,
 -0.00510028+0.04632039j,  0.00303227-0.00022976j,  0.00683309-0.00296729j,
  0.00820293+0.04020409j,  0.03173612+0.00050384j, -0.00889259+0.03072661j,
  0.00683112-0.02744609j,  0.0097564 +0.0164851j , -0.04029189+0.02276613j,
 -0.00470488-0.01883683j, -0.00389341+0.02889182j, -0.00024194+0.06182734j,
  0.04589331-0.07419272j,  0.01001529+0.05602292j,  0.01406233-0.00914521j,
  0.02297499-0.01755439j, -0.01538851+0.02703972j,  0.067551  -0.01156464j,
 -0.0481491 -0.06218329j, -0.03639401+0.01535315j, -0.00994877-0.0292396j ,
  0.0084837 +0.00401217j, -0.00530551-0.02984954j,  0.02174539+0.02691292j,
 -0.04655771-0.00494114j, -0.00781318+0.02760002j, -0.01840294+0.02424302j,
 -0.00805603+0.00458183j, -0.00912867+0.02135031j,  0.00827599+0.03772488j,
 -0.03129614-0.02784252j,  0.02393032-0.02527049j,  0.03592369+0.00748281j,
 -0.04754279-0.04189615j,  0.04684527-0.03841735j,  0.01952893-0.00404261j,
 -0.00319402-0.01995072j,  0.00090044-0.00377732j,  0.00599676+0.00224949j,
  0.0826531 +0.03441934j,  0.08023516+0.04723054j, -0.00715485+0.03100564j,
  0.00806316+0.02831259j,  0.00409237-0.03479336j,  0.05179641+0.01093601j,
  0.01400418+0.01378801j, -0.03224005-0.02327652j, -0.0400082 -0.00978357j,
  0.04261898+0.02764139j, -0.03365846-0.05408946j, -0.0131119 +0.00820225j,
  0.01526903-0.03782518j, -0.04081407+0.02884832j,  0.00696432-0.01722945j,
 -0.01918484-0.01343625j,  0.03528819+0.02012563j,  0.02223004-0.01684592j,
  0.03126133+0.06433416j, -0.02227023+0.02358j   ,  0.05323863+0.04801361j,
 -0.03892885+0.0741465j , -0.02153493+0.05576822j, -0.0211492 +0.03636754j,
  0.02921574-0.01743899j, -0.04854692-0.00834195j,  0.0082912 -0.03395765j,
  0.03060664+0.05798254j, -0.06722761-0.03293819j,  0.04520553-0.02827998j,
  0.00022039-0.008339j  ,  0.02403683-0.01916389j, -0.00453763+0.03639279j,
 -0.00063477-0.04341657j, -0.01366741+0.01174952j, -0.01032682-0.00670608j,
 -0.0208541 -0.01330704j, -0.02871179+0.01488267j,  0.02094811-0.01286907j,
 -0.00123146-0.02036243j,  0.00037732+0.02661726j,  0.01759579+0.01863718j,
 -0.00286589-0.00228462j, -0.01082022+0.01415987j,  0.00427502-0.01789228j,
 -0.00012414+0.00955633j,  0.01022139-0.04214995j, -0.01015519-0.04166757j,]

@pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("checkpoint_move", ['nonreentrant',False])
def test_3x3_D1_Z2_spinlessf_conv(ctm_init, truncate_multiplets_mode, checkpoint_move, additional_imports):
    config_kwargs, torch, gradcheck = additional_imports
    A0, _, cost_function= prepare_3x3(additional_imports, cost_function_f)
    test_elems= torch.cat([A0[loc]._data.clone() for loc in A0.keys()])
    slices, start = {}, 0
    for loc in A0.keys():
        slices[loc] = (slice(start, start + len(A0[loc]._data)), slice(0, len(A0[loc]._data)))
        start += len(A0[loc]._data)
    test_elems.requires_grad_()

    # It should take 35 steps to converge
    loc_cost_f= lambda x : cost_function(x, slices, 50, ctm_init=ctm_init, fix_signs=True,
                                         truncate_multiplets_mode=truncate_multiplets_mode,
                                         checkpoint_move=checkpoint_move)

    l0= loc_cost_f(test_elems)
    l0.backward()
    assert np.allclose(np.asarray(REF_3x3_D1_Z2_spinlessf_complex_grad), test_elems.grad.numpy(), rtol=1e-03, atol=1e-05)

@pytest.mark.parametrize("ctm_init", ['dl', 'eye'])
@pytest.mark.parametrize("truncate_multiplets_mode", ["truncate", "expand"])
@pytest.mark.parametrize("projector_svd_method", ["fullrank", "block_arnoldi", "block_propack"])
def test_3x3_D1_Z2_spinlessf_fp(ctm_init, truncate_multiplets_mode, projector_svd_method, additional_imports):
    config_kwargs, torch, gradcheck = additional_imports
    A, A_grad_expected, cost_f= prepare_3x3(additional_imports, cost_function_fp)

    test_elems= torch.cat([A[loc]._data.clone() for loc in A.keys()])
    slices, start = {}, 0
    for loc in A.keys():
        slices[loc] = (slice(start, start + len(A[loc]._data)), slice(0, len(A[loc]._data)))
        start += len(A[loc]._data)
    test_elems.requires_grad_()

    loc_cost_f= lambda x : cost_f(x, slices, max_sweeps=50, ctm_init=ctm_init, \
        fix_signs=True, truncate_multiplets_mode=truncate_multiplets_mode, projector_svd_method=projector_svd_method)

    l0= loc_cost_f(test_elems)
    l0.backward()

    assert np.allclose(np.asarray(REF_3x3_D1_Z2_spinlessf_complex_grad), test_elems.grad.numpy(), rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--long_tests"])
