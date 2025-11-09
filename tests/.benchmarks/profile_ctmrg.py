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
import numpy as np
import time
import pytest
import math
import copy
try:
    import yastn
except ModuleNotFoundError as e:
    import os 
    import sys
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, dir_path+"/../../")
    import yastn
from yastn.backend import backend_np
from yastn.backend import backend_torch
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Peps, Peps2Layers, RectangularUnitcell

tol = 1e-12  #pylint: disable=invalid-name

torch_test = pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                                reason="Uses torch.autograd.gradcheck().")

def distributeU1_exponential(D, p = 0.25):
    λ = (1-p)/(1+p)
    D0 = math.ceil(p*D)
    if (D-D0) % 2 == 1 :
        D0 = 2 if D0 == 1 else D0-1
    sectors = [(0, D0)]
    Drem = D - D0
    n = 1
    while Drem > 0 :
        pn = p * λ**n
        Dn = math.ceil(pn*D)
        sectors = sectors + [(n, Dn), (-n, Dn)]
        Drem -= 2*Dn
        n += 1
    sectors.sort(key = lambda d : d[0])
    return sectors


def profile_ctmrg(on_site_t, X, config_profile):
    USE_TORCH_NVTX= ("torch" in config_profile.backend.BACKEND_ID)

    geometry= RectangularUnitcell(pattern=[[0,]])
    psi = Peps(geometry, tensors=dict(zip(geometry.sites(), [on_site_t, ])))

    # grow X until saturation
    D= max([ sum(l.D) for l in on_site_t.get_legs()[:4]]) # over aux legs
    nsteps= 2
    X= nsteps*(D**2)
    env = fpeps.EnvCTM(psi, init='eye')
    info = env.ctmrg_(opts_svd = {"D_total": X, 'fix_signs': False, 'tol': 1.0e-12}, max_sweeps= nsteps, 
                            truncation_f=None, use_qr=False, checkpoint_move=False)
    
    # clone current env
    env2= env.clone()

    # switch backends 
    b= on_site_t.clone()
    b.config= config_profile
    psi2= Peps(geometry, tensors=dict(zip(geometry.sites(), [b, ])))
    env2.psi= Peps2Layers(psi2)
    for t in "tl", "tr", "bl", "br", "t", "l", "r", "b":
        setattr(getattr(env2[0,0],t),"config",config_profile)

    opts_svd = {"D_total": X, 'fix_signs': False, 'tol': 1.0e-12}
    max_sweeps= 5
    corner_tol= 1.0e-8
    

    max_dsv, converged, history = None, False, []
    if USE_TORCH_NVTX:
        env2.profiling_mode= "NVTX"
    for ctm_step in env2.iterate_(opts_svd=opts_svd, moves='hv', method='2site', max_sweeps=max_sweeps, 
                                  iterator_step=1, corner_tol=corner_tol, truncation_f=None, use_qr=False, checkpoint_move=False):
        sweep, max_dsv, max_D, converged = ctm_step
        print(f'Sweep = {sweep:03d}; max_diff_corner_singular_values = {max_dsv:0.2e} max_D {max_D} max_X {env2.effective_chi()}')


@torch_test
def test_ctmrg_U1_torch(config_kwargs,D : int=3, X : int=None):
    """
    """
    config = yastn.make_config(sym='U1', **config_kwargs)
    _cfg= copy.deepcopy(config_kwargs)
    _cfg["backend"]= backend_torch 
    config_torch = yastn.make_config(sym='U1', **_cfg)
    config.backend.random_seed(1)

    # make on-site tensor
    X= 2*D**2 if X is None else X
    aux_ts,aux_Ds= tuple(zip(*distributeU1_exponential(D)))
    l0= yastn.Leg(config_torch, s=1, t=aux_ts, D=aux_Ds)
    print(f"aux leg {l0}")
    lp= yastn.Leg(config_torch, s=1, t=(-1,1), D=(1,1) )
    a= yastn.rand(config_torch, legs=[l0.conj(), l0.conj(), l0, l0, lp], n=1)

    # 
    profile_ctmrg(a, 2*(D**2), config)


@torch_test
def test_ctmrg_Z2_torch(config_kwargs, D : int=4, X : int=None):
    """
    """
    config = yastn.make_config(sym='Z2', **config_kwargs)
    _cfg= copy.deepcopy(config_kwargs)
    _cfg["backend"]= backend_torch 
    config_torch = yastn.make_config(sym='Z2', **_cfg)
    config.backend.random_seed(1)

    # make on-site tensor
    X= 2*D**2 if X is None else X
    aux_ts,aux_Ds= (0,1), (D-D//2,D//2)
    l0= yastn.Leg(config_torch, s=1, t=aux_ts, D=aux_Ds)
    print(f"CTM D {D} X {X} aux leg {l0}")
    lp= yastn.Leg(config_torch, s=1, t=(0,), D=(2,) )
    a= yastn.rand(config_torch, legs=[l0.conj(), l0.conj(), l0, l0, lp], n=1)

    # 
    profile_ctmrg(a, X, config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='profile ctmrg', allow_abbrev=False)
    parser.add_argument("--backend", help='backend', default='np', choices=['np','torch','torch_cpp'], action='store')
    parser.add_argument("--device", help='cpu or cuda', default='cpu', action='store')
    parser.add_argument("--tensordot_policy", choices=['fuse_to_matrix', 'fuse_contracted', 'no_fusion'], default='fuse_to_matrix', action='store')
    parser.add_argument("--default_fusion", choices=['hard', 'meta'], default='hard', action='store')
    parser.add_argument("--D", type=int, default=3, help="iPEPS bond dimension")
    parser.add_argument("--X", type=int, default=None, help="environment bond dimension")
    parser.add_argument("--sym", type=str, default='Z2', choices=['U1','Z2'], help="symmetry")
    args, unknown_args = parser.parse_known_args()

    config_kwargs=  {'backend': args.backend, 'default_device': args.device,
            'default_fusion': args.default_fusion, 'tensordot_policy': args.tensordot_policy,}
    
    if args.sym == 'Z2':
        test_ctmrg_Z2_torch(config_kwargs, D=args.D, X=args.X)
    if args.sym == 'U1':
        test_ctmrg_U1_torch(config_kwargs, D=args.D, X=args.X)





