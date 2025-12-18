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
import sys
import os
try:
    import yastn
except ModuleNotFoundError as e:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, dir_path+"/../../")
    import yastn
from yastn.backend import backend_np
from yastn.backend import backend_torch
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Peps, Peps2Layers, RectangularUnitcell
from yastn.tn.fpeps.envs._env_ctm_dist_mp import iterate_D_
from itertools import product
import json
import time
import logging

logger = logging.getLogger()
for h in list(logger.handlers): logger.removeHandler(h)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
    
tol = 1e-12  #pylint: disable=invalid-name
YASTN_LOG_LEVEL = int(os.getenv("YASTN_LOG_LEVEL","0"))

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


def profile_ctmrg(on_site_t, X, config_profile, Nx=1, Ny=1, svd_policy="fullrank", devices=None, **kwargs):
    USE_TORCH_NVTX= ("torch" in config_profile.backend.BACKEND_ID)

    if kwargs.get("to_dense",False):
        on_site_t= on_site_t.to_nonsymmetric()
        _kwargs= config_profile._asdict()
        _kwargs["sym"]= yastn.sym.sym_none
        config_profile = yastn.make_config(**_kwargs)
    geometry= RectangularUnitcell(pattern=[ [col for col in range(row*Ny,(row+1)*Ny)] for row in range(Nx) ],)
    psi = Peps(geometry, tensors=dict(zip(geometry.sites(), [ on_site_t.clone() for _ in range(len(geometry.sites())) ])))

    # grow X until saturation
    D= max([ sum(l.D) for l in on_site_t.get_legs()[:4]]) # over aux legs
    nsteps= max(2,X//(D**2))
    env = fpeps.EnvCTM(psi, init='eye')

    info = env.ctmrg_(opts_svd = {"policy": svd_policy, "D_total": X, 'fix_signs': False, 'tol': 1.0e-12}, max_sweeps= nsteps, 
                            truncation_f=None, use_qr=False, checkpoint_move=False)
    
    # clone current env
    env2= env.clone()

    # switch backends 
    b= on_site_t.clone()
    b.config= config_profile
    psi2= Peps(geometry, tensors=dict(zip(geometry.sites(), [ b.clone() for _ in range(len(geometry.sites())) ] )))
    env2.psi= Peps2Layers(psi2)
    for site in geometry.sites():
        for t in "tl", "tr", "bl", "br", "t", "l", "r", "b":
            setattr(getattr(env2[site],t),"config",config_profile)

    opts_svd = {"policy": svd_policy, "D_total": X, 'fix_signs': False, 'tol': 1.0e-12} 
    max_sweeps= kwargs.get("max_sweeps",5)
    corner_tol= kwargs.get("ctm_conv_crit",-1)
    

    max_dsv, converged, history = None, False, []
    t0= time.perf_counter()
    if USE_TORCH_NVTX: env2.profiling_mode= "NVTX"

    if len(devices)==1:
        ctm_gen= env2.iterate_(opts_svd=opts_svd, moves='hv', method='2site', max_sweeps=max_sweeps, 
                                  iterator_step=1, corner_tol=corner_tol, truncation_f=None, use_qr=False, checkpoint_move=False)
    elif len(devices)>1:
        ctm_gen= iterate_D_(env2, opts_svd=opts_svd, moves='hv', method='2site', max_sweeps=max_sweeps,
                               iterator_step=1, corner_tol=corner_tol, truncation_f=None, use_qr=False, checkpoint_move=False, 
                               devices=devices)
    for ctm_step in ctm_gen:
        sweep, max_dsv, max_D, converged = ctm_step
        t1= time.perf_counter()
        print(f'Sweep = {sweep:03d} t {t1-t0} [s] max_diff_corner_singular_values = {max_dsv:0.2e} max_D {max_D} max_X {env2.effective_chi()}')
        print("\n".join([f"Corner {c} {getattr(env2[0,0],c).get_legs(0)}" for c in ["tl", "tr", "bl", "br"]]))
        t0=t1

        if config_profile.backend.BACKEND_ID in ["torch_cpp"] and YASTN_LOG_LEVEL>1:
            # for i,v in enumerate(config_profile.backend.cutensor_cache_stats().values()):
            #     print(f"Cutensor plan cache {i}: {v}")
            print(f"cutensor.cache_stats: "+str(list(config_profile.backend.cutensor_cache_stats().values())))


def uniform_init_(a, target_cfg: yastn.tensor._auxliary._config):
    cfg_uni= yastn.make_config(sym=target_cfg.sym, backend='np', default_device='cpu', default_fusion='none',)
    cfg_uni.backend.random_seed(1)

    a_uni= yastn.rand(cfg_uni, legs= a.get_legs(), n=a.n)
    a._data= target_cfg.backend.to_tensor( a_uni._data, device=target_cfg.default_device, dtype=target_cfg.default_dtype )

@torch_test
def test_ctmrg_U1xU1(config_kwargs,D : int=3, X : int=None, u1_charges : list[int]=None, u1_Ds: list[int]=None, 
                           input_shape_file=None, **kwargs):
    """
    """
    config = yastn.make_config(sym='U1xU1', **config_kwargs)

    # make on-site tensor
    if input_shape_file:
        with open(input_shape_file, 'r') as f:
            shape_data= json.load(f)

            assert shape_data["symmetry"]=="U1xU1", "Input shape file symmetry mismatch"
            a= yastn.rand(config, 
                          legs= [ yastn.Leg(config, s= shape_data[lid]["signature"], t=shape_data[lid]["charges"], D=shape_data[lid]["dimensions"]) 
                                    for lid in ["a_leg_t","a_leg_l","a_leg_b","a_leg_r","a_leg_s"] 
                            ])
            D= min( sum(l.D) for l in a.get_legs()[:4])
            X= 2*D**2 if X is None else X            
    else:
        if u1_charges and u1_Ds and len(u1_charges)==len(u1_Ds):
            aux_ts,aux_Ds= u1_charges,u1_Ds
        else:    
            aux_ts,aux_Ds= tuple(zip(*distributeU1_exponential(D)))
        # product of U1s
        aux_ts= list(product(aux_ts,aux_ts))
        aux_Ds= [ x*y for x,y in product(aux_Ds,aux_Ds) ]
        D= sum(aux_Ds)
        X= 2*D**2 if X is None else X
        l0= yastn.Leg(config, s=1, t=aux_ts, D=aux_Ds)
        lp= yastn.Leg(config, s=1, t=(((1,-1),(-1,1))), D=(1,1) )
        a= yastn.rand(config, legs=[l0.conj(), l0.conj(), l0, l0, lp], n=config.sym.zero())
    
    uniform_init_(a, config)
    print(a)
    profile_ctmrg(a, X, config, **kwargs)


@torch_test
def test_ctmrg_U1(config_kwargs, D : int=3, X : int=None, u1_charges : list[int]=None, u1_Ds: list[int]=None, 
                        input_shape_file=None, **kwargs):
    """
    """
    config = yastn.make_config(sym='U1', **config_kwargs)

    # make on-site tensor
    if input_shape_file:
        with open(input_shape_file, 'r') as f:
            shape_data= json.load(f)

            assert shape_data["symmetry"]=="U1", "Input shape file symmetry mismatch"
            a= yastn.rand(config, 
                          legs= [ yastn.Leg(config, s= shape_data[lid]["signature"], t=shape_data[lid]["charges"], D=shape_data[lid]["dimensions"]) 
                                    for lid in ["a_leg_t","a_leg_l","a_leg_b","a_leg_r","a_leg_s"] 
                            ])
            D= min( sum(l.D) for l in a.get_legs()[:4])
            X= 2*D**2 if X is None else X
    else:
        if u1_charges and u1_Ds and len(u1_charges)==len(u1_Ds):
            aux_ts,aux_Ds= u1_charges,u1_Ds
            D= sum(aux_Ds)
        else:    
            aux_ts,aux_Ds= tuple(zip(*distributeU1_exponential(D)))
        X= 2*D**2 if X is None else X
        l0= yastn.Leg(config, s=1, t=aux_ts, D=aux_Ds)
        lp= yastn.Leg(config, s=1, t=(-1,1), D=(1,1) )
        a= yastn.rand(config, legs=[l0.conj(), l0.conj(), l0, l0, lp], n=0)

    uniform_init_(a, config)
    print(a)
    profile_ctmrg(a, X, config, **kwargs)


@torch_test
def test_ctmrg_Z2(config_kwargs, D : int=4, X : int=None, **kwargs):
    """
    """
    config = yastn.make_config(sym='Z2', **config_kwargs)
    config.backend.random_seed(1)

    # make on-site tensor
    X= 2*D**2 if X is None else X
    aux_ts,aux_Ds= (0,1), (D-D//2,D//2)
    l0= yastn.Leg(config, s=1, t=aux_ts, D=aux_Ds)
    print(f"CTM D {D} X {X} aux leg {l0}")
    lp= yastn.Leg(config, s=1, t=(0,), D=(2,) )
    a= yastn.rand(config, legs=[l0.conj(), l0.conj(), l0, l0, lp], n=1)

    # 
    uniform_init_(a, config)
    profile_ctmrg(a, X, config, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='profile ctmrg', allow_abbrev=False)
    parser.add_argument("--backend", help='backend', default='np', choices=['np','torch','torch_cpp'], action='store')
    parser.add_argument("--devices", help='cpu or (list of) cuda. Default is cpu', default=None, dest='devices', nargs="+")
    parser.add_argument("--tensordot_policy", choices=['fuse_to_matrix', 'fuse_contracted', 'no_fusion'], default='fuse_to_matrix', action='store')
    parser.add_argument("--default_fusion", choices=['hard', 'meta'], default='hard', action='store')
    parser.add_argument("--svd_policy", type=str, default='fullrank', choices=["fullrank", "qr", "randomized", "block_arnoldi", "block_propack"], help="svd driver")
    parser.add_argument("--Nx", type=int, default=1, help="rows of the unit cell")
    parser.add_argument("--Ny", type=int, default=1, help="columns of the unit cell")
    parser.add_argument("--D", type=int, default=3, help="iPEPS bond dimension")
    parser.add_argument("--X", type=int, default=None, help="environment bond dimension")
    parser.add_argument("--sym", type=str, default='Z2', choices=['U1','Z2', 'U1xU1'], help="symmetry")
    parser.add_argument("--max_sweeps", type=int, default=5, help="Number of ctmrg sweeps ran under profiling")
    parser.add_argument("--to_dense", action='store_true', help="Run profile case as dense")
    parser.add_argument("--ctm_conv_crit", type=float, default=-1, help="Covergence criterion for ctmrg. "\
        +"Terminate once max change in corner singular values is below this value. -1 means run until max_sweeps.")
    parser.add_argument(
        "--u1_charges",
        dest="u1_charges",
        default=None,
        type=int,
        help="U(1) charges assigned to the states in the virtual space",
        nargs="+",
    )
    parser.add_argument(
        "--u1_Ds",
        dest="u1_Ds",
        default=None,
        type=int,
        help="dimensions of U(1) sectors in the virtual space",
        nargs="+",
    )
    parser.add_argument("--input_shape_file", type=str, default=None)
    args = parser.parse_args()

    # process devices
    ctm_devices= ['cpu'] if args.devices is None else args.devices
    
    config_kwargs=  {'backend': args.backend, 'default_device': args.devices[0],
            'default_fusion': args.default_fusion, 'tensordot_policy': args.tensordot_policy,}
    
    kwargs= dict( Nx=args.Nx, Ny=args.Ny, svd_policy=args.svd_policy, max_sweeps=args.max_sweeps, to_dense=args.to_dense,
                 ctm_conv_crit=args.ctm_conv_crit,) 
    if args.sym == 'Z2':
        test_ctmrg_Z2(config_kwargs,  D=args.D, X=args.X, devices=ctm_devices, **kwargs)
    if args.sym == 'U1':
        test_ctmrg_U1(config_kwargs, D=args.D, X=args.X, u1_charges=args.u1_charges, u1_Ds=args.u1_Ds, 
                            input_shape_file=args.input_shape_file, devices=ctm_devices, **kwargs)
    if args.sym == 'U1xU1':
        test_ctmrg_U1xU1(config_kwargs, D=args.D, X=args.X, u1_charges=args.u1_charges, u1_Ds=args.u1_Ds, 
                               input_shape_file=args.input_shape_file, devices=ctm_devices, **kwargs)