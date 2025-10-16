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
import yastn
from yastn.backend import backend_torch
from yastn.backend import backend_np
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

def compute_halfs_2site_(proj, site, dirn, env, opts_svd, **kwargs):
    r"""
    Calculate new projectors for CTM moves from 4x4 extended corners.
    """
    psi = env.psi
    sites = [psi.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    if None in sites:
        return

    use_qr = kwargs.get("use_qr", True)
    psh= kwargs.pop("proj_history", None)
    svd_predict_spec= lambda s0,p0,s1,p1: opts_svd.get('D_block', float('inf')) if psh is None else \
        env._partial_svd_predict_spec(getattr(psh[s0],p0), getattr(psh[s1],p1), opts_svd.get('sU', 1))

    tl, tr, bl, br = sites

    cor_tl = env[tl].l @ env[tl].tl @ env[tl].t
    cor_tl = yastn.tensordot(cor_tl, psi[tl], axes=((2, 1), (0, 1)))
    cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))

    cor_bl = env[bl].b @ env[bl].bl @ env[bl].l
    cor_bl = yastn.tensordot(cor_bl, psi[bl], axes=((2, 1), (1, 2)))
    cor_bl = cor_bl.fuse_legs(axes=((0, 3), (1, 2)))

    cor_tr = env[tr].t @ env[tr].tr @ env[tr].r
    cor_tr = yastn.tensordot(cor_tr, psi[tr], axes=((1, 2), (0, 3)))
    cor_tr = cor_tr.fuse_legs(axes=((0, 2), (1, 3)))

    cor_br = env[br].r @ env[br].br @ env[br].b
    cor_br = yastn.tensordot(cor_br, psi[br], axes=((2, 1), (2, 3)))
    cor_br = cor_br.fuse_legs(axes=((0, 2), (1, 3)))

    if ('l' in dirn) or ('r' in dirn):
        cor_tt = cor_tl @ cor_tr  # b(left) b(right)
        cor_bb = cor_br @ cor_bl  # t(right) t(left)
        return cor_tt, cor_bb

    if ('t' in dirn) or ('b' in dirn):
        cor_ll = cor_bl @ cor_tl  # l(bottom) l(top)
        cor_rr = cor_tr @ cor_br  # r(top) r(bottom)
        return cor_ll, cor_rr
    
    return None, None


@torch_test
def test_enlarged_corner_U1(config_kwargs):
    """ test tensordot for different symmetries. """
    config = yastn.make_config(sym='U1', **config_kwargs)
    _cfg= copy.deepcopy(config_kwargs)
    _cfg["backend"]= backend_torch 
    config_torch = yastn.make_config(sym='U1', **_cfg)
    config.backend.random_seed(1)

    # make on-site tensor
    D=9
    aux_ts,aux_Ds= tuple(zip(*distributeU1_exponential(D)))
    nsteps= 2
    X= nsteps*(D**2)
    l0= yastn.Leg(config_torch, s=1, t=aux_ts, D=aux_Ds)
    lp= yastn.Leg(config_torch, s=1, t=(-1,0,1), D=(1,1,1) )
    a= yastn.ones(config_torch, legs=[l0.conj(), l0.conj(), l0, l0, lp], n=1)

    geometry= RectangularUnitcell(pattern=[[0,]])
    psi = Peps(geometry, tensors=dict(zip(geometry.sites(), [a, ])))

    # grow X until saturation
    env = fpeps.EnvCTM(psi, init='eye')
    info = env.ctmrg_(opts_svd = {"D_total": X, 'fix_signs': False}, max_sweeps= nsteps, 
                            truncation_f=None, use_qr=False, checkpoint_move=False)
    
    h1l_ref,h2l_ref= compute_halfs_2site_(None, (0,0), 'l', env, None)

    # clone current env
    env2= env.clone()

    # do 1 more step on original env for reference
    # info = env.ctmrg_(opts_svd = {"D_total": X, 'fix_signs': False}, max_sweeps= 1, 
    #                         truncation_f=None, use_qr=False, checkpoint_move=False)

    # switch backends 
    # setattr(a,"config",config)
    b= a.clone()
    b.config= config
    psi2= Peps(geometry, tensors=dict(zip(geometry.sites(), [b, ])))
    env2.psi= Peps2Layers(psi2)
    for t in "tl", "tr", "bl", "br", "t", "l", "r", "b":
        setattr(getattr(env2[0,0],t),"config",config)
    # import pdb; pdb.set_trace()
      
    h1l,h2l= compute_halfs_2site_(None, (0,0), 'l', env2, None)

    # import pdb; pdb.set_trace()

    # info = env2.ctmrg_(opts_svd = {"D_total": X, 'fix_signs': False}, max_sweeps= 1, 
    #                         truncation_f=None, use_qr=False, checkpoint_move=False)
    # import pdb; pdb.set_trace()

    # compare envs (could be gauged differently?)
    pass


if __name__ == '__main__':
    pass