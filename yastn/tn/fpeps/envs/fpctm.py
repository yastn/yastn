# Copyright 2026 The YASTN Authors. All Rights Reserved.
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

from ._env_dataclasses import EnvCTM_projectors
from ...mps._umps import biorthogonalize_left, eigs_implicit_v2
from .geometry import Lattice
from ....tensor import tensordot, ncon
from ._env_contractions import corner2x2

def _update_fpctm_projectors(env: EnvCTM) -> "Lattice[EnvCTM_projectors]":
    proj_new = Lattice(env.geometry, objects={site: EnvCTM_projectors() for site in env.sites()})
    # Biorthogonalize the top and bottom edges
    #   
    # r \ c   0      1      2  ... 
    # 0   A[0,0] A[0,1] A[0,2] ...   
    # 1   A[1,0] A[1,1] A[1,2] ...  
    # 2   A[2,0] A[2,1] A[2,2] ...
    # ...
    #
    # -- C_LU  -- T_t [r,c]   --     -- P_L[r,c]      -- C_LU --
    #             |              =      |
    # -- C_DL  -- T_b [r-1,c] --     -- Pbar_L[r-1,c] -- C_DL --
    #
    pinv_cutoff, eps = 1e-12, 1e-12
    for site in env.sites:
        r,c= site
        # left projectors
        #
        # T[r,.] A[r-1,.]
        # A[r,.] T[r-1,.]
        #
        #
        #   C_LU[r,c] -- t[r,c] -- t[r,c+1]     =             -- P_L[r,c] -- C_LU[r,c+1] -- t[r,c+1]
        #    T_l[r,c] -- A[r,c]                      T_l[r,c] --   A[r,c] -- 
        #
        #  T_l[r-1,c] -- A[r-1,c]               =    T_l[r-1,c] -- A[r-1,c] --
        # C_LU[r-1,c] -- b[r-1,c] -- b[r-1,c+1]            -- Pbar_L[r-1,c] -- C_LU[r-1,c+1] -- t[r-1,c+1]       
        #
        P_L, Pbar_L, C_LU, C_DL= biorthogonalize_left(env.boundary_mps(r,'t'), 
                                                      env.boundary_mps(r-1,'b'), 
                                                      C_init=None, pinv_cutoff=pinv_cutoff, eps=eps)
        proj_new[r,c].hlt= P_L
        proj_new[r-1,c].hlb= Pbar_L

        # right projectors
        # 
        #      A[r,.] T_t[r+1,.]  
        #    T_b[r,.]   A[r+1,.]
        #
        # -- t[r+1,c-1] -- t[r+1,c] -- C_UR[r+1,c] =  -- t[r+1,c-1] -- C_UR[r+1,c-1] -- Pbar_R[r+1,c] --
        #               -- A[r+1,c] -- T_R[r+1,c]                                    -- A[r+1,c]      -- T_R[r+1,c] 
        #    
        #                -- A[r,c] -- T_r[r,c]                                 -- A[r,c]   -- T_r[r,c]
        #    -- b[r,c-1] -- b[r,c] -- C_RD[r,c]  =  -- b[r,c-1] -- C_RD[r,c-1] -- P_R[r,c] -- 
        P_R, Pbar_R, C_RD, C_UR= biorthogonalize_left( env.boundary_mps(r, 'b').reverse_sites(), 
                                                       env.boundary_mps(r+1, 't').reverse_sites(), 
                                                       C_init=None, pinv_cutoff=pinv_cutoff, eps=eps)
        proj_new[r,c].hrb= P_R
        proj_new[r+1,c].hrt= Pbar_R

        # top projectors
        #
        #      A[.,c] T_r[.,c]
        #  T_l[.,c+1]   A[.,c+1]
        P_t, Pbar_t, C_UR, C_LU= biorthogonalize_left( env.boundary_mps(c, 'r'), 
                                                       env.boundary_mps(r+1, 'l'), 
                                                       C_init=None, pinv_cutoff=pinv_cutoff, eps=eps)
        proj_new[r,c].vtl= P_t
        proj_new[r,c+1].vtr= Pbar_t

        # bottom projectors
        #
        #    T_l[.,c]     A[.,c]
        #      A[.,c-1] T_r[.,c-1]
        P_b, Pbar_b, C_DL, C_RD= biorthogonalize_left( env.boundary_mps(c, 'l'), 
                                                       env.boundary_mps(r-1, 'r'), 
                                                       C_init=None, pinv_cutoff=pinv_cutoff, eps=eps)
        proj_new[r,c].vbl= P_b
        proj_new[r,c-1].vbr= Pbar_b
    return proj_new

def _update_fpctm_env(env: EnvCTM) -> EnvCTM:
    # Update the environment tensors using the projectors
          
    psi = env.psi
    env_new = EnvCTM(env.psi, init=None)

    # T-tensors
    for site in env.sites:
        r,c= site
    
        def fpop_l(T_l):
            for _c in range(env.Ny):
                T_l = T_l @ env.proj[r,c+_c].hlt
                T_l = tensordot(psi[r,c+_c], T_l, axes=((0, 1), (2, 1)))
                T_l = tensordot(env.proj[r,c+_c].hlb, T_l, axes=((0, 1), (2, 0)))
            return T_l

        evals, evecs= eigs_implicit_v2(fpop_l, k=1, eigenvectors=True, V0= env[r,c].l)
        env_new[r,c].l = evecs[0].remove_leg(0)

        def fpop_r(T_r):
            for _c in range(env.Ny):
                T_r = T_r @ env.proj[r,c-_c].hrb
                T_r = tensordot(psi[r,c-_c], T_r, axes=((2, 3), (2, 1)))
                T_r = tensordot(env.proj[r,c-_c].hrt, T_r, axes=((0, 1), (2, 0)))
            return T_r
        evals, evecs= eigs_implicit_v2(fpop_r, k=1, eigenvectors=True, V0= env[r,c].r)
        env_new[r,c].r = evecs[0].remove_leg(0)

        def fpop_t(T_t):
            for _r in range(env.Nx):
                T_t = tensordot(env.proj[r+_r,c].vtl, T_t, axes=(0, 0))
                T_t = tensordot(T_t, psi[r+_r,c], axes=((2, 0), (0, 1)))
                T_t = tensordot(T_t, env.proj[r+_r,c].vtr, axes=((1, 3), (0, 1)))
            return T_t
        evals, evecs= eigs_implicit_v2(fpop_t, k=1, eigenvectors=True, V0= env[r,c].t)
        env_new[r,c].t = evecs[0].remove_leg(0)

        def fpop_b(T_b):
            for _r in range(env.Nx):
                T_b = tensordot(env.proj[r-_r,c].vbr, T_b, axes=(0, 0))
                T_b = tensordot(T_b, psi[r-_r,c], axes=((2, 0), (2, 3)))
                T_b = tensordot(T_b, env.proj[r-_r,c].vbl, axes=((1, 3), (0, 1)))
            return T_b
        evals, evecs= eigs_implicit_v2(fpop_b, k=1, eigenvectors=True, V0= env[r,c].b)
        env_new[r,c].b = evecs[0].remove_leg(0)
    
    # C-tensors
    for site in env.sites:
        r,c= site

        def fpop_tl(C_tl):
            C_tl= corner2x2('tl', env_new[r,c].l, C_tl, env_new[r,c].t, psi[r,c])
            # C_lu= tensordot( C_lu, env.proj[r,c].hlb, axes=((0, 2), (0, 1)) )
            # C_lu= tensordot( C_lu, env.proj[r,c].vtr, axes=((0, 1), (0, 1)) )
            C_tl = ncon( [C_tl, env.proj[r,c].hlb, env.proj[r,c].vtr],
                         [[0,1,2,3], [0, 2, -1], [1, 3, -2]], )
            return C_tl
        evals, evecs= eigs_implicit_v2(fpop_tl, k=1, eigenvectors=True, V0= env[r,c].tl)
        env_new[r,c].tl = evecs[0].remove_leg(0)

        def fpop_tr(C_ru):
            C_ru = corner2x2('tr', env_new[r,c].t, C_ru, env_new[r,c].r, psi[r,c])
            C_ru = ncon( [C_ru, env.proj[r,c].vtl, env.proj[r,c].hrb],
                         [[0,1,2,3], [0, 2, -1], [1, 3, -2]], )
            return C_ru
        evals, evecs= eigs_implicit_v2(fpop_tr, k=1, eigenvectors=True, V0= env[r,c].tr)
        env_new[r,c].tr = evecs[0].remove_leg(0)

        def fpop_br(C_br):
            C_br = corner2x2('br', env_new[r,c].r, C_br, env_new[r,c].b, psi[r,c])
            C_br = ncon( [C_br, env.proj[r,c].hrt, env.proj[r,c].vbl],
                         [[0,1,2,3], [0, 2, -1], [1, 3, -2]], )
            return C_br
        evals, evecs= eigs_implicit_v2(fpop_br, k=1, eigenvectors=True, V0= env[r,c].br)
        env_new[r,c].br = evecs[0].remove_leg(0)

        def fpop_bl(C_bl):
            C_bl = corner2x2('bl', env_new[r,c].b, C_bl, env_new[r,c].l, psi[r,c])
            C_bl = ncon( [C_bl, env.proj[r,c].vbr, env.proj[r,c].hlt],
                         [[0,1,2,3], [0, 2, -1], [1, 3, -2]], )
            return C_bl
        evals, evecs= eigs_implicit_v2(fpop_bl, k=1, eigenvectors=True, V0= env[r,c].bl)
        env_new[r,c].bl = evecs[0].remove_leg(0)

    return env_new