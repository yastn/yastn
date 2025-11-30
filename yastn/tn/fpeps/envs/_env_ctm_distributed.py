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
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Sequence
import time

import numpy as np
from yastn._from_dict import from_dict
from yastn.tn.fpeps._geometry import Site
from ....tensor import Tensor, YastnError, Leg, tensordot, qr, vdot
from ._env_ctm import CTMRG_out, EnvCTM, proj_corners, trivial_projectors_, update_env_, update_storage_
from ...._split_combine_dict import split_data_and_meta, combine_data_and_meta
import logging
import os
import torch
logger = logging.getLogger(__name__)


def _validate_devices_list(devices: list[str] | None) -> None:
    if devices:
        assert len(devices) > 0, "At least one device must be provided."
    if len(devices) < 2:
        raise YastnError("At least two devices must be provided for distributed CTM.")


def iterate_T_(env, opts_svd=None, moves='hv', method='2site', max_sweeps=1, iterator=False, corner_tol=None, truncation_f: Callable = None, **kwargs):
        r"""
        Perform CTMRG updates :meth:`yastn.tn.fpeps.EnvCTM.update_` until convergence.
        Convergence can be measured based on singular values of CTM environment corner tensors.

        Outputs iterator if ``iterator`` is given, which allows
        inspecting ``env``, e.g., calculating expectation values,
        outside of ``ctmrg_`` function after every sweeps.

        Parameters
        ----------
        opts_svd: dict
            A dictionary of options to pass to SVD truncation algorithm.
            This sets EnvCTM bond dimension.

        moves: str
            Specify a sequence of moves forming a single sweep.
            Individual moves are 'l', 'r', 't', 'b', 'h', or 'v'.
            Horizontal 'h' and vertical 'v' moves have all sites updated simultaneously.
            Left 'l', right 'r', top 't', and bottom 'b' are executed causally,
            row after row or column after column.
            Argument specifies a sequence of individual moves, where sensible options are 'hv' and 'lrtb'.
            The default is 'hv'.

        method: str
            '2site', '1site'. The default is '2site'.

                * '2site' uses the standard 4x4 enlarged corners, enabling enlargement of EnvCTM bond dimensions. When some PEPS bonds are rank-1, it recognizes it to use 5x4 corners to prevent artificial collapse of EnvCTM bond dimensions to 1, which is important for hexagonal lattice.
                * '1site' uses smaller 4x2 corners. It is significantly faster, but is less stable and  does not allow for EnvCTM bond dimension growth.

        max_sweeps: int
            The maximal number of sweeps.

        iterator: bool
            If True, ``ctmrg_`` returns a generator that would yield output after every sweep.
            The default is False, in which case  ``ctmrg_`` sweeps are performed immediately.

        corner_tol: float
            Convergence tolerance for the change of singular values of all corners in a single update.
            The default is ``None``, in which case convergence is not checked and it is up to user to implement
            convergence check.

        truncation_f:
            Custom projector truncation function with signature ``truncation_f(S: Tensor)->Tensor``, consuming
            rank-1 tensor with singular values. If provided, truncation parameters passed to SVD decomposition
            are ignored.

        checkpoint_move: str | bool
            Whether to use checkpointing for the CTM updates. The default is ``False``.
            Otherwise, in case of PyTorch backend it can be set to 'reentrant' for reentrant checkpointing
            or 'nonreentrant' for non-reentrant checkpointing, see https://pytorch.org/docs/stable/checkpoint.html.

        use_qr: bool
            Whether to include intermediate QR decomposition while calculating projectors.
            The default is ``True``.

        Returns
        -------
        Generator if iterator is True.

        CTMRG_out(NamedTuple)
            NamedTuple including fields:

                * ``sweeps`` number of performed ctmrg updates.
                * ``max_dsv`` norm of singular values change in the worst corner in the last sweep.
                * ``max_D`` largest bond dimension of environment tensors virtual legs.
                * ``converged`` whether convergence based on ``corner_tol`` has been reached.
        """
        if "checkpoint_move" in kwargs:
            if "torch" in env.config.backend.BACKEND_ID:
                assert kwargs["checkpoint_move"] in ['reentrant', 'nonreentrant', False], f"Invalid choice for {kwargs['checkpoint_move']}"
        kwargs["truncation_f"] = truncation_f
        kwargs["iterator_step"] = kwargs.get("iterator_step", int(iterator))
        tmp = _ctmrg_iterator_T_(env, opts_svd, moves, method, max_sweeps, corner_tol, **kwargs)
        return tmp if kwargs["iterator_step"] else next(tmp)


def _ctmrg_iterator_T_(env, opts_svd, moves, method, max_sweeps, corner_tol, **kwargs):
    """ Generator for iterate_ (or its alias ctmrg_). """
    iterator_step = kwargs.get("iterator_step", 0)
    max_dsv, converged, history = None, False, []
    
    devices= kwargs.get('devices', None)
    _validate_devices_list(devices)

    max_workers= len(devices) # one is the main thread
    # with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ctmrg" ) as thread_pool_executor:`
    import torch.multiprocessing as mp
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn'), initializer=None, initargs=(), 
                             max_tasks_per_child=None) as thread_pool_executor:
        logger.info(f"ctmrg_T main loop max_workers={max_workers} on devices={devices}")
        for sweep in range(1, max_sweeps + 1):
            if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_push(f"update_")
            update_T_(thread_pool_executor, env, opts_svd=opts_svd, moves=moves, method=method, **kwargs)
            if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_pop()
            
            # use default CTM convergence check
            if corner_tol is not None:
                # Evaluate convergence of CTM by computing the difference of environment corner spectra between consecutive CTM steps.
                corner_sv = env.calculate_corner_svd()
                max_dsv = max((corner_sv[k] - history[-1][k]).norm().item() for k in corner_sv) if history else float('Nan')
                history.append(corner_sv)
                converged = max_dsv < corner_tol
                logging.info(f'Sweep = {sweep:03d}; max_diff_corner_singular_values = {max_dsv:0.2e}')

                if converged:
                    break

            if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
                yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)
    yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)


def update_T_(thread_pool_executor, env, opts_svd, moves='hv', method='2site', **kwargs):
    r"""
    Perform one step of CTMRG update. Environment tensors are updated in place.

    The function performs a CTMRG update for a square lattice using the corner transfer matrix
    renormalization group (CTMRG) algorithm. The update is performed in two steps: a horizontal move
    and a vertical move. The projectors for each move are calculated first, and then the tensors in
    the CTM environment are updated using the projectors. The boundary conditions of the lattice
    determine whether trivial projectors are needed for the move.

    Parameters
    ----------
    opts_svd: dict
        A dictionary of options to pass to SVD truncation algorithm.
        This sets EnvCTM bond dimension.

    moves: str
        Specify a sequence of moves forming a single sweep.
        Individual moves are 'l', 'r', 't', 'b', 'h', or 'v'.
        Horizontal 'h' and vertical 'v' moves have all sites updated simultaneously.
        Left 'l', right 'r', top 't', and bottom 'b' are executed causally,
        row after row or column after column.
        Argument specifies a sequence of individual moves, where sensible options are 'hv' and 'lrtb'.
        The default is 'hv'.

    method: str
        '2site' or '1site'. The default is '2site'.
        '2site' uses the standard 4x4 enlarged corners, allowing to enlarge EnvCTM bond dimension.
        '1site' uses smaller 4x2 corners. It is significantly faster, but is less stable and
        does not allow to grow EnvCTM bond dimension.

    checkpoint_move: bool
        Whether to use (reentrant) checkpointing for the move. The default is ``False``

    Returns
    -------
    proj: Peps structure loaded with CTM projectors related to all lattice site.
    """
    if 'tol' not in opts_svd and 'tol_block' not in opts_svd:
        opts_svd['tol'] = 1e-14
    if method not in ('1site', '2site'):
        raise YastnError(f"CTM update {method=} not recognized. Should be '1site' or '2site'")

    checkpoint_move = kwargs.get('checkpoint_move', False)
    for d in moves:
        if checkpoint_move:
            def f_update_core_(move_d, loc_im, *inputs_t):
                loc_env = EnvCTM.from_dict(combine_data_and_meta(inputs_t, loc_im))
                _update_core_T_(thread_pool_executor, loc_env, move_d, opts_svd, method=method, **kwargs)
                out_data, out_meta = split_data_and_meta(loc_env.to_dict(level=0))
                return out_data, out_meta

            if "torch" in env.config.backend.BACKEND_ID:
                inputs_t, inputs_meta = split_data_and_meta(env.to_dict(level=0))

                if checkpoint_move == 'reentrant':
                    use_reentrant = True
                elif checkpoint_move == 'nonreentrant':
                    use_reentrant = False
                checkpoint_F = env.config.backend.checkpoint
                out_data, out_meta = checkpoint_F(f_update_core_, d, inputs_meta, *inputs_t, \
                                    **{'use_reentrant': use_reentrant, 'debug': False})
            else:
                raise RuntimeError(f"CTM update: checkpointing not supported for backend {env.config.BACKEND_ID}")

            # reconstruct env from output tensors
            env.update_from_dict_(combine_data_and_meta(out_data, out_meta))
        else:
            _update_core_T_(thread_pool_executor, env, d, opts_svd, method=method, **kwargs)
    return env


def _update_core_T_(executor, env, move: str, opts_svd: dict, **kwargs):
    r"""
    Core function updating CTM environment tensors pefrorming specified move.

        #GPUs <= len(sites)
            Invoke update_projectors_ on multiple devices in parallel, each invocation receiving one GPU
        #GPUS >= 2*len(sites)
            Invoke update_projectors_ on multiple devices in parallel, each invocation receiving two GPUs. 

        To avoid nested process creation, we execute update in two-stage process:

            Stage 1: compute enlarged corners and enlarged halfs of the system
            Stage 2: compute projectors
    """
    assert move in ['h', 'v', 'l', 'r', 't', 'b'], "Invalid move"
    if (move in 'hv') or (len(env.sites()) < env.Nx * env.Ny):
        # For horizontal and vertical moves,
        # and unit cell with a nontrivial pattern like CheckerboardLattice or RectangularUnitcell,
        # all sites are updated simultaneously.
        shift_proj = None
        jobs = [env.sites()] # 1 group with all non-equivalent sites
    elif move == 'l':  # Move done sequentially, column after column. Number of jobs = Ny groups of Nx jobs
        shift_proj = 'l'
        jobs = [[Site(nx, ny) for nx in range(env.Nx)] for ny in range(env.Ny)]
    elif move == 'r':  # Move done sequentially, column after column.
        shift_proj = None
        jobs = [[Site(nx, ny) for nx in range(env.Nx)] for ny in range(env.Ny-1, -1, -1)]
    elif move == 't':  # Move done sequentially, row after row.
        shift_proj = 't'
        jobs = [[Site(nx, ny) for ny in range(env.Ny)] for nx in range(env.Nx)]
    elif move == 'b':  # Move done sequentially, row after row.
        shift_proj = None
        jobs = [[Site(nx, ny) for ny in range(env.Ny)] for nx in range(env.Nx-1, -1, -1)]

    def _partition_devices(num_jobs : int) -> Sequence[Sequence[str]]:
        devices = kwargs.get('devices', None)

        if len(devices) <= num_jobs:
            reps = (num_jobs + len(devices) - 1) // len(devices)
            return [ [d] for d in (devices * reps)[:num_jobs] ]
        elif len(devices) >= 2 * num_jobs:
            return [ devices[2*i:2*i+2] for i in range(num_jobs) ]
        else:
            return [ devices[i:i+1]+devices[i+num_jobs:i+1+num_jobs] for i in range(num_jobs) ]

    corner_sites= lambda site: tuple(env.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1)))

    svd_predict_spec= lambda s0,p0,s1,p1,sign: opts_svd.get('D_block', float('inf')) \
        if env.proj is None or (getattr(env.proj[s0],p0) is None or getattr(env.proj[s1],p1) is None) else \
        env._partial_svd_predict_spec(getattr(env.proj[s0],p0).get_legs(-1), getattr(env.proj[s1],p1).get_legs(-1), sign)    

    stage1_futures= {}
    for site_group in jobs:
        sites_proj = [env.nn_site(site, shift_proj) for site in site_group] if shift_proj else site_group
        sites_proj = [site for site in sites_proj if site is not None] # handles boundaries of finite PEPS
        
        #
        # Projectors
        device_groups= _partition_devices(len(sites_proj))
        logger.info(f"_update_core_T_ {move} {device_groups} ")

        if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_push(f"update_projectors_")
        
        # Stage 1: compute enlarged corners and halfs
        
        env_d= env.to_dict(level=1)
        stage1_futures= {
                executor.submit(
                    projectors_stage1, env_d, *corner_sites(site), 
                                      move, device_groups[i][0], **kwargs
                ): (i,site) for i,site in enumerate(sites_proj)
            }
        
        # Stage 2: compute projectors
        stage2_futures = {}
        for f1 in as_completed(stage1_futures):
            i,site = stage1_futures[f1]
            tl,tr,bl,br= corner_sites(site)

            # launch stage-2 for z0 and z1 immediately
            half1_d, half2_d = f1.result()
            if move in 'h':
                opts_svd["D_blocks"]= svd_predict_spec(tr, "hrb", br, "hrt", half1_d['struct']['s'][1])
                f2_0 = executor.submit(projectors_move_MP_, 'rh', half1_d, half2_d, 
                                       device_groups[i][0], env.config.default_device, opts_svd, **kwargs)
                opts_svd["D_blocks"]= svd_predict_spec(tl, "hlb", bl, "hlt", half1_d['struct']['s'][1])
                f2_1 = executor.submit(projectors_move_MP_, 'lh', half1_d, half2_d, 
                                       device_groups[i][1], env.config.default_device, opts_svd, **kwargs)
            elif move in 'v':
                opts_svd["D_block"]= svd_predict_spec(tl, "vtr", tr, "vtl", half1_d['struct']['s'][1])
                f2_0 = executor.submit(projectors_move_MP_, 'tv', half1_d, half2_d, 
                                       device_groups[i][0], env.config.default_device, opts_svd, **kwargs)
                opts_svd["D_block"]= svd_predict_spec(bl, "vbr", br, "vbl", half1_d['struct']['s'][1])
                f2_1 = executor.submit(projectors_move_MP_, 'bv', half1_d, half2_d, 
                                       device_groups[i][1], env.config.default_device, opts_svd, **kwargs)
            stage2_futures[f2_0] = (i, site, 'rh' if move=='h' else 'tv')
            stage2_futures[f2_1] = (i, site, 'lh' if move=='h' else 'bv')

        for f2 in stage2_futures:
            p1_d,p2_d= f2.result()  # wait for all to complete
            i,site,proj_pair= stage2_futures[f2]
            tl,tr,bl,br= corner_sites(site)
            if proj_pair=='rh':
                env.proj[tr].hrb, env.proj[br].hrt= from_dict(p1_d), from_dict(p2_d)
            elif proj_pair=='lh':
                env.proj[tl].hlb, env.proj[bl].hlt= from_dict(p1_d), from_dict(p2_d)
            elif proj_pair=='tv':
                env.proj[tl].thl, env.proj[tr].thr= from_dict(p1_d), from_dict(p2_d)
            elif proj_pair=='bv':
                env.proj[bl].bhl, env.proj[br].bhr= from_dict(p1_d), from_dict(p2_d)


        if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_pop()

        # _profile_transfer= os.getenv("YASTN_CTMRG_PROFILE_TRANSFER", "0")
        # if _profile_transfer in ["1",]:
        #     logger.info("YASTN_CTMRG_PROFILE_TRANSFER enabled")
        #     return

        # fill trivial projectors on edges (if any)
        trivial_projectors_(env, move, sites_proj)
        
        #
        # Update move
        env_tmp = EnvCTM(env.psi, init=None)  # empty environments
        if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_push(f"update_env_")
        for site in site_group:
            update_env_(env_tmp, site, env, move)
        if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_pop()
        
        update_storage_(env, env_tmp)


def update_projectors_T_(thread_pool_executor, env, site, move, opts_svd, **kwargs):
    r"""
    Calculate new projectors for CTM moves passing to specific method to create enlarged corners.
    """
    sites = [env.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    # tl, tr, bl, br = sites
    if None in sites:
        return
    method = kwargs.get('method', '2site')

    # if method == '2site':
    #     return update_2site_projectors_(proj, *sites, move, env, opts_svd, **kwargs)
    if method == '1site':
        raise NotImplementedError("1site method not implemented in distributed CTM yet.")
    elif method == '2site' and kwargs.get('devices', None):
        if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.mark(f"update_extended_2site_projectors_T_ {site} {move}")
        # return thread_pool_executor.submit(
        #     update_extended_2site_projectors_T_(thread_pool_executor, env, *sites, move, opts_svd, **kwargs)
        # )
        return update_extended_2site_projectors_MP_(thread_pool_executor, env, *sites, move, opts_svd, **kwargs)
        
    
def update_extended_2site_projectors_T_(thread_pool_executor,
        env_source, tl, tr, bl, br, move, opts_svd, 
        devices: list[str] | None = None, **kwargs):
    r"""
    If move is 'hv' use up to two devices to schedule the computations.
    NOTE: cusolver's svd is thread-blocking, it necessary to create multiple threads/processes 
    """
    if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_push(f"update_extended_2site_projectors_T_ {move}")
    logger.info(f"update_extended_2site_projectors_T_ {move} devices {devices} ")
    use_qr = kwargs.get("use_qr", True)
    kwargs["profiling_mode"]= env_source.profiling_mode
    psh = env_source.proj
    svd_predict_spec= lambda s0,p0,s1,p1,sign: opts_svd.get('D_block', float('inf')) \
        if psh is None or (getattr(psh[s0],p0) is None or getattr(psh[s1],p1) is None) else \
        env_source._partial_svd_predict_spec(getattr(psh[s0],p0).get_legs(-1), getattr(psh[s1],p1).get_legs(-1), sign)

    # This part is shared between between l and r projectors for h move 
    # and between t and b projectors for v move
    # 

    # variation 2
    def _transfer(ts,env,dest):
        data, meta= zip( *(split_data_and_meta(t.to_dict(level=0)) for t in ts) )
        stacks= np.cumsum([0,]+[len(d) for d in data])
        data_views= np.cumsum( [0,]+[d.view(-1).shape[0] for d in sum(data,())] )
        data_dest= env.config.backend.move_to( torch.cat( sum(data,())   ), device=dest, non_blocking=True )

        return tuple( from_dict( combine_data_and_meta( 
            tuple(data_dest[data_views[k]:data_views[k+1]] for k in range(stacks[i],stacks[i+1])),  m ) ) 
                     for i,m in enumerate(meta) )
        
    if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_push(f"contract {move}")
    
    env= env_source
    psi= env.psi
    ts= ( env_source[tl].l, env_source[tl].tl, env_source[tl].t, env_source.psi[tl],
            env_source[bl].b, env_source[bl].bl, env_source[bl].l, env_source.psi[bl],
            env_source[tr].t, env_source[tr].tr, env_source[tr].r, env_source.psi[tr],
            env_source[br].r, env_source[br].br, env_source[br].b, env_source.psi[br], )
    if devices and devices[0] != env_source.config.default_device:
        ts= _transfer(ts, env_source, devices[0])

    cor_tl = c2x2('tl',*ts[0:4])
    cor_bl = c2x2('bl',*ts[4:8])
    cor_tr = c2x2('tr',*ts[8:12])
    cor_br = c2x2('br',*ts[12:16])

    _profile_transfer= os.getenv("YASTN_CTMRG_PROFILE_TRANSFER", "0")
    if _profile_transfer in ["1",]:
        logger.info("YASTN_CTMRG_PROFILE_TRANSFER enabled")
        env_source.config.backend.cuda.nvtx.range_pop()
        return []

    if any(x in move for x in 'lrh'):
        cor_tt = cor_tl @ cor_tr  # b(left) b(right)
        cor_bb = cor_br @ cor_bl  # t(right) t(left)
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()

    def move_rh():
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_push(f"move_rh")
        sl = psi[tl].get_shape(axes=2)
        ltl = env.nn_site(tl, d='l')
        lbl = env.nn_site(bl, d='l')
        if sl == 1 and ltl and lbl:
            cor_ltl = env[ltl].l @ env[ltl].tl @ env[ltl].t
            cor_ltl = tensordot(cor_ltl, psi[ltl], axes=((2, 1), (0, 1)))
            cor_ltl = tensordot(cor_ltl, env[tl].t, axes=(1, 0))
            cor_ltl = tensordot(cor_ltl, psi[tl], axes=((3, 2), (0, 1)))
            cor_ltl = cor_ltl.fuse_legs(axes=((0, 1, 3), (2, 4)))

            cor_lbl = env[lbl].b @ env[lbl].bl @ env[lbl].l
            cor_lbl = tensordot(cor_lbl, psi[lbl], axes=((2, 1), (1, 2)))
            cor_lbl = env[bl].b @ cor_lbl
            cor_lbl = tensordot(cor_lbl, psi[bl], axes=((4, 1), (1, 2)))
            cor_lbl = cor_lbl.fuse_legs(axes=((0, 4), (1, 2, 3)))

            cor_ltt = cor_ltl @ cor_tr  # b(left) b(right)
            cor_lbb = cor_br @ cor_lbl  # t(right) t(left)
            _, r_t = qr(cor_ltt, axes=(0, 1)) if use_qr else (None, cor_ltt)
            _, r_b = qr(cor_lbb, axes=(1, 0)) if use_qr else (None, cor_lbb.T)
        else:
            _, r_t = qr(cor_tt, axes=(0, 1)) if use_qr else (None, cor_tt)
            _, r_b = qr(cor_bb, axes=(1, 0)) if use_qr else (None, cor_bb.T)
        
        opts_svd["D_block"]= svd_predict_spec(tr, "hrb", br, "hrt", r_t.s[1])
        hrb, hrt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)
        env_source.proj[tr].hrb, env_source.proj[br].hrt= hrb.to(device=env_source.config.default_device,non_blocking=True), \
            hrt.to(device=env_source.config.default_device,non_blocking=True)
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()

    def move_lh():
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_push(f"move_lh")
        cor_tt_t= cor_tt
        cor_bb_t= cor_bb
        if devices and len(devices) > 1:
            cor_tt_t = cor_tt.to(device=devices[1],non_blocking=True) if devices else cor_tt
            cor_bb_t = cor_bb.to(device=devices[1],non_blocking=True) if devices else cor_bb
        sr = psi[tr].get_shape(axes=2)
        rtr = env.nn_site(tr, d='r')
        rbr = env.nn_site(br, d='r')
        if sr == 1 and rtr and rbr:   
            env_t= env
            psi_t= env.psi 
            if devices and len(devices) > 1:
                env_t= env_source.to(device=devices[1],non_blocking=True) if devices else env_source
                psi_t = env_t.psi
            cor_rtr = env_t[rtr].t @ env_t[rtr].tr @ env_t[rtr].r
            cor_rtr = tensordot(cor_rtr, psi_t[rtr], axes=((1, 2), (0, 3)))
            cor_rtr = env_t[tr].t @ cor_rtr
            cor_rtr = tensordot(cor_rtr, psi_t[tr], axes=((1, 3), (0, 3)))
            cor_rtr = cor_rtr.fuse_legs(axes=((0, 3), (1, 2, 4)))

            cor_rbr = env_t[rbr].r @ env_t[rbr].br @ env_t[rbr].b
            cor_rbr = tensordot(cor_rbr, psi_t[rbr], axes=((2, 1), (2, 3)))
            cor_rbr = tensordot(cor_rbr, env_t[br].b, axes=(1, 0))
            cor_rbr = tensordot(cor_rbr, psi_t[br], axes=((3, 2), (2, 3)))
            cor_rbr = cor_rbr.fuse_legs(axes=((0, 1, 3), (2, 4)))

            cor_rtt = cor_tl @ cor_rtr  # b(left) b(right)
            cor_rbb = cor_rbr @ cor_bl  # t(right) t(left)
            _, r_t = qr(cor_rtt, axes=(1, 0)) if use_qr else (None, cor_rtt.T)
            _, r_b = qr(cor_rbb, axes=(0, 1)) if use_qr else (None, cor_rbb)
        else:
            _, r_t = qr(cor_tt_t, axes=(1, 0)) if use_qr else (None, cor_tt_t.T)
            _, r_b = qr(cor_bb_t, axes=(0, 1)) if use_qr else (None, cor_bb_t)

        opts_svd["D_block"]= svd_predict_spec(tl, "hlb", bl, "hlt", r_t.s[1])
        hlb, hlt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)
        env_source.proj[tl].hlb, env_source.proj[bl].hlt= hlb.to(device=env_source.config.default_device,non_blocking=True), \
            hlt.to(device=env_source.config.default_device,non_blocking=True)
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()

    res= []
    if any(x in move for x in 'lh'):
        if len(devices)>1 and devices[0]!=devices[1]:
            # res.append( ('lh', thread_pool_executor.submit(move_lh), time.perf_counter()) )
            res.append( ('lh', thread_pool_executor.submit(move_lh_mp, cor_tt, cor_bb, env_source, tl, tr, bl, br,
                    use_qr, svd_predict_spec,
                    opts_svd, devices, **kwargs), 
                         time.perf_counter()) )
            if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"thread(move_lh)")
            logger.info(f"update_extended_2site_projectors_T_ {move}")
        else:
            if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"move_lh")
            move_lh()
    if any(x in move for x in 'rh'):
        # res.append( ('rh', thread_pool_executor.submit(move_rh), time.perf_counter()) )
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"move_rh")
        move_rh()

    if any(x in move for x in 'tbv'):
        cor_ll = cor_bl @ cor_tl  # l(bottom) l(top)
        cor_rr = cor_tr @ cor_br  # r(top) r(bottom)
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()
    else:
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()
        return res

    def move_tv():
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_push(f"move_tv")
        sb = psi[bl].get_shape(axes=3)
        bbl = env.nn_site(bl, d='b')
        bbr = env.nn_site(br, d='b')
        if sb == 1 and bbl and bbr:
            cor_bbl = env[bbl].b @ env[bbl].bl @ env[bbl].l
            cor_bbl = tensordot(cor_bbl, psi[bbl], axes=((2, 1), (1, 2)))
            cor_bbl = tensordot(cor_bbl, env[bl].l, axes=(1, 0))
            cor_bbl = tensordot(cor_bbl, psi[bl], axes=((3, 1), (1, 2)))
            cor_bbl = cor_bbl.fuse_legs(axes=((0, 1, 4), (2, 3)))

            cor_bbr = env[bbr].r @ env[bbr].br @ env[bbr].b
            cor_bbr = tensordot(cor_bbr, psi[bbr], axes=((2, 1), (2, 3)))
            cor_bbr = env[br].r @ cor_bbr
            cor_bbr = tensordot(cor_bbr, psi[br], axes=((3, 1), (2, 3)))
            cor_bbr = cor_bbr.fuse_legs(axes=((0, 3), (1, 2, 4)))

            cor_bll = cor_bbl @ cor_tl  # l(bottom) l(top)
            cor_brr = cor_tr @ cor_bbr  # r(top) r(bottom)
            _, r_l = qr(cor_bll, axes=(0, 1)) if use_qr else (None, cor_bll)
            _, r_r = qr(cor_brr, axes=(1, 0)) if use_qr else (None, cor_brr.T)
        else:
            _, r_l = qr(cor_ll, axes=(0, 1)) if use_qr else (None, cor_ll)
            _, r_r = qr(cor_rr, axes=(1, 0)) if use_qr else (None, cor_rr.T)

        opts_svd["D_block"]= svd_predict_spec(tl, "vtr", tr, "vtl", r_l.s[1])
        vtr, vtl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)
        env_source.proj[tl].vtr, env_source.proj[tr].vtl= vtr.to(device=env_source.config.default_device,non_blocking=True), \
            vtl.to(device=env_source.config.default_device,non_blocking=True)
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()

    def move_bv():
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_push(f"move_bv")
        cor_ll_t= cor_ll
        cor_rr_t= cor_rr
        if devices and len(devices) > 1:
            cor_ll_t = cor_ll.to(device=devices[1],non_blocking=True) if devices else cor_ll
            cor_rr_t = cor_rr.to(device=devices[1],non_blocking=True) if devices else cor_rr
        st = psi[tl].get_shape(axes=3)
        ttl = env.nn_site(tl, d='t')
        ttr = env.nn_site(tr, d='t')
        if st == 1 and ttl and ttr:
            env_t= env
            psi_t = env.psi
            if devices and len(devices) > 1:
                env_t= env_source.to(device=devices[1],non_blocking=True) if devices else env_source
                psi_t = env_t.psi
            cor_ttl = env_t[ttl].l @ env_t[ttl].tl @ env_t[ttl].t
            cor_ttl = tensordot(cor_ttl, psi_t[ttl], axes=((2, 1), (0, 1)))
            cor_ttl = env_t[tl].l @ cor_ttl
            cor_ttl = tensordot(cor_ttl, psi_t[tl], axes=((3, 1), (0, 1)))
            cor_ttl = cor_ttl.fuse_legs(axes=((0, 3), (1, 2, 4)))

            cor_ttr = env_t[ttr].t @ env_t[ttr].tr @ env_t[ttr].r
            cor_ttr = tensordot(cor_ttr, psi_t[ttr], axes=((1, 2), (0, 3)))
            cor_ttr = tensordot(cor_ttr, env_t[tr].r, axes=(1, 0))
            cor_ttr = tensordot(cor_ttr, psi_t[tr], axes=((2, 3), (0, 3)))
            cor_ttr = cor_ttr.fuse_legs(axes=((0, 1, 3), (2, 4)))

            cor_tll = cor_bl @ cor_ttl  # l(bottom) l(top)
            cor_trr = cor_ttr @ cor_br  # r(top) r(bottom)
            _, r_l = qr(cor_tll, axes=(1, 0)) if use_qr else (None, cor_tll.T)
            _, r_r = qr(cor_trr, axes=(0, 1)) if use_qr else (None, cor_trr)
        else:
            _, r_l = qr(cor_ll_t, axes=(1, 0)) if use_qr else (None, cor_ll_t.T)
            _, r_r = qr(cor_rr_t, axes=(0, 1)) if use_qr else (None, cor_rr_t)

        opts_svd["D_block"]= svd_predict_spec(bl, "vbr", br, "vbl", r_l.s[1])
        vbr, vbl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)
        env_source.proj[bl].vbr, env_source.proj[br].vbl= vbr.to(device=env_source.config.default_device,non_blocking=True), \
            vbl.to(device=env_source.config.default_device,non_blocking=True)
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()

    if any(x in move for x in 'bv'):
        if len(devices)>1 and devices[0]!=devices[1]:
            # res.append( ('bv', thread_pool_executor.submit(move_bv), time.perf_counter()) )
            res.append( ('bv', thread_pool_executor.submit(move_bv_mp, cor_ll, cor_rr, env_source, tl, tr, bl, br, 
               use_qr, svd_predict_spec,
               opts_svd, devices, **kwargs                         
                                                           ), time.perf_counter()) )
            if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"thread(move_bv)")
            logger.info(f"update_extended_2site_projectors_T_ {move}")
        else:
            if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"move_bv")
            move_bv()
    if any(x in move for x in 'tv'):
        # res.append( ('tv', thread_pool_executor.submit(move_tv), time.perf_counter()) )
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"move_tv")
        move_tv()
    if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()
    return res


def update_extended_2site_projectors_MP_(
        env_source, tl, tr, bl, br, move, opts_svd, 
        devices: list[str] | None = None, **kwargs):
    r"""
    If move is 'hv' use up to two devices to schedule the computations.
    NOTE: cusolver's svd is thread-blocking, it necessary to create multiple threads/processes 
    """
    if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_push(f"update_extended_2site_projectors_T_ {move}")
    logger.info(f"update_extended_2site_projectors_T_ {move} devices {devices} ")
    use_qr = kwargs.get("use_qr", True)
    kwargs["profiling_mode"]= env_source.profiling_mode
    psh = env_source.proj
    svd_predict_spec= lambda s0,p0,s1,p1,sign: opts_svd.get('D_block', float('inf')) \
        if psh is None or (getattr(psh[s0],p0) is None or getattr(psh[s1],p1) is None) else \
        env_source._partial_svd_predict_spec(getattr(psh[s0],p0).get_legs(-1), getattr(psh[s1],p1).get_legs(-1), sign)
        
    env= env_source.to(device=devices[0],non_blocking=True) if devices and devices[0] != env_source.config.default_device else env_source
    psi= env.psi


    def move_rh():
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_push(f"move_rh")
        sl = psi[tl].get_shape(axes=2)
        ltl = env.nn_site(tl, d='l')
        lbl = env.nn_site(bl, d='l')
        if sl == 1 and ltl and lbl:
            cor_ltl = env[ltl].l @ env[ltl].tl @ env[ltl].t
            cor_ltl = tensordot(cor_ltl, psi[ltl], axes=((2, 1), (0, 1)))
            cor_ltl = tensordot(cor_ltl, env[tl].t, axes=(1, 0))
            cor_ltl = tensordot(cor_ltl, psi[tl], axes=((3, 2), (0, 1)))
            cor_ltl = cor_ltl.fuse_legs(axes=((0, 1, 3), (2, 4)))

            cor_lbl = env[lbl].b @ env[lbl].bl @ env[lbl].l
            cor_lbl = tensordot(cor_lbl, psi[lbl], axes=((2, 1), (1, 2)))
            cor_lbl = env[bl].b @ cor_lbl
            cor_lbl = tensordot(cor_lbl, psi[bl], axes=((4, 1), (1, 2)))
            cor_lbl = cor_lbl.fuse_legs(axes=((0, 4), (1, 2, 3)))

            cor_ltt = cor_ltl @ cor_tr  # b(left) b(right)
            cor_lbb = cor_br @ cor_lbl  # t(right) t(left)
            _, r_t = qr(cor_ltt, axes=(0, 1)) if use_qr else (None, cor_ltt)
            _, r_b = qr(cor_lbb, axes=(1, 0)) if use_qr else (None, cor_lbb.T)
        else:
            _, r_t = qr(cor_tt, axes=(0, 1)) if use_qr else (None, cor_tt)
            _, r_b = qr(cor_bb, axes=(1, 0)) if use_qr else (None, cor_bb.T)
        
        opts_svd["D_block"]= svd_predict_spec(tr, "hrb", br, "hrt", r_t.s[1])
        hrb, hrt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)
        env_source.proj[tr].hrb, env_source.proj[br].hrt= hrb.to(device=env_source.config.default_device,non_blocking=True), \
            hrt.to(device=env_source.config.default_device,non_blocking=True)
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()

    res= []
    if any(x in move for x in 'lh'):
        if len(devices)>1 and devices[0]!=devices[1]:
            # res.append( ('lh', thread_pool_executor.submit(move_lh), time.perf_counter()) )
            res.append( ('lh', thread_pool_executor.submit(move_lh_mp, cor_tt, cor_bb, env_source, tl, tr, bl, br,
                    use_qr, svd_predict_spec,
                    opts_svd, devices, **kwargs), 
                         time.perf_counter()) )
            if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"thread(move_lh)")
            logger.info(f"update_extended_2site_projectors_T_ {move}")
        else:
            if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"move_lh")
            move_lh()
    if any(x in move for x in 'rh'):
        # res.append( ('rh', thread_pool_executor.submit(move_rh), time.perf_counter()) )
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"move_rh")
        move_rh()

    if any(x in move for x in 'tbv'):
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()
    else:
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()
        return res

    


    if any(x in move for x in 'bv'):
        if len(devices)>1 and devices[0]!=devices[1]:
            # res.append( ('bv', thread_pool_executor.submit(move_bv), time.perf_counter()) )
            res.append( ('bv', thread_pool_executor.submit(move_bv_mp, cor_ll, cor_rr, env_source, tl, tr, bl, br, 
               use_qr, svd_predict_spec,
               opts_svd, devices, **kwargs                         
                                                           ), time.perf_counter()) )
            if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"thread(move_bv)")
            logger.info(f"update_extended_2site_projectors_T_ {move}")
        else:
            if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"move_bv")
            move_bv()
    if any(x in move for x in 'tv'):
        # res.append( ('tv', thread_pool_executor.submit(move_tv), time.perf_counter()) )
        if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.mark(f"move_tv")
        move_tv()
    if env_source.profiling_mode in ["NVTX",]: env_source.config.backend.cuda.nvtx.range_pop()
    return res


def projectors_move_MP_(proj_pair, h1_d, h2_d, device, ret_device, opt_svd, **kwargs):
    profiling_mode= kwargs.get("profiling_mode", None)
    h1= from_dict(h1_d).to(device=device,non_blocking=True), 
    h2= from_dict(h2_d).to(device=device,non_blocking=True)
    if profiling_mode in ["NVTX",]: h1.config.backend.cuda.nvtx.range_push(f"{proj_pair}")
    if proj_pair in ['rh']:
        res= projectors_move_rh(h1, h2, opt_svd, **kwargs)
    elif proj_pair in ['lh']:
        res= projectors_move_lh_mp(h1, h2, opt_svd, **kwargs)
    elif proj_pair in ['tv']:          
        res= projectors_move_tv(h1, h2, opt_svd, **kwargs)
    elif proj_pair in ['bv']:
        res= projectors_move_bv_mp(h1, h2, opt_svd, **kwargs)
    else:
        raise ValueError(f"projectors_move_MP_ invalid proj_pair {proj_pair}")
    res= tuple(r.to_dict(level=1).to(device=ret_device) for r in res)
    if profiling_mode in ["NVTX",]: h1.config.backend.cuda.nvtx.range_pop()
    return res

def projectors_move_rh(cor_tt, cor_bb, opts_svd, **kwargs):
    use_qr = kwargs.get("use_qr", True)

    _, r_t = qr(cor_tt, axes=(0, 1)) if use_qr else (None, cor_tt)
    _, r_b = qr(cor_bb, axes=(1, 0)) if use_qr else (None, cor_bb.T)
    
    hrb, hrt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)
    return hrb, hrt
    
def projectors_move_lh_mp(cor_tt, cor_bb, opts_svd, **kwargs):
    use_qr = kwargs.get("use_qr", True)
    
    _, r_t = qr(cor_tt, axes=(1, 0)) if use_qr else (None, cor_tt.T)
    _, r_b = qr(cor_bb, axes=(0, 1)) if use_qr else (None, cor_bb)

    hlb, hlt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)
    return hlb, hlt

def projectors_move_tv(cor_ll, cor_rr, opts_svd, **kwargs):
    use_qr = kwargs.get("use_qr", True)

    _, r_l = qr(cor_ll, axes=(0, 1)) if use_qr else (None, cor_ll)
    _, r_r = qr(cor_rr, axes=(1, 0)) if use_qr else (None, cor_rr.T)

    vtr, vtl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)
    return vtr, vtl

def projectors_move_bv_mp(cor_ll, cor_rr, opts_svd, **kwargs):
    use_qr = kwargs.get("use_qr", True)
    
    _, r_l = qr(cor_ll, axes=(1, 0)) if use_qr else (None, cor_ll.T)
    _, r_r = qr(cor_rr, axes=(0, 1)) if use_qr else (None, cor_rr)
    
    vbr, vbl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)
    return vbr, vbl


def projectors_stage1(env_source_dict, tl, tr, bl, br, move, device, **kwargs):
    print(f"projectors_stage1 move {move} device {device} {tl},{tr},{bl},{br}")
    env= from_dict(env_source_dict)
    env= env.to(device=device,non_blocking=True) if device != env.config.default_device else env
    ts= ( env[tl].l, env[tl].tl, env[tl].t, env.psi[tl],
            env[bl].b, env[bl].bl, env[bl].l, env.psi[bl],
            env[tr].t, env[tr].tr, env[tr].r, env.psi[tr],
            env[br].r, env[br].br, env[br].b, env.psi[br], )
    
    res= ()
    if any(x in move for x in 'lrh'):
        cor_tt, cor_bb= contractions_2x2_lhr(ts) 
        res+= (cor_tt, cor_bb)
    if any(x in move for x in 'tvb'):
        cor_ll, cor_rr= contractions_2x2_tvb(ts) 
        res+= (cor_ll, cor_rr)
    res= tuple(r.to_dict(level=1) for r in res)
    return res

def contractions_2x2_lhr(ts):
    cor_tl = c2x2('tl',*ts[0:4])
    cor_bl = c2x2('bl',*ts[4:8])
    cor_tr = c2x2('tr',*ts[8:12])
    cor_br = c2x2('br',*ts[12:16])

    cor_tt = cor_tl @ cor_tr  # b(left) b(right)
    cor_bb = cor_br @ cor_bl  # t(right) t(left)
    return cor_tt, cor_bb

def contractions_2x2_tvb(ts):
    cor_tl = c2x2('tl',*ts[0:4])
    cor_bl = c2x2('bl',*ts[4:8])
    cor_tr = c2x2('tr',*ts[8:12])
    cor_br = c2x2('br',*ts[12:16])

    cor_ll = cor_bl @ cor_tl  # l(bottom) l(top)
    cor_rr = cor_tr @ cor_br  # r(top) r(bottom)
    return cor_ll, cor_rr

def c2x2(id_c2x2, t1, c, t2, onsite_t, mode='fuse'):
    if id_c2x2 == 'tl':
        return c2x2_tl(t1, c, t2, onsite_t, mode=mode)
    elif id_c2x2 == 'bl': 
        return c2x2_bl(t1, c, t2, onsite_t, mode=mode)
    elif id_c2x2 == 'tr':
        return c2x2_tr(t1, c, t2, onsite_t, mode=mode)
    elif id_c2x2 == 'br':
        return c2x2_br(t1, c, t2, onsite_t, mode=mode)

def c2x2_tl(t_left, c_topleft, t_top, onsite_t, mode='fuse'):
    cor_tl = t_left @ c_topleft @ t_top
    cor_tl = tensordot(cor_tl, onsite_t, axes=((2, 1), (0, 1)))
    if mode == 'fuse':
        cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))
    return cor_tl

def c2x2_bl(t_bottom, c_bottomleft, t_left, onsite_t, mode='fuse'):
    cor_bl = t_bottom @ c_bottomleft @ t_left
    cor_bl = tensordot(cor_bl, onsite_t, axes=((2, 1), (1, 2)))
    if mode == 'fuse': 
        cor_bl = cor_bl.fuse_legs(axes=((0, 3), (1, 2)))
    return cor_bl

def c2x2_tr(t_top, c_topright, t_right, onsite_t, mode='fuse'):
    cor_tr = t_top @ c_topright @ t_right
    cor_tr = tensordot(cor_tr, onsite_t, axes=((1, 2), (0, 3)))
    if mode == 'fuse':
        cor_tr = cor_tr.fuse_legs(axes=((0, 2), (1, 3)))
    return cor_tr

def c2x2_br(t_right, c_bottomright, t_bottom, onsite_t, mode='fuse'):
    cor_br = t_right @ c_bottomright @ t_bottom
    cor_br = tensordot(cor_br, onsite_t, axes=((2, 1), (2, 3)))
    if mode == 'fuse':
        cor_br = cor_br.fuse_legs(axes=((0, 2), (1, 3)))
    return cor_br