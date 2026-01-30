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
import logging
from typing import Callable, Sequence

from ._env_contractions import halves_4x4_lhr, halves_4x4_tvb, update_env_fetch_args, update_env_dir
from ._env_ctm import CTMRG_out, EnvCTM, proj_corners, update_storage_
from .. import Site, DoublePepsTensor

from ...._from_dict import from_dict
from ...._split_combine_dict import split_data_and_meta, combine_data_and_meta
from ....tensor import Tensor, YastnError, qr


logger = logging.getLogger(__name__)


def _validate_devices_list(devices: list[str] | None) -> None:
    if devices is None:
        raise YastnError("Devices list must be provided for distributed CTM.")
    if devices:
        assert len(devices) > 0, "At least one device must be provided."
    if len(devices) < 2:
        raise YastnError("At least two devices must be provided for distributed CTM.")


def iterate_D_(env, opts_svd=None, moves='hv', method='2x2', max_sweeps=1, iterator=False, corner_tol=None, truncation_f: Callable = None, **kwargs):
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
            '2x2', '1x2' in mathod. The default is '2x2'.

                * '2x2' uses the standard 2x2 enlarged corners forming 4x4 patch, enabling enlargement of EnvCTM bond dimensions. When some PEPS bonds are rank-1, it recognizes it to use 5x4 corners to prevent artificial collapse of EnvCTM bond dimensions to 1, which is important for hexagonal lattice.
                * '1x2' uses smaller 1x2 corners forming 2x4 patch. It is significantly faster, but is less stable and  does not allow for EnvCTM bond dimension growth.

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
        tmp = _ctmrg_iterator_D_(env, opts_svd, moves, method, max_sweeps, corner_tol, **kwargs)
        return tmp if kwargs["iterator_step"] else next(tmp)


def _ctmrg_iterator_D_(env, opts_svd, moves, method, max_sweeps, corner_tol, **kwargs):
    """ Generator for iterate_ (or its alias ctmrg_). """
    iterator_step = kwargs.get("iterator_step", 0)
    max_dsv, converged, history = None, False, []

    devices= kwargs.get('devices', None)
    _validate_devices_list(devices)

    mp= env.config.backend.mp
    max_workers= len(devices) # one is the main thread
    task_queue = mp.Queue()
    stage1_queue = mp.Queue()
    stage2_queue = mp.Queue()
    ctmrg_mp_context= (task_queue, stage1_queue, stage2_queue)
    procs=[]
    for i in range(max_workers):
        p = mp.Process(target=_ctmrg_worker_mp, args=(i, devices, *ctmrg_mp_context))
        p.start()
        procs.append(p)

    try:
        logger.info(f"ctmrg_T main loop max_workers={max_workers} on devices={devices}")
        for sweep in range(1, max_sweeps + 1):
            if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_push(f"update_")
            update_D_(ctmrg_mp_context, env, opts_svd=opts_svd, moves=moves, method=method, **kwargs)
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
    except Exception as e:
        logger.error(f"ctmrg_T main loop exception: {e}")
        raise e
    finally:
        for _ in range(len(procs)):
            task_queue.put(None)
        for p in procs:
            p.join()

    yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)


def update_D_(ctmrg_mp_context, env, opts_svd, moves='hv', method='2x2', **kwargs):
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
        '2x2' or '1x2' in method. The default is '2x2'.
        '2x2' uses the standard 2x2 enlarged corners, allowing to enlarge EnvCTM bond dimension.
        '1x2' uses smaller 1x2 corners. It is significantly faster, but is less stable and
        does not allow to grow EnvCTM bond dimension.

    checkpoint_move: bool
        Whether to use (reentrant) checkpointing for the move. The default is ``False``

    Returns
    -------
    proj: Peps structure loaded with CTM projectors related to all lattice site.
    """
    if 'tol' not in opts_svd and 'tol_block' not in opts_svd:
        opts_svd['tol'] = 1e-14

    checkpoint_move = kwargs.get('checkpoint_move', False)
    for d in moves:
        if checkpoint_move:
            def f_update_core_(move_d, loc_im, *inputs_t):
                loc_env = EnvCTM.from_dict(combine_data_and_meta(inputs_t, loc_im))
                _update_core_D_(ctmrg_mp_context, loc_env, move_d, opts_svd, method=method, **kwargs)
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
            _update_core_D_(ctmrg_mp_context, env, d, opts_svd, method=method, **kwargs)
    return env


def _update_core_D_(ctmrg_mp_context, env, move: str, opts_svd: dict, **kwargs):
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

        # TODO load-balancing
        if len(devices) <= num_jobs:
            reps = (num_jobs + len(devices) - 1) // len(devices)
            return [ [d]*2 for d in (devices * reps)[:num_jobs] ]
        elif len(devices) >= 2 * num_jobs:
            return [ devices[2*i:2*i+2] for i in range(num_jobs) ]
        else:
            return [ devices[i:i+1]+devices[i+num_jobs:i+1+num_jobs] for i in range(num_jobs) ]

    corner_sites= lambda site: tuple(env.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1)))

    svd_predict_spec= lambda s0,p0,s1,p1,sign: opts_svd.get('D_block', float('inf')) \
        if env.proj is None or (getattr(env.proj[s0],p0) is None or getattr(env.proj[s1],p1) is None) else \
        env._partial_svd_predict_spec(getattr(env.proj[s0],p0).get_legs(-1), getattr(env.proj[s1],p1).get_legs(-1), sign)

    task_queue, stage1_queue, stage2_queue= ctmrg_mp_context
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
        for i,site in enumerate(sites_proj):
            task_queue.put( ("projectors_stage1",
                             (i, site, env_d, move), kwargs) )

        # blocking wait for all stage-1 to complete
        for _ in range(len(sites_proj)):
            i,site, (half1_d, half2_d) = stage1_queue.get()
            h1,h2= from_dict(half1_d).clone(), from_dict(half2_d).clone()
            del half1_d, half2_d
            tl,tr,bl,br= corner_sites(site)

            if move in 'h':
                opts_svd["D_blocks"]= svd_predict_spec(tr, "hrb", br, "hrt", h1.s[1])
                task_queue.put( ("projectors_move_MP_",
                                 ( i, site, 'rh', h1.to_dict(level=1), h2.to_dict(level=1),
                                   env.config.default_device, opts_svd), kwargs) )
                opts_svd["D_blocks"]= svd_predict_spec(tl, "hlb", bl, "hlt", h1.s[0])
                task_queue.put( ("projectors_move_MP_",
                                 ( i, site, 'lh', h1.to_dict(level=1), h2.to_dict(level=1),
                                   env.config.default_device, opts_svd), kwargs) )
            elif move in 'v':
                opts_svd["D_block"]= svd_predict_spec(tl, "vtr", tr, "vtl", h1.s[1])
                task_queue.put( ("projectors_move_MP_",
                                 ( i, site, 'tv', h1.to_dict(level=1), h2.to_dict(level=1),
                                   env.config.default_device, opts_svd), kwargs) )
                opts_svd["D_block"]= svd_predict_spec(bl, "vbr", br, "vbl", h1.s[0])
                task_queue.put( ("projectors_move_MP_",
                                 ( i, site, 'bv', h1.to_dict(level=1), h2.to_dict(level=1),
                                   env.config.default_device, opts_svd), kwargs) )

        for _ in range(len(sites_proj)*2):
            i,site,proj_pair,(p1_d,p2_d)= stage2_queue.get()
            p1,p2= from_dict(p1_d).clone(), from_dict(p2_d).clone()
            del p1_d, p2_d
            tl,tr,bl,br= corner_sites(site)
            if proj_pair=='rh':
                env.proj[tr].hrb, env.proj[br].hrt= p1,p2
            elif proj_pair=='lh':
                env.proj[tl].hlb, env.proj[bl].hlt= p1,p2
            elif proj_pair=='tv':
                env.proj[tl].vtr, env.proj[tr].vtl= p1,p2
            elif proj_pair=='bv':
                env.proj[bl].vbr, env.proj[br].vbl= p1,p2


        if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_pop()

        # fill trivial projectors on edges (if any)
        env._trivial_projectors_(move, sites_proj)

        #
        # Update move
        if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_push(f"update_env_D_")
        
        # NOTE Here, we assumme that master process has up-to-date view of all on-site tensors, projectors, and env tensors
        #
        # Compute updated tensors in parallel and send back to default device
        moves= 'lr'*('h'==move) + 'tb'*('v'==move) + move*(move in 'lrtb')
        for i,site in enumerate(site_group):
            for mv in moves:
                job_ts= update_env_fetch_args(site, env, mv)
                job_ts_d= tuple(t.to_dict(level=1) if isinstance(t,(Tensor,DoublePepsTensor)) else t for t in job_ts)
                task_queue.put( ("update_env_move_MP_",
                    (i, site, mv, env.config.default_device, job_ts_d), \
                        {'profiling_mode': kwargs.get('profiling_mode', None)}) )
        
        # blocking wait for all updates to complete and assignment to env_tmp
        env_tmp = EnvCTM(env.psi, init=None)  # empty environments
        for i,_s in enumerate(site_group):
            for _mv in moves:
                _i, site, mv, tmp_env_ts_d = stage1_queue.get()
                tmp_env_ts= tuple(from_dict(t_d).clone() for t_d in tmp_env_ts_d)
                del tmp_env_ts_d
                if mv=='l':
                    env_tmp[site].l, env_tmp[site].tl, env_tmp[site].bl= tmp_env_ts
                elif mv=='r':
                    env_tmp[site].r, env_tmp[site].tr, env_tmp[site].br= tmp_env_ts
                elif mv=='t':
                    env_tmp[site].t, env_tmp[site].tl, env_tmp[site].tr= tmp_env_ts
                elif mv=='b':
                    env_tmp[site].b, env_tmp[site].bl, env_tmp[site].br= tmp_env_ts
                
        if env.profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_pop()

        update_storage_(env, env_tmp)


def _ctmrg_worker_mp(i:int, devices:Sequence[str],
                     task_queue: mp.Queue, stage1_queue: mp.Queue, stage2_queue: mp.Queue, **kwargs):
    r"""
    Worker process for distributed CTMRG.

    Listens on task_queue for tasks of the form:

        (function_name: str, args: tuple, kwargs: dict)

    Executes the function with given arguments and puts the result on done_queue.
    """
    device= devices[i % len(devices)]
    while True:
        task = task_queue.get()
        if task is None:
            break  # Exit signal
        function_name, args, func_kwargs = task
        if function_name == "projectors_stage1":
            projectors_stage1(stage1_queue, device, *args, **func_kwargs)
        elif function_name == "projectors_move_MP_":
            projectors_move_MP_(stage2_queue, device, *args, **func_kwargs)
        elif function_name == "update_env_move_MP_":
            update_env_move_MP_(stage1_queue, device, *args, **func_kwargs)
        else:
            raise ValueError(f"Unknown function name: {function_name}")
        del args, func_kwargs, task


def projectors_move_MP_(out_queue, device, i, site, proj_pair, h1_d, h2_d,
                        ret_device, opt_svd, **kwargs):
    r"""
    Stage 1 of CTM projector calculation: compute enlarged corners and halfs.

    Parameters
    ----------
    out_queue: mp.Queue
        multiprocessing Queue to put the result into.
    device: str
        Device to perform calculations on.
    i: int
        Index of the job.
    site: Site
        Reference lattice site for which the projectors are calculated.
    proj_pair: str
        Projector pair to calculate: 'rh', 'lh', 'tv', or 'bv
    h1_d, h2_d: dict
        pair of rank-2 tensors from which to construct the projectors serialized to dictionaries.
    ret_device: str
        Device to put the result onto.
    opt_svd: dict
        Options for SVD truncation.
    """
    profiling_mode= kwargs.get("profiling_mode", None)
    h1= from_dict(h1_d).clone().to(device=device,non_blocking=True)
    h2= from_dict(h2_d).clone().to(device=device,non_blocking=True)
    del h1_d, h2_d

    if profiling_mode in ["NVTX",]: h1.config.backend.cuda.nvtx.range_push(f"{proj_pair}")
    if proj_pair in ['rh']:
        res= projectors_move_rh(h1, h2, opt_svd, **kwargs)
    elif proj_pair in ['lh']:
        res= projectors_move_lh(h1, h2, opt_svd, **kwargs)
    elif proj_pair in ['tv']:
        res= projectors_move_tv(h1, h2, opt_svd, **kwargs)
    elif proj_pair in ['bv']:
        res= projectors_move_bv(h1, h2, opt_svd, **kwargs)
    else:
        raise ValueError(f"projectors_move_MP_ invalid proj_pair {proj_pair}")
    res= tuple(r.to(device=ret_device).to_dict(level=1) for r in res)
    out_queue.put( (i, site, proj_pair, res) )
    if profiling_mode in ["NVTX",]: h1.config.backend.cuda.nvtx.range_pop()
    del h1, h2, res

def projectors_move_rh(cor_tt, cor_bb, opts_svd, **kwargs):
    use_qr = kwargs.get("use_qr", True)

    _, r_t = qr(cor_tt, axes=(0, 1)) if use_qr else (None, cor_tt)
    _, r_b = qr(cor_bb, axes=(1, 0)) if use_qr else (None, cor_bb.T)

    hrb, hrt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)
    return hrb, hrt

def projectors_move_lh(cor_tt, cor_bb, opts_svd, **kwargs):
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

def projectors_move_bv(cor_ll, cor_rr, opts_svd, **kwargs):
    use_qr = kwargs.get("use_qr", True)

    _, r_l = qr(cor_ll, axes=(1, 0)) if use_qr else (None, cor_ll.T)
    _, r_r = qr(cor_rr, axes=(0, 1)) if use_qr else (None, cor_rr)

    vbr, vbl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)
    return vbr, vbl


def projectors_stage1(out_queue,device,
                      i,site,env_d, move, **kwargs):
    r"""
    Stage 1 of CTM projector calculation: compute enlarged corners and halfs.

    Parameters
    ----------
    out_queue: mp.Queue
        multiprocessing Queue to put the result into.
    device: str
        Device to perform calculations on.
    i: int
        Index of the job.
    site: Site
        Reference lattice site for which the projectors are calculated.
    env_d: dict
        EnvCTM serialized to dictionary.
    tl, tr, bl, br: Site
        Lattice sites corresponding to corners.
    move: str
        CTM move direction: 'h', 'v', 'l', 'r', 't', or 'b'.
    """
    profiling_mode= kwargs.get("profiling_mode", None)
    env= from_dict(env_d).clone().to(device=device,non_blocking=True)
    env = env.detach()
    env.psi = env.psi.detach()

    if profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_push(f"projectors_stage1")
    tl, tr, bl, br= tuple(env.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1)))
    ts= ( env[tl].l, env[tl].tl, env[tl].t, env.psi[tl],
            env[bl].b, env[bl].bl, env[bl].l, env.psi[bl],
            env[tr].t, env[tr].tr, env[tr].r, env.psi[tr],
            env[br].r, env[br].br, env[br].b, env.psi[br], )

    cor_tt, cor_bb, cor_ll, cor_rr= None, None, None, None
    res= ()
    if any(x in move for x in 'lrh'):
        cor_tt, cor_bb= halves_4x4_lhr(ts)
        res+= (cor_tt, cor_bb)
    if any(x in move for x in 'tvb'):
        cor_ll, cor_rr= halves_4x4_tvb(ts)
        res+= (cor_ll, cor_rr)
    res= tuple(r.to_dict(level=1) for r in res)
    out_queue.put( (i,site, res) )
    if profiling_mode in ["NVTX",]: env.config.backend.cuda.nvtx.range_pop()
    del env, ts
    del cor_tt, cor_bb, cor_ll, cor_rr, res


def update_env_move_MP_(out_queue, device, i, site, move, ret_device, args_d, **kwargs):
    r"""
    Update CTM environment tensors performing specified move.
    Parameters
    ----------
    out_queue: mp.Queue
        multiprocessing Queue to put the result into.
    device: str
        Device to perform calculations on.
    i: int
        Index of the job.
    site: Site
        Reference lattice site for which the environment tensors are updated.
    move: str
        CTM move direction: 'l', 'r', 't', or 'b'.
    ret_device: str
        Device to put the result onto.
    args_d: tuple(dict|Site)
        Tensors required for the update serialized to dictionaries. 
    """
    profiling_mode= kwargs.get("profiling_mode", None)
    args= tuple(from_dict(t_d).clone().to(device=device,non_blocking=True) if isinstance(t_d, dict) else t_d for t_d in args_d)
    del args_d

    if profiling_mode in ["NVTX",]: args[0].config.backend.cuda.nvtx.range_push(f"{site} {move}")
    res= update_env_dir(move, *args)
    res= tuple(r.to(device=ret_device).to_dict(level=1) for r in res)

    out_queue.put( (i, site, move, res) )
    if profiling_mode in ["NVTX",]: args[0].config.backend.cuda.nvtx.range_pop()
    del args, res

