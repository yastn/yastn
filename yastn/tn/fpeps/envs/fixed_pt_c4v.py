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
import copy
import logging
import time

import numpy as np
from scipy.optimize import minimize
import torch
from dataclasses import replace

from ._ctm_opts import CTMOpts, FixedPointOpts
from ._env_ctm_c4v import EnvCTM_c4v
from .fixed_pt import fast_env_T_gauge_multi_sites, NoFixedPointError, _concat_data, _assemble_dict_from_1d, extract_phase
from .rdm import *
from .._geometry import Lattice
from ....initialize import zeros, eye
from ....tensor import tensordot, diag
from ...._split_combine_dict import split_data_and_meta, combine_data_and_meta

log = logging.getLogger("FixedPoint_c4v")


def env_raw_data_c4v(env):
    '''
    Combine all env_c4v raw tensors into a 1d tensor.
    '''
    data_list = []
    slice_list = []
    numel = 0
    for site in env.sites():
        for dirn in ["tl", "t"]:
            data_list.append(getattr(env[site], dirn)._data)
            slice_list.append((numel, len(data_list[-1])))
            numel += len(data_list[-1])

    return torch.cat(data_list), slice_list

def find_gauge_c4v(env_old, env, verbose=False):
    r"""
    Find the gauge transformation matrix sigma that connects env and env_old.
    T: environment tensor for A; T': environment tensor for B
    #   --[leg1]--T_new-----T_new^'--[leg2]--sigma--- == ---sigma--[leg3]---T_old---T_old^' ---[leg4]---
    #               |          |                                              |         |
    #               A          B                                              A         B
    Args:
        env_old (EnvCTM_c4v): CTM_c4v environment
        env (EnvCTM_c4v): CTM_c4v environment after a single CTMRG step

    Returns:
        Tensor: gauge transformation matrix sigma.
    """
    site = env.sites()[0]
    T_olds, T_news = [env_old[site].t, env_old[site].t.flip_signature()], [env[site].t, env[site].t.flip_signature()]
    zero_modes = fast_env_T_gauge_multi_sites(env.psi.config, T_olds, T_news)
    if len(zero_modes) == 0:
        return None

    site = env.sites()[0]
    sigma = zero_modes[0]
    sigma._data.detach_()

    # Note: The sigma matrix for T_new^' can be obtained by flipping signatures of sigma matrix
    site = env.sites()[0]
    sigma_p = sigma.flip_signature() # flip signatures but keep the same charge sectors
    fixed_t = tensordot(
        tensordot(sigma, env[site].t, axes=(0, 0), conj=(1, 0)), sigma_p, axes=(2, 0),
    )
    T_old = env_old[site].t

    v1, v2 = T_old._data, fixed_t._data
    T_phase = extract_phase(v1, v2)
    if verbose:
        fixed_t._data = fixed_t._data * torch.exp(1j*T_phase).to(fixed_t._data.dtype)
        print("T diff:", (fixed_t - T_old).norm() / T_old.norm())

    fixed_C = tensordot(
        tensordot(sigma_p, env[site].tl, axes=(0, 0), conj=(1, 0)),
        sigma,
        axes=(1, 0),
    )
    C_old = env_old[site].tl
    v1, v2 = C_old._data, fixed_C._data
    C_phase = extract_phase(v1, v2)
    if verbose:
        fixed_C._data = fixed_C._data * torch.exp(1j*C_phase).to(fixed_C._data.dtype)
        print("C diff:", (fixed_C - C_old).norm() / C_old.norm())

    return sigma, T_phase, C_phase

def fp_ctmrg_c4v(env: EnvCTM_c4v,
            ctm_opts_fwd: dict | None = None,
            ctm_opts_fp: dict | None = None, *,
            opts: FixedPointOpts | None = None):
    r"""
    Compute the fixed-point environment for the given state using CTMRG.
    First, run CTMRG until convergence then find the gauge transformation guaranteeing element-wise
    convergence of the environment tensors.
    Enables backward differentiation through the fixed-point iteration to compute the gradients of the environment tensors 
    with respect to the state parameters, via `Neumann series expansion
    <https://en.wikipedia.org/wiki/Neumann_series>`_ of the fixed-point iteration.

    Parameters
    ----------
    env: EnvCTM_c4v
        C4v-symmetric CTM environment.

    ctm_opts_fwd: dict | None
        Options for the forward CTMRG convergence.
        See :class:`yastn.tn.fpeps.envs.CTMOpts` for the accepted keys.

    ctm_opts_fp: dict | None
        Overrides for the gauge-fixing CTM step, applied on top of
        ``ctm_opts_fwd`` which it otherwise inherits -- including ``max_sweeps``
        and ``corner_tol``, which the Neumann backward loop then uses as its
        budget and tolerance.

    opts: FixedPointOpts | None
        The two option dicts above, pre-resolved.

    Returns
    -------
    EnvCTM_c4v
        Environment at the fixed point.
    """
    if opts is None:
        opts = FixedPointOpts.from_legacy_dicts(ctm_opts_fwd, ctm_opts_fp)
    # Order leaves by site2index to match the backward's dA order (see the detailed
    # note in fixed_pt.py:fp_ctmrg). C4v is single-unique-site so this is an identity
    # here, but it keeps the apply-input / returned-gradient layout invariant explicit.
    ket = env.psi.ket
    raw_peps_params= tuple( ket[s]._data for s in sorted(ket.sites(), key=ket.site2index) )
    env_converged, env_t_meta, env_slices, env_1d = FixedPoint_c4v.apply(env, opts, *raw_peps_params)
    env_t_dict = _assemble_dict_from_1d(env_t_meta, env_1d, env_slices)
    env_converged.env = Lattice.from_dict(env_t_dict)
    return env_converged


class FixedPoint_c4v(torch.autograd.Function):
    ctm_log, t_ctm, t_check = None, None, None

    @staticmethod
    def compute_rdms(env):
        rdms = []
        start = time.time()
        for site in env.sites():
            rdms.append(rdm1x1(site, env.psi.ket, env)[0])
            rdms.append(rdm1x2(site, env.psi.ket, env)[0])
            rdms.append(rdm2x1(site, env.psi.ket, env)[0])
            rdms.append(rdm2x1(site, env.psi.ket, env)[0])
            rdms.append(rdm2x2_diagonal(site, env.psi.ket, env)[0])
            rdms.append(rdm2x2_anti_diagonal(site, env.psi.ket, env)[0])
            rdms.append(rdm2x2(site, env.psi.ket, env)[0])
        end = time.time()
        print(f"rdm calculations take {end-start:.1f}s")
        return rdms

    @staticmethod
    def fixed_point_iter(sigma, T_phase, C_phase, opts, env_dict, env_meta, env_slices, psi_meta, env_data, psi_data):
        env_t_dict = _assemble_dict_from_1d(env_meta, env_data, env_slices)
        psi_dict = combine_data_and_meta(psi_data, psi_meta)
        env_dict['env'], env_dict['psi'] = env_t_dict, psi_dict
        env_in = EnvCTM_c4v.from_dict(env_dict)

        env_in.update_(opts=opts.fp)


        sigma_p = sigma.flip_signature() # flip signatures but keep the same charge sectors
        site = env_in.sites()[0]
        fixed_t = tensordot(
            tensordot(sigma, env_in[site].t, axes=(0, 0), conj=(1, 0)),
            sigma_p,
            axes=(2, 0),
        )
        fixed_t._data = fixed_t._data * torch.exp(1j*T_phase).to(fixed_t._data.dtype)
        setattr(env_in[site], "t", fixed_t)

        fixed_C = tensordot(
            tensordot(sigma_p, env_in[site].tl, axes=(0, 0), conj=(1, 0)),
            sigma,
            axes=(1, 0),
        )
        fixed_C._data = fixed_C._data * torch.exp(1j*C_phase).to(fixed_C._data.dtype)
        setattr(env_in[site], "tl", fixed_C)

        env_out_dict = env_in.to_dict(level=0)
        env_t_data, _ = split_data_and_meta(env_out_dict['env'])
        return (_concat_data(env_t_data)[0], )

    def get_converged_env(env, opts: CTMOpts):
        r"""
        Run forward CTMRG to convergence.

        Deliberately separate from :meth:`FixedPoint.get_converged_env`: the c4v
        fixed point is its own algorithm, and detaches before measuring the
        corner spectra.
        """
        t_ctm, t_check = 0.0, 0.0
        converged, conv_history, max_dsv = False, [], None
        check = opts.conv_check if opts.conv_check is not None else opts.corner_tol

        # The inner iterator runs without a convergence test of its own; the
        # test happens below, so that conv_history reaches the caller.
        ctm_itr = env.ctmrg_(opts=replace(opts, corner_tol=None, conv_check=None,
                                          iterator_step=1))

        sweep = 0
        for sweep in range(opts.max_sweeps):
            t0 = time.perf_counter()
            next(ctm_itr)
            t1 = time.perf_counter()
            t_ctm += t1 - t0

            t2 = time.perf_counter()
            converged, max_dsv, conv_history = env.detach().ctm_conv_corner_spec(conv_history, check)
            t_check += time.perf_counter() - t2
            if opts.verbosity > 2:
                log.log(logging.INFO, f"CTM iter {len(conv_history)} |delta_C| {max_dsv} t {t1-t0} [s]")

            if converged:
                log.info(f"CTM converged: sweeps {sweep+1} t_ctm {t_ctm} [s] t_check {t_check} [s]"
                         + f" history {[r['max_dsv'] for r in conv_history]}.")
                break

        return env, converged, conv_history, t_ctm, t_check

    @staticmethod
    def forward(ctx, env: EnvCTM_c4v, opts: FixedPointOpts, *state_params):
        r"""
        Compute the fixed-point environment for the given state using CTMRG.
        First, run CTMRG until convergence then find the gauge transformation guaranteeing element-wise
        convergence of the environment tensors.

        Args:
            env (EnvCTM_c4v): Current environment to converge.
            opts (FixedPointOpts): resolved options. ``opts.fwd`` drives the forward
                convergence; ``opts.fp`` -- which inherits from it -- drives the
                gauge-fixing step and the Neumann backward budget.
            state_params (Sequence[Tensor]): tensors of underlying Peps state

        Returns:
            EnvCTM_c4v: Environment at fixed point.
            Sequence[Tensor]: raw environment data for the backward pass.
        """

        ctm_env_out, converged, *FixedPoint_c4v.ctm_log, FixedPoint_c4v.t_ctm, FixedPoint_c4v.t_check = FixedPoint_c4v.get_converged_env(
            env, opts.fwd,
        )
        if not converged:
            raise NoFixedPointError(code=1, message="No fixed point found: CTM forward does not converge!")

        # NOTE we need to find the gauge transformation that connects two set of environment tensors
        # obtained from CTMRG with the desired svd policy chosen for CTMRG fixed-point (differentiated) step

        # opts.fp already inherits from opts.fwd with the fp overrides merged in.
        ctx.verbosity = opts.verbosity

        env_converged = ctm_env_out.copy()
        t0= time.perf_counter()
        ctm_env_out.update_(opts=opts.fp)
        t1= time.perf_counter()
        log.info(f"{type(ctx).__name__}.forward FP CTM step t {t1-t0} [s]")

        t0 = time.perf_counter()
        sigma, T_phase, C_phase = find_gauge_c4v(env_converged, ctm_env_out, verbose=False)
        t1 = time.perf_counter()
        if sigma is None:
            raise NoFixedPointError(code=1, message="No fixed point found: fail to find the gauge matrix!")
        log.info(f"{type(ctx).__name__}.forward FP gauge-fixing t {t1-t0} [s]")
        env_dict = env_converged.to_dict(level=0)
        env_data, env_meta = split_data_and_meta(env_dict)

        sigma_d = sigma.to_dict(level=0)
        sigma_data, sigma_meta = split_data_and_meta(sigma_d)
        ctx.save_for_backward(*env_data, *sigma_data, T_phase, C_phase)
        ctx.env_meta, ctx.sigma_meta = env_meta, sigma_meta
        ctx.opts = opts

        env_t_data, env_t_meta = split_data_and_meta(env_dict['env'])
        env_1d, env_slices = _concat_data(env_t_data)

        return env_converged, env_t_meta, env_slices, env_1d

    @staticmethod
    def backward(ctx, none0, none1, none2, *grad_env):
        verbosity = ctx.verbosity
        grads = grad_env
        dA = grad_env

        env_data= ctx.saved_tensors[:-1]
        sigma_data, T_phase, C_phase = ctx.saved_tensors[-3], ctx.saved_tensors[-2], ctx.saved_tensors[-1]

        env_dict = combine_data_and_meta(env_data, ctx.env_meta)
        sigma_d = combine_data_and_meta((sigma_data,), ctx.sigma_meta)
        sigma = Tensor.from_dict(sigma_d)

        _env_data, _env_meta = split_data_and_meta(env_dict['env'])
        _env_ts, _env_slices = _concat_data(_env_data)
        _psi_data, _psi_meta = split_data_and_meta(env_dict['psi'])

        prev_grad_tmp = None
        diff_ave = None
        # Compute vjp only
        with torch.enable_grad():
            if verbosity > 2 and _env_ts.is_cuda:
                torch.cuda.memory._dump_snapshot(f"{type(ctx).__name__}_backward_prevjp_CUDAMEM.pickle")
            # _, dfdC_vjp = torch.func.vjp(lambda x: FixedPoint_c4v.fixed_point_iter(env, sigma, ctx.ctm_opts_fp, _env_slices, x, psi_data), _env_ts)
            # _, dfdA_vjp = torch.func.vjp(lambda x: FixedPoint_c4v.fixed_point_iter(env, sigma, ctx.ctm_opts_fp, _env_slices, _env_ts, x), psi_data)
            _, df_vjp = torch.func.vjp(lambda x,y: FixedPoint_c4v.fixed_point_iter(sigma, T_phase, C_phase, ctx.opts, env_dict, _env_meta, _env_slices, _psi_meta, x, y), _env_ts, _psi_data)
            dfdC_vjp= lambda x: (df_vjp(x)[0],)
            dfdA_vjp= lambda x: (df_vjp(x)[1],)
            if verbosity > 2 and _env_ts.is_cuda:
                torch.cuda.memory._dump_snapshot(f"{type(ctx).__name__}_backward_postvjp_CUDAMEM.pickle")

        # Budget and tolerance are the FP step's, carried over from the forward
        # by design: this loop reverses that CTM step.
        neumann_max_iter, neumann_tol = ctx.opts.fp.max_sweeps, ctx.opts.fp.corner_tol
        alpha = 0.4
        for step in range(neumann_max_iter):
            grads = dfdC_vjp(grads)
            if all([torch.norm(grad, p=torch.inf) < neumann_tol for grad in grads]):
                break
            else:
                dA = tuple(dA[i] + grads[i] for i in range(len(grads)))
            # for grad in grads:
            #     print(torch.norm(grad, p=torch.inf))

            if step % 10 == 0:
                grad_tmp = torch.cat(dfdA_vjp(dA)[0])
                if prev_grad_tmp is not None:
                    grad_diff = torch.norm(grad_tmp[0] - prev_grad_tmp[0])
                    # print("full grad diff", grad_diff)
                    if grad_diff < neumann_tol:
                        log.log(logging.INFO, f"Fixed_pt: The norm of the full grad diff is below {neumann_tol}.")
                        break
                    if diff_ave is not None:
                        if grad_diff > diff_ave:
                            # print("Full grad diff is no longer decreasing!")
                            log.log(logging.INFO, f"Fixed_pt: Full grad diff is no longer decreasing.")
                            break
                        else:
                            diff_ave = alpha*grad_diff + (1-alpha)*diff_ave
                    else:
                        diff_ave = grad_diff
                prev_grad_tmp = grad_tmp

        dA = dfdA_vjp(dA)[0]
        # one grad per forward input: (env, opts, *state_params)
        return None, None, *dA
