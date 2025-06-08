import time, logging
import torch
import numpy as np
from scipy.optimize import minimize
from collections import namedtuple

import torch.utils.checkpoint


from .... import zeros, eye, tensordot, einsum, diag
from ._env_ctm import ctm_conv_corner_spec
from ._env_ctm_c4v import EnvCTM_c4v, decompress_env_c4v_1d
from .fixed_pt import env_T_gauge_multi_sites, fast_env_T_gauge_multi_sites, NoFixedPointError, real_to_complex, complex_to_real
from .rdm import *


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

def refill_env_c4v(env, data, slice_list):
    ind = 0
    for site in env.sites():
        for dirn in ["tl", "t"]:
            getattr(env[site], dirn)._data = torch.narrow(data, 0, *slice_list[ind])
            ind += 1

def refill_state_c4v(state, data):
    assert len(state.sites()) == len(data), "Number of sites in state and data do not match"
    for site,d in zip(state.sites(), data):
        state[site]._data = d
    return state

def find_coeff(zero_modes, dtype=torch.complex128):
    def unitary_loss(cs):
        loss = 0.0
        # assemble unitaries
        legs = zero_modes[0].get_legs()
        identity = diag(eye(config=zero_modes[0].config, legs=(legs[0], legs[0].conj())))
        unitary = zeros(config=zero_modes[0].config, legs=legs, dtype='complex128' if dtype is torch.complex128 else 'float64')
        for j in range(len(zero_modes)):
            unitary = unitary + cs[j] * zero_modes[j]
        loss += ((tensordot(unitary, unitary, axes=(1, 1), conj=(0, 1)) - identity).norm(p='fro')**2/ identity.norm(p='fro')**2).item().real
        return loss

    if dtype == torch.complex128:
        cs = np.ones(len(zero_modes), dtype=np.float64)
        res = minimize(fun=lambda z: unitary_loss(real_to_complex(z)), x0=complex_to_real(cs), method='SLSQP', options={"eps":1e-9, "ftol":1e-14})
        # print(res.message)
        res = real_to_complex(res.x)
    else:
        cs = np.zeros(len(zero_modes), dtype=np.float64)
        cs += np.random.rand(*cs.shape)*0.5
        res = minimize(fun=unitary_loss, x0=cs, method='SLSQP', jac='3-point', options={"eps": 1e-9, "ftol":1e-14})
        # print(res.message)
        res = res.x
    return res

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
    # if len(zero_modes) == 0: # fast method fails
    #     zero_modes = env_T_gauge_multi_sites(env.psi.config, T_olds, T_news)
    if len(zero_modes) == 0:
        return None

    cs = find_coeff(zero_modes, dtype=zero_modes[0].dtype)

    site = env.sites()[0]
    dtype = "complex128" if zero_modes[0].dtype is torch.complex128 else "float64"
    sigma = zeros(zero_modes[0].config, legs=zero_modes[0].get_legs(), dtype=dtype)
    for i, c in enumerate(cs):
        sigma += c * zero_modes[i]
    sigma._data.detach_()

    # Note: The sigma matrix for T_new^' can be obtained by flipping signatures of sigma matrix
    site = env.sites()[0]
    sigma_p = sigma.flip_signature() # flip signatures but keep the same charge sectors
    fixed_t = tensordot(
        tensordot(sigma, env[site].t, axes=(0, 0), conj=(1, 0)), sigma_p, axes=(2, 0),
    )
    T_old = env_old[site].t
    if verbose:
        print("T diff:", (fixed_t - T_old).norm() / T_old.norm())

    fixed_C = tensordot(
        tensordot(sigma_p, env[site].tl, axes=(0, 0), conj=(1, 0)),
        sigma,
        axes=(1, 0),
    )

    C_old = env_old[site].tl
    if verbose:
        print("C diff:", (fixed_C - C_old).norm() / C_old.norm())

    return sigma

def fp_ctmrg_c4v(env: EnvCTM_c4v, \
            ctm_opts_fwd : dict= {'opts_svd': {}, 'corner_tol': 1e-8, 'max_sweeps': 100,
                'method': "2site", 'use_qr': False, 'svd_policy': 'fullrank', 'D_block': None}, \
            ctm_opts_fp: dict= {'svd_policy': 'fullrank'}):
    r"""
    Compute the fixed-point environment for the given state using CTMRG.
    First, run CTMRG until convergence then find the gauge transformation guaranteeing element-wise
    convergence of the environment tensors.

    Args:
        env (EnvCTM_c4v): CTM environment
        ctm_opts_fwd (dict): Options for forward CTMRG convergence.
        ctm_opts_fp (dict): Options for fixing the gauge transformation.

    Returns:
        EnvCTM_c4v: Environment at fixed point.
        Sequence[Tensor]: raw environment data for the backward pass.
    """
    raw_peps_params= tuple( env.psi.ket[s]._data for s in env.psi.ket.sites() )
    env_converged, env_slices, env_1d = FixedPoint_c4v.apply(env, ctm_opts_fwd, ctm_opts_fp, *raw_peps_params)
    return env_converged, env_slices, env_1d, FixedPoint_c4v.t_ctm


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
    def fixed_point_iter(env_in, sigma, ctm_opts_fp, slices, env_data, psi_data):
        refill_env_c4v(env_in, env_data, slices)
        refill_state_c4v(env_in.psi.ket, psi_data)

        env_in.update_(
            **ctm_opts_fp
        )


        sigma_p = sigma.flip_signature() # flip signatures but keep the same charge sectors
        site = env_in.sites()[0]
        fixed_t = tensordot(
            tensordot(sigma, env_in[site].t, axes=(0, 0), conj=(1, 0)),
            sigma_p,
            axes=(2, 0),
        )
        setattr(env_in[site], "t", fixed_t)

        fixed_C = tensordot(
            tensordot(sigma_p, env_in[site].tl, axes=(0, 0), conj=(1, 0)),
            sigma,
            axes=(1, 0),
        )
        setattr(env_in[site], "tl", fixed_C)

        env_out_data, slices = env_raw_data_c4v(env_in)
        return (env_out_data, )

    @torch.no_grad()
    def ctm_conv_check(env, history, corner_tol):
        converged, max_dsv, history = ctm_conv_corner_spec(env, history, corner_tol)
        # print("max_dsv:", max_dsv)
        log.log(logging.INFO, f"CTM iter {len(history)} |delta_C| {max_dsv}")
        return converged, history

    def get_converged_env(
        env,
        method="2site",
        svd_policy='fullrank',
        D_block=None,
        max_sweeps=100,
        opts_svd=None,
        corner_tol=1e-8,
        **kwargs
    ):
        t_ctm, t_check = 0.0, 0.0
        t_ctm_prev = time.perf_counter()
        converged, conv_history = False, []
        for sweep in range(max_sweeps):
            env.update_(
                opts_svd=opts_svd, method=method, checkpoint_move=False, policy=svd_policy, D_block=D_block, **kwargs
            )
            t_ctm_after = time.perf_counter()
            t_ctm += t_ctm_after - t_ctm_prev
            t_ctm_prev = t_ctm_after

            converged, conv_history = FixedPoint_c4v.ctm_conv_check(env, conv_history, corner_tol)
            if converged:
                break

        return env, converged, conv_history, t_ctm, t_check

    @staticmethod
    def forward(ctx, env: EnvCTM_c4v, ctm_opts_fwd : dict, ctm_opts_fp: dict, *state_params):
        r"""
        Compute the fixed-point environment for the given state using CTMRG.
        First, run CTMRG until convergence then find the gauge transformation guaranteeing element-wise
        convergence of the environment tensors.

        Args:
            env (EnvCTM_c4v): Current environment to converge.
            ctm_opts_fwd (dict): Options for forward CTMRG convergence.
            ctm_opts_fp (dict): Options for fixing the gauge transformation.
            state_params (Sequence[Tensor]): tensors of underlying Peps state

        Returns:
            EnvCTM_c4v: Environment at fixed point.
            Sequence[Tensor]: raw environment data for the backward pass.
        """

        ctm_env_out, converged, *FixedPoint_c4v.ctm_log, FixedPoint_c4v.t_ctm, FixedPoint_c4v.t_check = FixedPoint_c4v.get_converged_env(
            env,
            **ctm_opts_fwd,
        )
        if not converged:
            raise NoFixedPointError(code=1, message="No fixed point found: CTM forward does not converge!")

        # note that we need to find the gauge transformation that connects two set of environment tensors
        # obtained from CTMRG with the 'full' svd, because the backward uses the full svd backward.
        _ctm_opts_fp = dict(ctm_opts_fwd)
        if ctm_opts_fp is not None:
            _ctm_opts_fp.update(ctm_opts_fp)
        env_converged = ctm_env_out.copy()
        ctx.proj = ctm_env_out.update_(**_ctm_opts_fp)

        t_gauge_prev = time.perf_counter()
        sigma = find_gauge_c4v(env_converged, ctm_env_out, verbose=False)
        t_gauge_after = time.perf_counter()
        if sigma is None:
            raise NoFixedPointError(code=1, message="No fixed point found: fail to find the gauge matrix!")
        log.log(logging.INFO, f"t_gauge: {t_gauge_after - t_gauge_prev:.1f}s")

        env_data, env_meta = env_converged.compress_env_1d()
        ctx.save_for_backward(*env_data)
        ctx.env_meta = env_meta
        ctx.ctm_opts_fp = _ctm_opts_fp
        ctx.sigma = sigma
        env_1d, env_slices= env_raw_data_c4v(env_converged)

        return env_converged, env_slices, env_1d

    @staticmethod
    def backward(ctx, none0, none1, *grad_env):
        grads = grad_env
        dA = grad_env

        # env_data = torch.utils.checkpoint.detach_variable(ctx.saved_tensors)[0] # only one element in the tuple
        env_data = ctx.saved_tensors
        env= decompress_env_c4v_1d(env_data, ctx.env_meta)
        psi_data = tuple(env.psi.ket[s]._data for s in env.psi.ket.sites())
        _env_ts, _env_slices= env_raw_data_c4v(env)

        prev_grad_tmp = None
        diff_ave = None
        # Compute vjp only
        with torch.enable_grad():
            _, dfdC_vjp = torch.func.vjp(lambda x: FixedPoint_c4v.fixed_point_iter(env, ctx.sigma, ctx.ctm_opts_fp, _env_slices, x, psi_data), _env_ts)
            _, dfdA_vjp = torch.func.vjp(lambda x: FixedPoint_c4v.fixed_point_iter(env, ctx.sigma, ctx.ctm_opts_fp, _env_slices, _env_ts, x), psi_data)
        # fixed_point_iter changes the data of psi, so we need to refill the state to recover the previous state
        refill_state_c4v(env.psi.ket, psi_data)

        alpha = 0.4
        for step in range(ctx.ctm_opts_fp['max_sweeps']):
            grads = dfdC_vjp(grads)
            # with torch.enable_grad():
            #     grads = torch.autograd.grad(FixedPoint_c4v.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data, psi_data), env_data, grad_outputs=grads)
            if all([torch.norm(grad, p=torch.inf) < ctx.ctm_opts_fp["corner_tol"] for grad in grads]):
                break
            else:
                dA = tuple(dA[i] + grads[i] for i in range(len(grads)))
            # for grad in grads:
            #     print(torch.norm(grad, p=torch.inf))

            if step % 10 == 0:
                # with torch.enable_grad():
                #     grad_tmp = torch.autograd.grad(FixedPoint_c4v.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data, psi_data), psi_data, grad_outputs=dA)
                grad_tmp = torch.cat(dfdA_vjp(dA)[0])
                if prev_grad_tmp is not None:
                    grad_diff = torch.norm(grad_tmp[0] - prev_grad_tmp[0])
                    # print("full grad diff", grad_diff)
                    if grad_diff < ctx.ctm_opts_fp["corner_tol"]:
                        # print("The norm of the full grad diff is below 1e-10.")
                        log.log(logging.INFO, f"Fixed_pt: The norm of the full grad diff is below {ctx.ctm_opts_fp["corner_tol"]}.")
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
        return None, None, None, *dA
