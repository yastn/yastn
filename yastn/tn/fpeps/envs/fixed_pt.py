import time, logging
import torch
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab, lgmres
from collections import namedtuple


from .... import Tensor, ones, zeros, eye, YastnError, Leg, tensordot, einsum, diag
from ._env_ctm import ctm_conv_corner_spec
from .rdm import *


log = logging.getLogger("ctmrg")


def env_raw_data(env):
    '''
    Combine all env raw tensors into a 1d tensor.
    '''
    data_list = []
    slice_list = []
    numel = 0
    for site in env.sites():
        for dirn in ["tl", "tr", "bl", "br", "t", "l", "b", "r"]:
            data_list.append(getattr(env[site], dirn)._data)
            slice_list.append((numel, len(data_list[-1])))
            numel += len(data_list[-1])

    return torch.cat(data_list), slice_list

def refill_env(env, data, slice_list):
    ind = 0
    for site in env.sites():
        for dirn in ["tl", "tr", "bl", "br", "t", "l", "b", "r"]:
            getattr(env[site], dirn)._data = torch.narrow(data, 0, *slice_list[ind])
            ind += 1

def extract_dsv_t(t, history):
    max_dsv = max((history[t][k] - history[t+1][k]).norm().item() for k in history[t]) if len(history)>t else float('Nan')
    return max_dsv

def extract_dsv(history):
    max_dsv_history = [extract_dsv(t, history) for t in range(len(history) - 1)]
    return max_dsv_history

def running_ave(history, window_size=5):
    cumsum = np.cumsum(np.insert(history, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

class NoFixedPointError(Exception):
    def __init__(self, code):
        super().__init__("No FixedPoint found")  # Pass message to the base class
        self.code = code                         # Add a custom attribute for error codes

# -----------simulataneous optim ---------
def env_T_gauge(config, T_old, T_new):
    #
    #   ----T_new--[leg1]--sigma--- == ---sigma--[leg2]---T_old ---
    #          |                                          |
    #          |                                          |

    leg1 = T_new.get_legs(axes=2)
    leg2 = T_old.get_legs(axes=0)
    T_old = T_old.transpose(axes=(2, 1, 0))

    identity1 = diag(eye(config=config, legs=(leg1, leg1.conj())))
    identity2 = diag(eye(config=config, legs=(leg2, leg2.conj())))

    M = (
        einsum(
            "ij, kl -> ikjl ",
            tensordot(T_new, T_new, axes=([0, 1], [0, 1]), conj=(0, 1)),
            identity2,
        )
        + einsum(
            "ij, kl -> ikjl",
            identity1,
            tensordot(T_old, T_old, axes=([0, 1], [0, 1]), conj=(0, 1)),
        )
        - einsum("imj, kml -> iljk", T_new.conj(), T_old)
        - einsum("imj, kml -> jkil", T_new, T_old.conj())
    )
    M = M.fuse_legs(axes=((2, 3), (0, 1)))
    s, u = M.eigh(axes=(0, 1))
    u = u.unfuse_legs(axes=(0,))
    s_zeros = s <= 1e-10
    # s_zeros = s <= 1e-14
    # s_dense = s.to_dense().diag()
    # print(s_dense[s_dense < 1e-6])
    modes = u @ s_zeros  # set eigenvectors with non-zero eigenvalues to zero
    zero_modes = []
    # collect zero eigenvectors
    for i in range(modes.get_legs(axes=2).tD[(0,)]):
        zero_mode = zeros(
            config=config, legs=(leg1.conj(), leg2.conj())
        )  # 1 ---sigma--- 2
        non_zero = False
        for charge_sector in modes.get_blocks_charge():
            if charge_sector[-1] == 0:  # the total charge of sigma should be zero
                block = modes[charge_sector]
                if torch.norm(block[:, :, i]) > 1e-8:
                    non_zero = True
                    zero_mode.set_block(ts=charge_sector[:-1], val=block[:, :, i])
        if non_zero:
            zero_modes.append(zero_mode)
    return zero_modes
    # if len(zero_modes) == 0:
    #   return None

    # cs = find_coeff(zero_modes, zero_modes[0].dtype)
    # zero_mode = zeros(zero_modes[0].config, legs=zero_modes[0].get_legs(), dtype=zero_modes[0].dtype)
    # for i in range(len(cs)):
    #   zero_mode += cs[i] * zero_modes[i]
    # return zero_mode

def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))

def find_coeff(env_old_local, env_local, zero_modes_dict, dtype=torch.complex128):
    def fix_phase_manual(cs):
        unitaries = []
        for i, dirn in enumerate(("t", "l", "b", "r")):
            legs = zero_modes_dict[dirn][0].get_legs()
            unitary = zeros(config=zero_modes_dict[dirn][0].config, legs=legs, dtype='complex128' if dtype is torch.complex128 else 'float64')
            for j in range(len(zero_modes_dict[dirn])):
                unitary = unitary + cs[i][j] * zero_modes_dict[dirn][j]
            unitaries.append(unitary)

        for dirn in ["tl", "bl", "br", "tr"]:
            C_old = getattr(env_old_local, dirn)
            C_new = getattr(env_local, dirn)
            if dirn == "tl":
                fixed_C = tensordot(
                    tensordot(unitaries[1], C_new, axes=(0, 0), conj=(1, 0)),
                    unitaries[0],
                    axes=(1, 0),
                )
                nonzero_locs = torch.abs(fixed_C._data) > 1e-6
                phase = torch.mean(
                    fixed_C._data[nonzero_locs] / C_old._data[nonzero_locs]
                )
                for i in range(len(cs[0])):
                    cs[0][i] = cs[0][i] / phase
                unitaries[0] = unitaries[0]/phase

                # fixed_C = tensordot(
                #   tensordot(unitaries[1], C_new, axes=(0, 0), conj=(1, 0)),
                #   unitaries[0],
                #   axes=(1, 0),
                # )

            if dirn == "bl":
                fixed_C = tensordot(
                    tensordot(unitaries[2], C_new, axes=(0, 0), conj=(1, 0)),
                    unitaries[1],
                    axes=(1, 0),
                )
                nonzero_locs = torch.abs(fixed_C._data) > 1e-6
                phase = torch.mean(
                    fixed_C._data[nonzero_locs] / C_old._data[nonzero_locs]
                )
                for i in range(len(cs[2])):
                    cs[2][i] = cs[2][i]*phase
                unitaries[2] = unitaries[2]*phase

                # fixed_C = tensordot(
                #   tensordot(unitaries[2], C_new, axes=(0, 0), conj=(1, 0)),
                #   unitaries[1],
                #   axes=(1, 0),
                # )

            if dirn == "br":
                fixed_C = tensordot(
                    tensordot(unitaries[3], C_new, axes=(0, 0), conj=(1, 0)),
                    unitaries[2],
                    axes=(1, 0),
                )
                nonzero_locs = torch.abs(fixed_C._data) > 1e-6
                phase = torch.mean(
                    fixed_C._data[nonzero_locs] / C_old._data[nonzero_locs]
                )
                for i in range(len(cs[3])):
                    cs[3][i] = cs[3][i]*phase
                unitaries[3] = unitaries[3]*phase

                # fixed_C = tensordot(
                #   tensordot(unitaries[3], C_new, axes=(0, 0), conj=(1, 0)),
                #   unitaries[2],
                #   axes=(1, 0),
                # )

            if dirn == "tr":
                fixed_C = tensordot(
                    tensordot(unitaries[0], C_new, axes=(0, 0), conj=(1, 0)),
                    unitaries[3],
                    axes=(1, 0),
                )
                # pass
        return cs


    def phase_loss(phases, cs):
        unitaries = []
        for i, dirn in enumerate(("t", "l", "b", "r")):
            legs = zero_modes_dict[dirn][0].get_legs()
            unitary = zeros(config=zero_modes_dict[dirn][0].config, legs=legs, dtype='complex128' if dtype is torch.complex128 else 'float64')
            for j in range(len(zero_modes_dict[dirn])):
                unitary = unitary + cs[i][j] * zero_modes_dict[dirn][j]
            unitary = unitary*np.exp(1j*phases[i])
            unitaries.append(unitary)

        loss = 0.0
        for dirn in ["tl", "bl", "br", "tr"]:
            C_old = getattr(env_old_local, dirn)
            C_new = getattr(env_local, dirn)
            if dirn == "tl":
                fixed_C = tensordot(
                    tensordot(unitaries[1], C_new, axes=(0, 0), conj=(1, 0)),
                    unitaries[0],
                    axes=(1, 0),
                )

            if dirn == "bl":
                fixed_C = tensordot(
                    tensordot(unitaries[2], C_new, axes=(0, 0), conj=(1, 0)),
                    unitaries[1],
                    axes=(1, 0),
                )

            if dirn == "br":
                fixed_C = tensordot(
                    tensordot(unitaries[3], C_new, axes=(0, 0), conj=(1, 0)),
                    unitaries[2],
                    axes=(1, 0),
                )

            if dirn == "tr":
                fixed_C = tensordot(
                    tensordot(unitaries[0], C_new, axes=(0, 0), conj=(1, 0)),
                    unitaries[3],
                    axes=(1, 0),
                )

            loss += (fixed_C - C_old).norm(p='fro')**2
        return loss

    def unitary_loss(cs):
        loss = 0.0
        # assemble unitaries
        ind = 0
        for i, dirn in enumerate(("t", "l", "b", "r")):
            legs = zero_modes_dict[dirn][0].get_legs()
            identity = diag(eye(config=zero_modes_dict[dirn][0].config, legs=(legs[0], legs[0].conj())))
            unitary = zeros(config=zero_modes_dict[dirn][0].config, legs=legs, dtype='complex128' if dtype is torch.complex128 else 'float64')
            for j in range(len(zero_modes_dict[dirn])):
                unitary = unitary + cs[ind] * zero_modes_dict[dirn][j]
                ind += 1
            loss += ((tensordot(unitary, unitary, axes=(1, 1), conj=(0, 1)) - identity).norm(p='fro')**2/ identity.norm(p='fro')**2).item().real
            # loss += ((tensordot(unitary, unitary, axes=(1, 1), conj=(0, 1)) - identity).norm(p='fro')**2).item().real
        return loss

    if dtype == torch.complex128:
        cs = np.concatenate([np.ones(len(zero_modes_dict[dirn]), dtype=np.complex128) for dirn in ("t", "l", "b", "r")])
        res = minimize(fun=lambda z: unitary_loss(real_to_complex(z)), x0=complex_to_real(cs), method='cg', tol=1e-12)
        print(res.message)
        res = real_to_complex(res.x)
    else:
        cs = np.concatenate([np.ones(len(zero_modes_dict[dirn]), dtype=np.float64) for dirn in ("t", "l", "b", "r")])
        cs += np.random.rand(*cs.shape)*0.1
        # cs = np.concatenate([np.random.rand(len(zero_modes_dict[dirn])) for dirn in ("t", "l", "b", "r")])
        res = minimize(fun=unitary_loss, x0=cs, method='cg', tol=1e-10)
        print(res.message)
        res = res.x

    cs= []
    ind = 0
    for i, dirn in enumerate(("t", "l", "b", "r")):
        cs.append(np.array(res[ind:ind+len(zero_modes_dict[dirn])]))
        ind += len(zero_modes_dict[dirn])

    print("fix phase")
    if dtype == torch.complex128:
        phases = np.random.rand(4)
        res = minimize(fun=phase_loss, x0=phases, args=(cs, ), method='cg', tol=1e-12)
        phases = res.x
        print("phases: ", phases)
        for i, dirn in enumerate(("t", "l", "b", "r")):
            for j in range(len(zero_modes_dict[dirn])):
                cs[i][j] = cs[i][j] * np.exp(1j*phases[i])
    else:
        cs = fix_phase_manual(cs)

    cs = torch.utils.checkpoint.detach_variable(tuple(cs))
    return cs


def find_gauge(env_old, env):
    Gauge = namedtuple("Gauge", "t l b r")
    zero_modes_dict = {}
    sigma_dict = {}
    for site in env.sites():
        sigma_list = []
        for k in ["t", "l", "b", "r"]:
            T_old = getattr(env_old[site], k)
            T_new = getattr(env[site], k)
            zero_modes = env_T_gauge(env.psi.config, T_old, T_new)
            if len(zero_modes) == 0:
                return None
            zero_modes_dict[k] = zero_modes

        cs = find_coeff(env_old[site], env[site], zero_modes_dict, dtype=zero_modes_dict["t"][0].dtype)
        for i, dirn in enumerate(["t", "l", "b", "r"]):
            dtype = "complex128" if zero_modes_dict[dirn][0].dtype is torch.complex128 else "float64"
            zero_mode = zeros(zero_modes_dict[dirn][0].config, legs=zero_modes_dict[dirn][0].get_legs(), dtype=dtype)
            for j in range(len(cs[i])):
                zero_mode += cs[i][j] * zero_modes_dict[dirn][j]
            sigma_list.append(zero_mode)

        for i, dirn in enumerate(["t", "l", "b", "r"]):
            T_old = getattr(env_old[site], dirn)
            T_new = getattr(env[site], dirn)
            fixed_t = tensordot(
                tensordot(sigma_list[i], T_new, axes=(0, 0), conj=(1, 0)),
                sigma_list[i],
                axes=(2, 0),
            )
            print("T diff:", (fixed_t - T_old).norm() / T_old.norm())

        for k in ["tl", "bl", "br", "tr"]:
            C_old = getattr(env_old[site], k)
            C_new = getattr(env[site], k)
            if k == "tl":
                fixed_C = tensordot(
                    tensordot(sigma_list[1], C_new, axes=(0, 0), conj=(1, 0)),
                    sigma_list[0],
                    axes=(1, 0),
                )
            if k == "bl":
                fixed_C = tensordot(
                    tensordot(sigma_list[2], C_new, axes=(0, 0), conj=(1, 0)),
                    sigma_list[1],
                    axes=(1, 0),
                )
            if k == "br":
                fixed_C = tensordot(
                    tensordot(sigma_list[3], C_new, axes=(0, 0), conj=(1, 0)),
                    sigma_list[2],
                    axes=(1, 0),
                )
            if k == "tr":
                fixed_C = tensordot(
                    tensordot(sigma_list[0], C_new, axes=(0, 0), conj=(1, 0)),
                    sigma_list[3],
                    axes=(1, 0),
                )
            print("C diff:", (fixed_C - C_old).norm() / C_old.norm())

        for sigma in sigma_list:
            sigma._data.detach_()
        sigma_dict[site] = Gauge(*sigma_list)

    return sigma_dict


class FixedPoint(torch.autograd.Function):
    ctm_env_out, ctm_log, t_ctm, t_check = None, None, None, None

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
    def fixed_point_iter(env_in, sigma_dict, opts_svd, slices, env_data):
        refill_env(env_in, env_data, slices)
        env_in.update_(
            opts_svd=opts_svd, method="2site", use_qr=False, checkpoint_move=False, svd_policy='fullrank',
        )

        for site in env_in.sites():
            for dirn in ["t", "l", "b", "r"]:
                sigma = getattr(sigma_dict[site], dirn)
                fixed_t = tensordot(
                    tensordot(sigma, getattr(env_in[site], dirn), axes=(0, 0), conj=(1, 0)),
                    sigma,
                    axes=(2, 0),
                )
                setattr(env_in[site], dirn, fixed_t)

        for dirn in ["tl", "bl", "br", "tr"]:
            C_new = getattr(env_in[site], dirn)
            sigma_t, sigma_l, sigma_b, sigma_r = sigma_dict[site].t, sigma_dict[site].l, sigma_dict[site].b, sigma_dict[site].r
            if dirn == "tl":
                fixed_C = tensordot(
                    tensordot(sigma_l, C_new, axes=(0, 0), conj=(1, 0)),
                    sigma_t,
                    axes=(1, 0),
                )
            if dirn == "bl":
                fixed_C = tensordot(
                    tensordot(sigma_b, C_new, axes=(0, 0), conj=(1, 0)),
                    sigma_l,
                    axes=(1, 0),
                )
            if dirn == "br":
                fixed_C = tensordot(
                    tensordot(sigma_r, C_new, axes=(0, 0), conj=(1, 0)),
                    sigma_b,
                    axes=(1, 0),
                )
            if dirn == "tr":
                fixed_C = tensordot(
                    tensordot(sigma_t, C_new, axes=(0, 0), conj=(1, 0)),
                    sigma_r,
                    axes=(1, 0),
                )
            setattr(env_in[site], dirn, fixed_C)

        env_out_data, slices = env_raw_data(env_in)
        return env_out_data

    @torch.no_grad()
    def ctm_conv_check(env, history, corner_tol):
        converged, max_dsv, history = ctm_conv_corner_spec(env, history, corner_tol)
        print("max_dsv:", max_dsv)
        log.log(logging.INFO, f"CTM iter {len(history)} |delta_C| {max_dsv}")
        return converged, history

    def get_converged_env(
        env,
        method="2site",
        svd_policy='fullrank',
        D_block=None,
        max_sweeps=100,
        iterator_step=1,
        opts_svd=None,
        corner_tol=1e-8,
    ):
        t_ctm, t_check = 0.0, 0.0
        t_ctm_prev = time.perf_counter()
        converged, conv_history = False, []

        for sweep in range(max_sweeps):
            env.update_(
                opts_svd=opts_svd, method=method, use_qr=False, checkpoint_move=False, policy=svd_policy, D_block=D_block,
            )
            t_ctm_after = time.perf_counter()
            t_ctm += t_ctm_after - t_ctm_prev
            t_ctm_prev = t_ctm_after

            converged, conv_history = FixedPoint.ctm_conv_check(env, conv_history, corner_tol)
            if converged:
                break
        print(f"t_ctm: {t_ctm:.1f}s")

        return env, converged, conv_history, t_ctm, t_check

    @staticmethod
    def forward(
        ctx, env_params, slices, yastn_config, env, opts_svd, chi, corner_tol, ctm_args, *state_params
    ):
        refill_env(env, env_params, slices)
        ctm_env_out, converged, *FixedPoint.ctm_log, FixedPoint.t_ctm, FixedPoint.t_check = FixedPoint.get_converged_env(
            env,
            max_sweeps=ctm_args.ctm_max_iter,
            iterator_step=1,
            opts_svd=opts_svd,
            corner_tol=corner_tol,
            # svd_policy='arnoldi',
            svd_policy='fullrank',
            D_block=chi,
        )

        env_old = ctm_env_out.copy()
        # rdm_old = FixedPoint.compute_rdms(env_old)
        FixedPoint.ctm_env_out = env_old
        # note that we need to find the gauge transformation that connects two set of environment tensors
        # obtained from CTMRG with the 'full' svd, because the backward uses the full svd backward.
        ctx.proj = ctm_env_out.update_(
            opts_svd=opts_svd, method="2site", use_qr=False, checkpoint_move=False, svd_policy='fullrank',
        )
        # rdm_new = FixedPoint.compute_rdms(ctm_env_out)
        # for r1, r2 in zip(rdm_old, rdm_new):
        #     print("rdm diff:", (r1-r2).norm())

        sigma_dict = find_gauge(env_old, ctm_env_out)
        if sigma_dict is None:
            raise NoFixedPointError(code=1)

        env_out_data, slices = env_raw_data(env_old)
        ctx.save_for_backward(env_out_data)
        ctx.yastn_config = yastn_config
        ctx.ctm_args = ctm_args
        ctx.opts_svd = opts_svd
        ctx.ctm_args = ctm_args
        ctx.slices = slices
        FixedPoint.slices = slices

        ctx.env = env_old
        ctx.sigma_dict = sigma_dict
        return env_out_data

    @staticmethod
    def backward(ctx, *grad_env):
        print("Backward called")
        grads = grad_env[0]
        dA = list(grad_env)
        env_data = torch.utils.checkpoint.detach_variable(ctx.saved_tensors)[0] # only one element in the tuple

        prev_grad_tmp = None
        diff_ave = None
        alpha = 0.4
        for step in range(ctx.ctm_args.ctm_max_iter):
            # Compute vjp only
            with torch.enable_grad():
                grads = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data), env_data, grad_outputs=grads)
            if all([torch.norm(grad, p=torch.inf) < 1e-8 for grad in grads]):
                break
            else:
                for i in range(len(grads)):
                    dA[i] = dA[i] + grads[i]

            for grad in grads:
                print(torch.norm(grad, p=torch.inf))

            if step % 1 == 0:
                with torch.enable_grad():
                    grad_tmp = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data), ctx.env.psi.ket.get_parameters(), grad_outputs=dA)
                if prev_grad_tmp is not None:
                    grad_diff = torch.norm(grad_tmp[0] - prev_grad_tmp[0])
                    print("full grad diff", grad_diff)
                    if diff_ave is not None:
                        if grad_diff > diff_ave:
                            print("grad diff is no longer decreasing!")
                            break
                        else:
                            diff_ave = alpha*grad_diff + (1-alpha)*diff_ave
                    else:
                        diff_ave = grad_diff
                prev_grad_tmp = grad_tmp

        with torch.enable_grad():
            dA = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data), ctx.env.psi.ket.get_parameters(), grad_outputs=dA)

        # --------option 2----------
        # Solve equations of the form Ax = b. Takes extremely long time if df/dC has eigenvalues close to 1

        # env_data = torch.utils.checkpoint.detach_variable(ctx.saved_tensors)[0] # only one element in the tuple
        # grads = grad_env[0]
        # def mat_vec_A(u):
        #   u = torch.from_numpy(u)
        #   with torch.enable_grad():
        #       res = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data), env_data, grad_outputs=u)
        #   return (u - res[0]).numpy()

        # def rmat_vec_A(u):
        #   u = torch.from_numpy(u)
        #   u = u.conj()
        #   f = lambda x: FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, x)
        #   with torch.enable_grad():
        #       output, res = torch.autograd.functional.jvp(f, env_data, u)
        #   return (u - res).conj().numpy()

        # A = LinearOperator(matvec=mat_vec_A, rmatvec=rmat_vec_A, shape=(env_data.size(dim=0) , grads.size(dim=0)), dtype=np.complex128 if env_data.dtype==torch.complex128 else np.float64)

        # u, info = lgmres(A, grads, M=None, rtol=1e-7, atol=0.0)
        # # res = lsqr(A, grads, atol=1e-7, btol=1e-7, show=True, x0=np.random.rand(*u.shape))
        # # u = res[0]

        # u = torch.from_numpy(u)
        # with torch.enable_grad():
        #   dA = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data), ctx.env.psi.ket.get_parameters(), grad_outputs=u)

        return None, None, None, None, None, None, None, None, *dA
