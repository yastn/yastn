import time, logging
import torch
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh, eigs, svds, ArpackNoConvergence
import primme

from collections import namedtuple

import torch.utils.checkpoint


from .... import Tensor, zeros, eye, YastnError, tensordot, einsum, diag, rand, qr
from ._env_ctm import ctm_conv_corner_spec, decompress_env_1d, EnvCTM_projectors
from .... import zeros, decompress_from_1d
from ....tensor._tests import _test_axes_match
from .rdm import *
log = logging.getLogger(__name__)


class NoFixedPointError(Exception):
    def __init__(self, code, message=None):
        if message is None:
            message = "No fixed point found"
        self.message = message
        super().__init__(self.message)  # Pass message to the base class
        self.code = code          				 # Add a custom attribute for error codes


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

def refill_state(state, data):
    # def unflatten(flattened_list, nested_iterable):
    #     def unflatten_helper(flattened_iter, nested_iterable):
    #         if isinstance(nested_iterable, dict):
    #             for k, v in nested_iterable.items():
    #                 unflatten_helper(flattened_iter, v)
    #         elif isinstance(nested_iterable, (list, tuple, set)):
    #             for v in nested_iterable:
    #                 unflatten_helper(flattened_iter, v)
    #         else:
    #             nested_iterable._data = next(flattened_iter)

    #     flattened_iter = iter(flattened_list)
    #     unflatten_helper(flattened_iter, nested_iterable)

    # unflatten(data, state.parameters)
    # state.sync_()
    assert len(state.sites()) == len(data), "Number of sites in state and data do not match"
    for site,d in zip(state.sites(), data):
        state[site]._data = d
    return state

def extract_dsv_t(t, history):
    max_dsv = max((history[t][k] - history[t+1][k]).norm().item() for k in history[t]) if len(history)>t else float('Nan')
    return max_dsv

def extract_dsv(history):
    max_dsv_history = [extract_dsv(t, history) for t in range(len(history) - 1)]
    return max_dsv_history

def running_ave(history, window_size=5):
    cumsum = np.cumsum(np.insert(history, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def null_space(A, batch_k=2, maxiter=None, ncv=None):
    '''
    Run primme.eigsh sequentially to find the null space of A.

    Parameters:
    - A: sparse matrix (hermitian)
    - batch_k: number of eigenvalues to compute in each run of primme.eigsh
    - maxiter: maxiter in primme.eigsh
    - ncv: ncv in primme.eigsh

    returns:
    - zero_vs: null space of A
    '''
    maxiter = maxiter if maxiter is not None else A.shape[0] * 10
    zero_vs = np.empty((A.shape[0], 0), dtype=A.dtype)
    while True:
        # find eigenvectors that are orthogonal to the current zero_vs
        w, vs = primme.eigsh(A, k=batch_k, which='SA', v0=None, maxiter=maxiter, ncv=ncv, lock=zero_vs, method="PRIMME_DYNAMIC")
        zero_vs = np.hstack((zero_vs, vs[:, w < 1e-8]))
        if len(w[w<1e-8]) < len(w): # if non-zero eigenvalues are found, stop
            break
    return zero_vs


def robust_eigsh(A, initial_maxiter, max_maxiter, k=3, retry_factor=3, sigma=None, **kwargs):
    """
    Run eigsh with retries on non-convergence.

    Parameters:
    - A: sparse matrix (symmetric)
    - initial_maxiter: starting maxiter value [typical value: A.shape[0] * 10]
    - max_maxiter: maximum maxiter value
    - retry_factor: multiplier to increase maxiter on retry
    - k: number of eigenvalues
    - sigma: shift for shift-invert
    - kwargs: other args passed to eigsh

    Returns:
    - eigenvalues, eigenvectors
    """
    try:
        return eigsh(A, k=k, sigma=sigma, maxiter=initial_maxiter, **kwargs)
    except ArpackNoConvergence as e:
        # if all zero eigenvalues converge already, no need to retry
        if len(e.eigenvalues [e.eigenvalues < 1e-8]) < len(e.eigenvalues):
            return e.eigenvalues, e.eigenvectors
        elif initial_maxiter > max_maxiter:
            raise e
        else:
            log.log(logging.INFO, f"Eigsh Warning: No convergence after {initial_maxiter} iterations. Retrying with more iterations...")
            return robust_eigsh(A, initial_maxiter * retry_factor, max_maxiter, k=k, retry_factor=retry_factor, sigma=sigma, **kwargs)

def env_T_gauge_multi_sites(config, T_olds, T_news):
    #  Robust gauge-fixing from https://arxiv.org/abs/2311.11894
    #
    #   --[leg1]--T_new_1...T_new_L--[leg2]--sigma--- == ---sigma--[leg3]---T_old_1...T_old_L ---[leg4]---
    #               |          |                                              |         |
    #               |          |                                              |         |

    # ========full-matrix impl===========
    # leg1 = T_news[0].get_legs(axes=0)
    # leg2 = T_news[-1].get_legs(axes=2)
    # leg3 = T_olds[0].get_legs(axes=0)
    # leg4 = T_olds[-1].get_legs(axes=2)
    # # T_old = T_old.transpose(axes=(2, 1, 0)) # exchange the left and right indices

    # identity1 = diag(eye(config=config, legs=(leg1, leg1.conj())))
    # identity2 = diag(eye(config=config, legs=(leg2, leg2.conj())))
    # identity3 = diag(eye(config=config, legs=(leg3, leg3.conj())))
    # identity4 = diag(eye(config=config, legs=(leg4, leg4.conj())))

    # M1, M2 = identity1, identity4
    # M3, M4 = einsum("ji, kl -> ikjl", identity1, identity3), einsum("ij, lk -> ikjl", identity2, identity4)

    # for i in range(len(T_olds)):
    #     #     2    0
    #     #     |    |
    #     #    M1    |
    #     #     |    |
    #     #     \-- T_new_i^*----1
    #     tmp = tensordot(T_news[i], M1, axes=(0, 0), conj=(1, 0))
    #     #     /-- T_new_i------1---
    #     #     |    |
    #     #    M1    |
    #     #     |    |
    #     #     \-- T_new_i^*----0---
    #     M1 = tensordot(tmp, T_news[i], axes=([0, 2], [1, 0]))

    #     #          1           2
    #     #          |           |
    #     #          |           M2
    #     #          |           |
    #     #      0-- T_old_i^*----
    #     tmp = tensordot(T_olds[-1-i], M2, axes=(2, 0), conj=(1, 0))
    #     #      1-- T_old_i------
    #     #          |           |
    #     #          |           M2
    #     #          |           |
    #     #      0-- T_old_i^*----
    #     M2 = tensordot(tmp, T_olds[-1-i], axes=([1, 2], [1, 2]))

    #     #   1---        --- 2
    #     #       \------/
    #     #       |  M3  |      3
    #     #       /------\      |
    #     #  0---/       \-- T_new_i^*----4
    #     tmp = tensordot(M3, T_news[i], axes=(2, 0), conj=(0, 1))
    #     #   1---        --- T_old_i-----3
    #     #       \------/      |
    #     #       |  M3  |      |
    #     #       /------\      |
    #     #  0---/       \-- T_new_i^*----2
    #     M3 = tensordot(tmp, T_olds[i], axes=([2, 3], [0, 1]))

    #     #   3---T_new_i---        -----0
    #     #         |       \------/
    #     #         |       |  M4  |
    #     #         4       /------\
    #     #            2---/       \-----1
    #     tmp = tensordot(M4, T_news[-1-i], axes=(2, 2))
    #     #  2----T_new_i---        -----0
    #     #         |       \------/
    #     #         |       |  M4  |
    #     #         |       /------\
    #     #  3--T_old_i^*--/       \-----1
    #     M4 = tensordot(tmp, T_olds[-1-i], axes=([2, 4], [2, 1]), conj=(0, 1))

    # M = einsum("ij, kl -> ikjl", M1.transpose(), identity3) + einsum("ij, kl -> ikjl", identity2, M2.transpose()) - M3 - M4
    # M = M.fuse_legs(axes=((2, 3), (0, 1)))
    # s, u = M.eigh(axes=(0, 1))
    # u = u.unfuse_legs(axes=(0,))
    # s_zeros = s <= 1e-10
    # # s_zeros = s <= 1e-14
    # # s_dense = s.to_dense().diag()
    # # print(s_dense[s_dense < 1e-6])
    # modes = u @ s_zeros  # set eigenvectors with non-zero eigenvalues to zero
    # zero_modes = []
    # # collect zero eigenvectors
    # for i in range(modes.get_legs(axes=2).tD[(0,)]):
    #     zero_mode = zeros(
    #         config=config, legs=(leg2.conj(), leg3.conj())
    #     )  # 1 ---sigma--- 2
    #     non_zero = False
    #     for charge_sector in modes.get_blocks_charge():
    #         if charge_sector[-1] == 0:  # the total charge of sigma should be zero
    #             block = modes[charge_sector]
    #             if torch.norm(block[:, :, i]) > 1e-8:
    #                 non_zero = True
    #                 zero_mode.set_block(ts=charge_sector[:-1], val=block[:, :, i])
    #     if non_zero:
    #         zero_modes.append(zero_mode)
    # =======================================

    # ========linear-operator impl===========
    # leg1 = T_news[0].get_legs(axes=0)
    # leg4 = T_olds[-1].get_legs(axes=2)
    leg2 = T_news[-1].get_legs(axes=2) # leg1 = leg2.conj()
    leg3 = T_olds[0].get_legs(axes=0) # leg3 = leg4.conj()

    v0= zeros(config=T_news[0].config, legs=(leg2.conj(), leg3.conj()), n=T_news[0].config.sym.zero())
    _, meta= v0.compress_to_1d(meta=None)


    M1 = diag(eye(config=config, legs=(leg2.conj(), leg2)))
    M2 = diag(eye(config=config, legs=(leg3.conj(), leg3)))

    # The matrices M1 and M2 can be precomputed explicitly with a cost of O(chi^3 D^2)
    for i in range(len(T_olds)):
        #     2    0
        #     |    |
        #    M1    |
        #     |    |
        #     \-- T_new_i^*----1
        tmp = tensordot(T_news[i], M1, axes=(0, 0), conj=(1, 0))
        #     /-- T_new_i------1---
        #     |    |
        #    M1    |
        #     |    |
        #     \-- T_new_i^*----0---
        M1 = tensordot(tmp, T_news[i], axes=([0, 2], [1, 0]))

        #          1           2
        #          |           |
        #          |           M2
        #          |           |
        #      0-- T_old_i^*----
        tmp = tensordot(T_olds[-1-i], M2, axes=(2, 0), conj=(1, 0))
        #      1-- T_old_i------
        #          |           |
        #          |           M2
        #          |           |
        #      0-- T_old_i^*----
        M2 = tensordot(tmp, T_olds[-1-i], axes=([1, 2], [1, 2]))

    M1 = M1.transpose()
    M2 = M2.transpose()

    # take care of negative strides
    to_tensor= lambda x: T_news[0].config.backend.to_tensor(x if np.sum(np.array(x.strides)<0)==0 else x.copy() , dtype=T_news[0].yastn_dtype, device=T_news[0].device)
    to_numpy= lambda x: T_news[0].config.backend.to_numpy(x)

    def mv(v): # Av
        sigma = decompress_from_1d(to_tensor(v), meta)
        sigma3, sigma4 = sigma, sigma

        # Cost: O(chi^3 D^2)
        for i in range(len(T_olds)):
            #           ---[1]-> 0
            #   ------/
            #  sigma3 |      1
            #   ------\      |
            #          [0]-- T_new_i^*----2
            sigma3 = tensordot(sigma3, T_news[i], axes=(0, 0), conj=(0, 1))

            #           -- T_old_i-----1
            #   ------/      |
            #  sigma3 |      |
            #   ------\      |
            #          -- T_new_i^*----0
            sigma3 = tensordot(sigma3, T_olds[i], axes=([0, 1], [0, 1]))

            #   1---T_new_i---
            #         |       \------
            #         |       | sigma4
            #         2       /------
            #            0---/
            sigma4 = tensordot(sigma4, T_news[-1-i], axes=(0, 2))
            #  0----T_new_i---
            #         |       \------
            #         |       | sigma4
            #         |       /------
            #  1--T_old_i^*--/
            sigma4 = tensordot(sigma4, T_olds[-1-i], axes=([0, 2], [2, 1]), conj=(0, 1))

        res = tensordot(M1, sigma, axes=(0, 0)) + tensordot(sigma, M2, axes=(1, 0)) -sigma3 - sigma4
        res_data, res_meta= res.compress_to_1d(meta=meta)
        return to_numpy(res_data)

    A = LinearOperator((v0.size, v0.size), matvec=mv)
    # w, vs= robust_eigsh(A, k=6, which='SA', v0=None, initial_maxiter=v0.size*10, max_maxiter=v0.size*90, ncv=60)
    # zero_v = vs[:, w<1e-8]
    zero_v = null_space(A, batch_k=2, maxiter=v0.size*10, ncv=20)
    zero_modes = [decompress_from_1d(to_tensor(v), meta) for v in zero_v.T]
    # =======================================

    return zero_modes

def fast_env_T_gauge_multi_sites(config, T_olds, T_news):
    #  Generic gauge-fixing from https://arxiv.org/abs/2311.11894
    #
    #   ----T_new_1...T_new_L----sigma--- == ---sigma--T_old_1...T_old_L ----
    #          |          |                               |         |
    #          |          |                               |         |

    # TODO check if T_olds and T_news are compatible
    for To, Tn in zip(T_olds, T_news):
        try:
            mask_needed, _ = _test_axes_match(To, Tn)
        except YastnError:
            raise NoFixedPointError(code=1, message="No fixed point found: T tensors' symmetry sectors change after a CTM step!")
        if mask_needed: # charge sectors missing
            raise NoFixedPointError(code=1, message="No fixed point found: T tensors' symmetry sectors change after a CTM step!")

    leg = T_news[0].get_legs(axes=0)
    eigs_maxiter, ncv = 1000, 40
    def right_leading_eig(Ts, Ms, ncv=40):
        # Compute the leading left eigenvector of the operator
        #   --[leg1]---T_1--... --T_L--[leg2]--
        #               |          |
        #               |          |
        #        ------M_1---...--M_L-------
        v0= zeros(config=Ts[0].config, legs=(leg.conj(), leg), n=Ts[0].config.sym.zero())
        _, meta_L= v0.compress_to_1d(meta=None)

        v1= zeros(config=Ts[0].config, legs=(leg, leg.conj()), n=Ts[0].config.sym.zero())
        _, meta_R= v1.compress_to_1d(meta=None)

        to_tensor= lambda x: Ts[0].config.backend.to_tensor(x if np.sum(np.array(x.strides)<0)==0 else x.copy() , dtype=Ts[0].yastn_dtype, device=Ts[0].device)
        to_numpy= lambda x: Ts[0].config.backend.to_numpy(x)

        def vm(v):
            rho = decompress_from_1d(to_tensor(v.conj()), meta_L)

            # Cost: O(chi^3 D^2)
            for i in range(len(Ts)):
                #           ---[0]-> 0
                #   ------/
                #   rho   |      1
                #   ------\      |
                #          [1]-- M_i-----2
                rho = tensordot(rho, Ms[i], axes=(1, 0), conj=(0, 1))
                #           -- T_i-----0
                #   ------/      |
                #    rho  |      |
                #   ------\      |
                #          -----M_i----1
                rho = tensordot(Ts[i], rho, axes=([0, 1], [0, 1]))

            rho_data, rho_meta= rho.compress_to_1d(meta=meta_L)
            return to_numpy(rho_data).conj()

        def mv(v):
            rho = decompress_from_1d(to_tensor(v), meta_R)
            # Cost: O(chi^3 D^2)
            for i in range(len(Ts)):
                #     1---T_i-----
                #         |       \------
                #         |       | rho
                #         2       /------
                #            0---/
                rho = tensordot(rho, Ts[-1-i], axes=(0, 2))
                #    0----T_i-----
                #         |       \------
                #         |       | rho
                #         |       /------
                #    1----M_i^*--/
                rho = tensordot(rho, Ms[-1-i], axes=([0, 2], [2, 1]), conj=(0, 1))

            rho_data, rho_meta= rho.compress_to_1d(meta=meta_R)
            return to_numpy(rho_data)

        A = LinearOperator((v0.size, v0.size), matvec=mv, rmatvec=vm)
        w, vs = eigs(A, k=1, which='LM', ncv=ncv, maxiter=eigs_maxiter)
        return w, decompress_from_1d(to_tensor(vs[:, 0]), meta_R)

        # svds is slightly worse in precision compared to eigs
        # u, s, vh = svds(A, k=1, which='LM')
        # return s, decompress_from_1d(to_tensor(vh[0, :].conj()), meta_R)

    def normalize_QR(Q, R):
        # make the diagonal entries of R matrix positive
        R_sign = R.diag()
        R_sign._data = R_sign._data.conj()/abs(R_sign._data)
        return Q@R_sign.H, R_sign@R


    retry_cnt, retry_limit = 0, 20
    while True:
        if retry_cnt > retry_limit:
            return []
        # initialize Ms with random tensors
        Ms = []
        for i in range(len(T_news)):
            Ms.append(rand(config=config, legs=T_news[i].get_legs(), n=T_news[i].n))
        try:
            w_old, v_old = right_leading_eig(T_olds, Ms, ncv=ncv)
            w_new, v_new = right_leading_eig(T_news, Ms, ncv=ncv)
        except ArpackNoConvergence:
            ncv = ncv + 20
            eigs_maxiter += 1000
            retry_cnt += 1
            log.log(logging.INFO, f"Eigs Warning: No convergence after {eigs_maxiter} iterations with ncv={ncv:d}. Retrying with a differet M...")
            continue

        Q_old, R_old = qr(v_old, axes=(0, 1), sQ=-v_old.get_signature()[0]) # one in one out R
        Q_new, R_new = qr(v_new, axes=(0, 1), sQ=-v_new.get_signature()[0]) # one in one out R
        Q_old, R_old = normalize_QR(Q_old, R_old)
        Q_new, R_new = normalize_QR(Q_new, R_new)
        sigma = Q_new @ Q_old.H

        # Rerun the procedure with a different random MPS if the leading eigenvalue is not unique
        if (sigma@v_old - v_new).norm(p='inf') < 1e-5:
            break
        retry_cnt += 1

    return [sigma]


def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))

def find_coeff_multi_sites(env_old, env, zero_modes_dict, dtype=torch.complex128, verbose=False):
    Gauge = namedtuple("Gauge", "t l b r")

    phases_ind = {}
    fixed_env = env.copy()
    phase_loss_precomputed = False

    def phase_loss(phases, cs_dict):
        nonlocal phase_loss_precomputed
        sigma_dict = {}
        ind = 0
        exp_phases = np.exp(1j*phases)
        exp_phases = np.concatenate([exp_phases, np.ones(1)])

        loss = torch.zeros(1, dtype=torch.float64)
        if not phase_loss_precomputed:
            for site in env.sites():
                sigma_list = []
                site_ind = env.site2index(site)
                cs = cs_dict[site_ind]
                for i, dirn in enumerate(["t", "l", "b", "r"]):
                    dtype = "complex128" if zero_modes_dict[(site_ind, dirn)][0].dtype is torch.complex128 else "float64"
                    zero_mode = zeros(zero_modes_dict[(site_ind, dirn)][0].config, legs=zero_modes_dict[(site_ind, dirn)][0].get_legs(), dtype=dtype)
                    for j in range(len(cs[i])):
                        zero_mode += cs[i][j] * zero_modes_dict[(site_ind, dirn)][j]
                    # if ind < len(phases):
                    #     # sigma_list.append(zero_mode*torch.exp(1j*phases[ind]))
                    #     # sigma_list.append(zero_mode*exp_phases[ind])
                    # else:
                    #     sigma_list.append(zero_mode) # the last phase can be set to 0
                    phases_ind[(site_ind, dirn)] = ind
                    sigma_list.append(zero_mode)
                    ind += 1
                sigma_dict[site_ind] = Gauge(*sigma_list)

            for site in env_old.sites():
                site_ind = env_old.site2index(site)
                site_t, site_l, site_b, site_r = env_old.site2index(env_old.nn_site(site, "t")), env_old.site2index(env_old.nn_site(site, "l")), env_old.site2index(env_old.nn_site(site, "b")), env_old.site2index(env_old.nn_site(site, "r"))

                sigma1_t, sigma1_l, sigma1_b, sigma1_r = sigma_dict[site_ind].t, sigma_dict[site_ind].l, sigma_dict[site_ind].b, sigma_dict[site_ind].r
                sigma2_l, sigma2_b, sigma2_r, sigma2_t = sigma_dict[site_t].l, sigma_dict[site_l].b, sigma_dict[site_b].r, sigma_dict[site_r].t
                for dirn in ["t", "l", "b", "r"]:
                    # T_old = getattr(env_old[site], dirn)
                    T_new = getattr(env[site], dirn)
                    sigma1 = getattr(sigma_dict[site_ind], dirn)
                    if dirn == "t":
                        sigma2 = getattr(sigma_dict[site_r], dirn)
                    elif dirn == "r":
                        sigma2 = getattr(sigma_dict[site_b], dirn)
                    elif dirn == "b":
                        sigma2 = getattr(sigma_dict[site_l], dirn)
                    elif dirn == "l":
                        sigma2 = getattr(sigma_dict[site_t], dirn)

                    fixed_T = tensordot(
                        tensordot(sigma1, T_new, axes=(0, 0), conj=(1, 0)),
                        sigma2,
                        axes=(2, 0),
                    )
                    setattr(fixed_env[site], dirn, fixed_T.to('cpu'))

                    # loss += (fixed_T - T_old).norm(p='fro')**2


                for dirn in ["tl", "bl", "br", "tr"]:
                    # C_old = getattr(env_old[site], dirn)
                    C_new = getattr(env[site], dirn)
                    if dirn == "tl":
                        fixed_C = tensordot(
                            tensordot(sigma2_l, C_new, axes=(0, 0), conj=(1, 0)),
                            sigma1_t,
                            axes=(1, 0),
                        )
                    if dirn == "bl":
                        fixed_C = tensordot(
                            tensordot(sigma2_b, C_new, axes=(0, 0), conj=(1, 0)),
                            sigma1_l,
                            axes=(1, 0),
                        )
                    if dirn == "br":
                        fixed_C = tensordot(
                            tensordot(sigma2_r, C_new, axes=(0, 0), conj=(1, 0)),
                            sigma1_b,
                            axes=(1, 0),
                        )
                    if dirn == "tr":
                        fixed_C = tensordot(
                            tensordot(sigma2_t, C_new, axes=(0, 0), conj=(1, 0)),
                            sigma1_r,
                            axes=(1, 0),
                        )
                    setattr(fixed_env[site], dirn, fixed_C.to('cpu'))
                    # loss += (fixed_C - C_old).norm(p='fro')**2
            phase_loss_precomputed = True

        for site in env_old.sites():
            site_ind = env_old.site2index(site)
            site_t, site_l, site_b, site_r = env_old.site2index(env_old.nn_site(site, "t")), env_old.site2index(env_old.nn_site(site, "l")), env_old.site2index(env_old.nn_site(site, "b")), env_old.site2index(env_old.nn_site(site, "r"))

            phase1_t, phase1_l, phase1_b, phase1_r = exp_phases[phases_ind[(site_ind, "t")]], exp_phases[phases_ind[(site_ind, "l")]], exp_phases[phases_ind[(site_ind, "b")]], exp_phases[phases_ind[(site_ind, "r")]]
            phase2_l, phase2_b, phase2_r, phase2_t = exp_phases[phases_ind[(site_t, "l")]], exp_phases[phases_ind[(site_l, "b")]], exp_phases[phases_ind[(site_b, "r")]], exp_phases[phases_ind[(site_r, "t")]]
            for dirn in ["t", "l", "b", "r"]:
                if dirn == "t":
                    phase1, phase2 = phase1_t, phase2_t
                elif dirn == "l":
                    phase1, phase2 = phase1_l, phase2_l
                elif dirn == "b":
                    phase1, phase2 = phase1_b, phase2_b
                elif dirn == "r":
                    phase1, phase2 = phase1_r, phase2_r

                fixed_T = getattr(fixed_env[site], dirn)
                T_old = getattr(env_old[site], dirn)
                loss += (phase1.conj()*fixed_T*phase2 - T_old.to('cpu')).norm(p='fro')**2


            for dirn in ["tl", "bl", "br", "tr"]:
                if dirn == "tl":
                    phase1, phase2 = phase1_t, phase2_l
                if dirn == "bl":
                    phase1, phase2 = phase1_l, phase2_b
                if dirn == "br":
                    phase1, phase2 = phase1_b, phase2_r
                if dirn == "tr":
                    phase1, phase2 = phase1_r, phase2_t

                fixed_C = getattr(fixed_env[site], dirn)
                C_old = getattr(env_old[site], dirn)
                loss += (phase2.conj()*fixed_C*phase1 - C_old.to('cpu')).norm(p='fro')**2

        return loss

    def unitary_loss(cs_data):
        loss = 0.0
        # assemble unitaries
        ind = 0
        for site in env.sites():
            site_ind = env.site2index(site)
            for i, dirn in enumerate(("t", "l", "b", "r")):
                legs = zero_modes_dict[(site_ind, dirn)][0].get_legs()
                identity = diag(eye(config=zero_modes_dict[(site_ind, dirn)][0].config, legs=(legs[0], legs[0].conj())))
                unitary = zeros(config=zero_modes_dict[(site_ind, dirn)][0].config, legs=legs, dtype='complex128' if dtype is torch.complex128 else 'float64')
                for j in range(len(zero_modes_dict[(site_ind, dirn)])):
                    unitary = unitary + cs_data[ind] * zero_modes_dict[(site_ind, dirn)][j]
                    ind += 1
                loss += ((tensordot(unitary, unitary, axes=(1, 1), conj=(0, 1)) - identity).norm(p='fro')**2/ identity.norm(p='fro')**2).item().real
        return loss

    if dtype == torch.complex128:
        cs_data = []
        ind2row = {}
        start = 0
        for site in env.sites():
            site_ind = env.site2index(site)
            num = 0
            for dirn in ("t", "l", "b", "r"):
                cs_data.append(np.ones(len(zero_modes_dict[(site_ind, dirn)]), dtype=np.complex128))
                num += len(zero_modes_dict[(site_ind, dirn)])
            ind2row[site_ind] = slice(start, start+num)
            start += num
        cs_data = np.concatenate(cs_data)
        res = minimize(fun=lambda z: unitary_loss(real_to_complex(z)), x0=complex_to_real(cs_data),
                       jac='3-point', method='SLSQP', options={"eps":1e-9, "ftol":1e-14})
        res = real_to_complex(res.x)
    else:
        cs_data = []
        ind2row = {}
        start = 0
        for site in env.sites():
            site_ind = env.site2index(site)
            num = 0
            for dirn in ("t", "l", "b", "r"):
                cs_data.append(np.zeros(len(zero_modes_dict[(site_ind, dirn)]), dtype=np.float64))
                num += len(zero_modes_dict[(site_ind, dirn)])
            ind2row[site_ind] = slice(start, start+num)
            start += num
        cs_data = np.concatenate(cs_data)
        cs_data += np.random.rand(*cs_data.shape)*0.1
        res = minimize(fun=unitary_loss, x0=cs_data,
                       jac='3-point', method='SLSQP', options={"eps":1e-9, "ftol":1e-14})
        res = res.x

    cs_dict = {}
    ind = 0
    for site in env.sites():
        cs = []
        site_ind = env.site2index(site)
        for i, dirn in enumerate(("t", "l", "b", "r")):
            cs.append(np.array(res[ind:ind+len(zero_modes_dict[(site_ind, dirn)])]))
            ind += len(zero_modes_dict[(site_ind, dirn)])
        cs_dict[env.site2index(site)] = cs

    print("fix phase")
    start = time.time()
    phases = np.random.rand(4*len(cs_dict)-1)
    res = minimize(fun=phase_loss, x0=phases, args=(cs_dict,), method='cg', tol=1e-14)
    phases = res.x
    end = time.time()
    if verbose:
        print(f"scipy takes {end-start:.1f}s with loss fn ", res.fun)
        print("phases: ", phases)

    # assemble the coefficients dict
    if isinstance(phases, torch.Tensor):
        exp_phases = torch.exp(1j*phases).to(dtype)
    else:
        if dtype == torch.complex128:
            exp_phases = np.exp(1j*phases).astype(np.complex128)
        else:
            exp_phases = np.exp(1j*phases).astype(np.float64)
    ind = 0
    for site in env.sites():
        site_ind = env.site2index(site)
        for i, dirn in enumerate(("t", "l", "b", "r")):
            for j in range(len(zero_modes_dict[(site_ind, dirn)])):
                if ind < len(phases):
                    # cs_dict[site_ind][i][j] = cs_dict[site_ind][i][j] * np.exp(1j*phases[ind])
                    cs_dict[site_ind][i][j] = cs_dict[site_ind][i][j] * exp_phases[ind]
            ind += 1
        cs_dict[site_ind] = torch.utils.checkpoint.detach_variable(tuple(cs_dict[site_ind]))
    return cs_dict

def find_gauge_multi_sites(env_old, env, verbose=False):
    Gauge = namedtuple("Gauge", "t l b r")
    zero_modes_dict = {}
    sigma_dict = {}
    for site in env.sites():
        site_ind = env.site2index(site)
        for k, (N, d) in zip(["t", "l", "b", "r"], [(env.Ny, "r"), (env.Nx, "t"), (env.Ny, "l"), (env.Nx, "b")]):
            tmp_site = site
            T_olds, T_news = [], []
            for _ in range(N):
                T_olds.append(getattr(env_old[tmp_site], k))
                T_news.append(getattr(env[tmp_site], k))
                tmp_site = env.nn_site(tmp_site, d)

            zero_modes = fast_env_T_gauge_multi_sites(env.psi.config, T_olds, T_news)
            # if len(zero_modes) == 0: # fast method fails
            #     zero_modes = env_T_gauge_multi_sites(env.psi.config, T_olds, T_news)
            if len(zero_modes) == 0:
                return None
            zero_modes_dict[(site_ind, k)] = zero_modes

    cs_dict = find_coeff_multi_sites(env_old, env, zero_modes_dict, dtype=zero_modes_dict[(0, "t")][0].dtype, verbose=True)
    def is_diagonal(matrix, tol=1e-6):
        print(torch.diag(matrix))
        off_diag = matrix - torch.diag(torch.diag(matrix))  # Remove diagonal elements
        return torch.all(torch.abs(off_diag) < tol)  # Check if all off-diagonal elements are near zero
    for site in env.sites():
        sigma_list = []
        site_ind = env.site2index(site)
        for i, dirn in enumerate(["t", "l", "b", "r"]):
            dtype = "complex128" if zero_modes_dict[(site_ind, dirn)][0].dtype is torch.complex128 else "float64"
            zero_mode = zeros(zero_modes_dict[(site_ind, dirn)][0].config, legs=zero_modes_dict[(site_ind, dirn)][0].get_legs(), dtype=dtype)
            cs = cs_dict[site_ind]
            for j in range(len(cs[i])):
                zero_mode += cs[i][j] * zero_modes_dict[(site_ind, dirn)][j]
            sigma_list.append(zero_mode)

        for sigma in sigma_list:
            sigma._data.detach_()
        sigma_dict[site_ind] = Gauge(*sigma_list)

    for site in env.sites():
        site_ind = env.site2index(site)
        site_t, site_l, site_b, site_r = env.site2index(env.nn_site(site, "t")), env.site2index(env.nn_site(site, "l")), env.site2index(env.nn_site(site, "b")), env.site2index(env.nn_site(site, "r"))
        for dirn in ["t", "l", "b", "r"]:
            sigma1 = getattr(sigma_dict[site_ind], dirn)
            if dirn == "t":
                sigma2 = getattr(sigma_dict[site_r], dirn)
            elif dirn == "r":
                sigma2 = getattr(sigma_dict[site_b], dirn)
            elif dirn == "b":
                sigma2 = getattr(sigma_dict[site_l], dirn)
            elif dirn == "l":
                sigma2 = getattr(sigma_dict[site_t], dirn)

            fixed_t = tensordot(
                tensordot(sigma1, getattr(env[site], dirn), axes=(0, 0), conj=(1, 0)),
                sigma2,
                axes=(2, 0),
            )
            T_old = getattr(env_old[site], dirn)
            if verbose:
                print("T diff:", (fixed_t - T_old).norm() / T_old.norm())

        sigma1_t, sigma1_l, sigma1_b, sigma1_r = sigma_dict[site_ind].t, sigma_dict[site_ind].l, sigma_dict[site_ind].b, sigma_dict[site_ind].r
        sigma2_l, sigma2_b, sigma2_r, sigma2_t = sigma_dict[site_t].l, sigma_dict[site_l].b, sigma_dict[site_b].r, sigma_dict[site_r].t
        for dirn in ["tl", "bl", "br", "tr"]:
            C_new = getattr(env[site], dirn)
            if dirn == "tl":
                fixed_C = tensordot(
                    tensordot(sigma2_l, C_new, axes=(0, 0), conj=(1, 0)),
                    sigma1_t,
                    axes=(1, 0),
                )
            if dirn == "bl":
                fixed_C = tensordot(
                    tensordot(sigma2_b, C_new, axes=(0, 0), conj=(1, 0)),
                    sigma1_l,
                    axes=(1, 0),
                )
            if dirn == "br":
                fixed_C = tensordot(
                    tensordot(sigma2_r, C_new, axes=(0, 0), conj=(1, 0)),
                    sigma1_b,
                    axes=(1, 0),
                )
            if dirn == "tr":
                fixed_C = tensordot(
                    tensordot(sigma2_t, C_new, axes=(0, 0), conj=(1, 0)),
                    sigma1_r,
                    axes=(1, 0),
                )

            C_old = getattr(env_old[site], dirn)
            if verbose:
                print("C diff:", (fixed_C - C_old).norm() / C_old.norm())
    return sigma_dict

def fp_ctmrg(env: EnvCTM, \
            ctm_opts_fwd : dict= {'method': "2site", 'use_qr': False, 'corner_tol': 1e-8, 'max_sweeps': 100, 
                'svd_policy': 'fullrank', 'opts_svd': {}, 'D_block': None, 'verbosity': 0}, \
            ctm_opts_fp: dict= {'svd_policy': 'fullrank'}):
    r"""
    Compute the fixed-point environment for the given state using CTMRG.
    First, run CTMRG until convergence then find the gauge transformation guaranteeing element-wise
    convergence of the environment tensors.

    Args:
        env (EnvCTM): CTM environment
        ctm_opts_fwd (dict): Options for forward CTMRG convergence.
        ctm_opts_fp (dict): Options for fixing the gauge transformation.

    Returns:
        EnvCTM: Environment at fixed point.
        Sequence[Tensor]: raw environment data for the backward pass.
    """
    raw_peps_params= tuple( env.psi.ket[s]._data for s in env.psi.ket.sites() )
    return FixedPoint.apply(env, ctm_opts_fwd, ctm_opts_fp, *raw_peps_params)

@torch.no_grad()
def fp_ctm_conv_check(env, history, corner_tol, verbosity=0):
    converged, max_dsv, history = ctm_conv_corner_spec(env, history, corner_tol)
    if verbosity>2:
        log.log(logging.INFO, f"CTM iter {len(history)} |delta_C| {max_dsv}")
    return converged, history

class FixedPoint(torch.autograd.Function):
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
    def fixed_point_iter(env_in, sigma_dict, ctm_opts_fp, slices, env_data, psi_data):
        refill_env(env_in, env_data, slices)
        refill_state(env_in.psi.ket, psi_data)
        env_in.update_(
            **ctm_opts_fp
        )

        for site in env_in.sites():
            site_ind = env_in.site2index(site)
            site_t, site_l, site_b, site_r = env_in.site2index(env_in.nn_site(site, "t")), env_in.site2index(env_in.nn_site(site, "l")), env_in.site2index(env_in.nn_site(site, "b")), env_in.site2index(env_in.nn_site(site, "r"))
            for dirn in ["t", "l", "b", "r"]:
                sigma1 = getattr(sigma_dict[site_ind], dirn)
                if dirn == "t":
                    sigma2 = getattr(sigma_dict[site_r], dirn)
                elif dirn == "r":
                    sigma2 = getattr(sigma_dict[site_b], dirn)
                elif dirn == "b":
                    sigma2 = getattr(sigma_dict[site_l], dirn)
                elif dirn == "l":
                    sigma2 = getattr(sigma_dict[site_t], dirn)

                fixed_t = tensordot(
                    tensordot(sigma1, getattr(env_in[site], dirn), axes=(0, 0), conj=(1, 0)),
                    sigma2,
                    axes=(2, 0),
                )
                setattr(env_in[site], dirn, fixed_t)

            sigma1_t, sigma1_l, sigma1_b, sigma1_r = sigma_dict[site_ind].t, sigma_dict[site_ind].l, sigma_dict[site_ind].b, sigma_dict[site_ind].r
            sigma2_l, sigma2_b, sigma2_r, sigma2_t = sigma_dict[site_t].l, sigma_dict[site_l].b, sigma_dict[site_b].r, sigma_dict[site_r].t
            # sigma_t, sigma_l, sigma_b, sigma_r = sigma_dict[site].t, sigma_dict[site].l, sigma_dict[site].b, sigma_dict[site].r
            for dirn in ["tl", "bl", "br", "tr"]:
                C_new = getattr(env_in[site], dirn)
                if dirn == "tl":
                    fixed_C = tensordot(
                        tensordot(sigma2_l, C_new, axes=(0, 0), conj=(1, 0)),
                        sigma1_t,
                        axes=(1, 0),
                    )
                if dirn == "bl":
                    fixed_C = tensordot(
                        tensordot(sigma2_b, C_new, axes=(0, 0), conj=(1, 0)),
                        sigma1_l,
                        axes=(1, 0),
                    )
                if dirn == "br":
                    fixed_C = tensordot(
                        tensordot(sigma2_r, C_new, axes=(0, 0), conj=(1, 0)),
                        sigma1_b,
                        axes=(1, 0),
                    )
                if dirn == "tr":
                    fixed_C = tensordot(
                        tensordot(sigma2_t, C_new, axes=(0, 0), conj=(1, 0)),
                        sigma1_r,
                        axes=(1, 0),
                    )
                setattr(env_in[site], dirn, fixed_C)

        env_out_data, slices = env_raw_data(env_in)
        return (env_out_data, )

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
        converged, conv_history = False, []
        
        ctm_itr= env.ctmrg_(iterator_step=1, method=method,  max_sweeps=max_sweeps, 
                   svd_policy=svd_policy, opts_svd=opts_svd, D_block=D_block,
                   corner_tol=None, **kwargs)

        for sweep in range(max_sweeps):
            t0 = time.perf_counter()
            ctm_out_info= next(ctm_itr)
            t_ctm += time.perf_counter()-t0

            t0 = time.perf_counter()
            converged, conv_history = fp_ctm_conv_check(env, conv_history, corner_tol, \
                                                        verbosity= kwargs.get('verbosity',0))
            t_check += time.perf_counter()-t0

            if converged: 
                log.info(f"CTM converged sweeps {sweep+1} history {[r['max_dsv'] for r in conv_history]}.")
                break

        return env, converged, conv_history, t_ctm, t_check

    @staticmethod
    def forward(ctx, env: EnvCTM, ctm_opts_fwd : dict, ctm_opts_fp: dict, *state_params):
        r"""
        Compute the fixed-point environment for the given state using CTMRG.
        First, run CTMRG until convergence then find the gauge transformation guaranteeing element-wise
        convergence of the environment tensors.

        Args:
            env (EnvCTM): Current environment to converge.
            ctm_opts_fwd (dict): Options for forward CTMRG convergence.
            ctm_opts_fp (dict): Options for fixing the gauge transformation.
            state_params (Sequence[Tensor]): tensors of underlying Peps state

        Returns:
            EnvCTM: Environment at fixed point.
            Sequence[Tensor]: raw environment data for the backward pass.
        """

        ctm_env_out, converged, *FixedPoint.ctm_log, FixedPoint.t_ctm, FixedPoint.t_check = FixedPoint.get_converged_env(
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

        sigma_dict = find_gauge_multi_sites(env_converged, ctm_env_out, verbose=True)
        if sigma_dict is None:
            raise NoFixedPointError(code=1, message="No fixed point found: fail to find the gauge matrix!")

        env_data, env_meta = env_converged.compress_env_1d()
        ctx.save_for_backward(*env_data)
        ctx.env_meta = env_meta
        ctx.ctm_opts_fp = _ctm_opts_fp

        # TODO: use save_for_backward
        ctx.sigma_dict = sigma_dict
        env_1d, env_slices= env_raw_data(env_converged)

        return env_converged, env_slices, env_1d

    @staticmethod
    def backward(ctx, none0, none1, *grad_env):
        grads = grad_env
        dA = grad_env

        # env_data = torch.utils.checkpoint.detach_variable(ctx.saved_tensors)[0] # only one element in the tuple
        env_data = ctx.saved_tensors
        env= decompress_env_1d(env_data, ctx.env_meta)
        psi_data = tuple(env.psi.ket[s]._data for s in env.psi.ket.sites())
        _env_ts, _env_slices= env_raw_data(env)

        prev_grad_tmp = None
        diff_ave = None
        # Compute vjp only
        with torch.enable_grad():
            _, dfdC_vjp = torch.func.vjp(lambda x: FixedPoint.fixed_point_iter(env, ctx.sigma_dict, ctx.ctm_opts_fp, _env_slices, x, psi_data), _env_ts)
            _, dfdA_vjp = torch.func.vjp(lambda x: FixedPoint.fixed_point_iter(env, ctx.sigma_dict, ctx.ctm_opts_fp, _env_slices, _env_ts, x), psi_data)
        # fixed_point_iter changes the data of psi, so we need to refill the state to recover the previous state
        refill_state(env.psi.ket, psi_data)

        alpha = 0.4
        for step in range(ctx.ctm_opts_fp['max_sweeps']):
            grads = dfdC_vjp(grads)
            # with torch.enable_grad():
            #     grads = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data, psi_data), env_data, grad_outputs=grads)
            if all([torch.norm(grad, p=torch.inf) < ctx.ctm_opts_fp["corner_tol"] for grad in grads]):
                break
            else:
                dA = tuple(dA[i] + grads[i] for i in range(len(grads)))
            # for grad in grads:
            #     print(torch.norm(grad, p=torch.inf))

            if step % 1 == 0:
                # with torch.enable_grad():
                #     grad_tmp = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data, psi_data), psi_data, grad_outputs=dA)
                grad_tmp = torch.cat(dfdA_vjp(dA)[0])
                if prev_grad_tmp is not None:
                    grad_diff = torch.norm(grad_tmp[0] - prev_grad_tmp[0])
                    print("full grad diff", grad_diff)
                    if grad_diff < ctx.ctm_opts_fp["corner_tol"]:
                        # print("The norm of the full grad diff is below 1e-10.")
                        log.log(logging.INFO, f"Fixed_pt: The norm of the full grad diff is below {ctx.ctm_opts_fp['corner_tol']}.")
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
        # with torch.enable_grad():
        #     dA = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data, psi_data), psi_data, grad_outputs=dA)

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

        return None, None, None, *dA
