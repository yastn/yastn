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
import time, logging
from typing import Mapping
import torch
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh, eigs, ArpackNoConvergence
# import primme

import torch.utils.checkpoint


from .rdm import *
from ._env_ctm import ctm_conv_corner_spec
from ._env_dataclasses import Gauge
from .._geometry import Lattice
from ....tensor import Tensor, zeros, eye, tensordot, diag, rand, qr, split_data_and_meta, combine_data_and_meta
log = logging.getLogger(__name__)


class NoFixedPointError(Exception):
    def __init__(self, code, message=None):
        if message is None:
            message = "No fixed point found"
        self.message = message
        super().__init__(self.message)  # Pass message to the base class
        self.code = code          				 # Add a custom attribute for error codes
log = logging.getLogger("FixedPoint")

class EnvGauge():
    def __init__(self, geometry):
        r"""
        Gauge matrices used in fixed-point AD.
        """
        self.geometry = geometry
        for name in ["dims", "sites", "nn_site", "site2index", "Nx", "Ny"]:
            setattr(self, name, getattr(self.geometry, name))
        self.gauge = Lattice(self.geometry, objects={site: Gauge() for site in self.sites()})

    def __getitem__(self, site):
        return self.gauge[site]

    def __setitem__(self, site, obj):
        self.gauge[site] = obj

    def to_dict(self, level=2):
        return {'type': type(self).__name__,
                'dict_ver': 1,
                'gauge': self.gauge.to_dict(level=level)}

    @classmethod
    def from_dict(cls, d, config=None):
        if d['dict_ver'] == 1:
            if cls.__name__ != d['type']:
                raise YastnError(f"{cls.__name__} does not match d['type'] == {d['type']}")
            gauge = Lattice.from_dict(d['gauge'], config=config)
            g = cls(gauge.geometry)
            g.gauge = gauge
            return g

def _concat_data(env_data):
    '''
    Combine all env raw tensors into a 1d tensor.
    '''
    slice_list = []
    numel = 0
    for t in env_data:
        slice_list.append((numel, len(t)))
        numel += len(t)
    return torch.cat(env_data), slice_list

def _split_data(data1d, slice_list):
    '''
    Recover a tuple of tensors from a 1d tensor and a given slice_list.
    '''
    env_data = tuple(torch.narrow(data1d, 0, *s) for s in slice_list )
    return env_data

def _assemble_dict_from_1d(meta, data1d, slices):
    ts = _split_data(data1d, slices)
    return combine_data_and_meta(ts, meta)

def extract_dsv_t(t, history):
    max_dsv = max((history[t][k] - history[t+1][k]).norm().item() for k in history[t]) if len(history)>t else float('Nan')
    return max_dsv

def extract_dsv(history):
    max_dsv_history = [extract_dsv(t, history) for t in range(len(history) - 1)]
    return max_dsv_history

def running_ave(history, window_size=5):
    cumsum = np.cumsum(np.insert(history, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

# def null_space(A, batch_k=2, maxiter=None, ncv=None):
#     '''
#     Run primme.eigsh sequentially to find the null space of A.

#     Parameters:
#     - A: sparse matrix (hermitian)
#     - batch_k: number of eigenvalues to compute in each run of primme.eigsh
#     - maxiter: maxiter in primme.eigsh
#     - ncv: ncv in primme.eigsh

#     returns:
#     - zero_vs: null space of A
#     '''
#     maxiter = maxiter if maxiter is not None else A.shape[0] * 10
#     zero_vs = np.empty((A.shape[0], 0), dtype=A.dtype)
#     while True:
#         # find eigenvectors that are orthogonal to the current zero_vs
#         w, vs = primme.eigsh(A, k=batch_k, which='SA', v0=None, maxiter=maxiter, ncv=ncv, lock=zero_vs, method="PRIMME_DYNAMIC")
#         zero_vs = np.hstack((zero_vs, vs[:, w < 1e-8]))
#         if len(w[w<1e-8]) < len(w): # if non-zero eigenvalues are found, stop
#             break
#     return zero_vs


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

# def env_T_gauge_multi_sites(config, T_olds, T_news):
#     #  Robust gauge-fixing from https://arxiv.org/abs/2311.11894
#     #
#     #   --[leg1]--T_new_1...T_new_L--[leg2]--sigma--- == ---sigma--[leg3]---T_old_1...T_old_L ---[leg4]---
#     #               |          |                                              |         |
#     #               |          |                                              |         |

#     # ========full-matrix impl===========
#     # leg1 = T_news[0].get_legs(axes=0)
#     # leg2 = T_news[-1].get_legs(axes=2)
#     # leg3 = T_olds[0].get_legs(axes=0)
#     # leg4 = T_olds[-1].get_legs(axes=2)
#     # # T_old = T_old.transpose(axes=(2, 1, 0)) # exchange the left and right indices

#     # identity1 = diag(eye(config=config, legs=(leg1, leg1.conj())))
#     # identity2 = diag(eye(config=config, legs=(leg2, leg2.conj())))
#     # identity3 = diag(eye(config=config, legs=(leg3, leg3.conj())))
#     # identity4 = diag(eye(config=config, legs=(leg4, leg4.conj())))

#     # M1, M2 = identity1, identity4
#     # M3, M4 = einsum("ji, kl -> ikjl", identity1, identity3), einsum("ij, lk -> ikjl", identity2, identity4)

#     # for i in range(len(T_olds)):
#     #     #     2    0
#     #     #     |    |
#     #     #    M1    |
#     #     #     |    |
#     #     #     \-- T_new_i^*----1
#     #     tmp = tensordot(T_news[i], M1, axes=(0, 0), conj=(1, 0))
#     #     #     /-- T_new_i------1---
#     #     #     |    |
#     #     #    M1    |
#     #     #     |    |
#     #     #     \-- T_new_i^*----0---
#     #     M1 = tensordot(tmp, T_news[i], axes=([0, 2], [1, 0]))

#     #     #          1           2
#     #     #          |           |
#     #     #          |           M2
#     #     #          |           |
#     #     #      0-- T_old_i^*----
#     #     tmp = tensordot(T_olds[-1-i], M2, axes=(2, 0), conj=(1, 0))
#     #     #      1-- T_old_i------
#     #     #          |           |
#     #     #          |           M2
#     #     #          |           |
#     #     #      0-- T_old_i^*----
#     #     M2 = tensordot(tmp, T_olds[-1-i], axes=([1, 2], [1, 2]))

#     #     #   1---        --- 2
#     #     #       \------/
#     #     #       |  M3  |      3
#     #     #       /------\      |
#     #     #  0---/       \-- T_new_i^*----4
#     #     tmp = tensordot(M3, T_news[i], axes=(2, 0), conj=(0, 1))
#     #     #   1---        --- T_old_i-----3
#     #     #       \------/      |
#     #     #       |  M3  |      |
#     #     #       /------\      |
#     #     #  0---/       \-- T_new_i^*----2
#     #     M3 = tensordot(tmp, T_olds[i], axes=([2, 3], [0, 1]))

#     #     #   3---T_new_i---        -----0
#     #     #         |       \------/
#     #     #         |       |  M4  |
#     #     #         4       /------\
#     #     #            2---/       \-----1
#     #     tmp = tensordot(M4, T_news[-1-i], axes=(2, 2))
#     #     #  2----T_new_i---        -----0
#     #     #         |       \------/
#     #     #         |       |  M4  |
#     #     #         |       /------\
#     #     #  3--T_old_i^*--/       \-----1
#     #     M4 = tensordot(tmp, T_olds[-1-i], axes=([2, 4], [2, 1]), conj=(0, 1))

#     # M = einsum("ij, kl -> ikjl", M1.transpose(), identity3) + einsum("ij, kl -> ikjl", identity2, M2.transpose()) - M3 - M4
#     # M = M.fuse_legs(axes=((2, 3), (0, 1)))
#     # s, u = M.eigh(axes=(0, 1))
#     # u = u.unfuse_legs(axes=(0,))
#     # s_zeros = s <= 1e-10
#     # # s_zeros = s <= 1e-14
#     # # s_dense = s.to_dense().diag()
#     # # print(s_dense[s_dense < 1e-6])
#     # modes = u @ s_zeros  # set eigenvectors with non-zero eigenvalues to zero
#     # zero_modes = []
#     # # collect zero eigenvectors
#     # for i in range(modes.get_legs(axes=2).tD[(0,)]):
#     #     zero_mode = zeros(
#     #         config=config, legs=(leg2.conj(), leg3.conj())
#     #     )  # 1 ---sigma--- 2
#     #     non_zero = False
#     #     for charge_sector in modes.get_blocks_charge():
#     #         if charge_sector[-1] == 0:  # the total charge of sigma should be zero
#     #             block = modes[charge_sector]
#     #             if torch.norm(block[:, :, i]) > 1e-8:
#     #                 non_zero = True
#     #                 zero_mode.set_block(ts=charge_sector[:-1], val=block[:, :, i])
#     #     if non_zero:
#     #         zero_modes.append(zero_mode)
#     # =======================================

#     # ========linear-operator impl===========
#     # leg1 = T_news[0].get_legs(axes=0)
#     # leg4 = T_olds[-1].get_legs(axes=2)
#     leg2 = T_news[-1].get_legs(axes=2) # leg1 = leg2.conj()
#     leg3 = T_olds[0].get_legs(axes=0) # leg3 = leg4.conj()

#     v0= zeros(config=T_news[0].config, legs=(leg2.conj(), leg3.conj()), n=T_news[0].config.sym.zero())
#     _, meta= v0.compress_to_1d(meta=None)


#     M1 = diag(eye(config=config, legs=(leg2.conj(), leg2)))
#     M2 = diag(eye(config=config, legs=(leg3.conj(), leg3)))

#     # The matrices M1 and M2 can be precomputed explicitly with a cost of O(chi^3 D^2)
#     for i in range(len(T_olds)):
#         #     2    0
#         #     |    |
#         #    M1    |
#         #     |    |
#         #     \-- T_new_i^*----1
#         tmp = tensordot(T_news[i], M1, axes=(0, 0), conj=(1, 0))
#         #     /-- T_new_i------1---
#         #     |    |
#         #    M1    |
#         #     |    |
#         #     \-- T_new_i^*----0---
#         M1 = tensordot(tmp, T_news[i], axes=([0, 2], [1, 0]))

#         #          1           2
#         #          |           |
#         #          |           M2
#         #          |           |
#         #      0-- T_old_i^*----
#         tmp = tensordot(T_olds[-1-i], M2, axes=(2, 0), conj=(1, 0))
#         #      1-- T_old_i------
#         #          |           |
#         #          |           M2
#         #          |           |
#         #      0-- T_old_i^*----
#         M2 = tensordot(tmp, T_olds[-1-i], axes=([1, 2], [1, 2]))

#     M1 = M1.transpose()
#     M2 = M2.transpose()

#     # take care of negative strides
#     to_tensor= lambda x: T_news[0].config.backend.to_tensor(x if np.sum(np.array(x.strides)<0)==0 else x.copy() , dtype=T_news[0].yastn_dtype, device=T_news[0].device)
#     to_numpy= lambda x: T_news[0].config.backend.to_numpy(x)

#     def mv(v): # Av
#         sigma = decompress_from_1d(to_tensor(v), meta)
#         sigma3, sigma4 = sigma, sigma

#         # Cost: O(chi^3 D^2)
#         for i in range(len(T_olds)):
#             #           ---[1]-> 0
#             #   ------/
#             #  sigma3 |      1
#             #   ------\      |
#             #          [0]-- T_new_i^*----2
#             sigma3 = tensordot(sigma3, T_news[i], axes=(0, 0), conj=(0, 1))

#             #           -- T_old_i-----1
#             #   ------/      |
#             #  sigma3 |      |
#             #   ------\      |
#             #          -- T_new_i^*----0
#             sigma3 = tensordot(sigma3, T_olds[i], axes=([0, 1], [0, 1]))

#             #   1---T_new_i---
#             #         |       \------
#             #         |       | sigma4
#             #         2       /------
#             #            0---/
#             sigma4 = tensordot(sigma4, T_news[-1-i], axes=(0, 2))
#             #  0----T_new_i---
#             #         |       \------
#             #         |       | sigma4
#             #         |       /------
#             #  1--T_old_i^*--/
#             sigma4 = tensordot(sigma4, T_olds[-1-i], axes=([0, 2], [2, 1]), conj=(0, 1))

#         res = tensordot(M1, sigma, axes=(0, 0)) + tensordot(sigma, M2, axes=(1, 0)) -sigma3 - sigma4
#         res_data, res_meta= res.compress_to_1d(meta=meta)
#         return to_numpy(res_data)

#     A = LinearOperator((v0.size, v0.size), matvec=mv)
#     # w, vs= robust_eigsh(A, k=6, which='SA', v0=None, initial_maxiter=v0.size*10, max_maxiter=v0.size*90, ncv=60)
#     # zero_v = vs[:, w<1e-8]
#     zero_v = null_space(A, batch_k=2, maxiter=v0.size*10, ncv=20)
#     zero_modes = [decompress_from_1d(to_tensor(v), meta) for v in zero_v.T]
#     # =======================================

#     return zero_modes


def fast_env_T_gauge_multi_sites(config, T_olds, T_news):
    #  Generic gauge-fixing from https://arxiv.org/abs/2311.11894
    #
    #   ----T_new_1...T_new_L----sigma--- == ---sigma--T_old_1...T_old_L ----
    #          |          |                               |         |
    #          |          |                               |         |

    # TODO check if T_olds and T_news are compatible
    for To, Tn in zip(T_olds, T_news):
        if [l.tD for l in To.get_legs(axes=(0, 2))] != [l.tD for l in Tn.get_legs(axes=(0, 2))]:
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
        d_L = v0.to_dict(level=0)
        _, meta_L = split_data_and_meta(d_L)

        v1= zeros(config=Ts[0].config, legs=(leg, leg.conj()), n=Ts[0].config.sym.zero())
        d_R = v1.to_dict(level=0)
        _, meta_R = split_data_and_meta(d_R)

        to_tensor= lambda x: Ts[0].config.backend.to_tensor(x if np.sum(np.array(x.strides)<0)==0 else x.copy() , dtype=Ts[0].yastn_dtype, device=Ts[0].device)

        def vm(v):
            rho_d = combine_data_and_meta((to_tensor(v.conj()),), meta_L)
            rho = Tensor.from_dict(rho_d)

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

            # rho_data, rho_meta= rho.compress_to_1d(meta=meta_L)
            d_rho = rho.to_dict(level=2)
            rho_data, _ = split_data_and_meta(d_rho)
            return rho_data[0].conj()

        def mv(v):
            rho_d = combine_data_and_meta((to_tensor(v), ), meta_R)
            rho = Tensor.from_dict(rho_d)

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

            # rho_data, rho_meta= rho.compress_to_1d(meta=meta_R)
            d_rho = rho.to_dict(level=2)
            rho_data, _ = split_data_and_meta(d_rho)
            return rho_data[0]

        A = LinearOperator((v0.size, v0.size), matvec=mv, rmatvec=vm)
        w, vs = eigs(A, k=1, which='LM', ncv=ncv, maxiter=eigs_maxiter)
        return w, Tensor.from_dict(combine_data_and_meta((to_tensor(vs[:, 0]),), meta_R))

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


def compute_env_gauge_product(env, zero_modes_dict, cs_dict):
    # Given sigma matrices formed by cs_dict and zero_modes_dict, compute sigma-conjugated env tensors.
    # env_gauge, phases_ind = {}, {}
    env_gauge, phases_ind = EnvGauge(env.geometry), {}
    ind = 0
    fixed_env = env.copy()
    for site in env.sites():
        sigma_list = []
        site_ind = env.site2index(site)
        cs = cs_dict[site_ind]
        for i, dirn in enumerate(["t", "l", "b", "r"]):
            dtype = "complex128" if zero_modes_dict[(site_ind, dirn)][0].dtype is torch.complex128 else "float64"
            zero_mode = zeros(zero_modes_dict[(site_ind, dirn)][0].config, legs=zero_modes_dict[(site_ind, dirn)][0].get_legs(), dtype=dtype)
            for j in range(len(cs[i])):
                zero_mode += cs[i][j] * zero_modes_dict[(site_ind, dirn)][j]
            phases_ind[(site_ind, dirn)] = ind
            sigma_list.append(zero_mode)
            ind += 1
        env_gauge[site] = Gauge(*sigma_list)

    for site in env.sites():
        # site_ind = env.site2index(site)
        # site_t, site_l, site_b, site_r = env.site2index(env.nn_site(site, "t")), env.site2index(env.nn_site(site, "l")), env.site2index(env.nn_site(site, "b")), env.site2index(env.nn_site(site, "r"))

        site_t, site_l, site_b, site_r = env.nn_site(site, "t"), env.nn_site(site, "l"), env.nn_site(site, "b"), env.nn_site(site, "r")

        sigma1_t, sigma1_l, sigma1_b, sigma1_r = env_gauge[site].t, env_gauge[site].l, env_gauge[site].b, env_gauge[site].r
        sigma2_l, sigma2_b, sigma2_r, sigma2_t = env_gauge[site_t].l, env_gauge[site_l].b, env_gauge[site_b].r, env_gauge[site_r].t
        for dirn in ["t", "l", "b", "r"]:
            T_new = getattr(env[site], dirn)
            sigma1 = getattr(env_gauge[site], dirn)
            if dirn == "t":
                sigma2 = getattr(env_gauge[site_r], dirn)
            elif dirn == "r":
                sigma2 = getattr(env_gauge[site_b], dirn)
            elif dirn == "b":
                sigma2 = getattr(env_gauge[site_l], dirn)
            elif dirn == "l":
                sigma2 = getattr(env_gauge[site_t], dirn)

            fixed_T = tensordot(
                tensordot(sigma1, T_new, axes=(0, 0), conj=(1, 0)),
                sigma2,
                axes=(2, 0),
            )
            setattr(fixed_env[site], dirn, fixed_T)

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
            setattr(fixed_env[site], dirn, fixed_C)

    return env_gauge, phases_ind, fixed_env

def phase_loss(phases, env_old, fixed_env, phases_ind):
    exp_phases = torch.exp(1j * phases)
    exp_phases = torch.cat((exp_phases.new_ones(1), exp_phases))   #   … + an extra ‘1’

    # gather all indices / tensors that never change during optimisation
    sites      = env_old.sites()
    n_sites    = len(sites)
    cardinal = ("t", "l", "b", "r")          # 4 legs  (order fixed)
    corner   = ("tl", "bl", "br", "tr")      # 4 corners (order fixed)

    # indices for phase factors
    phase1_idx = torch.tensor(
        [phases_ind[(env_old.site2index(s), d)]      # (site, dir)
        for s in sites for d in cardinal],
        dtype=torch.long
    )

    neigh_map = {
        "t": lambda s: (env_old.site2index(env_old.nn_site(s, "r")), "t"),
        "l": lambda s: (env_old.site2index(env_old.nn_site(s, "t")), "l"),
        "b": lambda s: (env_old.site2index(env_old.nn_site(s, "l")), "b"),
        "r": lambda s: (env_old.site2index(env_old.nn_site(s, "b")), "r"),
    }
    phase2_idx = torch.tensor(
        [phases_ind[neigh_map[d](s)]
        for s in sites for d in cardinal],
        dtype=torch.long
    )

    fixed_T = [getattr(fixed_env[s], d) for s in sites for d in cardinal]
    T_old = [getattr(env_old[s], d) for s in sites for d in cardinal]
    fixed_C = [getattr(fixed_env[s], d) for s in sites for d in corner]
    C_old   = [getattr(env_old[s], d) for s in sites for d in corner]

    phase1 = exp_phases[phase1_idx].view(n_sites, 4)      # (N,4)
    phase2 = exp_phases[phase2_idx].view(n_sites, 4)      # (N,4)

    # compute loss
    # loss_T, loss_C = torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    # for i in range(n_sites):
    #     for j in range(4):
    #         diff_T = phase1[i][j].conj() * fixed_T[4*i+j] * phase2[i][j] - T_old[4*i+j]
    #         loss_T += diff_T.norm(p='fro')**2


    # prepare the phases used for contracting the corners
    # phase1_c = phase1
    # phase2_c = torch.roll(phase2, shifts=-1, dims=1) # [t l b r] -> [l b r t]

    # for i in range(n_sites):
    #     for j in range(4):
    #         diff_C = phase2_c[i][j].conj() * fixed_C[4*i+j] * phase1_c[i][j] - C_old[4*i+j]
    #         loss_C += diff_C.norm(p='fro')**2

    # ---vectorized_version
    fixed_T_1D, T_old_1D = torch.cat([T._data for T in fixed_T]), torch.cat([T._data for T in T_old])
    lengths = torch.tensor([len(T._data) for T in fixed_T], dtype=torch.long, device=phases.device)
    phase1_1D, phase2_1D = torch.repeat_interleave(torch.flatten(phase1), lengths), torch.repeat_interleave(torch.flatten(phase2), lengths)
    loss_T= (fixed_T_1D * phase1_1D.conj() * phase2_1D - T_old_1D).norm(p='fro')**2

    phase1_c = phase1
    phase2_c = torch.roll(phase2, shifts=-1, dims=1) # [t l b r] -> [l b r t]
    fixed_C_1D, C_old_1D = torch.cat([C._data for C in fixed_C]), torch.cat([C._data for C in C_old])
    lengths = torch.tensor([len(C._data) for C in fixed_C], dtype=torch.long, device=phases.device)
    phase1_c_1D, phase2_c_1D = torch.repeat_interleave(torch.flatten(phase1_c), lengths), torch.repeat_interleave(torch.flatten(phase2_c), lengths)
    loss_C= (fixed_C_1D * phase2_c_1D.conj() * phase1_c_1D - C_old_1D).norm(p='fro')**2

    return loss_T + loss_C

def phase_init_choice(phases, env_old, fixed_env, phases_ind):
    phases = np.concatenate((np.array([0.0]), phases))  # add an extra '1' phase
    is_fixed = np.zeros(len(phases), dtype=bool)
    is_fixed[0] = True  # the first phase is fixed to 1

    # gather all indices / tensors that never change during optimisation
    sites      = env_old.sites()
    # indices for phase factors

    T_neigh_map = {
        "t": lambda s: (env_old.nn_site(s, "r"), "t", "t"),
        "l": lambda s: (env_old.nn_site(s, "t"), "l", "l"),
        "b": lambda s: (env_old.nn_site(s, "l"), "b", "b"),
        "r": lambda s: (env_old.nn_site(s, "b"), "r", "r"),
    }
    C_neigh_map = {
        "t": lambda s: (env_old.nn_site(s, "t"), "l", "tl"),
        "l": lambda s: (env_old.nn_site(s, "l"), "b", "bl"),
        "b": lambda s: (env_old.nn_site(s, "b"), "r", "br"),
        "r": lambda s: (env_old.nn_site(s, "r"), "t", "tr"),
    }


    def extract_phase(v1, v2):
        # Given v1 = v2 e^{1j * phi}, find phi.
        v1, v2 = torch.as_tensor(v1, dtype=torch.complex128), torch.as_tensor(v2, dtype=torch.complex128)
        c = torch.dot(v1, v2.conj())
        return torch.angle(c)

    # BFS
    queue = [(sites[0], "t")]  # start from the first site and direction 't'
    while len(queue) > 0:
        site, dirn = queue.pop(0)
        new_site, new_dirn, env_dirn = T_neigh_map[dirn](site)
        ind, ind_new = phases_ind[(env_old.site2index(site), dirn)], phases_ind[(env_old.site2index(new_site), new_dirn)]
        if not is_fixed[ind_new]:
            phases[ind_new] = extract_phase(np.exp(1j*phases[ind])*getattr(env_old[site], env_dirn)._data, getattr(fixed_env[site], env_dirn)._data)
            is_fixed[ind_new] = True
            queue.append((new_site, new_dirn))

        new_site, new_dirn, env_dirn = C_neigh_map[dirn](site)
        ind, ind_new = phases_ind[(env_old.site2index(site), dirn)], phases_ind[(env_old.site2index(new_site), new_dirn)]
        if not is_fixed[ind_new]:
            phases[ind_new] = extract_phase(np.exp(1j*phases[ind])*getattr(fixed_env[site], env_dirn)._data, getattr(env_old[site], env_dirn)._data)
            is_fixed[ind_new] = True
            queue.append((new_site, new_dirn))


    return phases[1:]

def phase_loss_and_grad(phases, env_old, fixed_env, phases_ind):
    device = env_old[(0, 0)].t.device
    phases = torch.as_tensor(phases, dtype=torch.float64, device=device).requires_grad_(True)
    with torch.enable_grad():
        loss = phase_loss(phases, env_old, fixed_env, phases_ind)

    loss.backward()
    grad = phases.grad.detach().cpu().numpy()
    return loss.item(), grad

def unitary_loss(cs_data, env, zero_modes_dict, dtype=torch.complex128):
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

def find_coeff_multi_sites(env_old, env, zero_modes_dict, dtype=torch.complex128, verbose=False):
    # Fix the relative U(1) phases associated with the sigma matrices.

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
        res = minimize(fun=lambda z: unitary_loss(real_to_complex(z), env, zero_modes_dict, dtype), x0=complex_to_real(cs_data),\
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
                cs_data.append(np.ones(len(zero_modes_dict[(site_ind, dirn)]), dtype=np.float64))
                num += len(zero_modes_dict[(site_ind, dirn)])
            ind2row[site_ind] = slice(start, start+num)
            start += num
        cs_data = np.concatenate(cs_data)
        res = minimize(fun=unitary_loss, x0=cs_data,
                       args=(env, zero_modes_dict, dtype), jac='3-point', method='SLSQP', options={"eps":1e-9, "ftol":1e-14})
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
    start = time.time()
    _, phases_ind, fixed_env = compute_env_gauge_product(env, zero_modes_dict, cs_dict)
    init_phases = np.zeros(4*len(cs_dict)-1)
    phases = phase_init_choice(init_phases, env_old, fixed_env, phases_ind)
    # res = minimize(fun=phase_loss_and_grad, x0=phases, args=(env_old, fixed_env, phases_ind), jac=True, method='SLSQP', options={"ftol":1e-14, "maxiter":1000})
    # phases = res.x
    end = time.time()
    if verbose:
        # print(f"Fixing phase takes {end-start:.1f}s with loss fn ", res.fun)
        print(f"Fixing phase takes {end-start:.1f}s with loss fn ", phase_loss_and_grad(phases, env_old, fixed_env, phases_ind)[0])
        print("phases: ", phases)

    # assemble the coefficients dict
    if isinstance(phases, torch.Tensor):
        exp_phases = torch.exp(1j*phases).to(dtype)
        exp_phases = torch.cat((exp_phases.new_ones(1), exp_phases))   #   … + an extra ‘1’
    else:
        if dtype == torch.complex128:
            exp_phases = np.exp(1j*phases).astype(np.complex128)
        else:
            exp_phases = np.exp(1j*phases).astype(np.float64)
        exp_phases = np.concatenate((np.ones(1, dtype=exp_phases.dtype), exp_phases))   #   … + an extra ‘1’
    ind = 0
    for site in env.sites():
        site_ind = env.site2index(site)
        for i, dirn in enumerate(("t", "l", "b", "r")):
            for j in range(len(zero_modes_dict[(site_ind, dirn)])):
                cs_dict[site_ind][i][j] = cs_dict[site_ind][i][j] * exp_phases[ind]
            ind += 1
        cs_dict[site_ind] = torch.utils.checkpoint.detach_variable(tuple(cs_dict[site_ind]))
    return cs_dict

def find_gauge_multi_sites(env_old, env, verbose=False):
    zero_modes_dict = {}
    env_gauge = {}
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
            if len(zero_modes) == 0:
                return None
            zero_modes_dict[(site_ind, k)] = zero_modes

    cs_dict = find_coeff_multi_sites(env_old, env, zero_modes_dict, dtype=zero_modes_dict[(0, "t")][0].dtype, verbose=verbose)

    def is_diagonal(matrix, tol=1e-6):
        print(torch.diag(matrix))
        off_diag = matrix - torch.diag(torch.diag(matrix))  # Remove diagonal elements
        return torch.all(torch.abs(off_diag) < tol)  # Check if all off-diagonal elements are near zero

    env_gauge, _, fixed_env = compute_env_gauge_product(env, zero_modes_dict, cs_dict)
    if verbose:
        cardinal = ("t", "l", "b", "r")          # 4 legs  (order fixed)
        corner   = ("tl", "bl", "br", "tr")      # 4 corners (order fixed)
        for s in fixed_env.sites():
            for dirn in cardinal:
                print("T diff:", (getattr(fixed_env[s], dirn) - getattr(env_old[s], dirn)).norm() / getattr(env_old[s], dirn).norm())
            for dirn in corner:
                print("C diff:", (getattr(fixed_env[s], dirn) - getattr(env_old[s], dirn)).norm() / getattr(env_old[s], dirn).norm())

    return env_gauge


def fp_ctmrg(env: EnvCTM, \
            ctm_opts_fwd : dict= {'method': "2site", 'corner_tol': 1e-8, 'max_sweeps': 100, 'opts_svd': {}, 'verbosity': 0},
            ctm_opts_fp: dict= {'opts_svd': {'policy':'fullrank'}, "verbosity": 0})->tuple[EnvCTM,Sequence[torch.Tensor],Sequence[slice]]:
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
    env, env_t_meta, env_slices, env_1d = FixedPoint.apply(env, ctm_opts_fwd, ctm_opts_fp, *raw_peps_params)
    env_t_dict = _assemble_dict_from_1d(env_t_meta, env_1d, env_slices)
    env.env = Lattice.from_dict(env_t_dict)
    return env


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
    def fixed_point_iter(env_gauge, ctm_opts_fp, env_dict, env_meta, env_slices, psi_meta, env_data, psi_data):
        env_t_dict = _assemble_dict_from_1d(env_meta, env_data, env_slices)
        psi_dict = combine_data_and_meta(psi_data, psi_meta)
        env_dict['env'], env_dict['psi'] = env_t_dict, psi_dict
        env_in = EnvCTM.from_dict(env_dict)
        env_in.update_(
            **ctm_opts_fp
        )

        for site in env_in.sites():
            site_t, site_l, site_b, site_r = env_in.nn_site(site, "t"), env_in.nn_site(site, "l"), env_in.nn_site(site, "b"), env_in.nn_site(site, "r")
            for dirn in ["t", "l", "b", "r"]:
                sigma1 = getattr(env_gauge[site], dirn)
                if dirn == "t":
                    sigma2 = getattr(env_gauge[site_r], dirn)
                elif dirn == "r":
                    sigma2 = getattr(env_gauge[site_b], dirn)
                elif dirn == "b":
                    sigma2 = getattr(env_gauge[site_l], dirn)
                elif dirn == "l":
                    sigma2 = getattr(env_gauge[site_t], dirn)
                fixed_t = tensordot(
                    tensordot(sigma1, getattr(env_in[site], dirn), axes=(0, 0), conj=(1, 0)),
                    sigma2,
                    axes=(2, 0),
                )
                setattr(env_in[site], dirn, fixed_t)

            sigma1_t, sigma1_l, sigma1_b, sigma1_r = env_gauge[site].t, env_gauge[site].l, env_gauge[site].b, env_gauge[site].r
            sigma2_l, sigma2_b, sigma2_r, sigma2_t = env_gauge[site_t].l, env_gauge[site_l].b, env_gauge[site_b].r, env_gauge[site_r].t
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

        env_out_dict = env_in.to_dict(level=0)
        env_t_data, _ = split_data_and_meta(env_out_dict['env'])
        return (_concat_data(env_t_data)[0], )

    def get_converged_env(
        env,
        method="2site",
        max_sweeps=100,
        opts_svd=None,
        corner_tol=1e-8,
        **kwargs
    ):
        t_ctm, t_check = 0.0, 0.0
        converged, conv_history = False, []

        ctm_itr= env.ctmrg_(iterator_step=1, method=method,  max_sweeps=max_sweeps,
                   opts_svd=opts_svd,
                   corner_tol=None, **kwargs)

        for sweep in range(max_sweeps):
            t0 = time.perf_counter()
            ctm_out_info= next(ctm_itr)
            t1 = time.perf_counter()
            t_ctm += t1-t0

            t2 = time.perf_counter()
            converged, max_dsv, conv_history = ctm_conv_corner_spec(env, conv_history, corner_tol)
            t_check += time.perf_counter()-t2
            if kwargs.get('verbosity',0)>2:
                log.log(logging.INFO, f"CTM iter {len(conv_history)} |delta_C| {max_dsv} t {t1-t0} [s]")

            if converged:
                break

        log.info(f"CTM: convergence: {converged}, sweeps {sweep+1}, t_ctm {t_ctm} [s], t_check {t_check} [s]\n"
                +f"history {[r['max_dsv'] for r in conv_history]}.")

        return env, converged, conv_history, t_ctm, t_check

    @staticmethod
    def forward(ctx, env: EnvCTM, ctm_opts_fwd : dict, ctm_opts_fp: dict, *state_params):
        r"""
        Compute the fixed-point environment for the given state using CTMRG.
        First, run CTMRG until convergence then find the gauge transformation guaranteeing element-wise
        convergence of the environment tensors.

        Args:
            env (EnvCTM): Current environment to converge.
            ctm_opts_fwd (dict): Options for forward CTMRG convergence. The options should include:
                - opts_svd (dict): SVD options for the CTMRG step.
            ctm_opts_fp (dict): Options for fixed-point CTMRG step and for fixing the gauge transformation.
                - opts_svd (dict): SVD options for the fixed-point CTMRG step.
                                   Currently only 'policy': 'fullrank' is supported.
            state_params (Sequence[Tensor]): tensors of underlying Peps state

        Returns:
            EnvCTM: Environment at fixed point.
            Sequence[Tensor]: raw environment data for the backward pass.
        """

        # 1. Converge the environment using CTMRG
        ctm_env_out, converged, *FixedPoint.ctm_log, FixedPoint.t_ctm, FixedPoint.t_check = FixedPoint.get_converged_env(
            env,
            **ctm_opts_fwd,
        )
        if not converged:
            raise NoFixedPointError(code=1, message="No fixed point found: CTM forward does not converge!")

        # 2. Perform 1 extra CTM step to find the gauge transformation under which we have element-wise convergence
        #
        # TODO Use partial SVDs with appropriate backward
        _ctm_opts_fp = copy.deepcopy(ctm_opts_fwd)
        if ctm_opts_fp is not None:
            # NOTE svd is governed solely by opts_svd, expected under 'opts_svd' key in ctm_opts_fwd and ctm_opts_fp
            #      If ctm_opts_fp['opts_svd'] is set, we update ctm_opts_fwd['opts_svd'] with it.
            ctx.verbosity = ctm_opts_fp.get('verbosity', 0)
            _ctm_opts_fp['opts_svd'].update(ctm_opts_fp.get('opts_svd', {}))
            _ctm_opts_fp.update({k:v for k,v in ctm_opts_fp.items() if k not in ['opts_svd']})

        env_converged = ctm_env_out.copy()
        t0 = time.perf_counter()
        ctm_env_out.update_(**_ctm_opts_fp)
        t1 = time.perf_counter()
        log.info(f"{type(ctx).__name__}.forward FP CTM step t {t1-t0} [s]")

        # 3. Find the gauge transformation
        t0 = time.perf_counter()
        env_gauge = find_gauge_multi_sites(env_converged, ctm_env_out, verbose=True)
        if env_gauge is None:
            raise NoFixedPointError(code=1, message="No fixed point found: fail to find the gauge matrix!")
        t1 = time.perf_counter()
        log.info(f"{type(ctx).__name__}.forward FP gauge-fixing t {t1-t0} [s]")

        env_dict = env_converged.to_dict(level=0)
        env_data, env_meta = split_data_and_meta(env_dict)

        g_dict = env_gauge.to_dict(level=0)
        g_data, g_meta = split_data_and_meta(g_dict)
        ctx.save_for_backward(*env_data, *g_data)
        ctx.env_data_num = len(env_data)
        ctx.env_meta, ctx.g_meta = env_meta, g_meta
        ctx.ctm_opts_fp = _ctm_opts_fp

        # Note: one of the ouput of forward must be torch tensor to trigger backward
        env_t_data, env_t_meta = split_data_and_meta(env_dict['env'])
        env_1d, env_slices = _concat_data(env_t_data)
        return env_converged, env_t_meta, env_slices, env_1d

    @staticmethod
    def backward(ctx, none0, none1, none2, *grad_env):
        verbosity= ctx.verbosity
        grads = grad_env
        dA = grad_env

        env_data = ctx.saved_tensors[:ctx.env_data_num]
        g_data = ctx.saved_tensors[ctx.env_data_num:]

        env_dict = combine_data_and_meta(env_data, ctx.env_meta)
        g_dict = combine_data_and_meta(g_data, ctx.g_meta)
        env_gauge = EnvGauge.from_dict(g_dict)

        _env_data, _env_meta = split_data_and_meta(env_dict['env'])
        _env_ts, _env_slices = _concat_data(_env_data)
        _psi_data, _psi_meta = split_data_and_meta(env_dict['psi'])

        prev_grad_tmp = None
        diff_ave = None

        # Compute vjp only
        with torch.enable_grad():
            if verbosity > 2 and _env_ts.is_cuda:
                torch.cuda.memory._dump_snapshot(f"{type(ctx).__name__}_backward_prevjp_CUDAMEM.pickle")
            #_, dfdC_vjp = torch.func.vjp(lambda x: FixedPoint.fixed_point_iter(env, env_gauge, ctx.ctm_opts_fp, _env_slices, x, psi_data), _env_ts)
            #_, dfdA_vjp = torch.func.vjp(lambda x: FixedPoint.fixed_point_iter(env, env_gauge, ctx.ctm_opts_fp, _env_slices, _env_ts, x), psi_data)
            _, df_vjp = torch.func.vjp(lambda x,y: FixedPoint.fixed_point_iter(env_gauge, ctx.ctm_opts_fp, env_dict, _env_meta, _env_slices, _psi_meta, x, y), _env_ts, _psi_data)
            dfdC_vjp= lambda x: (df_vjp(x)[0],)
            dfdA_vjp= lambda x: (df_vjp(x)[1],)
            if verbosity > 2 and _env_ts.is_cuda:
                torch.cuda.memory._dump_snapshot(f"{type(ctx).__name__}_backward_postvjp_CUDAMEM.pickle")

        alpha = 0.4
        for step in range(ctx.ctm_opts_fp['max_sweeps']):
            grads = dfdC_vjp(grads)
            # with torch.enable_grad():
            #     grads = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.env_gauge, ctx.opts_svd, ctx.slices, env_data, psi_data), env_data, grad_outputs=grads)
            if all([torch.norm(grad, p=torch.inf) < ctx.ctm_opts_fp["corner_tol"] for grad in grads]):
                break
            else:
                dA = tuple(dA[i] + grads[i] for i in range(len(grads)))

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

        # --------option 2----------
        # Solve equations of the form Ax = b. Takes extremely long time if df/dC has eigenvalues close to 1

        # env_data = torch.utils.checkpoint.detach_variable(ctx.saved_tensors)[0] # only one element in the tuple
        # grads = grad_env[0]
        # def mat_vec_A(u):
        #   u = torch.from_numpy(u)
        #   with torch.enable_grad():
        #       res = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.env_gauge, ctx.opts_svd, ctx.slices, env_data), env_data, grad_outputs=u)
        #   return (u - res[0]).numpy()

        # def rmat_vec_A(u):
        #   u = torch.from_numpy(u)
        #   u = u.conj()
        #   f = lambda x: FixedPoint.fixed_point_iter(ctx.env, ctx.env_gauge, ctx.opts_svd, ctx.slices, x)
        #   with torch.enable_grad():
        #       output, res = torch.autograd.functional.jvp(f, env_data, u)
        #   return (u - res).conj().numpy()

        # A = LinearOperator(matvec=mat_vec_A, rmatvec=rmat_vec_A, shape=(env_data.size(dim=0) , grads.size(dim=0)), dtype=np.complex128 if env_data.dtype==torch.complex128 else np.float64)

        # u, info = lgmres(A, grads, M=None, rtol=1e-7, atol=0.0)
        # # res = lsqr(A, grads, atol=1e-7, btol=1e-7, show=True, x0=np.random.rand(*u.shape))
        # # u = res[0]

        # u = torch.from_numpy(u)
        # with torch.enable_grad():
        #   dA = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.env_gauge, ctx.opts_svd, ctx.slices, env_data), ctx.env.psi.ket.get_parameters(), grad_outputs=u)

        return None, None, None, *dA
