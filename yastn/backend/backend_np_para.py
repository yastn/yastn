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
"""Support of numpy as a data structure used by yastn."""
from itertools import groupby
from functools import reduce
import warnings
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from joblib import Parallel, delayed
from .backend_np import cuda_is_available, get_dtype, is_complex, get_device, random_seed, set_num_threads, grad, detach, detach_, clone, copy, to_numpy, get_shape, get_size, diag_create, diag_get, real, imag, max_abs, norm_matrix, count_nonzero, delete, insert, expm, first_element, item, sum_elements, norm, entropy, zeros, ones, rand, randint, to_tensor, to_mask, square_matrix_from_dict, requires_grad, requires_grad_, move_to, conj, trace, trace_with_mask, rsqrt, reciprocal, exp, sqrt, absolute, bitwise_not, safe_svd, svd_lowrank, fix_svd_signs, eig, argsort, eigs_which, embed_msk, embed_slc, allclose, add, sub, apxb, apply_slice, vdot, diag_1dto2d, diag_2dto1d, dot, dot_with_mask, dot_with_sum, dot_diag, mask_diag, transpose, transpose_and_merge, unmerge, merge_to_dense, merge_super_blocks, is_independent
from time import time

# non-deterministic initialization of random number generator
rng = {'rng': np.random.default_rng(None)}  # initialize random number generator
BACKEND_ID = "np"
DTYPE = {'float64': np.float64,
         'complex128': np.complex128,
         'bool': bool}




# def svd(data, meta, sizes, **kwargs):

#     start = time()

#     Udata = np.empty((sizes[0],), dtype=data.dtype)
#     Sdata = np.empty((sizes[1],), dtype=DTYPE['float64'])
#     Vdata = np.empty((sizes[2],), dtype=data.dtype)
#     for (sl, D, slU, DU, slS, slV, DV) in meta:
#         U, S, V = safe_svd(data[slice(*sl)].reshape(D))
#         Udata[slice(*slU)].reshape(DU)[:] = U
#         Sdata[slice(*slS)] = S
#         Vdata[slice(*slV)].reshape(DV)[:] = V

#     end = time()

#     print("svd time:", end - start, "s")

#     return Udata, Sdata, Vdata


def svd(data, meta, sizes, **kwargs):

    Udata = np.empty((sizes[0],), dtype=data.dtype)
    Sdata = np.empty((sizes[1],), dtype=DTYPE['float64'])
    Vdata = np.empty((sizes[2],), dtype=data.dtype)

    def svd_(matrix):
        U, S, V = safe_svd(matrix)
        return [U, S, V]

    complexity = np.array([D[0] * D[1] * min(D[0], D[1]) for _, D, _, _, _, _, _ in meta])
    # complexity = np.array([max(D[0], D[1]) for _, D, _, _, _, _, _ in meta])
    sorted_index = (np.argsort(complexity))[::-1]
    to_be_paralleled = np.sum(complexity >= 500 ** 3)

    if to_be_paralleled > 0:
        para_result = Parallel(n_jobs=min(4, to_be_paralleled), verbose=10)(delayed(svd_)(matrix) for matrix in [data[slice(*meta[ii][0])].reshape(*meta[ii][1]) for ii in sorted_index[:to_be_paralleled]])

        for i in range(to_be_paralleled):
            ii = sorted_index[i]
            sl, D, slU, DU, slS, slV, DV =  meta[ii]
            U, S, V = para_result[0]
            para_result.pop(0)
            Udata[slice(*slU)].reshape(DU)[:] = U
            Sdata[slice(*slS)] = S
            Vdata[slice(*slV)].reshape(DV)[:] = V

    for ii in sorted_index[to_be_paralleled:]:
        sl, D, slU, DU, slS, slV, DV =  meta[ii]
        U, S, V = safe_svd(data[slice(*sl)].reshape(D))
        Udata[slice(*slU)].reshape(DU)[:] = U
        Sdata[slice(*slS)] = S
        Vdata[slice(*slV)].reshape(DV)[:] = V


    return Udata, Sdata, Vdata

# def svdvals(data, meta, sizeS, **kwargs):
#     Sdata = np.empty((sizeS,), dtype=DTYPE['float64'])
#     for (sl, D, _, _, slS, _, _) in meta:
#         try:
#             S = scipy.linalg.svd(data[slice(*sl)].reshape(D), full_matrices=False, compute_uv=False)
#         except scipy.linalg.LinAlgError:  # pragma: no cover
#             S = scipy.linalg.svd(data[slice(*sl)].reshape(D), full_matrices=False, compute_uv=False, lapack_driver='gesvd')
#         Sdata[slice(*slS)] = S
#     return Sdata


def svdvals(data, meta, sizeS, **kwargs):

    Sdata = np.empty((sizeS,), dtype=DTYPE['float64'])

    def svdvals_(matrix):
        try:
            S = scipy.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        except scipy.linalg.LinAlgError:  # pragma: no cover
            S = scipy.linalg.svd(matrix, full_matrices=False, compute_uv=False, lapack_driver='gesvd')
        return S

    complexity = np.array([D[0] * D[1] * min(D[0], D[1]) for _, D, _, _, _, _, _ in meta])
    # complexity = np.array([max(D[0], D[1]) for _, D, _, _, _, _, _ in meta])
    sorted_index = (np.argsort(complexity))[::-1]
    to_be_paralleled = np.sum(complexity >= 500 ** 3)

    if to_be_paralleled > 0:
        para_result = Parallel(n_jobs=min(4, to_be_paralleled), verbose=10)(delayed(svdvals_)(matrix) for matrix in [data[slice(*meta[ii][0])].reshape(*meta[ii][1]) for ii in sorted_index[:to_be_paralleled]])

        for i in range(to_be_paralleled):
            ii = sorted_index[i]
            sl, D, slU, DU, slS, slV, DV =  meta[ii]
            S = para_result[0]
            para_result.pop(0)
            Sdata[slice(*slS)] = S

    for ii in sorted_index[to_be_paralleled:]:
        sl, D, slU, DU, slS, slV, DV =  meta[ii]
        Sdata[slice(*slS)] = svdvals_(data[slice(*sl)].reshape(D))

    return Sdata


# def eigh(data, meta=None, sizes=(1, 1)):
#     Sdata = np.zeros((sizes[0],), dtype=DTYPE['float64'])
#     Udata = np.zeros((sizes[1],), dtype=data.dtype)
#     if meta is not None:
#         for (sl, D, slU, DU, slS) in meta:
#             try:
#                 S, U = scipy.linalg.eigh(data[slice(*sl)].reshape(D))
#             except scipy.linalg.LinAlgError:  # pragma: no cover
#                 S, U = np.linalg.eigh(data[slice(*sl)].reshape(D))
#             Sdata[slice(*slS)] = S
#             Udata[slice(*slU)].reshape(DU)[:] = U
#         return Sdata, Udata
#     return np.linalg.eigh(data)  # S, U

def eigh(data, meta=None, sizes=(1, 1)):

    Sdata = np.zeros((sizes[0],), dtype=DTYPE['float64'])
    Udata = np.zeros((sizes[1],), dtype=data.dtype)

    def eigh_(matrix):
        try:
            S, U = scipy.linalg.eigh(matrix)
        except scipy.linalg.LinAlgError:  # pragma: no cover
            S, U = np.linalg.eigh(matrix)
        return [S, U]
    if meta is not None:

        complexity = np.array([D[0] for _, D, _, _, _ in meta])
        # complexity = np.array([max(D[0], D[1]) for _, D, _, _, _, _, _ in meta])
        sorted_index = (np.argsort(complexity))[::-1]
        to_be_paralleled = np.sum(complexity >= 500)

        if to_be_paralleled > 0:
            para_result = Parallel(n_jobs=min(8, to_be_paralleled), verbose=10)(delayed(eigh_)(matrix) for matrix in [data[slice(*meta[ii][0])].reshape(*meta[ii][1]) for ii in sorted_index[:to_be_paralleled]])

            for i in range(to_be_paralleled):
                ii = sorted_index[i]
                sl, D, slU, DU, slS =  meta[ii]
                S, U = para_result[0]
                para_result.pop(0)
                Sdata[slice(*slS)] = S
                Udata[slice(*slU)].reshape(DU)[:] = U

        for ii in sorted_index[to_be_paralleled:]:
            sl, D, slU, DU, slS =  meta[ii]
            S, U = eigh_(data[slice(*sl)].reshape(D))
            Sdata[slice(*slS)] = S
            Udata[slice(*slU)].reshape(DU)[:] = U

        return Sdata, Udata
    return np.linalg.eigh(data)  # S, U


# def qr(data, meta, sizes):
#     Qdata = np.empty((sizes[0],), dtype=data.dtype)
#     Rdata = np.empty((sizes[1],), dtype=data.dtype)
#     for (sl, D, slQ, DQ, slR, DR) in meta:
#         Q, R = scipy.linalg.qr(data[slice(*sl)].reshape(D), mode='economic')
#         sR = np.sign(np.real(np.diag(R)))
#         sR[sR == 0] = 1
#         Qdata[slice(*slQ)].reshape(DQ)[:] = Q * sR  # positive diag of R
#         Rdata[slice(*slR)].reshape(DR)[:] = sR.reshape([-1, 1]) * R
#     return Qdata, Rdata

def qr(data, meta, sizes):
    Qdata = np.empty((sizes[0],), dtype=data.dtype)
    Rdata = np.empty((sizes[1],), dtype=data.dtype)

    complexity = np.array([max(D[0], D[1]) * min(D[0], D[1]) ** 2 for _, D, _, _, _, _ in meta])
    # complexity = np.array([max(D[0], D[1]) for _, D, _, _, _, _, _ in meta])
    sorted_index = (np.argsort(complexity))[::-1]
    to_be_paralleled = np.sum(complexity >= 500 ** 3)

    def qr_(matrix):
        Q, R = scipy.linalg.qr(matrix, mode='economic')
        sR = np.sign(np.real(np.diag(R)))
        sR[sR == 0] = 1 # positive diag of R
        return [Q * sR, sR.reshape([-1, 1]) * R]

    if to_be_paralleled > 0:
        para_result = Parallel(n_jobs=4, verbose=10)(delayed(qr_)(matrix) for matrix in [data[slice(*meta[ii][0])].reshape(*meta[ii][1]) for ii in sorted_index[:to_be_paralleled]])

        for i in range(to_be_paralleled):
            ii = sorted_index[i]
            sl, D, slQ, DQ, slR, DR =  meta[ii]
            Q, R = para_result[0]
            para_result.pop(0)
            Qdata[slice(*slQ)].reshape(DQ)[:] = Q
            Rdata[slice(*slR)].reshape(DR)[:] = R

    for ii in sorted_index[to_be_paralleled:]:
        sl, D, slQ, DQ, slR, DR =  meta[ii]
        Q, R = qr_(data[slice(*sl)].reshape(D))
        Qdata[slice(*slQ)].reshape(DQ)[:] = Q
        Rdata[slice(*slR)].reshape(DR)[:] = R
    return Qdata, Rdata
