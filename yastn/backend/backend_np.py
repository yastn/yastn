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
try:
    import fbpca
except ModuleNotFoundError:  # pragma: no cover
    warnings.warn("fbpca not available", Warning)

# non-deterministic initialization of random number generator
rng = {'rng': np.random.default_rng(None)}  # initialize random number generator
BACKEND_ID = "numpy"
DTYPE = {'float64': np.float64,
         'complex128': np.complex128}


def cuda_is_available():
    return False


def get_dtype(t):
    return t.dtype


def is_complex(x):
    return np.iscomplexobj(x)


def get_device(x):
    return 'cpu'


def random_seed(seed):
    rng['rng'] = np.random.default_rng(seed)


def set_num_threads(num_threads):  # pragma: no cover
    warnings.warn("backend_np does not support set_num_threads.", Warning)
    pass


def grad(x):  # pragma: no cover
    warnings.warn("backend_np does not support automatic differentiation.", Warning)
    return None

###################################
#     single tensor operations    #
###################################


def detach(x):  # pragma: no cover
    return x


def detach_(x):  # pragma: no cover
    return x


def clone(x):
    return x.copy()


def copy(x):
    return x.copy()


def to_numpy(x):
    return x.copy()


def get_shape(x):
    return x.shape


def get_size(x):
    return x.size


def diag_create(x, p=0):
    return np.diag(x, k=p)


def diag_get(x):
    return np.diag(x).copy()


def real(x):
    return np.real(x)


def imag(x):
    return np.imag(x)


def max_abs(x):
    return np.abs(x).max()


def norm_matrix(x):
    return np.linalg.norm(x)


def count_nonzero(x):
    return np.count_nonzero(x)


def delete(x, sl):
    return np.delete(x, slice(*sl))


def insert(x, start, values):
    return np.insert(x, start, values)


def expm(x):
    return scipy.linalg.expm(x)


#########################
#    output numbers     #
#########################


def first_element(x):
    return x.ravel()[0]


def item(x):
    return x.item()


def sum_elements(data):
    """ sum of all elements of all tensors in A """
    return data.sum().reshape(1)


def norm(data, p):
    """ 'fro' for Frobenius; 'inf' for max(abs(A)) """
    if p == 'fro':
        return np.linalg.norm(data)
    return max(np.abs(data)) if len(data) > 0  else np.float64(0.)


def entropy(data, alpha, tol):
    """ von Neuman or Renyi entropy from data describing probability distribution."""
    Snorm = np.sum(data)
    if Snorm > 0:
        data = data / Snorm
        data = data[data > tol]
        if alpha == 1:
            return -1 * np.sum(data * np.log2(data))
        return np.log2(np.sum(data ** alpha)) / (1 - alpha)
    return 0.


##########################
#     setting values     #
##########################


def zeros(D, dtype='float64', **kwargs):
    return np.zeros(D, dtype=DTYPE[dtype])


def ones(D, dtype='float64', **kwargs):
    return np.ones(D, dtype=DTYPE[dtype])


def rand(D, dtype='float64', **kwargs):
    if dtype == 'float64':
        return 2 * rng['rng'].random(D) - 1
    return 2 * (rng['rng'].random(D) + 1j *  rng['rng'].random(D)) - (1 + 1j)  # dtype == 'complex128


def randint(low, high):
    return rng['rng'].integers(low, high)


def to_tensor(val, Ds=None, dtype='float64', **kwargs):
    # try:
    T = np.array(val, dtype=DTYPE[dtype])
    # except TypeError:
    #     T = np.array(val, dtype=DTYPE['complex128'])
    return T if Ds is None else T.reshape(Ds)


def to_mask(val):
    return val.nonzero()[0].ravel()


def square_matrix_from_dict(H, D=None, **kwargs):
    dtype = reduce(np.promote_types, (a.dtype for a in H.values()))
    T = np.zeros((D, D), dtype=dtype)
    for (i, j), v in H.items():
        if i < D and j < D:
            T[i, j] = v
    return T


def requires_grad_(data, requires_grad=True):  # pragma: no cover
    warnings.warn("backend_np does not support autograd.", Warning)
    pass


def requires_grad(data):  # pragma: no cover
    return False


def move_to(data, dtype, **kwargs):
    return data.astype(dtype=DTYPE[dtype]) if dtype in DTYPE else data


def conj(data):
    return data.conj()


def trace(data, order, meta, Dsize):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    newdata = np.zeros((Dsize,), dtype=data.dtype)
    for (sln, slo, Do, Drsh) in meta:
        newdata[slice(*sln)] += np.trace(data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh)).ravel()
    return newdata


def trace_with_mask(data, order, meta, Dsize, tcon, msk12):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    newdata = np.zeros((Dsize,), dtype=data.dtype)
    for (sln, slo, Do, Drsh), tt in zip(meta, tcon):
        temp = data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh)
        newdata[slice(*sln)] += np.sum(temp[msk12[tt][0], msk12[tt][1]], axis=0).ravel()
    return newdata


def rsqrt(data, cutoff=0):
    res = np.zeros_like(data)
    ind = np.abs(data) > cutoff
    res[ind] = 1. / np.sqrt(data[ind])
    return res


def reciprocal(data, cutoff=0):
    res = np.zeros_like(data)
    ind = np.abs(data) > cutoff
    res[ind] = 1. / data[ind]
    return res


def exp(data, step):
    return np.exp(step * data)


def sqrt(data):
    return np.sqrt(data)


def absolute(data):
    return np.abs(data)


def bitwise_not(data):
    return np.bitwise_not(data)


def svd_lowrank(data, meta, sizes, n_iter=60, k_fac=6, **kwargs):
    Udata = np.empty((sizes[0],), dtype=data.dtype)
    Sdata = np.empty((sizes[1],), dtype=DTYPE['float64'])
    Vdata = np.empty((sizes[2],), dtype=data.dtype)
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        k = slS[1] - slS[0]
        U, S, V = fbpca.pca(data[slice(*sl)].reshape(D), k=k, raw=True, n_iter=n_iter, l=k_fac * k)
        Udata[slice(*slU)].reshape(DU)[:] = U
        Sdata[slice(*slS)] = S
        Vdata[slice(*slV)].reshape(DV)[:] = V
    return Udata, Sdata, Vdata


def svd(data, meta, sizes, **kwargs):
    Udata = np.empty((sizes[0],), dtype=data.dtype)
    Sdata = np.empty((sizes[1],), dtype=DTYPE['float64'])
    Vdata = np.empty((sizes[2],), dtype=data.dtype)
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        try:
            U, S, V = scipy.linalg.svd(data[slice(*sl)].reshape(D), full_matrices=False)
        except scipy.linalg.LinAlgError:  # pragma: no cover
            U, S, V = scipy.linalg.svd(data[slice(*sl)].reshape(D), full_matrices=False, lapack_driver='gesvd')
        Udata[slice(*slU)].reshape(DU)[:] = U
        Sdata[slice(*slS)] = S
        Vdata[slice(*slV)].reshape(DV)[:] = V
    return Udata, Sdata, Vdata


def svdvals(data, meta, sizeS, **kwargs):
    Sdata = np.empty((sizeS,), dtype=DTYPE['float64'])
    for (sl, D, _, _, slS, _, _) in meta:
        try:
            S = scipy.linalg.svd(data[slice(*sl)].reshape(D), full_matrices=False, compute_uv=False)
        except scipy.linalg.LinAlgError:  # pragma: no cover
            S = scipy.linalg.svd(data[slice(*sl)].reshape(D), full_matrices=False, compute_uv=False, lapack_driver='gesvd')
        Sdata[slice(*slS)] = S
    return Sdata


def fix_svd_signs(Udata, Vdata, meta):
    Uamp = (abs(Udata) * (2 ** 40)).astype(np.int64)
    for (_, _, slU, DU, _, slV, DV) in meta:
        Utemp = Udata[slice(*slU)].reshape(DU)
        Vtemp = Vdata[slice(*slV)].reshape(DV)
        Utemp_amp = Uamp[slice(*slU)].reshape(DU)
        ii = np.argmax(Utemp_amp, axis=0).reshape(1, -1)
        phase = np.take_along_axis(Utemp, ii, axis=0)
        phase /= abs(phase)
        Utemp *= phase.conj().reshape(1, -1)
        Vtemp *= phase.reshape(-1, 1)
    return Udata, Vdata


def eigh(data, meta=None, sizes=(1, 1)):
    Sdata = np.zeros((sizes[0],), dtype=DTYPE['float64'])
    Udata = np.zeros((sizes[1],), dtype=data.dtype)
    if meta is not None:
        for (sl, D, slU, DU, slS) in meta:
            try:
                S, U = scipy.linalg.eigh(data[slice(*sl)].reshape(D))
            except scipy.linalg.LinAlgError:  # pragma: no cover
                S, U = np.linalg.eigh(data[slice(*sl)].reshape(D))
            Sdata[slice(*slS)] = S
            Udata[slice(*slU)].reshape(DU)[:] = U
        return Sdata, Udata
    return np.linalg.eigh(data)  # S, U


def eig(T):
    return np.linalg.eig(T)  # S, U


def qr(data, meta, sizes):
    Qdata = np.empty((sizes[0],), dtype=data.dtype)
    Rdata = np.empty((sizes[1],), dtype=data.dtype)
    for (sl, D, slQ, DQ, slR, DR) in meta:
        Q, R = scipy.linalg.qr(data[slice(*sl)].reshape(D), mode='economic')
        sR = np.sign(np.real(np.diag(R)))
        sR[sR == 0] = 1
        Qdata[slice(*slQ)].reshape(DQ)[:] = Q * sR  # positive diag of R
        Rdata[slice(*slR)].reshape(DR)[:] = sR.reshape([-1, 1]) * R
    return Qdata, Rdata


def argsort(data):
    return np.argsort(data)


def eigs_which(val, which):
    if which == 'LM':
        return (-abs(val)).argsort()
    # if which == 'SM':
    #     return abs(val).argsort()
    if which == 'LR':
        return (-val.real).argsort()
    # elif which == 'SR':
    return (val.real).argsort()


def embed_msk(data, msk, Dsize):
    newdata = np.zeros((Dsize,), dtype=data.dtype)
    newdata[msk] = data
    return newdata


def embed_slc(data, meta, Dsize):
    newdata = np.zeros((Dsize,), dtype=data.dtype)
    for sln, slo in meta:
        newdata[slice(*sln)] = data[slice(*slo)]
    return newdata


################################
#     two dict-s operations    #
################################


def allclose(Adata, Bdata, rtol, atol):
    return np.allclose(Adata, Bdata, rtol=rtol, atol=atol)


def add(Adata, Bdata, meta, Dsize):
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.zeros((Dsize,), dtype=dtype)
    for sl_c, sl_a in meta[0]:
        newdata[slice(*sl_c)] += Adata[slice(*sl_a)]
    for sl_c, sl_b in meta[1]:
        newdata[slice(*sl_c)] += Bdata[slice(*sl_b)]
    return newdata


def sub(Adata, Bdata, meta, Dsize):
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.zeros((Dsize,), dtype=dtype)
    for sl_c, sl_a in meta[0]:
        newdata[slice(*sl_c)] += Adata[slice(*sl_a)]
    for sl_c, sl_b in meta[1]:
        newdata[slice(*sl_c)] -= Bdata[slice(*sl_b)]
    return newdata


def apxb(Adata, Bdata, x, meta, Dsize):
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.zeros((Dsize,), dtype=dtype)
    for sl_c, sl_a in meta[0]:
        newdata[slice(*sl_c)] += Adata[slice(*sl_a)]
    for sl_c, sl_b in meta[1]:
        newdata[slice(*sl_c)] += x * Bdata[slice(*sl_b)]
    return newdata


def apply_slice(data, slcn, slco):
    Dsize = slcn[-1][1] if len(slcn) > 0 else 0
    newdata = np.zeros((Dsize,), dtype=data.dtype)
    for sn, so in zip(slcn, slco):
        newdata[slice(*sn)] = data[slice(*so)]
    return newdata


def vdot(Adata, Bdata):
    return Adata @ Bdata


def diag_1dto2d(Adata, meta, Dsize):
    newdata = np.zeros((Dsize,), dtype=Adata.dtype)
    for sln, slo in meta:
        newdata[slice(*sln)] = np.diag(Adata[slice(*slo)]).ravel()
    return newdata


def diag_2dto1d(Adata, meta, Dsize):
    newdata = np.zeros((Dsize,), dtype=Adata.dtype)
    for sln, slo, Do in meta:
        newdata[slice(*sln)] = np.diag(Adata[slice(*slo)].reshape(Do))
    return newdata


def dot(Adata, Bdata, meta_dot, Dsize):
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.empty((Dsize,), dtype=dtype)
    for (slc, Dc, sla, Da, slb, Db, ia, ib) in meta_dot:
        np.matmul(Adata[slice(*sla)].reshape(Da), \
                  Bdata[slice(*slb)].reshape(Db), \
                  out=newdata[slice(*slc)].reshape(Dc))
    return newdata


def dot_with_mask(Adata, Bdata, meta_dot, Dsize, msk_a, msk_b):
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.empty((Dsize,), dtype=dtype)
    for (slc, Dc, sla, Da, slb, Db, ia, ib) in meta_dot:
        np.matmul(Adata[slice(*sla)].reshape(Da)[:, msk_a[ia]], \
                  Bdata[slice(*slb)].reshape(Db)[msk_b[ib], :], \
                  out=newdata[slice(*slc)].reshape(Dc))
    return newdata


def dot_diag(Adata, Bdata, meta, Dsize, axis, a_ndim):
    dim = [1] * a_ndim
    dim[axis] = -1
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.empty((Dsize,), dtype=dtype)
    for sln, slb, Db, sla in meta:
        newdata[slice(*sln)].reshape(Db)[:] = Adata[slice(*sla)].reshape(dim) * Bdata[slice(*slb)].reshape(Db)
    return newdata


def mask_diag(Adata, Bdata, meta, Dsize, axis, a_ndim):
    slc1 = (slice(None),) * axis
    slc2 = (slice(None),) * (a_ndim - (axis + 1))
    newdata = np.zeros((Dsize,), dtype=Adata.dtype)
    for sln, sla, Da, slb in meta:
        cut = Bdata[slice(*slb)].nonzero()
        newdata[slice(*sln)] = Adata[slice(*sla)].reshape(Da)[slc1 + cut + slc2].ravel()
    return newdata


# dot_dict = {(0, 0): lambda x, y, out: np.matmul(x, y, out=out),
#             (0, 1): lambda x, y, out: np.matmul(x, y.conj(), out=out),
#             (1, 0): lambda x, y, out: np.matmul(x.conj(), y, out=out),
#             (1, 1): lambda x, y, out: np.matmul(x.conj(), y.conj(), out=out)}
#
#
# def dot_nomerge(Adata, Bdata, cc, oA, oB, meta, Dsize):
#     f = dot_dict[cc]  # proper conjugations
#     dtype = np.promote_types(Adata.dtype, Bdata.dtype)
#     newdata = np.zeros((Dsize,), dtype=dtype)
#     for (sln, sla, Dao, Dan, slb, Dbo, Dbn) in meta:
#         newdata[slice(*sln)] += f(Adata[slice(*sla)].reshape(Dao).transpose(oA).reshape(Dan), \
#                                   Bdata[slice(*slb)].reshape(Dbo).transpose(oB).reshape(Dbn), None).ravel()
#     return newdata


# def dot_nomerge_masks(Adata, Bdata, cc, oA, oB, meta, Dsize, tcon, ma, mb):
#     f = dot_dict[cc]  # proper conjugations
#     dtype = np.promote_types(Adata.dtype, Bdata.dtype)
#     newdata = np.zeros((Dsize,), dtype=dtype)
#     for (sln, sla, Dao, Dan, slb, Dbo, Dbn), tt in zip(meta, tcon):
#         newdata[slice(*sln)] += f(Adata[slice(*sla)].reshape(Dao).transpose(oA).reshape(Dan)[:, ma[tt]], \
#                                   Bdata[slice(*slb)].reshape(Dbo).transpose(oB).reshape(Dbn)[mb[tt], :], None).ravel()
#     return newdata

#####################################################
#     block merging, truncations and un-merging     #
#####################################################

def transpose(data, axes, meta_transpose):
    newdata = np.empty_like(data)
    for sln, Dn, slo, Do in meta_transpose:
        newdata[slice(*sln)].reshape(Dn)[:] = data[slice(*slo)].reshape(Do).transpose(axes)
    return newdata


def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    newdata = np.zeros((Dsize,), dtype=data.dtype)
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        assert tn == t1
        temp = newdata[slice(*sln)].reshape(Dn)
        for (_, slo, Do, Dslc, Drsh) in gr:
            slcs = tuple(slice(*x) for x in Dslc)
            temp[slcs] = data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh)
    return newdata


def unmerge(data, meta):
    newdata = np.empty_like(data)  # this does not introduce zero blocks
    for sln, Dn, slo, Do, sub_slc in meta:
        slcs = tuple(slice(*x) for x in sub_slc)
        newdata[slice(*sln)].reshape(Dn)[:] = data[slice(*slo)].reshape(Do)[slcs]
    return newdata


def merge_to_dense(data, Dtot, meta):
    newdata = np.zeros(Dtot, dtype=data.dtype)
    for (sl, Dss) in meta:
        newdata[tuple(slice(*Ds) for Ds in Dss)] = data[sl].reshape(tuple(Ds[1] - Ds[0] for Ds in Dss))
    return newdata.reshape(-1)


def merge_super_blocks(pos_tens, meta_new, meta_block, Dsize):
    dtype = reduce(np.promote_types, (a._data.dtype for a in pos_tens.values()))
    newdata = np.zeros((Dsize,), dtype=dtype)
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_block, key=lambda x: x[0])):
        assert tn == t1
        for (_, slo, Do, pos, Dslc) in gr:
            slcs = tuple(slice(*x) for x in Dslc)
            newdata[slice(*sln)].reshape(Dn)[slcs] = pos_tens[pos]._data[slice(*slo)].reshape(Do)
    return newdata


#############
#   tests   #
#############

def is_independent(x, y):
    """
    check if two arrays are identical, or share the same view.
    """
    return not ((x is y) or (x.base is y) or (x is y.base))
