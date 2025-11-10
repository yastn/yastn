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


# non-deterministic initialization of random number generator
rng = {'rng': np.random.default_rng(None)}  # initialize random number generator
BACKEND_ID = "np"
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
    return x if isinstance(x, (int, float, complex)) else x.copy()


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
    return max(np.abs(data)) if len(data) > 0 else np.float64(0.)


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
    return 2 * (rng['rng'].random(D) + 1j * rng['rng'].random(D)) - (1 + 1j)  # dtype == 'complex128


def randint(low, high):
    return rng['rng'].integers(low, high)


def to_tensor(val, Ds=None, dtype='float64', **kwargs):
    T = np.array(val, dtype=DTYPE[dtype])
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
    pass


def requires_grad(data):  # pragma: no cover
    return False


def move_to(data, dtype, **kwargs):
    return data.astype(dtype=DTYPE[dtype]) if dtype in DTYPE else data


def conj(data):
    return data.conj()


def trace(data, order, meta, Dsize):
    newdata = np.zeros(Dsize, dtype=data.dtype)
    for (sln, list_sln) in meta:
        tmp = newdata[slice(*sln)]
        for slo, Do, Drsh in list_sln:
            tmp += np.trace(data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh))
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


def safe_svd(a):
    try:
        U, S, V = scipy.linalg.svd(a, full_matrices=False)  # , lapack_driver='gesdd'
    except scipy.linalg.LinAlgError:  # pragma: no cover
        U, S, V = scipy.linalg.svd(a, full_matrices=False, lapack_driver='gesvd')
    return U, S, V


def svd_lowrank(data, meta, sizes, **kwargs):
    return svds_scipy(data, meta, sizes, None, 'arpack', **kwargs)

def svds_scipy(data, meta, sizes, thresh=None, solver='arpack', **kwargs):
    Udata = np.empty((sizes[0],), dtype=data.dtype)
    Sdata = np.empty((sizes[1],), dtype=DTYPE['float64'])
    Vdata = np.empty((sizes[2],), dtype=data.dtype)
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        k = slS[1] - slS[0]
        # Is block too small for iterative svd ?
        # TODO user defined threshold
        # the second condition is heuristic estimate when performing dense svd should be faster.
        if (k < min(D) - 1 and D[0] * D[1] > 5000) or (not(thresh is None) and min(D)*thresh > k):
            if solver == 'arpack':
                try:
                    U, S, V = scipy.sparse.linalg.svds(data[slice(*sl)].reshape(D), k=k, ncv=min(5 * k, min(D) - 1),
                                                    which='LM', maxiter=20 * min(D), solver='arpack')
                except scipy.sparse.linalg.ArpackError:
                    U, S, V = scipy.sparse.linalg.svds(data[slice(*sl)].reshape(D), k=k,
                                                    which='LM', maxiter=20 * k, solver='propack')
            if solver == 'propack':
                U, S, V = scipy.sparse.linalg.svds(data[slice(*sl)].reshape(D), k=k,
                                                    which='LM', maxiter=20 * k, solver='propack')
            ord = np.argsort(-S)
            U, S, V = U[:, ord], S[ord], V[ord, :]
        else:
            U, S, V = safe_svd(data[slice(*sl)].reshape(D))
        Udata[slice(*slU)].reshape(DU)[:] = U[:, :k]
        Sdata[slice(*slS)] = S[:k]
        Vdata[slice(*slV)].reshape(DV)[:] = V[:k, :]
    return Udata, Sdata, Vdata


def svd(data, meta, sizes, **kwargs):
    Udata = np.empty((sizes[0],), dtype=data.dtype)
    Sdata = np.empty((sizes[1],), dtype=DTYPE['float64'])
    Vdata = np.empty((sizes[2],), dtype=data.dtype)
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        U, S, V = safe_svd(data[slice(*sl)].reshape(D))
        Udata[slice(*slU)].reshape(DU)[:] = U
        Sdata[slice(*slS)] = S
        Vdata[slice(*slV)].reshape(DV)[:] = V
    return Udata, Sdata, Vdata


def eig_lowrank(data, meta, sizes, **kwargs):
    which= kwargs.get('which', 'LM')
    Udata = np.empty((sizes[0],), dtype=DTYPE['complex128'])
    Sdata = np.empty((sizes[1],), dtype=DTYPE['complex128'])
    Vdata = np.empty((sizes[2],), dtype=DTYPE['complex128'])
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        k = slS[1] - slS[0]
        if k < min(D) - 1 and D[0] * D[1] > 5000:
            # the second condition is heuristic estimate when performing dense eig should be faster.
            try:
                S, U= scipy.sparse.linalg.eigs(data[slice(*sl)].reshape(D), k=k, M=None, sigma=None,
                    which=which, v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None)
            except scipy.sparse.linalg.ArpackError as e:
                raise e
        else:
            S, V, U = scipy.linalg.eig(data[slice(*sl)].reshape(D), left=True, right=True)
        Udata[slice(*slU)].reshape(DU)[:] = U[:, :k]
        Sdata[slice(*slS)] = S[:k]
        Vdata[slice(*slV)].reshape(DV)[:] = V[:k, :]
    return Udata, Sdata, Vdata


def eig(data, meta=None, sizes=(1, 1), **kwargs):
    if meta is None:
        return np.linalg.eig(data)  # S, U
    # Assume worst case ?
    Udata = np.empty((sizes[0],), dtype=DTYPE['complex128'])
    Sdata = np.empty((sizes[1],), dtype=DTYPE['complex128'])
    Vdata = np.empty((sizes[2],), dtype=DTYPE['complex128'])
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        S, V, U = scipy.linalg.eig(data[slice(*sl)].reshape(D), left=True, right=True)
        #
        # in general diag(U.H @ U) = 1 but not U.H @ U = I, i.e. right eigenvectors are not orthogonal
        # same is true for left eigenvectors V, diag(V.H @ V) = 1 but not V.H @ V = I
        #
        # The solutions satisfy
        # M @ U / U = S (as cols)
        # V.H @ M / V.H = S (as rows)
        #
        # However, in general V and U are not biorthogonal, i.e. V.H @ U != I
        #
        # One can enforce biorthogonality by replacing V -> V @ (V.H @ U)^{-1}
        # TODO
        # If matrix has repeated/clustered eigenvalues or is defective, plain diagonal rescaling may be ill‑conditioned.

        tol = 1e-12 if np.iscomplexobj(data) else 1e-14
        try:
            # Column-wise overlaps d_j = v_j^H u_j
            d = np.sum(np.conjugate(V) * U, axis=0)

            # Guard against (near-)defective cases where an overlap is ~0
            # (cannot biorthonormalize a pair with zero overlap via diagonal scaling)
            if np.any(np.abs(d) < tol):
                raise ValueError("At least one left/right eigenvector pair has ~zero overlap; "
                            "biorthonormalization by simple scaling is ill-conditioned. "
                            "Matrix may be defective or numerically close to defective.")

            # Symmetric scaling: divide V by sqrt(d), and U by conj(sqrt(d)),
            # so that (U')^H V' has ones on the diagonal.
            s = np.sqrt(d)
            _U = U / s
            _V = (V / np.conjugate(s)).conj().T
        except ValueError as e:
            try:
                # V.H @ U != I -> solve U.H @ V = I for V
                _V = scipy.linalg.solve(U.conj().T, np.eye(len(S)), lower=False, overwrite_a=False, overwrite_b=False,
                                    check_finite=True, assume_a='gen', transposed=False)
                _U, _V = U, _V.conj().T
            except (scipy.linalg.LinAlgError, np.linalg.LinAlgError) as e:
                raise ValueError("Biorthonormalization of left/right eigenvector pairs failed.") from e

        if any( np.abs(np.sum(_V.T * _U, axis=0) - 1) > tol ):
            raise ValueError("Biorthonormalization of left/right eigenvector pairs failed.")

        s_order= eigs_which(S, which=kwargs.get('which', 'LM'))
        Udata[slice(*slU)].reshape(DU)[:] = _U[:,s_order]
        Sdata[slice(*slS)] = S[s_order]
        Vdata[slice(*slV)].reshape(DV)[:] = _V[s_order,:]
    return Udata, Sdata, Vdata


def eigvals(data, meta, sizeS, **kwargs):
    Sdata = np.empty((sizeS,), dtype=DTYPE['complex128'])
    for (sl, D, _, _, slS, _, _) in meta:
        S = scipy.linalg.eigvals(data[slice(*sl)].reshape(D), b=None, overwrite_a=False,
                                     check_finite=True, homogeneous_eigvals=False)
        Sdata[slice(*slS)]= S[eigs_which(S, which=kwargs.get('which', 'LM'))]
    return Sdata


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


def pinv(a, rcond=None, hermitian=False, out=None, atol=None, rtol=None):
    return np.linalg.pinv(a, rcond=rtol if not rtol is None else rcond, hermitian=hermitian)


def argsort(data):
    return np.argsort(data)

def maximum(x1, x2):
    return np.maximum(x1, x2)

def eigs_which(val, which):
    if which == 'LM':
        return (-abs(val)).argsort()
    if which == 'SM':
        return abs(val).argsort()
    if which == 'LR':
        return (-val.real).argsort()
    # elif which == 'SR':
    return (val.real).argsort()


################################
#     two dicts operations     #
################################


def allclose(Adata, Bdata, rtol, atol):
    return np.allclose(Adata, Bdata, rtol=rtol, atol=atol)


def add(datas, metas, Dsize):
    dtype = reduce(np.promote_types, (data.dtype for data in datas))
    newdata = np.zeros(Dsize, dtype=dtype)
    for data, meta in zip(datas, metas):
        for sl_c, sl_a in meta:
            newdata[slice(*sl_c)] += data[slice(*sl_a)]
    return newdata


def sub(Adata, Bdata, meta, Dsize):
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.zeros(Dsize, dtype=dtype)
    for sl_c, sl_a in meta[0]:
        newdata[slice(*sl_c)] += Adata[slice(*sl_a)]
    for sl_c, sl_b in meta[1]:
        newdata[slice(*sl_c)] -= Bdata[slice(*sl_b)]
    return newdata


def vdot(Adata, Bdata, meta):
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    tmp = np.empty(len(meta), dtype=dtype)
    for ii, (sla, slb) in enumerate(meta):
        tmp[ii] = np.dot(Adata[slice(*sla)], Bdata[slice(*slb)])
    return np.sum(tmp)


def dot(Adata, Bdata, meta_dot, Dsize):
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.empty(Dsize, dtype=dtype)
    for (slc, Dc, sla, Da, slb, Db) in meta_dot:
        np.dot(Adata[slice(*sla)].reshape(Da),
               Bdata[slice(*slb)].reshape(Db),
               out=newdata[slice(*slc)].reshape(Dc))
    return newdata


def transpose_dot_sum(Adata, Bdata, meta_dot, Areshape, Breshape, Aorder, Border, Dsize):
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.empty(Dsize, dtype=dtype)
    Ad = tuple(Adata[slice(*sl)].reshape(Di).transpose(Aorder).reshape(Dl, Dr) for sl, Di, Dl, Dr in Areshape)
    Bd = tuple(Bdata[slice(*sl)].reshape(Di).transpose(Border).reshape(Dl, Dr) for sl, Di, Dl, Dr in Breshape)
    for (sl, Dslc, list_tab) in meta_dot:
        block = newdata[slice(*sl)].reshape(Dslc)
        ta, tb = list_tab[0]
        np.dot(Ad[ta], Bd[tb], out=block)
        for ta, tb in list_tab[1:]:
            block += np.dot(Ad[ta], Bd[tb])
    return newdata


def dot_diag(Adata, Bdata, meta, Dsize, axis, a_ndim):
    dim = [1] * a_ndim
    dim[axis] = -1
    dtype = np.promote_types(Adata.dtype, Bdata.dtype)
    newdata = np.empty(Dsize, dtype=dtype)
    for sln, slb, Db, sla in meta:
        newdata[slice(*sln)].reshape(Db)[:] = Adata[slice(*sla)].reshape(dim) * Bdata[slice(*slb)].reshape(Db)
    return newdata


def negate_blocks(Adata, slices):
    newdata = Adata.copy()
    for slc in slices:
        newdata[slice(*slc)] *= -1
    return newdata


#####################################################
#     block merging, truncations and un-merging     #
#####################################################


def apply_mask(Adata, mask, meta, Dsize, axis, a_ndim):
    slc1 = (slice(None),) * axis
    slc2 = (slice(None),) * (a_ndim - (axis + 1))
    newdata = np.empty(Dsize, dtype=Adata.dtype)
    for sln, Dn, sla, Da, tm in meta:
        slcs = slc1 + (mask[tm],) + slc2
        newdata[slice(*sln)].reshape(Dn)[:] = Adata[slice(*sla)].reshape(Da)[slcs]
    return newdata


def embed_mask(Adata, mask, meta, Dsize, axis, a_ndim):
    slc1 = (slice(None),) * axis
    slc2 = (slice(None),) * (a_ndim - (axis + 1))
    newdata = np.zeros(Dsize, dtype=Adata.dtype)
    for sln, Dn, sla, Da, tm in meta:
        slcs = slc1 + (mask[tm],) + slc2
        newdata[slice(*sln)].reshape(Dn)[slcs] = Adata[slice(*sla)].reshape(Da)
    return newdata


def transpose(data, axes, meta_transpose):
    newdata = np.empty_like(data)
    for sln, Dn, slo, Do in meta_transpose:
        newdata[slice(*sln)].reshape(Dn)[:] = data[slice(*slo)].reshape(Do).transpose(axes)
    return newdata


def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    newdata = np.zeros(Dsize, dtype=data.dtype)
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
    newdata = np.zeros(Dsize, dtype=dtype)
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_block, key=lambda x: x[0])):
        assert tn == t1
        for (_, slo, Do, pos, Dslc) in gr:
            slcs = tuple(slice(*x) for x in Dslc)
            newdata[slice(*sln)].reshape(Dn)[slcs] = pos_tens[pos]._data[slice(*slo)].reshape(Do)
    return newdata


def diag_1dto2d(Adata, meta, Dsize):
    newdata = np.zeros(Dsize, dtype=Adata.dtype)
    for sln, slo in meta:
        newdata[slice(*sln)] = np.diag(Adata[slice(*slo)]).ravel()
    return newdata


def diag_2dto1d(Adata, meta, Dsize):
    newdata = np.zeros(Dsize, dtype=Adata.dtype)
    for sln, slo, Do in meta:
        newdata[slice(*sln)] = np.diag(Adata[slice(*slo)].reshape(Do))
    return newdata


#############
#   tests   #
#############


def is_independent(x, y):
    """
    check if two arrays are identical, or share the same view.
    """
    return (x is not y) and (x.base is not y) and (x is not y.base)
