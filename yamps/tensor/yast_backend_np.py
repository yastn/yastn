import logging
import numpy as np
import scipy as sp
import fbpca as pca
from itertools import product


logger = logging.getLogger('yast_backend_np')


_data_dtype = {'float64': np.float64,
               'complex128': np.complex128}


def random_seed(seed):
    np.random.seed(seed)

#######################################
#     single element calculations     #
#######################################

def copy(x):
    return x.copy()


def to_numpy(x):
    return x.copy()


def first_element(A):
    return A[next(iter(A))].flat[0] if A else 0.


def get_shape(x):
    return x.shape


def get_ndim(x):
    return x.ndim


def get_size(x):
    return x.size


def diag_create(x):
    return np.diag(x)


def diag_get(x):
    return np.diag(x)


def count_greater(x, cutoff):
    return np.sum(x > cutoff)

##########################
#     setting values     #
##########################

def zeros(D, dtype='float64'):
    return np.zeros(D, dtype=_data_dtype[dtype])


def ones(D, dtype='float64'):
    return np.ones(D, dtype=_data_dtype[dtype])


def randR(D, dtype='float64'):
    return 2 * np.random.random_sample(D).astype(_data_dtype[dtype]) - 1


def rand(D, dtype='float64'):
    if dtype == 'float64':
        return randR(D)
    elif dtype == 'complex128':
        return 2 * np.random.random_sample(D) - 1 + 1j * (2 * np.random.random_sample(D) - 1)


def to_tensor(val, Ds=None, dtype='float64'):
    return np.array(val, dtype=_data_dtype[dtype]) if Ds is None else np.array(val, dtype=_data_dtype[dtype]).reshape(Ds)

##################################
#     single dict operations     #
##################################

def conj(A):
    """ Conjugate dict of tensors forcing a copy. """
    return {t: x.copy().conj() for t, x in A.items()}


def trace(A, order, meta):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    Aout = {}
    for (tnew, told, Drsh) in meta:
        Atemp = np.trace(np.reshape(np.transpose(A[told], order), Drsh))
        if tnew in Aout:
            Aout[tnew] += Atemp
        else:
            Aout[tnew] = Atemp
    return Aout


def moveaxis_local(A, source, destination, meta_transpose):
    """ Move axis; Does not force a copy. """
    return {new: np.moveaxis(A[old], source, destination) for old, new in meta_transpose}


def transpose(A, axes, meta_transpose):
    """ Transpose forcing a copy. """
    return {new: np.transpose(A[old], axes=axes).copy() for old, new in meta_transpose}


def invsqrt(A):
    return {t: 1. / np.sqrt(x) for t, x in A.items()}


def invsqrt_diag(A):
    return {t: np.diag(1. / np.sqrt(np.diag(x))) for t, x in A.items()}


def inv(A):
    return {t: 1. / x for t, x in A.items()}


def inv_diag(A):
    return {t: np.diag(1. / np.diag(x)) for t, x in A.items()}


def exp(A, step):
    return {t: np.exp(step * x) for t, x in A.items()}


def exp_diag(A, step):
    return {t: np.diag(np.exp(step * np.diag(x))) for t, x in A.items()}


def sqrt(A):
    return {t: np.sqrt(x) for t, x in A.items()}


def svd(A, meta, opts):
    U, S, V = {}, {}, {}
    for (iold, iU, iS, iV) in meta:
        if opts['truncated_svd'] and min(A[iold].shape) > opts['kfac'] * opts['D_block']:
            U[iU], S[iS], V[iV] = pca.pca(A[iold], k=opts['D_block'], raw=True,
                                            n_iter=opts['nbit'], l=opts['kfac'] * opts['D_block'])
        else:
            try:
                U[iU], S[iS], V[iV] = sp.linalg.svd(A[iold], full_matrices=False)
            except sp.linalg.LinAlgError:
                U[iU], S[iS], V[iV] = sp.linalg.svd(A[iold], full_matrices=False, lapack_driver='gesvd')
    return U, S, V


def svd_S(A):
    S = {}
    for ind in A:
        try:
            S[ind] = sp.linalg.svd(A[ind], full_matrices=False, compute_uv=False)
        except sp.linalg.LinAlgError:
            S[ind] = sp.linalg.svd(A[ind], full_matrices=False, lapack_driver='gesvd', compute_uv=False)
    return S


def qr(A, meta):
    Q, R = {}, {}
    for (ind, indQ, indR) in meta:
        Q[indQ], R[indR] = sp.linalg.qr(A[ind], mode='economic')
        sR = np.sign(np.real(np.diag(R[indR])))
        sR[sR == 0] = 1
        # positive diag of R
        Q[indQ] = Q[indQ] * sR
        R[indR] = sR.reshape([-1, 1]) * R[indR]
    return Q, R
# def rq(A):
#     R, Q = {}, {}
#     for ind in A:
#         R[ind], Q[ind] = sp.linalg.rq(A[ind], mode='economic')
#         sR = np.sign(np.real(np.diag(R[ind])))
#         sR[sR == 0] = 1
#         # positive diag of R
#         R[ind], Q[ind] = R[ind] * sR, sR.reshape([-1, 1]) * Q[ind]
#     return R, Q


def eigh(A, meta):
    S, U = {}, {}
    for (ind, indS, indU) in meta:
        S[indS], U[indU] = np.linalg.eigh(A[ind])
    return S, U


def select_largest(S, D_keep, D_total, sorting):
    if sorting == 'svd':
        return np.hstack([S[ind][:D_keep[ind]] for ind in S]).argpartition(-D_total-1)[-D_total:]
    elif sorting == 'eigh':
        return np.hstack([S[ind][-D_keep[ind]:] for ind in S]).argpartition(-D_total-1)[-D_total:]


def range_largest(D_keep, D_total, sorting):
    if sorting == 'svd':
        return (0, D_keep)
    elif sorting == 'eigh':
        return (D_total - D_keep, D_total)


def maximum(A):
    val = [np.max(x) for x in A.values()]
    val.append(0.)
    return max(val)


def max_abs(A):
    val = [np.max(np.abs(x)) for x in A.values()]
    val.append(0.)
    return max(val)


def entropy(A, alpha=1, tol=1e-12):
    temp = 0.
    for x in A.values():
        temp += np.sum(np.abs(x) ** 2)
    normalization = np.sqrt(temp)

    entropy = 0.
    Smin = np.inf
    if normalization > 0:
        for x in A.values():
            Smin = min(Smin, min(x))
            x = x / normalization
            if alpha == 1:
                x = x[x > tol]
                entropy += -2 * sum(x * x * np.log2(x))
            else:
                entropy += x**(2 * alpha)
        if alpha != 1:
            entropy = np.log2(entropy) / (1 - alpha)
    return entropy, Smin, normalization


_norms = {'fro': np.linalg.norm, 'inf': lambda x: np.abs(x).max()}


def norm(A, ord):
    block_norm = [0.]
    for x in A.values():
        block_norm.append(_norms[ord](x))
    return _norms[ord](block_norm)

################################
#     two dicts operations     #
################################

def norm_diff(A, B, ord):
    block_norm = [0.]
    for ind in A:
        if ind in B:
            block_norm.append(_norms[ord](A[ind] - B[ind]))
        else:
            block_norm.append(_norms[ord](A[ind]))
    for ind in B:
        if ind not in B:
            block_norm.append(_norms[ord](B[ind]))
    return _norms[ord](block_norm)


def add(A, B, meta):
    """ C = A + B. meta = kab, ka, kb """
    C = {ind: A[ind] + B[ind] for ind in meta[0]}
    for ind in meta[1]:
        C[ind] = A[ind].copy()
    for ind in meta[2]:
        C[ind] = B[ind].copy()
    return C


def sub(A, B, meta):
    """ C = A - B. meta = kab, ka, kb """
    C = {ind: A[ind] - B[ind] for ind in meta[0]}
    for ind in meta[1]:
        C[ind] = A[ind].copy()
    for ind in meta[2]:
        C[ind] = -B[ind]
    return C


def apxb(A, B, x, meta):
    """ C = A + x * B. meta = kab, ka, kb """
    C = {ind: A[ind] + x * B[ind] for ind in meta[0]}
    for ind in meta[1]:
        C[ind] = A[ind].copy()
    for ind in meta[2]:
        C[ind] = x * B[ind]
    return C


def scalar(A, B, meta):
    out = 0.
    for ind in meta:
        out += (A[ind].conj().reshape(-1)) @ (B[ind].reshape(-1))


dot_dict = {(0, 0): lambda x, y: x @ y,
            (0, 1): lambda x, y: x @ y.conj(),
            (1, 0): lambda x, y: x.conj() @ y,
            (1, 1): lambda x, y: x.conj() @ y.conj()}


def dot(A, B, conj, meta_dot):
    f = dot_dict[conj]  # proper conjugations
    C = {}
    for (out, ina, inb) in meta_dot:
        C[out] = f(A[ina], B[inb])    
    return C


# dotdiag_dict = {(0, 0): lambda x, y, dim: x * y.reshape(dim),
#                 (0, 1): lambda x, y, dim: x * y.reshape(dim).conj(),
#                 (1, 0): lambda x, y, dim: x.conj() * y.reshape(dim),
#                 (1, 1): lambda x, y, dim: x.conj() * y.reshape(dim).conj()}

# def dot_diag(A, B, conj, to_execute, a_con, a_ndim):
#     dim = np.ones(a_ndim, int)
#     dim[a_con] = -1
#     f = dotdiag_dict[conj]
#     C = {}
#     for in1, in2, out in to_execute:
#         C[out] = f(A[in1], B[in2], dim)
#     return C


# def trace_dot_diag(A, B, conj, to_execute, axis1, axis2, a_ndim):
#     dim = np.ones(a_ndim, int)
#     dim[axis2] = -1
#     f = dotdiag_dict[conj]

#     C = {}
#     for in1, in2, out in to_execute:
#         temp = np.trace(f(A[in1], B[in2], dim), axis1=axis1, axis2=axis2)
#         try:
#             C[out] = C[out] + temp
#         except KeyError:
#             C[out] = temp
#     return C

#####################################################
#     block merging, truncations and un-merging     #
#####################################################

def merge_to_matrix(A, order, meta_new, meta_mrg, dtype):
    """ New dictionary of blocks after merging into matrix. """
    Anew = {u: np.zeros(Du, dtype=_data_dtype[dtype]) for (u, Du) in meta_new}
    for (tn, to, Dsl, Dl, Dsr, Dr) in meta_mrg:
        Anew[tn][slice(*Dsl), slice(*Dsr)] = A[to].transpose(order).reshape(Dl, Dr)
    return Anew


def merge_one_leg(A, axis, order, meta_new, meta_mrg, dtype):
    """
    outputs new dictionary of blocks after fusing one leg
    """
    Anew = {u: np.zeros(Du, dtype=_data_dtype[dtype]) for (u, Du) in meta_new}
    for (tn, Ds, to, Do) in meta_mrg:
        slc = [slice(None)] * len(Do)
        slc[axis] = slice(*Ds)
        Anew[tn][tuple(slc)] = A[to].transpose(order).reshape(Do)  
    return Anew


def merge_to_dense(A, Dtot, meta, dtype):
    """ outputs full tensor """
    Anew = np.zeros(Dtot, dtype=_data_dtype[dtype])
    for (ind, Dss) in meta:
        Anew[tuple(slice(*Ds) for Ds in Dss)] = A[ind]
    return Anew


def unmerge_from_matrix(A, meta):
    """ unmerge matrix into single blocks """
    Anew = {}
    for (ind, indm, sl, sr, D) in meta:
        Anew[ind] = A[indm][slice(*sl), slice(*sr)].reshape(D)
    return Anew


def unmerge_from_diagonal(A, meta):
    """ unmerge matrix into single blocks """
    Anew = {}
    for (inew, iold, slc) in meta:
        Anew[inew] = A[iold][slice(*slc)]
    return Anew


def unmerge_one_leg(A, axis, ndim, meta):
    """ unmerge single leg """
    Anew = {}
    for (told, tnew, Dsl, Dnew) in meta:
        slc = [slice(None)] * A[told].ndim
        slc[axis] = slice(*Dsl)
        Anew[tnew] = np.reshape(A[told][tuple(slc)], Dnew).copy()
    return Anew

##############
#  tests
##############

def is_independent(A, B):
    """
    check if two arrays are identical, or share the same view.
    """
    return (A is B) or (A.base is B) or (A is B.base)

##############
#  multi dict operations
##############

def block(td, meta, dtype):

    def to_block(li, lD, level):
        if level > 0:
            oi = []
            for ii, DD in zip(li, lD):
                oi.append(to_block(ii, DD, level - 1))
            return oi
        else:
            key = tuple(li)
            try:
                return td[key].A[ind]
            except KeyError:
                return np.zeros(lD, dtype=_data_dtype[dtype])

    A = {}
    # to_execute = [(ind, pos, legs_ind, legs_D), ... ]
    # ind: index of the merged blocks
    # pos: non-trivial tensors in the block, rest is zero
    # legs_ind: all elements for all dimensions to be cloked
    # legs_D: and bond dimensions (including common ones)

    for ind, legs_ind, legs_D in meta:
        all_ind = np.array(list(product(*legs_ind)), dtype=int)
        all_D = np.array(list(product(*legs_D)), dtype=int)

        shape_ind = [len(x) for x in legs_D] + [-1]
        all_ind = list(np.reshape(all_ind, shape_ind))
        all_D = list(np.reshape(all_D, shape_ind))

        temp = to_block(all_ind, all_D, len(shape_ind) - 1)
        A[ind] = np.block(temp)
    return A
