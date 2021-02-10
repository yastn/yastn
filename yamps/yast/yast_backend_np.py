import warnings
import numpy as np
import scipy as sp
try:
    import fbpca as pca 
except ImportError as e:
    warnings.warn("fbpca not available", Warning)


_data_dtype = {'float64': np.float64,
               'complex128': np.complex128}


def random_seed(seed):
    np.random.seed(seed)

###################################
#     single tensor operations    #
###################################

def detach(x):
    return x


def clone(x):
    return x.copy()


def copy(x):
    return x.copy()


def to_numpy(x):
    return x.copy()


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


def diag_diag(x):
    return np.diag(np.diag(x))


def count_greater(x, cutoff):
    return np.sum(x > cutoff)

#########################
#    output numbers     #
#########################

def first_element(x):
    return x.flat[0]


def item(x):
    return x.item()


def norm(A, ord):
    if ord == 'fro':
        return np.linalg.norm([np.linalg.norm(x) for x in A.values()])
    elif ord == 'inf':
        return max( [ np.abs(x).max() for x in A.values()] )


def norm_diff(A, B, ord, meta):
    """ norm(A - B); meta = kab, ka, kb """
    if ord == 'fro':
        return np.linalg.norm([np.linalg.norm(A[ind]-B[ind]) for ind in meta[0]] +\
                              [np.linalg.norm(A[ind]) for ind in meta[1]] +\
                              [np.linalg.norm(B[ind]) for ind in meta[2]])
    elif ord == 'inf':
        return max([np.abs(A[ind]-B[ind]).max() for ind in meta[0]] +\
                   [np.abs(A[ind]).max() for ind in meta[1]] +\
                   [np.abs(B[ind]).max() for ind in meta[2]])


def entropy(A, alpha=1, tol=1e-12):
    """ von Neuman or Renyi entropy from svd's"""
    normalization = np.sqrt(np.sum([np.sum(x ** 2) for x in A.values()]))
    if normalization > 0:
        Smin = min([min(x) for x in A.values()])
        entropy = []
        for x in A.values():
            x = x / normalization
            x = x[x > tol]
            if alpha == 1:
                entropy.append(-2 * np.sum(x * x * np.log2(x)))
            else:
                entropy.append(x**(2 * alpha))
        entropy = np.sum(entropy)
        if alpha != 1:
            entropy = np.log2(entropy) / (1 - alpha)
        return entropy, Smin, normalization
    return normalization, normalization, normalization  # this hould be 0., 0., 0.


##########################
#     setting values     #
##########################

def zero_scalar(dtype='float64', device='cpu'):
    return _data_dtype[dtype](0)


def zeros(D, dtype='float64', device='cpu'):
    return np.zeros(D, dtype=_data_dtype[dtype])


def ones(D, dtype='float64', device='cpu'):
    return np.ones(D, dtype=_data_dtype[dtype])


def randR(D, dtype='float64', device='cpu'):
    return 2 * np.random.random_sample(D).astype(_data_dtype[dtype]) - 1


def rand(D, dtype='float64', device='cpu'):
    if dtype == 'float64':
        return randR(D)
    elif dtype == 'complex128':
        return 2 * np.random.random_sample(D) - 1 + 1j * (2 * np.random.random_sample(D) - 1)


def to_tensor(val, Ds=None, dtype='float64', device='cpu'):
    return np.array(val, dtype=_data_dtype[dtype]) if Ds is None else np.array(val, dtype=_data_dtype[dtype]).reshape(Ds)

##################################
#     single dict operations     #
##################################

def move_to_device(A, device):
    return A


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


def transpose(A, axes, meta_transpose, inplace):
    """ Transpose; Force a copy if not inplace. """
    if inplace:
        return {new: np.transpose(A[old], axes=axes) for old, new in meta_transpose}
    return {new: np.transpose(A[old], axes=axes).copy() for old, new in meta_transpose}
    

def invsqrt(A, cutoff=0):
    res = {t: np.sqrt(x) for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0.
    return res


def invsqrt_diag(A, cutoff=0):
    res = {t: np.sqrt(np.diag(x)) for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0.
    return {t: np.diag(x) for t, x in res.items()}


def inv(A, cutoff=0):
    res = {t: 1./x for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0.
    return res


def inv_diag(A, cutoff=0):
    res = {t: 1./ np.diag(x) for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0.
    return {t: np.diag(x) for t, x in res.items()}


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


def eigh(A, meta):
    S, U = {}, {}
    for (ind, indS, indU) in meta:
        S[indS], U[indU] = np.linalg.eigh(A[ind])
    return S, U


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

################################
#     two dicts operations     #
################################

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
    return np.sum([(A[ind].conj().reshape(-1)) @ (B[ind].reshape(-1)) for ind in meta])


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

def merge_to_matrix(A, order, meta_new, meta_mrg, dtype, device='cpu'):
    """ New dictionary of blocks after merging into matrix. """
    Anew = {u: np.zeros(Du, dtype=_data_dtype[dtype]) for (u, Du) in meta_new}
    for (tn, to, Dsl, Dl, Dsr, Dr) in meta_mrg:
        Anew[tn][slice(*Dsl), slice(*Dsr)] = A[to].transpose(order).reshape(Dl, Dr)
    return Anew


def merge_one_leg(A, axis, order, meta_new, meta_mrg, dtype, device='cpu'):
    """
    outputs new dictionary of blocks after fusing one leg
    """
    Anew = {u: np.zeros(Du, dtype=_data_dtype[dtype]) for (u, Du) in meta_new}
    for (tn, Ds, to, Do) in meta_mrg:
        slc = [slice(None)] * len(Do)
        slc[axis] = slice(*Ds)
        Anew[tn][tuple(slc)] = A[to].transpose(order).reshape(Do)  
    return Anew


def merge_to_dense(A, Dtot, meta, dtype, device='cpu'):
    """ outputs full tensor """
    Anew = np.zeros(Dtot, dtype=_data_dtype[dtype])
    for (ind, Dss) in meta:
        Anew[tuple(slice(*Ds) for Ds in Dss)] = A[ind]
    return Anew

def merge_super_blocks(pos_tens, meta_new, meta_block, dtype, device='cpu'):
    """ Outputs new dictionary of blocks after creating super-tensor. """
    Anew = {u: np.zeros(Du, dtype=_data_dtype[dtype]) for (u, Du) in meta_new}
    for (tind, pos, Dslc) in meta_block: 
        slc = tuple(slice(*DD) for DD in Dslc)
        Anew[tind][slc] = pos_tens[pos].A[tind]# .copy() # is copy required?
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


def unmerge_one_leg(A, axis, meta):
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