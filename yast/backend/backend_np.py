"""Support of numpy as a data structure used by yast."""
from itertools import chain
import numpy as np
import scipy.linalg
try:
    import fbpca
except ModuleNotFoundError:
    import warnings
    warnings.warn("fbpca not available", Warning)

BACKEND_ID = "numpy"
DTYPE = {'float64': np.float64,
         'complex128': np.complex128}


def get_dtype(iterator):
    """ iterators of numpy arrays; returns np.complex128 if any array is complex else np.float64"""
    return np.complex128 if any(np.iscomplexobj(x) for x in iterator) else np.float64


def unique_dtype(t):
    dtypes = set(b.dtype for b in t.A.values())
    if len(dtypes) == 1:
        return str(tuple(dtypes)[0])
    return False


def random_seed(seed):
    np.random.seed(seed)


###################################
#     single tensor operations    #
###################################


def detach(x):
    return x


def detach_(x):
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


def get_device(x):
    return 'cpu'


def count_greater(x, cutoff):
    return np.sum(x > cutoff).item()


def real(x):
    return np.real(x)


def imag(x):
    return np.imag(x)


def max_abs(x):
    return np.abs(x).max()


def norm_matrix(x):
    return np.linalg.norm(x)


def expand_dims(x, axis):
    return np.expand_dims(x, axis)


def count_nonzero(x):
    return np.count_nonzero(x)


#########################
#    output numbers     #
#########################


def first_element(x):
    return x.flat[0]


def item(x):
    return x.item()


def sum_elements(A):
    """ sum of all elements of all tensors in A """
    return sum(x.sum() for x in A.values())


def norm(A, p):
    """ 'fro' for Frobenious; 'inf' for max(abs(A)) """
    if p == 'fro':
        return np.linalg.norm([np.linalg.norm(x) for x in A.values()])
    if p == 'inf':
        return max([np.abs(x).max() for x in A.values()])
    raise RuntimeError("Invalid norm type: %s" % str(p))


def norm_diff(A, B, meta, p):
    """ norm(A - B); meta = kab, ka, kb """
    if p == 'fro':
        return np.linalg.norm([np.linalg.norm(A[ind] - B[ind]) for ind in meta[0]]
                              + [np.linalg.norm(A[ind]) for ind in meta[1]]
                              + [np.linalg.norm(B[ind]) for ind in meta[2]])
    if p == 'inf':
        return max([np.abs(A[ind] - B[ind]).max() for ind in meta[0]]
                   + [np.abs(A[ind]).max() for ind in meta[1]]
                   + [np.abs(B[ind]).max() for ind in meta[2]])
    raise RuntimeError("Invalid norm type: %s" % str(p))


def entropy(A, alpha=1, tol=1e-12):
    """ von Neuman or Renyi entropy from svd's"""
    Snorm = np.sqrt(np.sum([np.sum(x ** 2) for x in A.values()]))
    if Snorm > 0:
        Smin = min([min(x) for x in A.values()])
        ent = []
        for x in A.values():
            x = x / Snorm
            x = x[x > tol]
            if alpha == 1:
                ent.append(-2 * np.sum(x * x * np.log2(x)))
            else:
                ent.append(np.sum(x**(2 * alpha)))
        ent = np.sum(ent)
        if alpha != 1:
            ent = np.log2(ent) / (1 - alpha)
        return ent, Smin, Snorm
    return Snorm, Snorm, Snorm  # this should be 0., 0., 0.


##########################
#     setting values     #
##########################


def dtype_scalar(x, dtype='float64', **kwargs):
    return DTYPE[dtype](x)


def zeros(D, dtype='float64', **kwargs):
    return np.zeros(D, dtype=DTYPE[dtype])


def ones(D, dtype='float64', **kwargs):
    return np.ones(D, dtype=DTYPE[dtype])


def randR(D, **kwargs):
    return 2 * np.random.random_sample(D).astype(DTYPE['float64']) - 1


def randC(D, **kwargs):
    return 2 * (np.random.random_sample(D) + 1j * np.random.random_sample(D)).astype(DTYPE['complex128']) - (1 + 1j)


def to_tensor(val, Ds=None, dtype='float64', **kwargs):
    # try:
    T = np.array(val, dtype=DTYPE[dtype])
    # except TypeError:
    #     T = np.array(val, dtype=DTYPE['complex128'])
    return T if Ds is None else T.reshape(Ds)


def to_mask(val):
    return val.nonzero()[0].ravel()


def square_matrix_from_dict(H, D=None, **kwargs):
    dtype = get_dtype(H.values())
    T = np.zeros((D, D), dtype=dtype)
    for (i, j), v in H.items():
        if i < D and j < D:
            T[i, j] = v
    return T


##################################
#     single dict operations     #
##################################

def requires_grad_(A, requires_grad=True):
    pass


def requires_grad(A):
    return False


def move_to(A, *args, **kwargs):
    return A


def conj(A, inplace):
    """ Conjugate dict of tensors; Force a copy in not in place. """
    if inplace:
        return {t: x.conj() for t, x in A.items()}
    return {t: x.copy().conj() for t, x in A.items()}


def trace(A, order, meta):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    Aout = {}
    for (tnew, told, Drsh, _) in meta:
        Atemp = np.trace(np.reshape(np.transpose(A[told], order), Drsh))
        if tnew in Aout:
            Aout[tnew] += Atemp
        else:
            Aout[tnew] = Atemp
    return Aout


def trace_with_mask(A, order, meta, msk12):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    Aout = {}
    for (tnew, told, Drsh, tt) in meta:
        Atemp = np.reshape(np.transpose(A[told], order), Drsh)
        Atemp = np.sum(Atemp[msk12[tt][0], msk12[tt][1]], axis=0)
        if tnew in Aout:
            Aout[tnew] += Atemp
        else:
            Aout[tnew] = Atemp
    return Aout


def transpose(A, axes, meta_transpose, inplace):
    """ Transpose; Force a copy if not inplace. """
    if inplace:
        return {new: np.transpose(A[old], axes=axes) for new, old in meta_transpose}
    return {new: np.transpose(A[old], axes=axes).copy() for new, old in meta_transpose}


def rsqrt(A, cutoff=0):
    res = {t: 1. / np.sqrt(x) for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1. / cutoff] = 0
    return res


def reciprocal(A, cutoff=0):
    res = {t: 1. / x for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1. / cutoff] = 0
    return res


def exp(A, step):
    return {t: np.exp(step * x) for t, x in A.items()}


def expm(A):
    return scipy.linalg.expm(A)


def sqrt(A):
    return {t: np.sqrt(x) for t, x in A.items()}


def absolute(A):
    return {t: np.abs(x) for t, x in A.items()}


def svd_lowrank(A, meta, D_block, n_iter, k_fac):
    U, S, V = {}, {}, {}
    for (iold, iU, iS, iV) in meta:
        k = min(min(A[iold].shape), D_block)
        U[iU], S[iS], V[iV] = fbpca.pca(A[iold], k=k, raw=True, n_iter=n_iter, l=k_fac * k)
    return U, S, V


def svd(A, meta):
    U, S, V = {}, {}, {}
    for (iold, iU, iS, iV) in meta:
        try:
            U[iU], S[iS], V[iV] = scipy.linalg.svd(A[iold], full_matrices=False)
        except scipy.linalg.LinAlgError:
            U[iU], S[iS], V[iV] = scipy.linalg.svd(A[iold], full_matrices=False, lapack_driver='gesvd')
    return U, S, V


def eigh(A, meta=None):
    S, U = {}, {}
    if meta is not None:
        for (ind, indS, indU) in meta:
            S[indS], U[indU] = np.linalg.eigh(A[ind])
    else:
        S, U = np.linalg.eigh(A)
    return S, U


def svd_S(A):
    S = {}
    for ind in A:
        try:
            S[ind] = scipy.linalg.svd(A[ind], full_matrices=False, compute_uv=False)
        except scipy.linalg.LinAlgError:
            S[ind] = scipy.linalg.svd(A[ind], full_matrices=False, lapack_driver='gesvd', compute_uv=False)
    return S


def qr(A, meta):
    Q, R = {}, {}
    for (ind, indQ, indR) in meta:
        Q[indQ], R[indR] = scipy.linalg.qr(A[ind], mode='economic')
        sR = np.sign(np.real(np.diag(R[indR])))
        sR[sR == 0] = 1
        # positive diag of R
        Q[indQ] = Q[indQ] * sR
        R[indR] = sR.reshape([-1, 1]) * R[indR]
    return Q, R


# def rq(A):
#     R, Q = {}, {}
#     for ind in A:
#         R[ind], Q[ind] = scipy.linalg.rq(A[ind], mode='economic')
#         sR = np.sign(np.real(np.diag(R[ind])))
#         sR[sR == 0] = 1
#         # positive diag of R
#         R[ind], Q[ind] = R[ind] * sR, sR.reshape([-1, 1]) * Q[ind]
#     return R, Q


def select_global_largest(S, D_keep, D_total, keep_multiplets, eps_multiplet, ordering):
    if ordering == 'svd':
        s_all = np.hstack([S[ind][:D_keep[ind]] for ind in S])
    elif ordering == 'eigh':
        s_all = np.hstack([S[ind][-D_keep[ind]:] for ind in S])
    Darg = D_total + int(keep_multiplets)
    order = s_all.argsort()[-1:-Darg-1:-1]
    if keep_multiplets:  # if needed, preserve multiplets within each sector
        s_all = s_all[order]
        gaps = np.abs(s_all)
        # compute gaps and normalize by larger singular value. Introduce cutoff
        gaps = np.abs(gaps[:len(s_all) - 1] - gaps[1:len(s_all)]) / gaps[0]  # / (gaps[:len(values) - 1] + 1.0e-16)
        gaps[gaps > 1.0] = 0.  # for handling vanishing values set to exact zero
        if gaps[D_total - 1] < eps_multiplet:
            # the chi is within the multiplet - find the largest chi_new < chi
            # such that the complete multiplets are preserved
            for i in range(D_total - 1, -1, -1):
                if gaps[i] > eps_multiplet:
                    order = order[:i + 1]
                    break
    return order


def eigs_which(val, which):
    if which == 'LM':
        return (-abs(val)).argsort()
    if which == 'SM':
        return abs(val).argsort()
    if which == 'LR':
        return (-val.real).argsort()
    # elif which == 'SR':
    return (val.real).argsort()


def range_largest(D_keep, D_total, ordering):
    if ordering == 'svd':
        return (0, D_keep)
    if ordering == 'eigh':
        return (D_total - D_keep, D_total)


def maximum(A):
    """ maximal element of A """
    return max(np.max(x) for x in A.values())


def embed(A, sl, tD):
    """ embeds old tensors A into larger zero blocks based on slices. """
    dtype = get_dtype(A.values())
    C = {}
    for t, val in A.items():
        C[t] = np.zeros(tD[t], dtype=dtype)
        C[t][sl[t]] = val
    return C


################################
#     two dicts operations     #
################################


def add(A, B, meta):
    """ C = A + B. meta = kab, ka, kb """
    C = {}
    for t, ab in meta:
        if ab == 'AB':
            C[t] = A[t] + B[t]
        elif ab == 'A':
            C[t] = A[t].copy()
        else:  # ab == 'B'
            C[t] = B[t].copy()
    return C


def sub(A, B, meta):
    """ C = A - B. meta = kab, ka, kb """
    C = {}
    for t, ab in meta:
        if ab == 'AB':
            C[t] = A[t] - B[t]
        elif ab == 'A':
            C[t] = A[t].copy()
        else:  # ab == 'B'
            C[t] = -B[t]
    return C


def apxb(A, B, x, meta):
    """ C = A + x * B. meta = kab, ka, kb """
    C = {}
    for t, ab in meta:
        if ab == 'AB':
            C[t] = A[t] + x * B[t]
        elif ab == 'A':
            C[t] = A[t].copy()
        else:  # ab == 'B'
            C[t] = x * B[t]
    return C


dot_dict = {(0, 0): lambda x, y: x @ y,
            (0, 1): lambda x, y: x @ y.conj(),
            (1, 0): lambda x, y: x.conj() @ y,
            (1, 1): lambda x, y: x.conj() @ y.conj()}


def vdot(A, B, cc, meta):
    f = dot_dict[cc]  # proper conjugations
    return np.sum([f(A[ind].reshape(-1), B[ind].reshape(-1)) for ind in meta])


def dot(A, B, cc, meta_dot):
    f = dot_dict[cc]  # proper conjugations
    C = {}
    for (out, ina, inb) in meta_dot:
        C[out] = f(A[ina], B[inb])
    return C


dotdiag_dict = {(0, 0): lambda x, y, dim: x * y.reshape(dim),
                (0, 1): lambda x, y, dim: x * y.reshape(dim).conj(),
                (1, 0): lambda x, y, dim: x.conj() * y.reshape(dim),
                (1, 1): lambda x, y, dim: x.conj() * y.reshape(dim).conj()}


def dot_diag(A, B, cc, meta, axis, a_ndim):
    dim = [1] * a_ndim
    dim[axis] = -1
    f = dotdiag_dict[cc]
    C = {}
    for ind_a, ind_b in meta:
        C[ind_a] = f(A[ind_a], B[ind_b], dim)
    return C


def mask_diag(A, B, meta, axis, a_ndim):
    slc1 = (slice(None),) * axis
    slc2 = (slice(None),) * (a_ndim - (axis + 1))
    Bslc = {k: v.nonzero() for k, v in B.items()}
    return {ind_a: A[ind_a][slc1 + Bslc[ind_b] + slc2] for ind_a, ind_b in meta}


def dot_nomerge(A, B, cc, oA, oB, meta):
    f = dot_dict[cc]  # proper conjugations
    C = {}
    for (ina, inb, out, Da, Db, Dout, _) in meta:
        temp = f(A[ina].transpose(oA).reshape(Da), B[inb].transpose(oB).reshape(Db)).reshape(Dout)
        try:
            C[out] += temp
        except KeyError:
            C[out] = temp
    return C


def dot_nomerge_masks(A, B, cc, oA, oB, meta, ma, mb):
    f = dot_dict[cc]  # proper conjugations
    C = {}
    for (ina, inb, out, Da, Db, Dout, tt) in meta:
        temp = f(A[ina].transpose(oA).reshape(Da)[:, ma[tt]], B[inb].transpose(oB).reshape(Db)[mb[tt], :]).reshape(Dout)
        try:
            C[out] += temp
        except KeyError:
            C[out] = temp
    return C
#####################################################
#     block merging, truncations and un-merging     #
#####################################################


def merge_blocks(A, order, meta_new, meta_mrg, *args, **kwargs):
    """ New dictionary of blocks after merging into n-dimensional array """
    dtype = get_dtype(A.values())
    Anew = {u: np.zeros(Du, dtype=dtype) for u, Du in zip(*meta_new)}
    for (tn, to, Dslc, Drsh) in meta_mrg:
        if to in A:
            slc = tuple(slice(*x) for x in Dslc)
            Anew[tn][slc] = A[to].transpose(order).reshape(Drsh)
    return Anew


def merge_to_dense(A, Dtot, meta, *args, **kwargs):
    """ Outputs full tensor. """
    dtype = get_dtype(A.values())
    Anew = np.zeros(Dtot, dtype=dtype)
    for (ind, Dss) in meta:
        Anew[tuple(slice(*Ds) for Ds in Dss)] = A[ind].reshape(tuple(Ds[1] - Ds[0] for Ds in Dss))
    return Anew


def merge_super_blocks(pos_tens, meta_new, meta_block, *args, **kwargs):
    """ Outputs new dictionary of blocks after creating super-tensor. """
    dtype = get_dtype(chain.from_iterable(t.A.values() for t in pos_tens.values()))
    Anew = {u: np.zeros(Du, dtype=dtype) for (u, Du) in meta_new}
    for (tind, pos, Dslc) in meta_block:
        slc = tuple(slice(*DD) for DD in Dslc)
        Anew[tind][slc] = pos_tens[pos].A[tind]  # .copy() # is copy required?
    return Anew


def unmerge_from_matrix(A, meta):
    """ unmerge matrix into single blocks """
    Anew = {}
    for (ind, indm, sl, sr, D) in meta:
        Anew[ind] = A[indm][slice(*sl), slice(*sr)].reshape(D)
    return Anew


def unmerge_from_array(A, meta):
    """ unmerge matrix into single blocks """
    Anew = {}
    for (tn, to, slc, D) in meta:
        sl = tuple(slice(*x) for x in slc)
        Anew[tn] = A[to][sl].reshape(D)
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
        Anew[tnew] = np.reshape(A[told][tuple(slc)], Dnew).copy()  # TODO check if this copy() is neccesary
    return Anew


#############
#   tests   #
#############


def is_complex(x):
    return np.iscomplexobj(x)


def is_independent(A, B):
    """
    check if two arrays are identical, or share the same view.
    """
    return (A is B) or (A.base is B) or (A is B.base)
