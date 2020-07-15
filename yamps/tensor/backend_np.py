import numpy as np
import scipy as sp
import fbpca as pca
from itertools import groupby
from itertools import product
from itertools import accumulate
from functools import reduce
from operator import mul
import numba as nb

_select_dtype = {'float64': np.float64,
                 'complex128': np.complex128}


def random_seed(seed):
    np.random.seed(seed)

##############
# single element calculations
##############


def copy(x):
    return x.copy()


def to_numpy(nparray):
    return nparray


def first_el(x):
    return x.flat[0]


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

##############
#  setting values
##############


def zeros(D, dtype='float64'):
    return np.zeros(D, dtype=_select_dtype[dtype])


def ones(D, dtype='float64'):
    return np.ones(D, dtype=_select_dtype[dtype])


def randR(D, dtype='float64'):
    return 2 * np.random.random_sample(D).astype(_select_dtype[dtype]) - 1


def rand(D, dtype='float64'):
    if dtype == 'float64':
        return randR(D, dtype=dtype)
    elif dtype == 'complex128':
        return 2 * np.random.random_sample(D) - 1 + 1j * (2 * np.random.random_sample(D) - 1)


def to_tensor(val, Ds=None, dtype='float64'):
    return np.array(val, dtype=_select_dtype[dtype]) if Ds is None else np.array(val, dtype=_select_dtype[dtype]).reshape(Ds)


##############
# single dict operations
##############


def conj(A):
    """ conjugate dict of tensors """
    cA = {}
    for ind in A:
        cA[ind] = A[ind].copy()
        np.conj(cA[ind], cA[ind])
    return cA


def trace_axis(A, to_execute, axis):
    """ sum dict of tensors according to: to_execute =[(ind_in, ind_out), ...]
        repeating ind_out are added"""
    cA = {}
    for old, new in to_execute:
        Atemp = np.sum(A[old], axis=axis)
        if new in cA:
            cA[new] = cA[new] + Atemp
        else:
            cA[new] = Atemp
    return cA


def trace(A, to_execute, in1, in2, out):
    """ trace dict of tensors according to: to_execute =[(ind_in, ind_out), ...]
        repeating ind_out are added"""
    cA = {}
    order = in1 + in2 + out
    for task in to_execute:
        D1 = tuple(A[task[0]].shape[ii] for ii in in1)
        D2 = tuple(A[task[0]].shape[ii] for ii in in2)
        D3 = tuple(A[task[0]].shape[ii] for ii in out)
        pD1 = reduce(mul, D1, 1)
        pD2 = reduce(mul, D2, 1)
        Atemp = np.reshape(np.transpose(A[task[0]], order), (pD1, pD2) + D3)
        if task[1] in cA:
            cA[task[1]] = cA[task[1]] + np.trace(Atemp)
        else:
            cA[task[1]] = np.trace(Atemp)
    return cA


def transpose_local(A, axes, to_execute):
    """
    transpose in place.
    """
    cA = {}
    for old, new in to_execute:
        cA[new] = A[old].transpose(axes)
    return cA


def transpose(A, axes, to_execute):
    """
    transpose forcing a copy.
    """
    cA = {}
    for old, new in to_execute:
        cA[new] = np.transpose(A[old], axes=axes).copy()
    return cA


def invsqrt(A):
    cA = {}
    for ind in A:
        cA[ind] = 1. / np.sqrt(A[ind])
    return cA


def inv(A):
    cA = {}
    for ind in A:
        cA[ind] = 1. / A[ind]
    return cA


def exp(A, step):
    cA = {}
    for ind in A:
        cA[ind] = np.exp(A[ind] * step)
    return cA


def sqrt(A):
    cA = {}
    for ind in A:
        cA[ind] = np.sqrt(A[ind])
    return cA


def svd(A, truncated=False, Dblock=np.inf, nbit=60, kfac=6):
    U, S, V = {}, {}, {}
    for ind in A:
        if truncated and min(A[ind].shape) > kfac * Dblock:
            U[ind], S[ind], V[ind] = pca.pca(A[ind], k=Dblock, raw=True, n_iter=nbit, l=kfac * Dblock)
        else:
            try:
                U[ind], S[ind], V[ind] = sp.linalg.svd(A[ind], full_matrices=False)
            except sp.linalg.LinAlgError:
                U[ind], S[ind], V[ind] = sp.linalg.svd(A[ind], full_matrices=False, lapack_driver='gesvd')
    return U, S, V


def svd_no_uv(A):
    S = {}
    for ind in A:
        try:
            S[ind] = sp.linalg.svd(A[ind], full_matrices=False, compute_uv=False)
        except sp.linalg.LinAlgError:
            S[ind] = sp.linalg.svd(A[ind], full_matrices=False, lapack_driver='gesvd', compute_uv=False)
    return S


def qr(A):
    Q, R = {}, {}
    for ind in A:
        Q[ind], R[ind] = sp.linalg.qr(A[ind], mode='economic')
        sR = np.sign(np.real(np.diag(R[ind])))
        sR[sR == 0] = 1
        # positive diag of R
        Q[ind], R[ind] = Q[ind] * sR, sR.reshape([-1, 1]) * R[ind]
    return Q, R


def rq(A):
    R, Q = {}, {}
    for ind in A:
        R[ind], Q[ind] = sp.linalg.rq(A[ind], mode='economic')
        sR = np.sign(np.real(np.diag(R[ind])))
        sR[sR == 0] = 1
        # positive diag of R
        R[ind], Q[ind] = R[ind] * sR, sR.reshape([-1, 1]) * Q[ind]
    return R, Q


def eigh(A):
    S, U = {}, {}
    for ind in A:
        S[ind], U[ind] = np.linalg.eigh(A[ind])
    return S, U


_norms = {'fro': np.linalg.norm, 'inf': lambda x: np.abs(x).max()}


def norm(A, ord):
    block_norm = [0.]
    for x in A.values():
        block_norm.append(_norms[ord](x))
    return _norms[ord](block_norm)


def entropy(A, alpha=1, tol=1e-12):
    temp = 0.
    for x in A.values():
        temp += np.sum(np.abs(x) ** 2)
    no = np.sqrt(temp)

    ent = 0.
    Smin = np.inf
    if no > 0:
        for x in A.values():
            Smin = min(Smin, min(x))
            x = x / no
            if alpha == 1:
                x = x[x > tol]
                ent += -2 * sum(x * x * np.log2(x))
            else:
                ent += x**(2 * alpha)
        if alpha != 1:
            ent = np.log2(ent) / (1 - alpha)
    return ent, Smin, no

##############
# two dicts operations
##############


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


def add(aA, bA, to_execute, x=1):
    """ add two dicts of tensors, according to to_execute = [(ind, meta), ...]
        meta = 0; add both
        meta = 1; only from aA
        meta = 2; only from bA"""
    cA = {}
    for ind in to_execute:
        if (ind[1] == 0):
            cA[ind[0]] = aA[ind[0]] + x * bA[ind[0]]
        elif (ind[1] == 1):
            cA[ind[0]] = aA[ind[0]].copy()
        elif (ind[1] == 2):
            cA[ind[0]] = x * bA[ind[0]].copy()
    return cA


def sub(aA, bA, to_execute):
    """ subtract two dicts of tensors, according to to_execute = [(ind, meta), ...]
        meta = 0; add both
        meta = 1; only from aA
        meta = 2; only from bA"""
    cA = {}
    for ind in to_execute:
        if (ind[1] == 0):
            cA[ind[0]] = aA[ind[0]] - bA[ind[0]]
        elif (ind[1] == 1):
            cA[ind[0]] = aA[ind[0]].copy()
        elif (ind[1] == 2):
            cA[ind[0]] = -bA[ind[0]]
    return cA


def dot_matrix(x, y):
    return x @ y


def dotC_matrix(x, y):
    return x @ y.conj()


def Cdot_matrix(x, y):
    return x.conj() @ y


def CdotC_matrix(x, y):
    return x.conj() @ y.conj()


dot_dict = {(0, 0): dot_matrix, (0, 1): dotC_matrix,
            (1, 0): Cdot_matrix, (1, 1): CdotC_matrix}


def dot_merged(A, B, conj):
    C = {}
    if conj == (0, 0):
        for ind in A:
            C[ind] = A[ind] @ B[ind]
    elif conj == (0, 1):
        for ind in A:
            C[ind] = A[ind] @ B[ind].conj()
    elif conj == (1, 0):
        for ind in A:
            C[ind] = A[ind].conj() @ B[ind]
    elif conj == (1, 1):
        for ind in A:
            C[ind] = (A[ind] @ B[ind]).conj()
    return C


# @nb.njit()
# def tr_rs(Al, a_all):
#     # sl = nb.typed.List()
#     # for ii in range(len(Al)):
#     #     sl.append(np.empty((pDl[ii], pDlc[ii])))
#     for ii in nb.prange(len(Al)):
#         Al[ii] = Al[ii].transpose(a_all).copy()
#     return Al
#       numba_list = nb.typed.List()
# [numba_list.append(x) for x in A.values()]
# numba_list = tr_rs(numba_list, a_all)
# Atemp = {in1: x.reshape(dl, dc) for in1, x, dl, dc in zip(A, numba_list, pDl, pDlc)}

# @profile
def dot(A, B, conj, to_execute, a_out, a_con, b_con, b_out, dtype='float64'):
    a_all = a_out + a_con  # order for transpose in A
    b_all = b_con + b_out  # order for transpose in B
    f = dot_dict[conj]  # proper conjugations
    if len(to_execute) == 1:
        in1, in2, out = to_execute[0]
        AA, BB = A[in1], B[in2]
        Dl = tuple(AA.shape[ii] for ii in a_out)
        Dc = tuple(AA.shape[ii] for ii in a_con)
        Dr = tuple(BB.shape[ii] for ii in b_out)
        pDl = reduce(mul, Dl, 1)
        pDc = reduce(mul, Dc, 1)
        pDr = reduce(mul, Dr, 1)
        C = {out: f(AA.transpose(a_all).reshape(pDl, pDc), BB.transpose(b_all).reshape(pDc, pDr)).reshape(Dl + Dr)}
    else:
        Andim = len(a_con) + len(a_out)
        Bndim = len(b_con) + len(b_out)
        DA = np.array([A[ind].shape for ind in A], dtype=np.int).reshape(len(A), Andim)  # bond dimensions of A
        DB = np.array([B[ind].shape for ind in B], dtype=np.int).reshape(len(B), Bndim)  # bond dimensions of B
        Dl = DA[:, np.array(a_out, dtype=np.int)]  # bond dimension on left legs
        Dlc = DA[:, np.array(a_con, dtype=np.int)]  # bond dimension on contracted legs
        Dcr = DB[:, np.array(b_con, dtype=np.int)]  # bond dimension on contracted legs
        Dr = DB[:, np.array(b_out, dtype=np.int)]  # bond dimension on right legs
        pDl = np.multiply.reduce(Dl, axis=1, dtype=np.int)  # their product
        pDlc = np.multiply.reduce(Dlc, axis=1, dtype=np.int)
        pDcr = np.multiply.reduce(Dcr, axis=1, dtype=np.int)
        pDr = np.multiply.reduce(Dr, axis=1, dtype=np.int)

        Atemp = {in1: A[in1].transpose(a_all).reshape(d1, d2) for in1, d1, d2 in zip(A, pDl, pDlc)}
        Btemp = {in2: B[in2].transpose(b_all).reshape(d1, d2) for in2, d1, d2 in zip(B, pDcr, pDr)}

        Dl = {in1: tuple(dl) for in1, dl in zip(A, Dl)}
        Dr = {in2: tuple(dr) for in2, dr in zip(B, Dr)}
        C = {}

        # DC = {}
        # for in1, in2, out in to_execute:   # can use if in place of try;exept
        #     temp = f(Atemp[in1], Btemp[in2])
        #     try:
        #         C[out] += temp
        #     except KeyError:
        #         C[out] = temp
        #         DC[out] = Dl[in1] + Dr[in2]
        # for out in C:
        #     C[out] = C[out].reshape(DC[out])

        # to_execute = groupby(sorted(to_execute, key=lambda x: x[2]), key=lambda x: x[2])
        # for out, execute in to_execute:
        #     execute = list(execute)
        #     le = len(execute)
        #     in1, in2, _ = execute[-1]
        #     dl = Dl[in1]
        #     dr = Dr[in2]
        #     if le > 1:
        #         pdl = Atemp[in1].shape[0]
        #         pdr = Btemp[in2].shape[1]
        #         temp = np.empty((le, pdl, pdr), dtype=_select_dtype[dtype])
        #         for ii in range(le):
        #             in1, in2, _ = execute[ii]
        #             np.matmul(Atemp[in1], Btemp[in2], out=temp[ii])
        #         C[out] = np.sum(temp, axis=0).reshape(dl + dr)
        #     else:
        #         C[out] = np.matmul(Atemp[in1], Btemp[in2]).reshape(dl + dr)

        to_execute = sorted(to_execute, key=lambda x: x[2])
        multiplied = [f(Atemp[in1], Btemp[in2]) for in1, in2, _ in to_execute[::-1]]
        to_execute = groupby(to_execute, key=lambda x: x[2])
        for out, execute in to_execute:
            C[out] = multiplied.pop()
            in1, in2, _ = next(execute)
            for _ in execute:
                C[out] += multiplied.pop()
            C[out] = C[out].reshape(Dl[in1] + Dr[in2])
    return C


def dotdiag_matrix(x, y, dim):
    return x * y.reshape(dim)


def dotdiagC_matrix(x, y, dim):
    return x * y.reshape(dim).conj()


def Cdotdiag_matrix(x, y, dim):
    return x.conj() * y.reshape(dim)


def CdotdiagC_matrix(x, y, dim):
    return x.conj() * y.reshape(dim).conj()


dotdiag_dict = {(0, 0): dotdiag_matrix, (0, 1): dotdiagC_matrix,
                (1, 0): Cdotdiag_matrix, (1, 1): CdotdiagC_matrix}


def dot_diag(A, B, conj, to_execute, a_con, a_ndim):
    dim = np.ones(a_ndim, int)
    dim[a_con] = -1
    f = dotdiag_dict[conj]

    C = {}
    for in1, in2, out in to_execute:
        C[out] = f(A[in1], B[in2], dim)
    return C


def trace_dot_diag(A, B, conj, to_execute, axis1, axis2, a_ndim):
    dim = np.ones(a_ndim, int)
    dim[axis2] = -1
    f = dotdiag_dict[conj]

    C = {}
    for in1, in2, out in to_execute:
        temp = np.trace(f(A[in1], B[in2], dim), axis1=axis1, axis2=axis2)
        try:
            C[out] = C[out] + temp
        except KeyError:
            C[out] = temp
    return C


##############
# block merging, truncations and un-merging
##############

# @profile
# def merge_blocks(A, to_execute, out_l, out_r, dtype='float64'):
#     """ merge blocks depending on the cut """
#     out_all = out_l + out_r
#     Amerged, order_l, order_r = {}, {}, {}
#     for tcut, tts in to_execute:
#         if len(tts) > 1:
#             Dl, Dr = {}, {}
#             for tl, tr, ind in tts:
#                 if tl not in Dl:
#                     s = A[ind].shape
#                     Dl[tl] = tuple(s[ii] for ii in out_l)
#                 if tr not in Dr:
#                     s = A[ind].shape
#                     Dr[tr] = tuple(s[ii] for ii in out_r)

#             ind_l, ind_r = sorted(Dl), sorted(Dr)
#             pDl = {tl: reduce(mul, Dl[tl], 1) for tl in ind_l}
#             pDr = {tr: reduce(mul, Dr[tr], 1) for tr in ind_r}

#             orl = {ind: (Dl[ind], slice(j - i, j)) for ind, i, j in zip(ind_l, pDl.values(), accumulate(pDl.values()))}
#             orr = {ind: (Dr[ind], slice(j - i, j)) for ind, i, j in zip(ind_r, pDr.values(), accumulate(pDr.values()))}

#             Atemp = np.zeros((sum(pDl.values()), sum(pDr.values())), dtype=_select_dtype[dtype])

#             for tl, tr, ind in tts:
#                 Atemp[orl[tl][1], orr[tr][1]] = A[ind].transpose(out_all).reshape(pDl[tl], pDr[tr])

#             order_l[tcut] = orl
#             order_r[tcut] = orr
#             Amerged[tcut] = Atemp
#         else:
#             tl, tr, ind = tts[0]
#             Dl = tuple(A[ind].shape[ii] for ii in out_l)
#             Dr = tuple(A[ind].shape[ii] for ii in out_r)
#             pDl = reduce(mul, Dl, 1)
#             pDr = reduce(mul, Dr, 1)
#             order_l[tcut] = {tl: (Dl, slice(None))}  # info for un-grouping
#             order_r[tcut] = {tr: (Dr, slice(None))}
#             Amerged[tcut] = np.transpose(A[ind], out_all).reshape(pDl, pDr)
#     return Amerged, order_l, order_r


# @profile
def merge_blocks(A, to_execute, out_l, out_r, dtype='float64'):
    """ merge blocks depending on the cut """
    out_all = out_l + out_r
    Andim = len(out_all)

    lA = [A[ind] for _, tts in to_execute for _, _, ind in tts]
    aDA = np.array([x.shape for x in lA], dtype=np.int).reshape(len(lA), Andim)
    aDl = aDA[:, np.array(out_l, dtype=np.int)]  # bond dimension on left legs
    aDr = aDA[:, np.array(out_r, dtype=np.int)]  # bond dimension on right legs
    apDl = np.multiply.reduce(aDl, axis=1, dtype=np.int)  # their product
    apDr = np.multiply.reduce(aDr, axis=1, dtype=np.int)

    Are = [x.transpose(out_all).reshape(pl, pr) for x, pl, pr in zip(lA, apDl, apDr)]
    Are = Are[::-1]

    Amerged, order_l, order_r = {}, {}, {}

    ii = 0  # assume here that tts are ordered according to tl
    for tcut, tts in to_execute:
        if len(tts) > 1:
            Dl, pDl, ind_tl, Dr, pDr, tr_list = [], [], [], {}, {}, {}
            for tl, group_tr in groupby(tts, key=lambda x: x[0]):
                Dl.append(tuple(aDl[ii]))
                pDl.append(apDl[ii])
                ind_tl.append(tl)
                tr_list = [x[1] for x in group_tr]
                for jj, tr in enumerate(tr_list):
                    if tr not in Dr:
                        Dr[tr] = tuple(aDr[ii + jj])
                        pDr[tr] = apDr[ii + jj]
                ii += len(tr_list)

            ind_tr = sorted(Dr)
            ipDr = [pDr[ind] for ind in ind_tr]

            orl = {tl: (x, slice(j - i, j)) for tl, x, i, j in zip(ind_tl, Dl, pDl, accumulate(pDl))}
            orr = {tr: (Dr[tr], slice(j - i, j)) for tr, i, j in zip(ind_tr, ipDr, accumulate(ipDr))}
            order_l[tcut] = orl
            order_r[tcut] = orr

            # version with slices;
            Atemp = np.zeros((sum(pDl), sum(ipDr)), dtype=_select_dtype[dtype])
            for tl, tr, _ in tts:
                Atemp[orl[tl][1], orr[tr][1]] = Are.pop()
            Amerged[tcut] = Atemp
        else:
            tl, tr, _ = tts[0]
            order_l[tcut] = {tl: (tuple(aDl[ii]), slice(None))}  # info for un-grouping
            order_r[tcut] = {tr: (tuple(aDr[ii]), slice(None))}
            ii += 1
            Amerged[tcut] = Are.pop()
    return Amerged, order_l, order_r

# @profile
# def merge_blocks(A, to_execute, out_l, out_r, dtype='float64'):
#     """ merge blocks depending on the cut """
#     out_all = out_l + out_r
#     Andim = len(out_all)

#     lA = [A[ind] for _, tts in to_execute for _, _, ind in tts]
#     aDA = np.array([x.shape for x in lA], dtype=np.int).reshape(len(lA), Andim)
#     aDl = aDA[:, np.array(out_l, dtype=np.int)]  # bond dimension on left legs
#     aDr = aDA[:, np.array(out_r, dtype=np.int)]  # bond dimension on right legs
#     apDl = np.multiply.reduce(aDl, axis=1, dtype=np.int)  # their product
#     apDr = np.multiply.reduce(aDr, axis=1, dtype=np.int)

#     Are = [x.transpose(out_all).reshape(pl, pr) for x, pl, pr in zip(lA, apDl, apDr)]
#     Are = Are[::-1]

#     Amerged, order_l, order_r = {}, {}, {}
#     Azeros = np.zeros((max(apDl), max(apDr)), dtype=_select_dtype[dtype])
#     ii = 0  # assume here that tts are ordered according to tl
#     for tcut, tts in to_execute:
#         if len(tts) > 1:
#             Dl, pDl, ind_tl, Dr, pDr, tr_list = [], [], [], {}, {}, {}
#             for tl, group_tr in groupby(tts, key=lambda x: x[0]):
#                 Dl.append(tuple(aDl[ii]))
#                 pDl.append(apDl[ii])
#                 ind_tl.append(tl)
#                 tr_list[tl] = [x[1] for x in group_tr]
#                 for jj, tr in enumerate(tr_list[tl]):
#                     if tr not in Dr:
#                         Dr[tr] = tuple(aDr[ii + jj])
#                         pDr[tr] = apDr[ii + jj]
#                 ii += len(tr_list[tl])

#             ind_tr = sorted(Dr)
#             ipDr = [pDr[ind] for ind in ind_tr]
#             order_l[tcut] = {tl: (x, slice(j - i, j)) for tl, x, i, j in zip(ind_tl, Dl, pDl, accumulate(pDl))}
#             order_r[tcut] = {tr: (Dr[tr], slice(j - i, j)) for tr, i, j in zip(ind_tr, ipDr, accumulate(ipDr))}

#             # version with stack;
#             form_matrix = []
#             for tl, dl in zip(ind_tl, pDl):
#                 form_matrix.append([Are.pop() if ir in tr_list[tl] else Azeros[:dl, :pDr[ir]] for ir in ind_tr])
#             Amerged[tcut] = np.block(form_matrix)
#         else:
#             tl, tr, _ = tts[0]
#             order_l[tcut] = {tl: (tuple(aDl[ii]), slice(None))}  # info for un-grouping
#             order_r[tcut] = {tr: (tuple(aDr[ii]), slice(None))}
#             ii += 1
#             Amerged[tcut] = Are.pop()
#     return Amerged, order_l, order_r


def group_legs(A, to_execute, axes, rest, new_axis, dtype='float64'):
    """ merge blocks depending on the cut """
    out_all = rest[:new_axis] + axes + rest[new_axis:]
    Anew, leg_order = {}, {}
    slc = [slice(None)] * (len(rest) + 1)

    for tcut, tts in to_execute:
        Din, pDin = {}, {}
        for _, tin, _, ind in tts:
            if tin not in Din:
                Din[tin] = tuple(A[ind].shape[ii] for ii in axes)

        ind_in = sorted(Din)
        pDin = {tin: reduce(mul, Din[tin], 1) for tin in ind_in}

        leg_order[tcut] = {ind: (Din[ind], slice(j - i, j)) for ind, i, j in zip(ind_in, pDin.values(), accumulate(pDin.values()))}
        cpDin = sum(pDin.values())
        Dnew = {}
        for _, tin, tnew, ind in tts:
            if tnew not in Anew:
                Dnew[tnew] = tuple(A[ind].shape[ii] for ii in rest)
                Anew[tnew] = np.zeros((*Dnew[tnew][:new_axis], cpDin, *Dnew[tnew][new_axis:]), dtype=_select_dtype[dtype])

            slc[new_axis] = leg_order[tcut][tin][1]
            Anew[tnew][tuple(slc)] = A[ind].transpose(out_all).reshape(*Dnew[tnew][:new_axis], pDin[tin], *Dnew[tnew][new_axis:])

    return Anew, leg_order


def ungroup_leg(A, axis, ndim, leg_order, to_execute):
    """ ungroup single leg """
    Anew = {}
    slc = [slice(None)] * ndim
    for tcut, tout, told, tnew in to_execute:
        Dn, slc[axis] = leg_order[tcut][tout]
        shp = A[told].shape
        shp = shp[:axis] + Dn + shp[axis + 1:]
        Anew[tnew] = np.reshape(A[told][tuple(slc)], shp).copy()
    return Anew


def slice_none(d):
    return {ind: slice(None) for ind in d}


def slice_S(S, tol=0., Dblock=np.inf, Dtotal=np.inf, decrease=True):
    r"""Gives slices for truncation of 1d matrices.

    decrease = True assumes that S[][0] is largest -- like in svd
    decrease = False assumes that S[][-1] is largest -- like in eigh
    """
    maxS, Dmax = 0., {}
    for ind in S:
        maxS = max(maxS, S[ind][0], S[ind][-1])
        Dmax[ind] = min(Dblock, S[ind].size)
    # truncate to given relative tolerance
    if (tol > 0) and (maxS > 0):
        for ind in Dmax:
            Dmax[ind] = min(Dmax[ind], np.sum(S[ind] >= maxS * tol))
    # truncate to total bond dimension
    if sum(Dmax[ind] for ind in Dmax) > Dtotal:
        if decrease:
            s_all = np.hstack([S[ind][:Dmax[ind]] for ind in Dmax])
        else:
            s_all = np.hstack([S[ind][-Dmax[ind]:] for ind in Dmax])
        order = s_all.argpartition(- Dtotal - 1)[-Dtotal:]
        low = 0
        for ind in Dmax:
            high = low + Dmax[ind]
            Dmax[ind] = np.sum((low <= order) & (order < high))
            low = high
    # give slices for truncation
    Dcut = {}
    if decrease:
        for ind in Dmax:
            if Dmax[ind] > 0:
                Dcut[ind] = slice(Dmax[ind])
    else:
        for ind in Dmax:
            if Dmax[ind] > 0:
                Dcut[ind] = slice(-Dmax[ind], None)
    return Dcut


def unmerge_blocks_diag(S, Dcut):
    Sout = {}
    for ind in Dcut:
        Sout[ind] = S[ind][Dcut[ind]]
    return Sout


def unmerge_blocks_left(U, order_l, Dcut):
    """ select non-zero sectors; and truncate u, s, v to newD """
    Uout = {}
    Dc = (-1,)
    for tcut in Dcut:  # fill blocks
        for tl, (Dl, slice_l) in order_l[tcut].items():
            Uout[tl] = np.reshape(U[tcut][slice_l, Dcut[tcut]], Dl + Dc)
    return Uout


def unmerge_blocks_right(V, order_r, Dcut):
    """ select non-zero sectors; and truncate u, s, v to newD"""
    Vout = {}
    Dc = (-1,)
    for tcut in Dcut:  # fill blocks
        for tr, (Dr, slice_r) in order_r[tcut].items():
            Vout[tr] = np.reshape(V[tcut][Dcut[tcut], slice_r], Dc + Dr)
    return Vout


def unmerge_blocks(C, order_l, order_r):
    Cout = {}
    for tcut in C:
        for tl, (Dl, slice_l) in order_l[tcut].items():
            for tr, (Dr, slice_r) in order_r[tcut].items():
                ind = tl + tr
                Cout[ind] = C[tcut][slice_l, slice_r].reshape(Dl + Dr)
    return Cout

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


def block(Ad, to_execute):

    def to_block(li, lD, level):
        if level > 0:
            oi = []
            for ii, DD in zip(li, lD):
                oi.append(to_block(ii, DD, level - 1))
            return oi
        else:
            key = tuple(li)
            try:
                return Ad[key][ind]
            except KeyError:
                return np.zeros(lD)

    A = {}
    # to_execute = [(ind, pos, legs_ind, legs_D), ... ]
    # ind: index of the merged blocks
    # pos: non-trivial tensors in the block, rest is zero
    # legs_ind: all elements for all dimensions to be cloked
    # legs_D: and bond dimensions (including common ones)

    for ind, pos, legs_ind, legs_D in to_execute:
        all_ind = np.array(list(product(*legs_ind)), dtype=int)
        all_D = np.array(list(product(*legs_D)), dtype=int)

        shape_ind = [len(x) for x in legs_D] + [-1]
        all_ind = list(np.reshape(all_ind, shape_ind))
        all_D = list(np.reshape(all_D, shape_ind))

        temp = to_block(all_ind, all_D, len(shape_ind) - 1)
        A[ind] = np.block(temp)

    return A

##############
#  multi dict operations
##############


def block(Ad, to_execute, dtype):

    def to_block(li, lD, level):
        if level > 0:
            oi = []
            for ii, DD in zip(li, lD):
                oi.append(to_block(ii, DD, level - 1))
            return oi
        else:
            key = tuple(li)
            try:
                return Ad[key][ind]
            except KeyError:
                return np.zeros(lD, dtype=_select_dtype[dtype])

    A = {}
    # to_execute = [(ind, pos, legs_ind, legs_D), ... ]
    # ind: index of the merged blocks
    # pos: non-trivial tensors in the block, rest is zero
    # legs_ind: all elements for all dimensions to be cloked
    # legs_D: and bond dimensions (including common ones)

    for ind, legs_ind, legs_D in to_execute:
        all_ind = np.array(list(product(*legs_ind)), dtype=int)
        all_D = np.array(list(product(*legs_D)), dtype=int)

        shape_ind = [len(x) for x in legs_D] + [-1]
        all_ind = list(np.reshape(all_ind, shape_ind))
        all_D = list(np.reshape(all_D, shape_ind))

        temp = to_block(all_ind, all_D, len(shape_ind) - 1)
        A[ind] = np.block(temp)
    return A
