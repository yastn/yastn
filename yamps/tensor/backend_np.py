import numpy as np
import scipy as sp
import fbpca as pca
from itertools import groupby
from itertools import product
from functools import reduce
from operator import mul


def random_seed(seed):
    np.random.seed(seed)


def find_round2(x):
    """get significand"""
    n = np.abs(np.float64(x)).view(np.int64)
    return 2.**((n >> 52) - 1023)

##############
# single element calculations
##############


def get_str(x):
    return str(x)


def copy(x):
    return x.copy()


def to_numpy(nparray):
    return nparray


def first_el(x):
    return x.flat[0]


def get_dtype(x):
    return x.dtype


def get_shape(x):
    return x.shape


def get_ndim(x):
    return x.ndim


def diag_create(x):
    return np.diag(x)


def diag_get(x):
    return np.diag(x)

##############
#  setting values
##############


def zeros(D, isdiag, dtype='float64'):
    if (dtype is 'float64') and (not isdiag):
        return np.zeros(D, dtype=np.float64)
    elif (dtype is 'float64') and (isdiag):
        return np.diag(np.zeros(D[0], dtype=np.float64))
    elif (dtype is 'complex128') and (not isdiag):
        return np.zeros(D, dtype=np.complex128)
    elif (dtype is 'complex128') and (isdiag):
        return np.diag(np.zeros(D[0], dtype=np.complex128))


def ones(D, isdiag, dtype='float64'):
    if (dtype is 'float64') and (not isdiag):
        return np.ones(D, dtype=np.float64)
    elif (dtype is 'float64') and (isdiag):
        return np.diag(np.ones(D[0], dtype=np.float64))
    elif (dtype is 'complex128') and (not isdiag):
        return np.ones(D, dtype=np.complex128)
    elif (dtype is 'complex128') and (isdiag):
        return np.diag(np.ones(D[0], dtype=np.complex128))


def rand(D, isdiag, dtype):
    if dtype is 'float64':
        if not isdiag:
            return 2*np.random.rand(*D)-1
        else:
            return np.diag(2*np.random.rand(D[0])-1)
    elif dtype is 'complex128':
        if not isdiag:
            return 2*np.random.rand(*D)-1 + 1j*(2*np.random.rand(*D)-1)
        else:
            return np.diag(2*np.random.rand(D[0])-1 + 1j*(2*np.random.rand(D[0])-1))


def to_tensor(val, isdiag=False, dtype='float64', Ds=None):
    if (dtype == 'float64') and (not isdiag):
        if Ds is not None:
            return np.reshape(np.array(val, dtype=np.float64), Ds)
        else:
            return np.array(val, dtype=np.float64)
    elif (dtype == 'float64') and (isdiag):
        return np.diag(np.array(val, dtype=np.float64))
    elif (dtype == 'complex128') and (not isdiag):
        if Ds is not None:
            return np.reshape(np.array(val, dtype=np.complex128), Ds)
        else:
            return np.array(val, dtype=np.complex128) 
    elif (dtype == 'complex128') and (isdiag):
        return np.diag(np.array(val, dtype=np.complex128))

##############
# single dict operations
##############


def conj(A):
    ''' conjugate dict of tensors '''
    cA = {}
    for ind in A:
        cA[ind] = A[ind].copy()
        np.conj(cA[ind], cA[ind])
    return cA


def trace(A, to_execute, in1, in2, out):
    ''' trace dict of tensors using to_execute =[(ind_in, ind_out), ...]
        adds repeating ind_out's '''
    cA = {}
    order = in1+in2+out
    for task in to_execute:
        D1 = tuple(A[task[0]].shape[ii] for ii in in1)
        D2 = tuple(A[task[0]].shape[ii] for ii in in2)
        D3 = tuple(A[task[0]].shape[ii] for ii in out)
        pD1 = reduce(mul, D1, 1)
        pD2 = reduce(mul, D2, 1)
        Atemp = np.reshape(np.transpose(A[task[0]], order), (pD1, pD2)+D3)
        if task[1] in cA:
            cA[task[1]] = cA[task[1]] + np.trace(Atemp)
        else:
            cA[task[1]] = np.trace(Atemp)
    return cA


def transpose(A, axes, to_execute):
    cA = {}
    for old, new in to_execute:
        cA[new] = np.transpose(A[old], axes=axes).copy()
    return cA


def invsqrt(A, isdiag=True):
    cA = {}
    if isdiag:
        for ind in A:
            cA[ind] = np.diag(1./np.sqrt(np.diag(A[ind])))
    else:
        for ind in A:
            cA[ind] = 1./np.sqrt(A[ind])
    return cA


def inv(A, isdiag=True):
    cA = {}
    if isdiag:
        for ind in A:
            cA[ind] = np.diag(1./np.diag(A[ind]))
    else:
        for ind in A:
            cA[ind] = 1./A[ind]
    return cA


def exp(A, step, isdiag=True):
    cA = {}
    if isdiag:
        for ind in A:
            cA[ind] = np.diag(np.exp(np.diag(A[ind])*step))
    else:
        for ind in A:
            cA[ind] = np.exp(A[ind]*step)
    return cA


def sqrt(A, isdiag):
    cA = {}
    if isdiag:
        for ind in A:
            cA[ind] = np.diag(np.sqrt(np.diag(A[ind])))
    else:
        for ind in A:
            cA[ind] = np.sqrt(A[ind])
    return cA


def svd(A, truncated=False, Dblock=np.inf, nbit=60, kfac=6):
    U, S, V = {}, {}, {}
    for ind in A:
        if truncated and min(A[ind].shape) > kfac*Dblock:
            U[ind], S[ind], V[ind] = pca.pca(A[ind], k=Dblock, raw=True, n_iter=nbit, l=kfac*Dblock)
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
        Q[ind], R[ind] = Q[ind]*sR, sR.reshape([-1, 1])*R[ind]
    return Q, R


def rq(A):
    R, Q = {}, {}
    for ind in A:
        R[ind], Q[ind] = sp.linalg.rq(A[ind], mode='economic')
        sR = np.sign(np.real(np.diag(R[ind])))
        sR[sR == 0] = 1
        # positive diag of R
        R[ind], Q[ind] = R[ind]*sR, sR.reshape([-1, 1])*Q[ind]
    return R, Q


def eigh(A):
    S, U = {}, {}
    for ind in A:
        S[ind], U[ind] = np.linalg.eigh(A[ind])
    return S, U


def norm(A, ord, round2=False):
    if ord == 'fro':
        f = lambda x: np.linalg.norm(x)
        block_norm = []
    elif ord == 'inf':
        f = lambda x: np.abs(x).max()
        block_norm = [0.]
    for x in A.values():
        block_norm.append(f(x))
    if not round2:
        return f(block_norm)
    else:
        return find_round2(f(block_norm))


def entropy(A, alpha=1):
    block_norm = []
    for x in A.values():
        if x.ndim == 2:
            x = np.diag(x)
        block_norm.append(np.linalg.norm(x))
    no = np.linalg.norm(block_norm)

    ent = 0.
    Smin = np.inf
    if no > 0:
        for x in A.values():
            if x.ndim == 2:
                x = np.diag(x)
            Smin = min(Smin, min(x))
            x = x/no
            if alpha == 1:
                x = x[x > 1e-12]
                ent += -2*sum(x*x*np.log2(x))
            else:
                ent += x**(2*alpha)
        if alpha != 1:
            ent = np.log2(ent)/(1-alpha)
    return ent, Smin, no

##############
# two dicts operations
##############


def norm_diff(A, B, ord):
    if ord == 'fro':
        f = lambda x: np.linalg.norm(x)
        block_norm = []
    elif ord == 'inf':
        f = lambda x: np.abs(x).max()
        block_norm = [0.]
    for ind in A:
        if ind in B:
            block_norm.append(f(A[ind]-B[ind]))
        else:
            block_norm.append(f(A[ind]))
    for ind in B:
        if ind not in B:
            block_norm.append(f(B[ind]))
    return f(block_norm)


def add(aA, bA, to_execute, x=1):
    ''' add two dicts of tensors, according to to_execute = [(ind, meta), ...]
        meta = 0; add both
        meta = 1; only from aA
        meta = 2; only from bA'''
    cA = {}
    for ind in to_execute:
        if (ind[1] == 0):
            cA[ind[0]] = aA[ind[0]]+x*bA[ind[0]]
        elif (ind[1] == 1):
            cA[ind[0]] = aA[ind[0]].copy()
        elif (ind[1] == 2):
            cA[ind[0]] = x*bA[ind[0]].copy()
    return cA


def sub(aA, bA, to_execute):
    ''' subtract two dicts of tensors, according to to_execute = [(ind, meta), ...]
        meta = 0; add both
        meta = 1; only from aA
        meta = 2; only from bA'''
    cA = {}
    for ind in to_execute:
        if (ind[1] == 0):
            cA[ind[0]] = aA[ind[0]]-bA[ind[0]]
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
            C[ind] = A[ind]@B[ind]
    elif conj == (0, 1):
        for ind in A:
            C[ind] = A[ind]@B[ind].conj()
    elif conj == (1, 0):
        for ind in A:
            C[ind] = A[ind].conj()@B[ind]
    elif conj == (1, 1):
        for ind in A:
            C[ind] = (A[ind]@B[ind]).conj()
    return C


def dot(A, B, conj, to_execute, a_out, a_con, b_con, b_out):
    a_all = a_out+a_con  # order for transpose in A
    b_all = b_con+b_out  # order for transpose in B
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
        C = {out: f(AA.transpose(a_all).reshape(pDl, pDc), BB.transpose(b_all).reshape(pDc, pDr)).reshape(Dl+Dr)}
    else:
        Andim = len(a_con) + len(a_out)
        Bndim = len(b_con) + len(b_out)
        DA = np.array([A[ind].shape for ind in A]).reshape(len(A), Andim)  # bond dimensions of A
        DB = np.array([B[ind].shape for ind in B]).reshape(len(B), Bndim)  # bond dimensions of B
        Dl = DA[:, np.array(a_out, dtype=np.int64)]  # bond dimension on left legs
        Dlc = DA[:, np.array(a_con, dtype=np.int64)]  # bond dimension on contracted legs
        Dcr = DB[:, np.array(b_con, dtype=np.int64)]  # bond dimension on contracted legs
        Dr = DB[:, np.array(b_out, dtype=np.int64)]  # bond dimension on right legs
        pDl = np.multiply.reduce(Dl, axis=1, dtype=np.int64)  # their product
        pDlc = np.multiply.reduce(Dlc, axis=1, dtype=np.int64)
        pDcr = np.multiply.reduce(Dcr, axis=1, dtype=np.int64)
        pDr = np.multiply.reduce(Dr, axis=1, dtype=np.int64)

        Atemp = {in1: (A[in1].transpose(a_all).reshape(d1, d2), tuple(dl)) for in1, d1, d2, dl in zip(A, pDl, pDlc, Dl)}
        Btemp = {in2: (B[in2].transpose(b_all).reshape(d1, d2), tuple(dr)) for in2, d1, d2, dr in zip(B, pDcr, pDr, Dr)}

        C, DC = {}, {}
        for in1, in2, out in to_execute:   # can use if in place of try;exept
            temp = f(Atemp[in1][0], Btemp[in2][0])
            try:
                C[out] = C[out] + temp
            except KeyError:
                C[out] = temp
                DC[out] = Atemp[in1][1]+Btemp[in2][1]
        for out in C:
            C[out] = C[out].reshape(DC[out])  # C[out].shape=DC[out]
        # CAN DIVIDE BELOW into multiplication, and than adding the same out...
        # multiplied = [[out, f(Atemp[in1][0], Btemp[in2][0]), Atemp[in1][1], Btemp[in2][1]] for in1, in2, out in to_execute]
        # multiplied = groupby(sorted(multiplied, key=lambda x: x[0]), key=lambda x: x[0])
        # C={}
        # for out, xx in multiplied:
        #     _, C[out], dl, dr = next(xx)
        #     for _, mat, _, _ in xx:
        #         C[out] = C[out] + mat
        #     C[out] = C[out].reshape(dl+dr)
    return C

##############
# block merging, truncations and un-merging
##############


def merge_blocks(A, to_execute, out_l, out_r):
    ''' merge blocks depending on the cut '''
    out_all = out_l + out_r
    Amerged, order_l, order_r = {}, {}, {}
    for tcut, tts in to_execute:
        if len(tts) > 1:
            Dl, Dr, pDl, pDr = {}, {}, {}, {}
            for _, tl, tr, ind in tts:
                if tl not in Dl:
                    Dl[tl] = [A[ind].shape[ii] for ii in out_l]
                    pDl[tl] = reduce(mul, Dl[tl], 1)
                if tr not in Dr:
                    Dr[tr] = [A[ind].shape[ii] for ii in out_r]
                    pDr[tr] = reduce(mul, Dr[tr], 1)

            dtype = A[ind].dtype  # all dtype's in A should be the same -- takes the last one

            ind_l, ind_r = sorted(Dl), sorted(Dr)
            cpDl = np.cumsum([0]+[pDl[ind] for ind in ind_l])
            order_l[tcut] = [(ind, Dl[ind], slice(cpDl[ii], cpDl[ii+1])) for ii, ind in enumerate(ind_l)]
            cpDr = np.cumsum([0]+[pDr[ind] for ind in ind_r])
            order_r[tcut] = [(ind, Dr[ind], slice(cpDr[ii], cpDr[ii+1])) for ii, ind in enumerate(ind_r)]

            Atemp = np.zeros((cpDl[-1], cpDr[-1]), dtype=dtype)

            jj, max_tts = 0, len(tts)
            _, tl, tr, ind = tts[jj]
            for il, _, sl in order_l[tcut]:
                for ir, _, sr in order_r[tcut]:
                    if (tr == ir) and (tl == il):
                        Atemp[sl, sr] = A[ind].transpose(out_all).reshape(pDl[il], pDr[ir])
                        jj += 1
                        if jj < max_tts:
                            _, tl, tr, ind = tts[jj]
            Amerged[tcut] = Atemp

            # version with stack; instead of filling in with slices
            # form_matrix = []
            # for il in ind_l:
            #     form_row = []
            #     for ir in ind_r:
            #         if (tr == ir) and (tl == il):
            #             form_row.append(np.transpose(A[ind], out_all).reshape(pDl[il], pDr[ir]))
            #             try:
            #                 _, tl, tr, ind = next(itts)
            #             except StopIteration:
            #                 pass
            #         else:
            #             form_row.append(np.zeros((pDl[il], pDr[ir])))
            #     form_matrix.append(np.hstack(form_row) if len(form_row) > 1 else form_row[0])
            # Amerged[tcut] = np.vstack(form_matrix) if len(form_matrix) > 1 else form_matrix[0]
        else:
            tcut, tl, tr, ind = tts[0]
            Dl = [A[ind].shape[ii] for ii in out_l]
            Dr = [A[ind].shape[ii] for ii in out_r]
            pDl = reduce(mul, Dl, 1)
            pDr = reduce(mul, Dr, 1)
            order_l[tcut] = [(tl, Dl, slice(None))]  # info for un-grouping
            order_r[tcut] = [(tr, Dr, slice(None))]
            Amerged[tcut] = np.transpose(A[ind], out_all).reshape(pDl, pDr)
    return Amerged, order_l, order_r


def slice_none(d):
    return {ind: slice(None) for ind in d}


def slice_S(S, tol=0., Dblock=np.inf, Dtotal=np.inf, decrease=True):
    """gives slices for truncation of 1d matrices
    decrease =True assumes that S[][0]  is largest -- like in svd
    decrease=False assumes that S[][-1] is largest -- like in eigh"""
    maxS, Dmax = 0., {}
    for ind in S:
        maxS = max(maxS, S[ind][0], S[ind][-1])
        Dmax[ind] = min(Dblock, S[ind].size)
    # truncate to given relative tolerance
    if (tol > 0) and (maxS > 0):
        for ind in Dmax:
            Dmax[ind] = min(Dmax[ind], np.sum(S[ind] >= maxS*tol))
    # truncate to total bond dimension
    if sum(Dmax[ind] for ind in Dmax) > Dtotal:
        if decrease:
            s_all = np.hstack([S[ind][:Dmax[ind]] for ind in Dmax])
        else:
            s_all = np.hstack([S[ind][-Dmax[ind]:] for ind in Dmax])
        order = s_all.argpartition(-Dtotal-1)[-Dtotal:]
        low = 0
        for ind in Dmax:
            high = low+Dmax[ind]
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


def unmerge_blocks_diag(S, order_s, Dcut):
    Sout = {}
    for tcut, ind in order_s:
        Sout[ind] = np.diag(S[tcut][Dcut[tcut]])
    return Sout


def unmerge_blocks_left(U, order_l, Dcut):
    ''' select non-zero sectors; and truncate u, s, v to newD '''
    Uout = {}
    Dc = [-1]
    for tcut in Dcut:  # fill blocks
        for tl, Dl, slice_l in order_l[tcut]:
            Uout[tl] = np.reshape(U[tcut][slice_l, Dcut[tcut]], Dl+Dc)
    return Uout


def unmerge_blocks_right(V, order_r, Dcut):
    ''' select non-zero sectors; and truncate u, s, v to newD'''
    Vout = {}
    Dc = [-1]
    for tcut in Dcut:  # fill blocks
        for tr, Dr, slice_r in order_r[tcut]:
            Vout[tr] = np.reshape(V[tcut][Dcut[tcut], slice_r], Dc+Dr)
    return Vout


def unmerge_blocks(C, order_l, order_r):
    Cout = {}
    for tcut in C:
        for (tl, Dl, slice_l), (tr, Dr, slice_r) in product(order_l[tcut], order_r[tcut]):
            ind = tl+tr
            Cout[ind] = C[tcut][slice_l, slice_r].reshape(Dl+Dr)
    return Cout

##############
#  multi dict operations
##############


def block(Ad, to_execute):

    def to_block(li, lD, level):
        if level > 0:
            oi = []
            for ii, DD in zip(li, lD):
                oi.append(to_block(ii, DD, level-1))
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

        shape_ind = [len(x) for x in legs_D]+[-1]
        all_ind = list(np.reshape(all_ind, shape_ind))
        all_D = list(np.reshape(all_D, shape_ind))

        temp = to_block(all_ind, all_D, len(shape_ind)-1)
        A[ind] = np.block(temp)

    return A
