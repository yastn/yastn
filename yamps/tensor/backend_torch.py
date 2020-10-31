import logging
import torch
import numpy as np
import scipy as sp
from itertools import groupby
from itertools import product
from itertools import accumulate
from functools import reduce
from operator import mul
from yamps.tensor.linalg.torch_svd_gesdd import SVDGESDD
import pdb

log= logging.getLogger('yamps.tensor.backend_torch')

_select_dtype = {'float64': torch.float64,
                 'complex128': torch.complex128}

def random_seed(seed):
    torch.random.manual_seed(seed)

def set_num_threads(num_threads):
    torch.set_num_threads(num_threads)

def detach_(A):
    for ind in A: A[ind]= A[ind].detach()

# ----- properties ------------------------------------------------------------

def is_independent(A, B):
    """
    check if two dicts are identical, or share the same view.
    """
    inAB= set(A.keys()) & set(B.keys())
    test= [A[k].storage().data_ptr()==B[k].storage().data_ptr() for k in inAB]
    return not any(test)

# ----- single element (of dict) operations -----------------------------------

def copy(x):
    return x.clone().detach()

def to_numpy(x):
    return x.detach().numpy()

# def get_str(x):
#     return str(x)

def first_el(x):
    return x.view(-1)[0]

def get_shape(x):
    return x.size()

def get_ndim(x):
    return len(x.size())

def get_size(x):
    return x.numel()

def diag_create(x):
    return torch.diag(x)

def diag_get(x):
    return torch.diag(x)

# ----- tensor creation ops ---------------------------------------------------

def zeros(D, dtype='float64', device='cpu'):
    return torch.zeros(D, dtype=_select_dtype[dtype], device=device)
    
def ones(D, dtype='float64', device='cpu'):
    return torch.ones(D, dtype=_select_dtype[dtype], device=device)

def rand(D, dtype='float64', device='cpu'):
    return torch.rand(D, dtype=_select_dtype[dtype], device=device)

# NOTE unnecessary ? Given that real/complex randoms are generated depending on dtype
def randR(D, dtype='float64', device='cpu'):
    return rand(D, dtype=dtype, device=device)

def to_tensor(val, Ds=None, dtype='float64', device='cpu'):
    T= torch.as_tensor(val, dtype=_select_dtype[dtype], device=device)
    return T if not Ds else T.reshape(Ds).contiguous()

def move_to_device(A, device):
    return { ind: b.to(device) for ind,b in A.items() }

def compress_to_1d(A):
    # get the total number of elements
    n= sum((t.numel() for t in A.values()))
    r1d= torch.zeros(n, dtype=next(iter(A.values())).dtype, \
        device=next(iter(A.values())).device)
    # store each block as 1D array within r1d in contiguous manner
    i=0
    meta= []
    for ind in A:
        r1d[i:i+A[ind].numel()]= A[ind].contiguous().view(-1)
        meta.append((ind, A[ind].size()))
        i+=A[ind].numel()
    return meta, r1d

def decompress_from_1d(r1d, charges_and_dims):
    i=0
    A={}
    for charges,dims in charges_and_dims:
        D= reduce(mul,dims,1)
        A[charges]= r1d[i:i+D].reshape(dims).contiguous()
        i+=D
    return A


# ===== single tensor/dict operations =========================================



# ----- leg manipulation -----

# TODO is it necessary to clone ?
def transpose(A, axes, to_execute):
    # cA = {}
    # for old, new in to_execute:
    #     cA[new] = A[old].permute(*axes).clone()
    # return cA
    return {new: A[old].permute(*axes).clone().contiguous() for old,new in to_execute}

def transpose_local(A, axes, to_execute):
    """
    transpose in place.
    """
    cA = {}
    for old, new in to_execute:
        cA[new] = A[old].permute(axes).contiguous()
    return cA

def group_legs(A, to_execute, axes, rest, new_axis, dtype=None):
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
                Anew[tnew] = torch.zeros((*Dnew[tnew][:new_axis], cpDin, *Dnew[tnew][new_axis:]), \
                    dtype=next(iter(A.values())).dtype, device=next(iter(A.values())).device)

            slc[new_axis] = leg_order[tcut][tin][1]
            Anew[tnew][tuple(slc)] = A[ind].permute(out_all).reshape(*Dnew[tnew][:new_axis], pDin[tin], *Dnew[tnew][new_axis:])

    return Anew, leg_order

def ungroup_leg(A, axis, ndim, leg_order, to_execute):
    """ ungroup single leg """
    Anew = {}
    slc = [slice(None)] * ndim
    for tcut, tout, told, tnew in to_execute:
        Dn, slc[axis] = leg_order[tcut][tout]
        shp = A[told].shape
        shp = shp[:axis] + Dn + shp[axis + 1:]
        #Anew[tnew]= np.reshape(A[told][tuple(slc)], shp).copy()
        Anew[tnew]= A[told][tuple(slc)].reshape(shp).clone()
    return Anew

# ----- linear algebra ------

def conj(A):
    return {ind: t.conj() for ind,t in A.items()}

def invsqrt(A):
    # cA = {}
    # if isdiag:
    #     for ind in A:
    #         cA[ind] = torch.diag(1./torch.sqrt(torch.diag(A[ind])))
    # else:
    #     for ind in A:
    #         cA[ind] = 1./torch.sqrt(A[ind])
    # return cA
    return {ind: t.rsqrt() for ind,t in A.items()}

def inv(A, isdiag = True):
    # cA = {}
    # if isdiag:
    #     for ind in A:
    #         cA[ind] = torch.diag(1./torch.diag(A[ind]))
    # else:
    #     for ind in A:
    #         cA[ind] = 1./A[ind]
    # return cA
    return {ind: 1./t for ind,t in A.items()}

def exp(A, step, isdiag = True):
    cA = {}
    if isdiag:
        for ind in A:
            cA[ind] = torch.diag(torch.exp(torch.diag(A[ind])*step))
    else:
        for ind in A:
            cA[ind] = torch.exp(A[ind]*step)
    return cA

def sqrt(A):
    # cA = {}
    # if isdiag:
    #     for ind in A:
    #         cA[ind] = torch.diag(torch.sqrt(torch.diag(A[ind])))
    # else:
    #     for ind in A:
    #         cA[ind] = torch.sqrt(A[ind])
    # return cA
    return {ind: t.sqrt() for ind,t in A.items()}

# TODO add customized svd with correct backward
def svd(A, truncated = False, Dblock=np.inf, nbit = 60, kfac = 6):
    U, S, V = {}, {}, {}
    for ind in A:
        # TO DO
        # if truncated and min(A[ind].shape) > kfac*Dblock:
        #     U[ind], S[ind], V[ind] =  pca.pca(A[ind], k=Dblock, raw=True, n_iter=nbit, l=kfac*Dblock)
        #else:
        # U[ind], S[ind], V[ind] = torch.svd(A[ind], some=True)
        U[ind], S[ind], V[ind] = SVDGESDD.apply(A[ind])
        V[ind] = V[ind].t().conj()
    return U, S, V

def qr(A):
    Q, R = {}, {}
    for ind in A:
        Q[ind], R[ind] = torch.qr(A[ind], some=True)
        sR = torch.sign(torch.diag(R[ind]))
        sR[sR == 0] = 1
        Q[ind], R[ind] = Q[ind]*sR, sR.reshape([-1,1])*R[ind] # positive diag of R
    return Q, R

def rq(A):
    Q, R = {}, {}
    for ind in A:
        QT, RT = torch.qr(torch.t(A[ind]), some=True)
        sR = torch.sign(torch.diag(RT))
        sR[sR == 0] = 1
        R[ind], Q[ind] = torch.t(RT)*sR, sR.reshape([-1,1])*torch.t(QT),  # positive diag of R
    return R, Q

def eigh(A):
    S, U = {}, {}
    for ind in A:
        S[ind], U[ind]  = torch.symeig(A[ind], eigenvectors=True)
    return S, U

# ----- rank reduction -----

def norm(A, ord='fro'):
    # if ord == 'fro':
    #     f = lambda x: torch.norm(x.flatten())
    #     block_norm=[]
    # elif ord == 'inf':
    #     f = lambda x: torch.abs(x).max()
    #     block_norm=[0.]
    # for x in A.values():
    #     block_norm.append(f(x))
    # return f(torch.tensor(block_norm)).item()
    if ord=='fro':
        # get the list of sums of squares, sum it and take square root 
        return torch.sum(torch.stack(\
            [torch.sum(t.abs()**2) for t in A.values()])).sqrt()
    elif ord=='inf':
        # get the list of sums of squares, sum it and take square root
        return torch.max(torch.stack([t.abs().max() for t in A.values()]))
    else:
        raise RuntimeError("Invalid metric: "+ord+". Choices: [\'fro\',\'inf\']")

def max_abs(A):
    return norm(A, ord='inf')

def trace(A, to_execute, in1, in2, out):
    cA = {}
    order = out+in1+in2
    for task in to_execute:
        D1 = tuple(A[task[0]].shape[ii] for ii in in1)
        D2 = tuple(A[task[0]].shape[ii] for ii in in2)
        D3 = tuple(A[task[0]].shape[ii] for ii in out)
        pD1 = reduce(mul, D1, 1)
        pD2 = reduce(mul, D2, 1)
        Atemp = torch.reshape(A[task[0]].permute(*order), D3+(pD1, pD2))
        if task[1] in cA:
            cA[task[1]] = cA[task[1]] + torch.sum(torch.diagonal(Atemp, dim1=-2, dim2=-1), dim=-1)
        else:
            cA[task[1]] = torch.sum(torch.diagonal(Atemp, dim1=-2, dim2=-1), dim=-1)
    return cA

# ----- pair tensor/dict operations -------------------------------------------
def norm_diff(A, B, metric='fro'):
    # get intersection and set differences
    inAnotB= set(A.keys())-set(B.keys())
    inBnotA= set(B.keys())-set(A.keys())
    inAB= set(B.keys()) & set(A.keys())
    if metric=='fro':
        # get the list of sums of squares, sum it and take square root 
        return torch.sum(torch.stack(\
            [torch.sum(A[k].abs()**2) for k in inAnotB]\
                + [torch.sum(B[k].abs()**2) for k in inBnotA]\
                + [torch.sum((A[k]-B[k]).abs()**2) for k in inAB] )).sqrt()
    elif metric=='inf':
        # get the list of sums of squares, sum it and take square root
        return torch.max(torch.stack(\
            [A[k].abs().max() for k in inAnotB]\
                + [B[k].abs().max() for k in inBnotA]\
                + [(A[k]-B[k]).abs().max() for k in inAB]))
    else:
        raise RuntimeError("Invalid metric: "+metric+". Choices: [\'fro\',\'inf\']")

def add(aA, bA, to_execute, x=1.):
    ''' add two dicts of tensors, according to to_execute = [(ind, meta), ...]
        meta = 0; add both
        meta = 1; only from aA
        meta = 2; only from bA'''
    # NOTE: scalar operation (+,-,/,*) produces a copy implicit
    cA = {}
    for ind in to_execute:
        if (ind[1] == 0):
            cA[ind[0]] = aA[ind[0]]+x*bA[ind[0]]
        elif (ind[1] == 1):
            cA[ind[0]] = aA[ind[0]].clone()
        elif (ind[1] == 2):
            cA[ind[0]] = x*bA[ind[0]]
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
            cA[ind[0]] = aA[ind[0]].clone()
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
    return (x @ y).conj()

dot_dict = {(0,0):dot_matrix, (0,1):dotC_matrix, (1,0):Cdot_matrix, (1,1):CdotC_matrix}

def dotdiag_matrix(x, y, dim):
    return x * y.reshape(dim)

def dotdiagC_matrix(x, y, dim):
    return x * y.reshape(dim).conj()

def Cdotdiag_matrix(x, y, dim):
    return x.conj() * y.reshape(dim)

def CdotdiagC_matrix(x, y, dim):
    return (x * y.reshape(dim)).conj()

dotdiag_dict = {(0, 0): dotdiag_matrix, (0, 1): dotdiagC_matrix,
                (1, 0): Cdotdiag_matrix, (1, 1): CdotdiagC_matrix}

def dot_diag(A, B, conj, to_execute, a_con, a_ndim):
    dim = [1]*a_ndim #torch.ones(a_ndim, dtype=torch.long)
    dim[a_con] = -1
    f = dotdiag_dict[conj]

    return {out: f(A[in1], B[in2], dim) for in1, in2, out in to_execute}

# def dot_merged(A, B, conj):
#     C = {}
#     for ind in A:
#         if A[ind].ndim > 0 and B[ind].ndim > 0:
#                     C[ind] = A[ind]@B[ind]
#         else:
#                     C[ind] = A[ind]*B[ind]
#     return C

# NOTE complex matrix-matrix multiplication missing
def dot_merged(A, B, conj):
    if conj == (0, 0):
        return {ind: A[ind] @ B[ind] for ind in A}
    elif conj == (0, 1):
        return {ind: A[ind] @ B[ind].conj() for ind in A}
    elif conj == (1, 0):
        return {ind: A[ind].conj() @ B[ind] for ind in A}
    elif conj == (1, 1):
        return {ind: (A[ind]@B[ind]).conj() for ind in A}

# def dot(A, B, conj, to_execute, a_out, a_con, b_con, b_out):
#     a_all = a_out+a_con  # order for transpose in A
#     b_all = b_con+b_out  # order for transpose in B
#     f = dot_dict[conj] # proper conjugations
#     if len(to_execute) == 1:
#         in1, in2, out = to_execute[0]
#         AA, BB = A[in1], B[in2]
#         Dl = tuple(AA.shape[ii] for ii in a_out)
#         Dc = tuple(AA.shape[ii] for ii in a_con)
#         Dr = tuple(BB.shape[ii] for ii in b_out)
#         pDl = reduce(mul, Dl, 1)
#         pDc = reduce(mul, Dc, 1)
#         pDr = reduce(mul, Dr, 1)
#         C = {out:torch.reshape(f(torch.reshape(AA.permute(*a_all),(pDl, pDc)), torch.reshape(BB.permute(*b_all),(pDc, pDr))),Dl+Dr)}
#     else:
#         Andim = len(a_con) + len(a_out)
#         Bndim = len(b_con) + len(b_out)
#         DA = np.array([A[ind].shape for ind in A]).reshape(len(A), Andim) # bond dimensions of A
#         DB = np.array([B[ind].shape for ind in B]).reshape(len(B), Bndim) # bond dimensions of B
#         Dl = DA[:, np.array(a_out, dtype=np.int64)] # bond dimension on left legs
#         Dlc = DA[:, np.array(a_con, dtype=np.int64)] # bond dimension on contracted legs
#         Dcr = DB[:, np.array(b_con, dtype=np.int64)] # bond dimension on contracted legs
#         Dr = DB[:, np.array(b_out, dtype=np.int64)] # bond dimension on right legs
#         pDl = np.multiply.reduce(Dl, axis=1, dtype=np.int64) # their product
#         pDlc = np.multiply.reduce(Dlc, axis=1, dtype=np.int64)
#         pDcr = np.multiply.reduce(Dcr, axis=1, dtype=np.int64)
#         pDr = np.multiply.reduce(Dr, axis=1, dtype=np.int64)

#         Atemp = {in1:(torch.reshape(A[in1].permute(*a_all),(d1, d2)), tuple(dl)) for in1, d1, d2, dl in zip(A, pDl, pDlc, Dl)}
#         Btemp = {in2:(torch.reshape(B[in2].permute(*b_all),(d1, d2)), tuple(dr)) for in2, d1, d2, dr in zip(B, pDcr, pDr, Dr)}

#         C, DC = {}, {}
#         for in1, in2, out in to_execute:   ## can use if in place of try;exept
#             temp = f(Atemp[in1][0], Btemp[in2][0])
#             try:
#                 C[out] = C[out] + temp
#             except KeyError:
#                 C[out] = temp
#                 DC[out] = Atemp[in1][1]+Btemp[in2][1]
#         for out in C:
#             C[out] = torch.reshape(C[out],DC[out]) #C[out].shape=DC[out]
#         # CAN DIVIDE BELOW into multiplication, and than adding the same out ...
#         # multiplied = [[out, f(Atemp[in1][0], Btemp[in2][0]), Atemp[in1][1], Btemp[in2][1]] for in1, in2, out in to_execute]
#         # multiplied = groupby(sorted(multiplied, key=lambda x: x[0]), key=lambda x: x[0])
#         # C={}
#         # for out, xx in multiplied:
#         #     _, C[out], dl, dr = next(xx)
#         #     for _, mat, _, _ in xx:
#         #         C[out] = C[out] + mat
#         #     C[out] = C[out].reshape(dl+dr)
#     return C

def dot(A, B, conj, to_execute, a_out, a_con, b_con, b_out, dtype=None):
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
        C = {out: f(AA.permute(a_all).reshape(pDl, pDc), BB.permute(b_all).reshape(pDc, pDr)).reshape(Dl + Dr)}
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

        Atemp = {in1: A[in1].permute(a_all).reshape(d1, d2) for in1, d1, d2 in zip(A, pDl, pDlc)}
        Btemp = {in2: B[in2].permute(b_all).reshape(d1, d2) for in2, d1, d2 in zip(B, pDcr, pDr)}

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

def merge_blocks(A, to_execute, out_l, out_r, dtype=None):
    """ merge blocks depending on the cut """
    out_all = out_l + out_r
    Andim = len(out_all)

    lA = [A[ind] for _, tts in to_execute for _, _, ind in tts]
    aDA = np.array([x.shape for x in lA], dtype=np.int).reshape(len(lA), Andim)
    aDl = aDA[:, np.array(out_l, dtype=np.int)]  # bond dimension on left legs
    aDr = aDA[:, np.array(out_r, dtype=np.int)]  # bond dimension on right legs
    apDl = np.multiply.reduce(aDl, axis=1, dtype=np.int)  # their product
    apDr = np.multiply.reduce(aDr, axis=1, dtype=np.int)

    Are = [x.permute(out_all).reshape(pl, pr) for x, pl, pr in zip(lA, apDl, apDr)]
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
            Atemp = torch.zeros((sum(pDl), sum(ipDr)), dtype=next(iter(A.values())).dtype,\
                device=next(iter(A.values())).device)
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

# def merge_blocks(A, to_execute, out_l, out_r):
#     ''' merge blocks depending on the cut '''
#     out_all = out_l + out_r
#     Amerged, order_l, order_r = {}, {}, {}
#     for tcut, tts in to_execute:
#         if len(tts) > 1:
#             Dl, Dr, pDl, pDr  = {}, {}, {}, {}
#             for _, tl, tr, ind in tts:
#                 if tl not in Dl:
#                     Dl[tl] = [A[ind].shape[ii] for ii in out_l]
#                     pDl[tl] = reduce(mul, Dl[tl], 1)
#                 if tr not in Dr:
#                     Dr[tr] = [A[ind].shape[ii] for ii in out_r]
#                     pDr[tr] = reduce(mul, Dr[tr], 1)

#             dtype = get_dtype(A[ind])  # all dtype's in A should be the same -- takes the last one

#             ind_l, ind_r = sorted(Dl), sorted(Dr)
#             cpDl = np.cumsum([0]+[pDl[ind] for ind in ind_l])
#             order_l[tcut] = [ (ind, Dl[ind], slice(cpDl[ii], cpDl[ii+1])) for ii, ind in enumerate(ind_l)]
#             cpDr = np.cumsum([0]+[pDr[ind] for ind in ind_r])
#             order_r[tcut] = [ (ind, Dr[ind], slice(cpDr[ii], cpDr[ii+1])) for ii, ind in enumerate(ind_r)]

#             Atemp = zeros( (cpDl[-1], cpDr[-1]), False,  dtype=dtype )

#             jj, max_tts = 0, len(tts)
#             _, tl, tr, ind = tts[jj]
#             for il, _, sl in order_l[tcut]:
#                 for ir, _, sr in order_r[tcut]:
#                     if (tr == ir) and (tl == il):
#                         Atemp[sl, sr] = torch.reshape(A[ind].permute(*out_all),(pDl[il], pDr[ir]))
#                         jj += 1
#                         if jj<max_tts:
#                             _, tl, tr, ind = tts[jj]
#             Amerged[tcut] = Atemp

#             # version with stack; instead of filling in with slices
#             # form_matrix = []
#             # for il in ind_l:
#             #     form_row = []
#             #     for ir in ind_r:
#             #         if (tr == ir) and (tl == il):
#             #             form_row.append(np.transpose(A[ind], out_all).reshape(pDl[il], pDr[ir]))
#             #             try:
#             #                 _, tl, tr, ind = next(itts)
#             #             except StopIteration:
#             #                 pass
#             #         else:
#             #             form_row.append(np.zeros((pDl[il], pDr[ir])))
#             #     form_matrix.append(np.hstack(form_row) if len(form_row) > 1 else form_row[0])
#             # Amerged[tcut] = np.vstack(form_matrix) if len(form_matrix) > 1 else form_matrix[0]
#         else:
#             tcut, tl, tr, ind = tts[0]
#             Dl = [A[ind].shape[ii] for ii in out_l]
#             Dr = [A[ind].shape[ii] for ii in out_r]
#             pDl = reduce(mul, Dl, 1)
#             pDr = reduce(mul, Dr, 1)
#             order_l[tcut] = [(tl, Dl, slice(None))]  # info for un-grouping
#             order_r[tcut] = [(tr, Dr, slice(None))]
#             if out_all == ():
#                 Amerged[tcut] = A[ind]
#             else:
#                 Amerged[tcut] = torch.reshape(A[ind].permute(*out_all),(pDl, pDr))
#     return Amerged, order_l, order_r

def slice_none(d):
    return {ind:slice(None) for ind in d}

@torch.no_grad()
def slice_S(S, tol=0., Dblock=np.inf, Dtotal=np.inf, decrease = True):
    """gives slices for truncation of 1d matrices
    decrease =True assumes that S[][0]  is largest -- like in svd
    decrease=False assumes that S[][-1] is largest -- like in eigh"""
    maxS, Dmax = 0., {}
    for ind in S:
        maxS = max(maxS, S[ind][0], S[ind][-1])
        # blocks of S are 1D, *get_shape(S[ind]) returns length of block
        Dmax[ind] = min(Dblock, *get_shape(S[ind]))
    # truncate to given relative tolerance
    if (tol > 0) and (maxS > 0):
        for ind in Dmax:
            Dmax[ind] = min(Dmax[ind], torch.sum(S[ind] >= maxS*tol))
    # truncate to total bond dimension
    if sum(Dmax[ind] for ind in Dmax) > Dtotal:
        if decrease:
            s_all = torch.cat([S[ind][:Dmax[ind]] for ind in Dmax])
        else:
            s_all = torch.cat([S[ind][-Dmax[ind]:] for ind in Dmax])
        order = s_all.numpy().argpartition(-Dtotal-1)[-Dtotal:]
        # values, indices= torch.topk(s_all, Dtotal)
        low = 0
        for ind in Dmax:
            high = low+Dmax[ind]
            Dmax[ind]= np.sum((low <= order) & (order < high))
            # Dmax[ind] = torch.sum((low <= order) & (order < high))
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

# def unmerge_blocks_diag(S, Dcut, order_s):
    # Sout = {}
    # for tcut, ind in order_s:
    #     Sout[ind] = torch.diag(S[tcut][Dcut[tcut]])
    # return Sout
    
def unmerge_blocks_diag(S, Dcut):
    Sout = {}
    for ind in Dcut:
        Sout[ind] = S[ind][Dcut[ind]]
    return Sout

def unmerge_blocks_left(U, order_l, Dcut):
    ''' select non-zero sectors; and truncate u, s, v to newD '''
    # Uout = {}
    # Dc = [-1]
    # for tcut in Dcut:  # fill blocks
    #     for tl, Dl, slice_l in order_l[tcut]:
    #         Uout[tl] = torch.reshape(U[tcut][slice_l, Dcut[tcut]], Dl+Dc)
    # return Uout
    Uout = {}
    Dc = (-1,)
    for tcut in Dcut:  # fill blocks
        for tl, (Dl, slice_l) in order_l[tcut].items():
            Uout[tl] = U[tcut][slice_l, Dcut[tcut]].reshape(Dl + Dc)
    return Uout

def unmerge_blocks_right(V, order_r, Dcut):
    ''' select non-zero sectors; and truncate u, s, v to newD'''
    # Vout = {}
    # Dc = [-1]
    # for tcut in Dcut:  # fill blocks
    #     for tr, Dr, slice_r in order_r[tcut]:
    #         Vout[tr] = torch.reshape(V[tcut][Dcut[tcut], slice_r], Dc+Dr)
    # return Vout
    Vout = {}
    Dc = (-1,)
    for tcut in Dcut:  # fill blocks
        for tr, (Dr, slice_r) in order_r[tcut].items():
            Vout[tr] = V[tcut][Dcut[tcut], slice_r].reshape(Dc + Dr)
    return Vout

def unmerge_blocks(C, order_l, order_r):
    # Cout = {}
    # for tcut in C:
    #     for (tl, Dl, slice_l), (tr, Dr, slice_r) in product(order_l[tcut], order_r[tcut]):
    #         ind = tl+tr
    #         Cout[ind] = torch.reshape(C[tcut][slice_l, slice_r],(Dl+Dr))
    # return Cout
    Cout = {}
    for tcut in C:
        for tl, (Dl, slice_l) in order_l[tcut].items():
            for tr, (Dr, slice_r) in order_r[tcut].items():
                ind = tl + tr
                Cout[ind] = C[tcut][slice_l, slice_r].reshape(Dl + Dr)
    return Cout
