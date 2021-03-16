"""Support of torch as a data structure used by yast."""
import torch
from .linalg.torch_svd_gesdd import SVDGESDD
from .linalg.torch_eig_sym import SYMEIG
from .linalg.torch_eig_arnoldi import SYMARNOLDI, SYMARNOLDI_2C


_data_dtype = {'float64': torch.float64,
               'complex128': torch.complex128}


def random_seed(seed):
    torch.random.manual_seed(seed)


def set_num_threads(num_threads):
    torch.set_num_threads(num_threads)


####################################
#     single tensor operations     #
####################################

def detach(x):
    return x.detach()

def detach_(x):
    x.detach_()

def clone(x):
    return x.clone()


def copy(x):
    return x.detach().clone()

def real(x): return x.real

def imag(x): return x.imag

def to_numpy(x):
    return x.detach().cpu().numpy()

def get_shape(x):
    return x.size()


def get_ndim(x):
    return len(x.size())


def get_size(x):
    return x.numel()


def diag_create(x):
    return torch.diag(x)  ## TODO: PROBLEM WITH COMPLEX NUMBERS


def diag_get(x):
    return torch.diag(x)  ## TODO: PROBLEM WITH COMPLEX NUMBERS


def diag_diag(x):
    return torch.diag(torch.diag(x))  ## TODO: PROBLEM WITH COMPLEX NUMBERS


def count_greater(x, cutoff):
    return torch.sum(x > cutoff)

#########################
#    output numbers     #
#########################

def first_element(x):
    return x.view(-1)[0]


def item(x):
    return x.item()


def norm(A, p):
    """ 'fro' for Frobenious; 'inf' for max(abs(A)) """
    if p == "fro":
        return torch.sum(torch.stack([torch.sum(t.abs()**2) for t in A.values()])).sqrt()
    elif p == "inf":
        return torch.max(torch.stack([t.abs().max() for t in A.values()]))
    else:
        raise RuntimeError("Invalid norm type: "+p)


def norm_diff(A, B, meta, p):
    """ norm(A - B); meta = kab, ka, kb """
    if p == 'fro':
        return torch.sum(torch.stack([torch.sum(A[k].abs() ** 2) for k in meta[1]] + \
                                     [torch.sum(B[k].abs() ** 2) for k in meta[2]] + \
                                     [torch.sum((A[k]-B[k]).abs() ** 2) for k in meta[0]])).sqrt()
    if p == 'inf':
        return torch.max(torch.stack([A[k].abs().max() for k in meta[1]] + \
                                     [B[k].abs().max() for k in meta[2]] + \
                                     [(A[k]-B[k]).abs().max() for k in meta[0]]))


def entropy(A, alpha=1, tol=1e-12):
    """ von Neuman or Renyi entropy from svd's"""
    Snorm = torch.sum(torch.stack([torch.sum(x ** 2) for x in A.values()])).sqrt()
    if Snorm > 0:
        ent = []
        Smin = min([min(x) for x in A.values()])
        for x in A.values():
            x = x / Snorm
            x = x[x > tol]
            if alpha == 1:
                ent.append(-2 * torch.sum(x * x * torch.log2(x)))
            else:
                ent.append(x**(2 * alpha))
        ent = torch.sum(torch.stack(ent))
        if alpha != 1:
            ent = torch.log2(ent) / (1 - alpha)
        return ent, Smin, Snorm
    return Snorm, Snorm, Snorm  # this should be 0., 0., 0.

##########################
#     setting values     #
##########################

def zero_scalar(dtype='float64', device='cpu'):
    return torch.tensor(0, dtype=_data_dtype[dtype], device=device)


def zeros(D, dtype='float64', device='cpu'):
    return torch.zeros(D, dtype=_data_dtype[dtype], device=device)


def ones(D, dtype='float64', device='cpu'):
    return torch.ones(D, dtype=_data_dtype[dtype], device=device)


def randR(D, dtype='float64', device='cpu'):
    x = 2 * torch.rand(D, dtype=_data_dtype[dtype], device=device) - 1
    return x if dtype=='float64' else torch.real(x)


def rand(D, dtype='float64', device='cpu'):
    return 2 * torch.rand(D, dtype=_data_dtype[dtype], device=device) - 1


def to_tensor(val, Ds=None, dtype='float64', device='cpu'):
    T = torch.as_tensor(val, dtype=_data_dtype[dtype], device=device)
    return T if Ds is None else T.reshape(Ds).contiguous()

##################################
#     single dict operations     #
##################################

def move_to_device(A, device):
    return {ind: x.to(device) for ind, x in A.items()}


def conj(A, inplace):
    """ Conjugate dict of tensors. Force a copy if not inplace. """
    if inplace:
        return {t: x.conj() for t, x in A.items()}
    return {t: x.conj() for t, x in A.items()}   # TODO is it a copy or not


def trace(A, order, meta):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    Aout = {}
    for (tnew, told, Drsh) in meta:
        Atemp = torch.reshape(A[told].permute(*order), Drsh)
        Atemp = torch.sum(torch.diagonal(Atemp, dim1=0, dim2=1), dim=-1)
        if tnew in Aout:
            Aout[tnew] += Atemp
        else:
            Aout[tnew] = Atemp
    return Aout


def transpose(A, axes, meta_transpose, inplace):
    """ Transpose; Force a copy if not inplace. """
    # check this inplace ...
    if inplace:
        return {new: A[old].permute(*axes) for old, new in meta_transpose}
    return {new: A[old].permute(*axes).clone().contiguous() for old, new in meta_transpose}
    

def rsqrt(A, cutoff=0):
    res = {t: x.rsqrt() for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0
    return res


def rsqrt_diag(A, cutoff=0):
    res = {t: torch.diag(x).rsqrt() for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0
    return {t: torch.diag(x) for t, x in res.items()}


def reciprocal(A, cutoff=0):
    res = {t: 1./x for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0
    return res


def reciprocal_diag(A, cutoff=0):
    res = {t: 1./torch.diag(x) for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0
    return {t: torch.diag(x) for t, x in res.items()}


def exp(A, step):
    return {t: torch.exp(step * x) for t, x in A.items()}


def exp_diag(A, step):
    return {t: torch.diag(torch.exp(step * torch.diag(x))) for t, x in A.items()}


def sqrt(A):
    return {t: torch.sqrt(x) for t, x in A.items()}


ad_decomp_reg = 1.0e-12


def svd(A, meta, opts):
    U, S, V = {}, {}, {}
    tn = next(iter(A.values()))
    reg = torch.as_tensor(ad_decomp_reg, dtype=tn.dtype, device=tn.device)
    for (iold, iU, iS, iV) in meta:
        U[iU], S[iS], V[iV] = SVDGESDD.apply(A[iold], reg)
        V[iV] = V[iV].t().conj()
    return U, S, V


def eigh(A, meta, order_by_magnitude=False):
    S, U = {}, {}
    if order_by_magnitude:
        tn = next(iter(A.values()))
        reg = torch.as_tensor(ad_decomp_reg, dtype=tn.dtype, device=tn.device)
        for ind in A:
            S[ind], U[ind] = SYMEIG.apply(A[ind], reg)
    else:
        for (ind, indS, indU) in meta:
            S[indS], U[indU] = torch.symeig(A[ind], eigenvectors=True)
    return S, U


def svd_S(A):
    S = {}
    tn = next(iter(A.values()))
    reg = torch.as_tensor(ad_decomp_reg, dtype=tn.dtype, device=tn.device)
    for ind in A:
        _, S[ind], _ = SVDGESDD.apply(A[ind], reg)
        # S[ind] = torch.svd(A[ind], some=True, compute_uv=False)
    return S


def qr(A, meta):
    Q, R = {}, {}
    for (ind, indQ, indR) in meta:
        Q[indQ], R[indR] = torch.qr(A[ind], some=True)
        sR = torch.sign(torch.diag(R[indR])) ##  PROBLEM WITH COMPLEX NUMBERS
        sR[sR == 0] = 1
        # positive diag of R
        Q[indQ] = Q[indQ] * sR
        R[indR] = sR.reshape([-1, 1]) * R[indR]
    return Q, R


# def rq(A):
#     R, Q = {}, {}
#     for ind in A:
#         R[ind], Q[ind] = torch.qr(torch.t(A[ind]), some=True)
#         sR = torch.sign(torch.real(torch.diag(R[ind])))
#         sR[sR == 0] = 1
#         # positive diag of R
#         R[ind], Q[ind] = torch.t(R[ind]) * sR, sR.reshape([-1, 1]) * torch.t(Q[ind])
#     return R, Q

@torch.no_grad()
def select_global_largest(S, D_keep, D_total, sorting, \
    keep_multiplets=False, eps_multiplet=1.0e-14):
    if sorting == 'svd':
        s_all = torch.cat([S[ind][:D_keep[ind]] for ind in S])
        values, order= torch.topk(s_all, D_total+int(keep_multiplets))
        # if needed, preserve multiplets within each sector
        if keep_multiplets:
            # regularize by discarding small values
            gaps=torch.abs(values.clone())
            # compute gaps and normalize by larger sing. value. Introduce cutoff
            # for handling vanishing values set to exact zero
            gaps=(gaps[:len(values)-1]-torch.abs(values[1:len(values)]))/(gaps[:len(values)-1]+1.0e-16)
            gaps[gaps > 1.0]= 0.

            if gaps[D_total-1] < eps_multiplet:
                # the chi is within the multiplet - find the largest chi_new < chi
                # such that the complete multiplets are preserved
                for i in range(D_total-1,-1,-1):
                    if gaps[i] > eps_multiplet:
                        #chi_new= i
                        order= order[:i+1]
                        break
        return order
        # return torch.from_numpy(s_all.cpu().numpy().argpartition(-D_total-1)[-D_total:])
    elif sorting == 'eigh':
        s_all = torch.cat([S[ind][-D_keep[ind]:] for ind in S])
        return torch.from_numpy(s_all.cpu().numpy().argpartition(-D_total-1)[-D_total:])


def range_largest(D_keep, D_total, sorting):
    if sorting == 'svd':
        return (0, D_keep)
    elif sorting == 'eigh':
        return (D_total - D_keep, D_total)


def maximum(A):
    return max(torch.max(x) for x in A.values())


def max_abs(A):
    return norm(A, p="inf")

################################
#     two dicts operations     #
################################

def add(A, B, meta):
    """ C = A + B. meta = kab, ka, kb """
    C = {ind: A[ind] + B[ind] for ind in meta[0]}
    for ind in meta[1]:
        C[ind] = A[ind].clone()
    for ind in meta[2]:
        C[ind] = B[ind].clone()
    return C


def sub(A, B, meta):
    """ C = A - B. meta = kab, ka, kb """
    C = {ind: A[ind] - B[ind] for ind in meta[0]}
    for ind in meta[1]:
        C[ind] = A[ind].clone()
    for ind in meta[2]:
        C[ind] = -B[ind]
    return C


def apxb(A, B, x, meta):
    """ C = A + x * B. meta = kab, ka, kb """
    C = {ind: A[ind] + x * B[ind] for ind in meta[0]}
    for ind in meta[1]:
        C[ind] = A[ind].clone()
    for ind in meta[2]:
        C[ind] = x * B[ind]
    return C


def vdot(A, B, meta):
    return torch.sum(torch.stack([(A[ind].conj().reshape(-1)) @ (B[ind].reshape(-1)) for ind in meta]))


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

#####################################################
#     block merging, truncations and un-merging     #
#####################################################

def merge_to_matrix(A, order, meta_new, meta_mrg, dtype, device='cpu'):
    """ New dictionary of blocks after merging into matrix. """
    Anew = {u: torch.zeros(Du, dtype=_data_dtype[dtype], device=device) for (u, Du) in meta_new}
    for (tn, to, Dsl, Dl, Dsr, Dr) in meta_mrg:
        Anew[tn][slice(*Dsl), slice(*Dsr)] = A[to].permute(order).reshape(Dl, Dr)
    return Anew


def merge_one_leg(A, axis, order, meta_new, meta_mrg, dtype, device='cpu'):
    """ Outputs new dictionary of blocks after fusing one leg. """
    Anew = {u: torch.zeros(Du, dtype=_data_dtype[dtype], device=device) for (u, Du) in meta_new}
    for (tn, Ds, to, Do) in meta_mrg:
        if to in A:
            slc = [slice(None)] * len(Do)
            slc[axis] = slice(*Ds)
            Anew[tn][tuple(slc)] = A[to].permute(order).reshape(Do)
    return Anew


def merge_to_dense(A, Dtot, meta, dtype, device='cpu'):
    """ Outputs full tensor. """
    Anew = torch.zeros(Dtot, dtype=_data_dtype[dtype], device=device)
    for (ind, Dss) in meta:
        Anew[tuple(slice(*Ds) for Ds in Dss)] = A[ind].reshape(tuple(Ds[1] - Ds[0] for Ds in Dss))
    return Anew

def merge_super_blocks(pos_tens, meta_new, meta_block, dtype, device='cpu'):
    """ Outputs new dictionary of blocks after creating super-tensor. """
    Anew = {u: torch.zeros(Du, dtype=_data_dtype[dtype], device=device) for (u, Du) in meta_new}
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
        Anew[tnew] = torch.reshape(A[told][tuple(slc)], Dnew).clone()   # is this clone neccesary -- there is a test where it is neccesary for numpy
    return Anew

##############
#  tests
##############

def is_complex(x):
    return x.is_complex()

def is_independent(A, B):
    """
    check if two arrays are identical, or share the same view.
    """
    return (A is B) or (A.storage().data_ptr() is B.storage().data_ptr())
