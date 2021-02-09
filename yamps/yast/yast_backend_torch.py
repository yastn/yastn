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


def clone(x):
    return x.clone()


def copy(x):
    return x.detach().clone()


def to_numpy(x):
    return x.detach().cpu().numpy()


def first_element(x):
    return x.view(-1)[0]


def item(x):
    return x.item()


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


def diag_diag(x):
    return torch.diag(torch.diag(x))


def count_greater(x, cutoff):
    return torch.sum(x > cutoff)

##########################
#     setting values     #
##########################

def zeros(D, dtype='float64', device='cpu'):
    return torch.zeros(D, dtype=_data_dtype[dtype], device=device)


def ones(D, dtype='float64', device='cpu'):
    return torch.ones(D, dtype=_data_dtype[dtype], device=device)


def randR(D, dtype='float64', device='cpu'):
    return 2 * torch.rand(D, dtype=_data_dtype[dtype], device=device) - 1


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


def conj(A):
    """ Conjugate dict of tensors forcing a copy. """
    # is it a copy or not
    return {t: x.conj() for t, x in A.items()}


def trace(A, order, meta):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    Aout = {}
    for (tnew, told, Drsh) in meta:
        Atemp = torch.reshape(A[told].permute(*order), Drsh)
        if tnew in Aout:
            Aout[tnew] += torch.sum(torch.diagonal(Atemp, dim1=0, dim2=1), dim=0)
        else:
            Aout[tnew] = torch.sum(torch.diagonal(Atemp, dim1=0, dim2=1), dim=0)
    return Aout


def transpose(A, axes, meta_transpose, inplace):
    """ Transpose; Force a copy if not inplace. """
    # check this inplace ...
    if inplace:
        return {new: A[old].permute(*axes) for old, new in meta_transpose}
    return {new: A[old].permute(*axes).clone().contiguous() for old, new in meta_transpose}
    

def invsqrt(A, cutoff=0):
    res = {t: x.rsqrt() for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0.
    return res


def invsqrt_diag(A, cutoff=0):
    res = {t: torch.diag(x).rsqrt() for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0.
    return {t: torch.diag(x) for t, x in res.items()}


def inv(A, cutoff=0):
    res = {t: 1./x for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0.
    return res


def inv_diag(A, cutoff=0):
    res = {t: 1./torch.diag(x) for t, x in A.items()}
    if cutoff > 0:
        for t in res:
            res[t][abs(res[t]) > 1./cutoff] = 0.
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
    for ind in A:
        S[ind] = torch.svd(A[ind], some=True, compute_uv=False)
    return S


def qr(A, meta):
    Q, R = {}, {}
    for (ind, indQ, indR) in meta:
        Q[indQ], R[indR] = torch.qr(A[ind], some=True)
        sR = torch.sign(torch.real(torch.diag(R[indR])))
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

def select_largest(S, D_keep, D_total, sorting):
    if sorting == 'svd':
        s_all = torch.cat([S[ind][:D_keep[ind]] for ind in S])
        return torch.from_numpy(s_all.cpu().numpy().argpartition(-D_total-1)[-D_total:])
    elif sorting == 'eigh':
        s_all = torch.cat([S[ind][-D_keep[ind]:] for ind in S])
        return torch.from_numpy(s_all.cpu().numpy().argpartition(-D_total-1)[-D_total:])


def range_largest(D_keep, D_total, sorting):
    if sorting == 'svd':
        return (0, D_keep)
    elif sorting == 'eigh':
        return (D_total - D_keep, D_total)


def maximum(A):
    val = [torch.max(x) for x in A.values()]   ## IS THIS FINE WITH GRAD
    val.append(0.)
    return max(val)


def max_abs(A):
    val = [norm(x, ord='inf') for x in A.values()]  ## IS THIS FINE WITH GRAD
    val.append(0.)
    return max(val)


def entropy(A, alpha=1, tol=1e-12):
    temp = 0.
    for x in A.values():
        temp += torch.sum(torch.abs(x) ** 2)
    normalization = torch.sqrt(temp)

    entropy = 0.
    Smin = 10000000
    if normalization > 0:
        for x in A.values():
            Smin = min(Smin, min(x))
            x = x / normalization
            if alpha == 1:
                x = x[x > tol]
                entropy += -2 * sum(x * x * torch.log2(x))
            else:
                entropy += x**(2 * alpha)
        if alpha != 1:
            entropy = torch.log2(entropy) / (1 - alpha)
    return entropy, Smin, normalization


def norm(A, ord):
    if ord=='fro':
        return torch.sum(torch.stack([torch.sum(t.abs()**2) for t in A.values()])).sqrt()
    elif ord=='inf':         
        return torch.max(torch.stack([t.abs().max() for t in A.values()]))
    else:
        raise RuntimeError("Invalid metric: "+ord+". Choices: [\'fro\',\'inf\']")


################################
#     two dicts operations     #
################################

def norm_diff(A, B, ord, meta):
    if ord=='fro':
        # get the list of sums of squares, sum it and take square root         
        return torch.sum(torch.stack([torch.sum(A[k].abs()**2) for k in meta[1]] + \
                                     [torch.sum(B[k].abs()**2) for k in meta[2]] + \
                                     [torch.sum((A[k]-B[k]).abs()**2) for k in meta[0]])).sqrt()
    elif ord=='inf':       
        return torch.max(torch.stack([A[k].abs().max() for k in meta[1]] + \
                                     [B[k].abs().max() for k in meta[2]] + \
                                     [(A[k]-B[k]).abs().max() for k in meta[0]]))
    else: 
        raise RuntimeError("Invalid metric: "+ord+". Choices: [\'fro\',\'inf\']") 


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


def scalar(A, B, meta):
    out = 0.
    for ind in meta:
        out += (A[ind].conj().reshape(-1)) @ (B[ind].reshape(-1))
    return out


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
        Anew[tn][slice(*Dsl), slice(*Dsr)] = A[to].permute(*order).reshape(Dl, Dr)
    return Anew


def merge_one_leg(A, axis, order, meta_new, meta_mrg, dtype, device='cpu'):
    """
    outputs new dictionary of blocks after fusing one leg
    """
    Anew = {u: torch.zeros(Du, dtype=_data_dtype[dtype], device=device) for (u, Du) in meta_new}
    for (tn, Ds, to, Do) in meta_mrg:
        slc = [slice(None)] * len(Do)
        slc[axis] = slice(*Ds)
        Anew[tn][tuple(slc)] = A[to].permute(*order).reshape(Do)  
    return Anew


def merge_to_dense(A, Dtot, meta, dtype, device='cpu'):
    """ outputs full tensor """
    Anew = torch.zeros(Dtot, dtype=_data_dtype[dtype], device=device)
    for (ind, Dss) in meta:
        Anew[tuple(slice(*Ds) for Ds in Dss)] = A[ind]
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

def is_independent(A, B):
    """
    check if two arrays are identical, or share the same view.
    """
    return (A is B) or (A.storage().data_ptr() is B.storage().data_ptr())
