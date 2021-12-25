"""Support of torch as a data structure used by yast."""
from itertools import chain
import torch
from .linalg.torch_svd_gesdd import SVDGESDD
from .linalg.torch_eig_sym import SYMEIG
# from .linalg.torch_eig_arnoldi import SYMARNOLDI, SYMARNOLDI_2C

BACKEND_ID = "torch"
DTYPE = {'float64': torch.float64,
         'complex128': torch.complex128}


def get_dtype(iterator):
    """ iterators of torch tensors; returns torch.complex128 if any tensor is complex else torch.float64"""
    return torch.complex128 if any(torch.is_complex(x) for x in iterator) else torch.float64


def unique_dtype(t):
    dtypes = set(b.dtype for b in t.A.values())
    if len(dtypes) == 1:
        return str(tuple(dtypes)[0])[len("torch."):]
    return False


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


def to_numpy(x):
    return x.detach().cpu().numpy()


def get_shape(x):
    return x.size()


def get_size(x):
    return x.numel()


def diag_create(x, p=0):
    return torch.diag(x, diagonal=p)


def diag_get(x):
    return torch.diag(x)


def get_device(x):
    return str(x.device)


@torch.no_grad()
def count_greater(x, cutoff):
    return torch.sum(x > cutoff).item()


def real(x):
    return torch.real(x) if torch.is_complex(x) else x


def imag(x):
    return torch.imag(x) if torch.is_complex(x) else 0 * x


def max_abs(x):
    return x.abs().max()


def norm_matrix(x):
    return torch.linalg.norm(x)


def expand_dims(x, axis):
    return torch.unsqueeze(x, axis)


def any_nonzero(x):
    return torch.any(x).item()


#########################
#    output numbers     #
#########################


def first_element(x):
    return x.view(-1)[0]


def item(x):
    return x.item()


def sum_elements(A):
    """ sum of all elements of all tensors in A """
    return sum(x.sum() for x in A.values())


def norm(A, p):
    """ 'fro' for Frobenious; 'inf' for max(abs(A)) """
    if p == "fro":
        return torch.sum(torch.stack([torch.sum(t.abs()**2) for t in A.values()])).sqrt()
    if p == "inf":
        return torch.max(torch.stack([t.abs().max() for t in A.values()]))
    raise RuntimeError("Invalid norm type: %s" % str(p))


def norm_diff(A, B, meta, p):
    """ norm(A - B); meta = kab, ka, kb """
    if p == 'fro':
        return torch.sum(torch.stack([torch.sum(A[k].abs() ** 2) for k in meta[1]]
                                     + [torch.sum(B[k].abs() ** 2) for k in meta[2]]
                                     + [torch.sum((A[k] - B[k]).abs() ** 2) for k in meta[0]])).sqrt()
    if p == 'inf':
        return torch.max(torch.stack([A[k].abs().max() for k in meta[1]]
                                     + [B[k].abs().max() for k in meta[2]]
                                     + [(A[k] - B[k]).abs().max() for k in meta[0]]))
    raise RuntimeError("Invalid norm type: %s" % str(p))


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
                ent.append(torch.sum(x**(2 * alpha)))
        ent = torch.sum(torch.stack(ent))
        if alpha != 1:
            ent = torch.log2(ent) / (1 - alpha)
        return ent, Smin, Snorm
    return Snorm, Snorm, Snorm  # this should be 0., 0., 0.


##########################
#     setting values     #
##########################


def dtype_scalar(x, dtype='float64', device='cpu'):
    return torch.tensor(x, dtype=DTYPE[dtype], device=device)


def zeros(D, dtype='float64', device='cpu'):
    return torch.zeros(D, dtype=DTYPE[dtype], device=device)


def ones(D, dtype='float64', device='cpu'):
    return torch.ones(D, dtype=DTYPE[dtype], device=device)


def randR(D, device='cpu'):
    return 2 * torch.rand(D, dtype=DTYPE['float64'], device=device) - 1


def randC(D, device='cpu'):
    return 2 * torch.rand(D, dtype=DTYPE['complex128'], device=device) - (1 + 1j)


def to_tensor(val, Ds=None, dtype='float64', device='cpu'):
    # try:
    T = torch.as_tensor(val, dtype=DTYPE[dtype], device=device)
    # except TypeError:
    #     T = torch.as_tensor(val, dtype=DTYPE['complex128'], device=device)
    return T if Ds is None else T.reshape(Ds).contiguous()


@torch.no_grad()
def to_mask(val):
    return torch.as_tensor(val).nonzero().ravel()


def square_matrix_from_dict(H, D=None, device='cpu'):
    dtype = get_dtype(H.values())
    T = torch.zeros((D, D), dtype=dtype, device=device)
    for (i, j), v in H.items():
        if i < D and j < D:
            T[i, j] = v
    return T


##################################
#     single dict operations     #
##################################

def requires_grad_(A, requires_grad=True):
    for b in A.values(): b.requires_grad_(requires_grad)


def requires_grad(A):
    return any([b.requires_grad for b in A.values()])


def move_to(A, *args, **kwargs):
    if "dtype" in kwargs:
        if kwargs["dtype"] is None:
            del kwargs["dtype"]
        else:
            kwargs["dtype"] = DTYPE[kwargs["dtype"]]
    return {ind: x.to(*args, **kwargs) for ind, x in A.items()}


def conj(A, inplace):
    """ Conjugate dict of tensors. Force a copy if not inplace. """
    if inplace:
        return {t: x.conj() for t, x in A.items()}
    return {t: x.conj() for t, x in A.items()}   # TODO is it a copy or not


def trace(A, order, meta):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    Aout = {}
    for (tnew, told, Drsh, _) in meta:
        Atemp = torch.reshape(A[told].permute(order), Drsh)
        Atemp = torch.sum(torch.diagonal(Atemp, dim1=0, dim2=1), dim=-1)
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
        Atemp = torch.reshape(A[told].permute(order), Drsh)
        Atemp = torch.sum(Atemp[msk12[tt][0], msk12[tt][1]], axis=0)
        if tnew in Aout:
            Aout[tnew] += Atemp
        else:
            Aout[tnew] = Atemp
    return Aout


def transpose(A, axes, meta_transpose, inplace):
    """ Transpose; Force a copy if not inplace. """
    # check this inplace ...
    if inplace:
        return {new: A[old].permute(axes) for old, new in meta_transpose}
    return {new: A[old].permute(axes).clone().contiguous() for old, new in meta_transpose}


def rsqrt(A, cutoff=0):
    res = {t: x.rsqrt() for t, x in A.items()}
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
    return {t: torch.exp(step * x) for t, x in A.items()}


def expm(A):
    return torch.matrix_exp(A)


def sqrt(A):
    return {t: torch.sqrt(x) for t, x in A.items()}


def absolute(A):
    return {t: torch.abs(x) for t, x in A.items()}


def svd_lowrank(A, meta, D_block, n_iter, k_fac):
    U, S, V = {}, {}, {}
    for (iold, iU, iS, iV) in meta:
        q = min(min(A[iold].shape), D_block * k_fac)
        U[iU], S[iS], V[iV] = torch.svd_lowrank(A[iold], q=q, niter=n_iter)
        V[iV] = V[iV].T.conj()
    return U, S, V


ad_decomp_reg = 1.0e-12


def svd(A, meta):
    U, S, V = {}, {}, {}
    tn = next(iter(A.values()))
    reg = torch.as_tensor(ad_decomp_reg, dtype=tn.dtype, device=tn.device)
    for (iold, iU, iS, iV) in meta:
        U[iU], S[iS], V[iV] = SVDGESDD.apply(A[iold], reg)
        V[iV] = V[iV].t().conj()
    return U, S, V


def eigh(A, meta=None, order_by_magnitude=False):
    S, U = {}, {}
    if meta is not None:
        if order_by_magnitude:
            tn = next(iter(A.values()))
            reg = torch.as_tensor(ad_decomp_reg, dtype=tn.dtype, device=tn.device)
            for ind in A:
                S[ind], U[ind] = SYMEIG.apply(A[ind], reg)
        else:
            for (ind, indS, indU) in meta:
                S[indS], U[indU] = torch.linalg.eigh(A[ind])
    else:
        S, U = torch.linalg.eigh(A)
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
        Q[indQ], R[indR] = torch.linalg.qr(A[ind])
        sR = torch.sign(real(R[indR].diag()))
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
def select_global_largest(S, D_keep, D_total, keep_multiplets, eps_multiplet, ordering):
    if ordering == 'svd':
        s_all = torch.cat([S[ind][:D_keep[ind]] for ind in S])
    elif ordering == 'eigh':
        s_all = torch.cat([S[ind][-D_keep[ind]:] for ind in S])
    values, order = torch.topk(s_all, D_total + int(keep_multiplets))
    if keep_multiplets:  # if needed, preserve multiplets within each sector
        gaps = torch.abs(values.clone())  # regularize by discarding small values
        # compute gaps and normalize by larger singular value. Introduce cutoff
        gaps = torch.abs(gaps[:len(values) - 1] - gaps[1:len(values)]) / gaps[0]  # / (gaps[:len(values) - 1] + 1.0e-16)
        gaps[gaps > 1.0] = 0.  # for handling vanishing values set to exact zero
        if gaps[D_total - 1] < eps_multiplet:
            # the chi is within the multiplet - find the largest chi_new < chi
            # such that the complete multiplets are preserved
            for i in range(D_total - 1, -1, -1):
                if gaps[i] > eps_multiplet:
                    order = order[:i + 1]
                    break
    return order


@torch.no_grad()
def eigs_which(val, which):
    if which == 'LM':
        return (-abs(val)).argsort()
    if which == 'SM':
        return abs(val).argsort()
    if which == 'LR':
        return (-real(val)).argsort()
    #elif which == 'SR':
    return (real(val)).argsort()


def range_largest(D_keep, D_total, ordering):
    if ordering == 'svd':
        return (0, D_keep)
    if ordering == 'eigh':
        return (D_total - D_keep, D_total)


def maximum(A):
    """ maximal element of A """
    return max(torch.max(x) for x in A.values())


def embed(A, sl, tD):
    """ embeds old tensors A into larger zero blocks based on slices. """
    dtype = get_dtype(A.values())
    C = {}
    for t, val in A.items():
        C[t] = torch.zeros(tD[t], dtype=dtype, device=val.device)
        C[t][sl[t]] = val
    return C


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


dot_dict = {(0, 0): lambda x, y: x @ y,
            (0, 1): lambda x, y: x @ y.conj(),
            (1, 0): lambda x, y: x.conj() @ y,
            (1, 1): lambda x, y: x.conj() @ y.conj()}


def vdot(A, B, conj, meta):
    f = dot_dict[conj]  # proper conjugations
    return torch.sum(torch.stack([f(A[ind].reshape(-1), B[ind].reshape(-1)) for ind in meta]))


def dot(A, B, conj, meta_dot):
    f = dot_dict[conj]  # proper conjugations
    C = {}
    for (out, ina, inb) in meta_dot:
        C[out] = f(A[ina], B[inb])
    return C


dotdiag_dict = {(0, 0): lambda x, y, dim: x * y.reshape(dim),
                (0, 1): lambda x, y, dim: x * y.reshape(dim).conj(),
                (1, 0): lambda x, y, dim: x.conj() * y.reshape(dim),
                (1, 1): lambda x, y, dim: x.conj() * y.reshape(dim).conj()}


def dot_diag(A, B, conj, to_execute, axis, a_ndim):
    dim = [1] * a_ndim
    dim[axis] = -1
    f = dotdiag_dict[conj]
    C = {}
    for in1, in2, out in to_execute:
        C[out] = f(A[in1], B[in2], dim)
    return C


def mask_diag(A, B, meta, axis, a_ndim):
    slc1 = (slice(None),) * axis
    slc2 = (slice(None),) * (a_ndim - (axis + 1))
    Bslc = {k: v.nonzero(as_tuple=True) for k, v in B.items()}
    return {out: A[in1][slc1 + Bslc[in2] + slc2] for in1, in2, out in meta}


def dot_nomerge(A, B, conj, oA, oB, meta):
    f = dot_dict[conj]  # proper conjugations
    C = {}
    for (ina, inb, out, Da, Db, Dout, _) in meta:
        temp = f(A[ina].permute(oA).reshape(Da), B[inb].permute(oB).reshape(Db)).reshape(Dout)
        try:
            C[out] += temp
        except KeyError:
            C[out] = temp
    return C


def dot_nomerge_masks(A, B, conj, oA, oB, meta, ma, mb):
    f = dot_dict[conj]  # proper conjugations
    C = {}
    for (ina, inb, out, Da, Db, Dout, tt) in meta:
        temp = f(A[ina].permute(oA).reshape(Da)[:, ma[tt]], B[inb].permute(oB).reshape(Db)[mb[tt], :]).reshape(Dout)
        try:
            C[out] += temp
        except KeyError:
            C[out] = temp
    return C

#####################################################
#     block merging, truncations and un-merging     #
#####################################################


def merge_blocks(A, order, meta_new, meta_mrg, device='cpu'):
    """ New dictionary of blocks after merging into matrix. """
    dtype = get_dtype(A.values())
    Anew = {u: torch.zeros(Du, dtype=dtype, device=device) for u, Du in zip(*meta_new)}
    for (tn, to, Dslc, Drsh) in meta_mrg:
        if to in A:
            slc = tuple(slice(*x) for x in Dslc)
            Anew[tn][slc] = A[to].permute(order).reshape(Drsh)
    return Anew


def merge_to_dense(A, Dtot, meta, device='cpu'):
    """ Outputs full tensor. """
    dtype = get_dtype(A.values())
    Anew = torch.zeros(Dtot, dtype=dtype, device=device)
    for (ind, Dss) in meta:
        Anew[tuple(slice(*Ds) for Ds in Dss)] = A[ind].reshape(tuple(Ds[1] - Ds[0] for Ds in Dss))
    return Anew


def merge_super_blocks(pos_tens, meta_new, meta_block, device='cpu'):
    """ Outputs new dictionary of blocks after creating super-tensor. """
    dtype = get_dtype(chain.from_iterable(t.A.values() for t in pos_tens.values()))
    Anew = {u: torch.zeros(Du, dtype=dtype, device=device) for (u, Du) in meta_new}
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
        Anew[tnew] = torch.reshape(A[told][tuple(slc)], Dnew).clone()
        # is this clone above neccesary ? -- there is a test where it is neccesary for numpy
    return Anew


#############
#   tests   #
#############


def is_complex(x):
    return x.is_complex()


def is_independent(A, B):
    """
    check if two arrays are identical, or share the same view.
    """
    return (A is B) or (A.storage().data_ptr() is B.storage().data_ptr() and A.numel() > 0)
