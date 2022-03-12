"""Support of torch as a data structure used by yast."""
from itertools import chain, groupby
import torch
from .linalg.torch_svd_gesdd import SVDGESDD
from .linalg.torch_eig_sym import SYMEIG
# from .linalg.torch_eig_arnoldi import SYMARNOLDI, SYMARNOLDI_2C

BACKEND_ID = "torch"
DTYPE = {'float64': torch.float64,
         'complex128': torch.complex128}


def _common_type(iterator):
    return torch.complex128 if any(x.is_complex() for x in iterator) else torch.float64


def get_dtype(t):
    return t.dtype


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


def count_nonzero(x):
    return torch.count_nonzero(x).item()


def delete(x, sl):
    return torch.cat([x[:sl[0]], x[sl[1]:]])


def insert(x, start, values):
    return torch.cat([x[:start], values, x[start:]])


def expm(x):
    return torch.matrix_exp(x)

#########################
#    output numbers     #
#########################


def first_element(x):
    return x.view(-1)[0]


def item(x):
    return x.item()


def sum_elements(data):
    """ sum of all elements of all tensors in A """
    return data.sum().reshape(1)


def norm(data, p):
    """ 'fro' for Frobenious; 'inf' for max(abs(A)) """
    if p == "fro":
        return data.norm()
    return data.abs().max() if len(data) > 0 else torch.tensor(0.) # else p == "inf":


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


def square_matrix_from_dict(H, D=None, **kwargs):
    dtype = _common_type(H.values())
    device = next(iter(H.values())).device
    T = torch.zeros((D, D), dtype=dtype, device=device)
    for (i, j), v in H.items():
        if i < D and j < D:
            T[i, j] = v
    return T


def requires_grad_(data, requires_grad=True):
    data.requires_grad_(requires_grad)


def requires_grad(data):
    return data.requires_grad


def move_to(data, *args, **kwargs):
    if "dtype" in kwargs:
        if kwargs["dtype"] is None:
            del kwargs["dtype"]
        else:
            kwargs["dtype"] = DTYPE[kwargs["dtype"]]
    return data.to(*args, **kwargs)


def conj(data):
    return data.conj()


def trace(data, order, meta, Dsize):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    newdata = torch.zeros((Dsize,), dtype=data.dtype)
    for (sln, slo, Do, Drsh) in meta:
        temp = data[slice(*slo)].reshape(Do).permute(order).reshape(Drsh)
        newdata[slice(*sln)] += torch.sum(torch.diagonal(temp, dim1=0, dim2=1), dim=-1).ravel()
    return newdata


def trace_with_mask(data, order, meta, Dsize, tcon, msk12):
    """ Trace dict of tensors according to meta = [(tnew, told, Dreshape), ...].
        Repeating tnew are added."""
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for (sln, slo, Do, Drsh), tt in zip(meta, tcon):
        temp = data[slice(*slo)].reshape(Do).permute(order).reshape(Drsh)
        newdata[slice(*sln)] += torch.sum(temp[msk12[tt][0], msk12[tt][1]], axis=0).ravel()
    return newdata


def transpose(data, axes, meta_transpose):
    newdata = torch.zeros_like(data)
    for sln, slo, Do in meta_transpose:
        newdata[slice(*sln)] = data[slice(*slo)].reshape(Do).permute(axes).ravel()
    return newdata


def rsqrt(data, cutoff=0):
    res = torch.zeros_like(data)
    ind = data.abs() > cutoff
    res[ind] = data[ind].rsqrt()
    return res


def reciprocal(data, cutoff=0):
    res = torch.zeros_like(data)
    ind = data.abs() > cutoff
    res[ind] = 1. / data[ind]
    return res


def exp(data, step):
    return torch.exp(step * data)


def sqrt(data):
    return torch.sqrt(data)


def absolute(data):
    return torch.abs(data)


def svd_lowrank(A, meta, D_block, n_iter=60, k_fac=6):
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


def embed_msk(data, msk, Dsize):
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    newdata[msk] = data
    return newdata


def embed_slc(data, meta, Dsize):
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for sln, slo in meta:
        newdata[slice(*sln)] = data[slice(*slo)]
    return newdata


################################
#     two dicts operations     #
################################


def add(Adata, Bdata, meta, Dsize):
    dtype = _common_type((Adata, Bdata))
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    for sl_c, sl_a, sl_b, ab in meta:
        if ab == 'AB':
            newdata[slice(*sl_c)] = Adata[slice(*sl_a)] + Bdata[slice(*sl_b)]
        elif ab == 'A':
            newdata[slice(*sl_c)] = Adata[slice(*sl_a)]
        else:  # ab == 'B'
            newdata[slice(*sl_c)] = Bdata[slice(*sl_b)]
    return newdata


def sub(Adata, Bdata, meta, Dsize):
    dtype = _common_type((Adata, Bdata))
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    for sl_c, sl_a, sl_b, ab in meta:
        if ab == 'AB':
            newdata[slice(*sl_c)] = Adata[slice(*sl_a)] - Bdata[slice(*sl_b)]
        elif ab == 'A':
            newdata[slice(*sl_c)] = Adata[slice(*sl_a)]
        else:  # ab == 'B'
            newdata[slice(*sl_c)] = -Bdata[slice(*sl_b)]
    return newdata


def apxb(Adata, Bdata, x, meta, Dsize):
    dtype = _common_type((Adata, Bdata))
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    for sl_c, sl_a, sl_b, ab in meta:
        if ab == 'AB':
            newdata[slice(*sl_c)] = Adata[slice(*sl_a)] + x * Bdata[slice(*sl_b)]
        elif ab == 'A':
            newdata[slice(*sl_c)] = Adata[slice(*sl_a)]
        else:  # ab == 'B'
            newdata[slice(*sl_c)] = x * Bdata[slice(*sl_b)]
    return newdata


dot_dict = {(0, 0): lambda x, y: x @ y,
            (0, 1): lambda x, y: x @ y.conj(),
            (1, 0): lambda x, y: x.conj() @ y,
            (1, 1): lambda x, y: x.conj() @ y.conj()}


def apply_slice(data, slcn, slco):
    Dsize = slcn[-1][1] if len(slcn) > 0 else 0
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for sn, so in zip(slcn, slco):
        newdata[slice(*sn)] = data[slice(*so)]
    return newdata


def vdot(Adata, Bdata, cc):
    f = dot_dict[cc]  # proper conjugations
    return f(Adata, Bdata)


def diag_1dto2d(data, meta, Dsize):
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for sln, slo in meta:
        newdata[slice(*sln)] = torch.diag(data[slice(*slo)]).ravel()
    return newdata


def diag_2dto1d(data, meta, Dsize):
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for sln, slo, Do in meta:
        newdata[slice(*sln)] = torch.diag(data[slice(*slo)].reshape(Do))
    return newdata


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


def dot_diag(Adata, Bdata, cc, meta, Dsize, axis, a_ndim):
    dim = [1] * a_ndim
    dim[axis] = -1
    f = dotdiag_dict[cc]
    dtype = _common_type((Adata, Bdata))
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    for sln, sla, Da, slb in meta:
        newdata[slice(*sln)] = f(Adata[slice(*sla)].reshape(Da), Bdata[slice(*slb)], dim).ravel()
    return newdata


def mask_diag(Adata, Bdata, meta, Dsize, axis, a_ndim):
    slc1 = (slice(None),) * axis
    slc2 = (slice(None),) * (a_ndim - (axis + 1))
    newdata = torch.zeros((Dsize,), dtype=Adata.dtype, device=Adata.device)
    for sln, sla, Da, slb in meta:
        cut = (Bdata[slice(*slb)].nonzero(),)
        newdata[slice(*sln)] = Adata[slice(*sla)].reshape(Da)[slc1 + cut + slc2].ravel()
    return newdata


def dot_nomerge(Adata, Bdata, cc, oA, oB, meta, Dsize):
    f = dot_dict[cc]  # proper conjugations
    dtype = _common_type((Adata, Bdata))
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    for (sln, sla, Dao, Dan, slb, Dbo, Dbn) in meta:
        newdata[slice(*sln)] += f(Adata[slice(*sla)].reshape(Dao).permute(oA).reshape(Dan), \
                                  Bdata[slice(*slb)].reshape(Dbo).permute(oB).reshape(Dbn)).ravel()
    return newdata



def dot_nomerge_masks(Adata, Bdata, cc, oA, oB, meta, Dsize, tcon, ma, mb):
    f = dot_dict[cc]  # proper conjugations
    dtype = _common_type((Adata, Bdata))
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    for (sln, sla, Dao, Dan, slb, Dbo, Dbn), tt in zip(meta, tcon):
        newdata[slice(*sln)] += f(Adata[slice(*sla)].reshape(Dao).permute(oA).reshape(Dan)[:, ma[tt]], \
                                  Bdata[slice(*slb)].reshape(Dbo).permute(oB).reshape(Dbn)[mb[tt], :]).ravel()
    return newdata


#####################################################
#     block merging, truncations and un-merging     #
#####################################################


def merge_to_2d(data, order, meta_new, meta_mrg, *args, **kwargs):
    Anew = {u: torch.zeros(Du, dtype=data.dtype, device=data.device) for u, Du in zip(*meta_new)}
    for (tn, slo, Do, Dslc, Drsh) in meta_mrg:
        Anew[tn][tuple(slice(*x) for x in Dslc)] = data[slice(*slo)].reshape(Do).permute(order).reshape(Drsh)
    return Anew


def merge_to_1d(data, order, meta_new, meta_mrg, Dsize, *args, **kwargs):
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for (tn, Dn, sln), (t1, gr) in zip(zip(*meta_new), groupby(meta_mrg, key=lambda x: x[0])):
        assert tn == t1
        temp = torch.zeros(Dn, dtype=data.dtype, device=data.device)
        for (_, slo, Do, Dslc, Drsh) in gr:
            temp[tuple(slice(*x) for x in Dslc)] = data[slice(*slo)].reshape(Do).permute(order).reshape(Drsh)
        newdata[slice(*sln)] = temp.ravel()
    return newdata


def merge_to_dense(data, Dtot, meta, *args, **kwargs):
    newdata = torch.zeros(Dtot, dtype=data.dtype, device=data.device)
    for (sl, Dss) in meta:
        newdata[tuple(slice(*Ds) for Ds in Dss)] = data[sl].reshape(tuple(Ds[1] - Ds[0] for Ds in Dss))
    return newdata.ravel()


def merge_super_blocks(pos_tens, meta_new, meta_block, Dsize):
    dtype = _common_type(pos_tens.values())
    device = next(iter(pos_tens.values()))._data.device
    newdata = torch.zeros((Dsize,), dtype=dtype, device=device)
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_block, key=lambda x: x[0])):
        assert tn == t1
        temp = torch.zeros(Dn, dtype=dtype, device=device)
        for (_, slo, Do, pos, Dslc) in gr:
            slc = tuple(slice(*x) for x in Dslc)
            temp[slc] = pos_tens[pos]._data[slice(*slo)].reshape(Do)
        newdata[slice(*sln)] = temp.ravel()
    return newdata


def unmerge_from_2d(A, meta, new_sl, Dsize, device='cpu'):
    dtype = next(iter(A.values())).dtype if len(A) > 0 else torch.float64
    newdata = torch.zeros((Dsize,), dtype=dtype, device=device)
    for (indm, sl, sr), snew in zip(meta, new_sl):
        newdata[slice(*snew)] = A[indm][slice(*sl), slice(*sr)].ravel()
    return newdata


def unmerge_from_2ddiag(A, meta, new_sl, Dsize, device='cpu'):
    dtype = next(iter(A.values())).dtype if len(A) > 0 else torch.float64
    newdata = torch.zeros((Dsize,), dtype=dtype, device=device)
    for (_, iold, slc), snew in zip(meta, new_sl):
        newdata[slice(*snew)] = A[iold][slice(*slc)]
    return newdata


def unmerge_from_1d(data, meta, new_sl, Dsize):
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for (slo, Do, sub_slc), snew in zip(meta, new_sl):
        newdata[slice(*snew)] = data[slice(*slo)].reshape(Do)[tuple(slice(*x) for x in sub_slc)].ravel()
    return newdata


#############
#   tests   #
#############


def is_complex(x):
    return x.is_complex()


def is_independent(x, y):
    """
    check if two arrays are identical, or share the same view.
    """
    return not ((x is y) or (x.numel() > 0 and x.storage().data_ptr() == y.storage().data_ptr()))
