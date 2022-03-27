"""Support of torch as a data structure used by yast."""
from itertools import chain, groupby
import numpy as np
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


def is_complex(x):
    return x.is_complex()


def get_device(x):
    return str(x.device)


def random_seed(seed):
    torch.random.manual_seed(seed)


def set_num_threads(num_threads):
    torch.set_num_threads(num_threads)


def grad(x):
    return x.grad

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


def entropy(data, alpha=1, tol=1e-12):
    """ von Neuman or Renyi entropy from svd's"""
    Snorm = data.norm()
    if Snorm > 0:
        Smin = min(data)
        data = data / Snorm
        data = data[data > tol]
        if alpha == 1:
            ent = -2 * torch.sum(data * data * torch.log2(data))
        else:
            ent = torch.sum(data **(2 * alpha))
        if alpha != 1:
            ent = torch.log2(ent) / (1 - alpha)
        return ent, Smin, Snorm
    return Snorm, Snorm, Snorm  # this should be 0., 0., 0.


##########################
#     setting values     #
##########################


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
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
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
    return kernel_transpose.apply(data, axes, meta_transpose)

class kernel_transpose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, axes, meta_transpose):
        ctx.axes= axes
        ctx.meta_transpose= meta_transpose
        
        newdata = torch.zeros_like(data)
        for slice_to, slice_from, Do in meta_transpose:
            newdata[slice(*slice_to)] = data[slice(*slice_from)].view(Do).permute(axes).ravel()
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        axes= ctx.axes
        meta_transpose= ctx.meta_transpose
        inv_axes= tuple(np.argsort(axes))
        newdata_b = torch.zeros_like(data_b)
        for slice_to, slice_from, D_from in meta_transpose:
            Drsh = tuple(D_from[n] for n in axes)
            newdata_b[slice(*slice_from)] = data_b[slice(*slice_to)].view(Drsh).permute(inv_axes).ravel()
        return newdata_b, None, None

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


def svd_lowrank(data, meta, Usize, Ssize, Vsize, D_block, n_iter=60, k_fac=6):
    Udata = torch.zeros((Usize,), dtype=data.dtype, device=data.device)
    Sdata = torch.zeros((Ssize,), dtype=data.dtype, device=data.device)
    Vdata = torch.zeros((Vsize,), dtype=data.dtype, device=data.device)
    reg = torch.as_tensor(ad_decomp_reg, dtype=data.dtype, device=data.device)
    for (sl, D, slU, slS, slV) in meta:
        q = min(min(D), D_block)
        U, S, V = torch.svd_lowrank(data[slice(*sl)].view(D), q=q, niter=n_iter)
        Udata[slice(*slU)] = U.ravel()
        Sdata[slice(*slS)] = S.ravel()
        Vdata[slice(*slV)] = V.t().conj().ravel()
    return Udata, Sdata, Vdata


ad_decomp_reg = 1.0e-12


def svd(data, meta, Usize, Ssize, Vsize):
    Udata = torch.zeros((Usize,), dtype=data.dtype, device=data.device)
    Sdata = torch.zeros((Ssize,), dtype=data.dtype, device=data.device)
    Vdata = torch.zeros((Vsize,), dtype=data.dtype, device=data.device)
    reg = torch.as_tensor(ad_decomp_reg, dtype=data.dtype, device=data.device)
    for (sl, D, slU, slS, slV) in meta:
        U, S, V = SVDGESDD.apply(data[slice(*sl)].view(D), reg)
        Udata[slice(*slU)] = U.ravel()
        Sdata[slice(*slS)] = S.ravel()
        Vdata[slice(*slV)] = V.t().conj().ravel()
    return Udata, Sdata, Vdata



def eigh(data, meta=None, Ssize=1, Usize=1, order_by_magnitude=False,):
    Udata = torch.zeros((Usize,), dtype=data.dtype, device=data.device)
    Sdata = torch.zeros((Ssize,), dtype=data.dtype, device=data.device)
    if meta is not None:
        if order_by_magnitude:
            reg = torch.as_tensor(ad_decomp_reg, dtype=data.dtype, device=data.device)
            f = lambda x: SYMEIG.apply(x, reg)
        else:
            f = lambda x: torch.linalg.eigh(x)
        for (sl, D, slS) in meta:
            S, U = f(data[slice(*sl)].view(D))
            Sdata[slice(*slS)] = S.ravel()
            Udata[slice(*sl)] = U.ravel()
        return Sdata, Udata
    return torch.linalg.eigh(data)  # S, U


def qr(data, meta, Qsize, Rsize):
    Qdata = torch.zeros((Qsize,), dtype=data.dtype, device=data.device)
    Rdata = torch.zeros((Rsize,), dtype=data.dtype, device=data.device)
    for (sl, D, slQ, slR) in meta:
        Q, R = torch.linalg.qr(data[slice(*sl)].view(D))
        sR = torch.sign(real(R.diag()))
        sR[sR == 0] = 1
        Qdata[slice(*slQ)] = (Q * sR).ravel()  # positive diag of R
        Rdata[slice(*slR)] = (sR.reshape([-1, 1]) * R).ravel()
    return Qdata, Rdata


@torch.no_grad()
def select_global_largest(Sdata, St, Ssl, D_keep, D_total, keep_multiplets, eps_multiplet, ordering):
    if ordering == 'svd':
        s_all = torch.cat([Sdata[slice(*sl)][:D_keep[t]] for t, sl in zip(St, Ssl)])
    elif ordering == 'eigh':
        s_all = torch.cat([Sdata[slice(*sl)][-D_keep[t]:] for t, sl in zip(St, Ssl)])
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


def dot(Adata, Bdata, cc, meta_dot, Dsize):
    dtype = _common_type((Adata, Bdata))
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    f = dot_dict[cc]  # proper conjugations
    for (slc, sla, Da, slb, Db) in meta_dot:
        newdata[slice(*slc)] = f(Adata[slice(*sla)].view(Da), \
                                 Bdata[slice(*slb)].view(Db)).ravel()
    return newdata


def dot_with_mask(Adata, Bdata, cc, meta_dot, Dsize, msk_a, msk_b):
    dtype = _common_type((Adata, Bdata))
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    f = dot_dict[cc]  # proper conjugations
    for (slc, sla, Da, slb, Db, ia, ib) in meta_dot:
        newdata[slice(*slc)] = f(Adata[slice(*sla)].view(Da)[:, msk_a[ia]], \
                                 Bdata[slice(*slb)].view(Db)[msk_b[ib], :]).ravel()
    return newdata


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


def merge_to_1d(data, order, meta_new, meta_mrg, Dsize):
    return kernel_merge_to_1d.apply(data, order, meta_new, meta_mrg, Dsize)


class kernel_merge_to_1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, order, meta_new, meta_mrg, Dsize):
        ctx.order= order
        ctx.meta_new= meta_new
        ctx.meta_mrg= meta_mrg
        ctx.D_source= data.numel()

        # Dsize - total size of fused representation (might include some zero-blocks)
        newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)

        # meta_new -> list of [(tn, Dn, sln), ...] where
        #             tn -> effective charge for block in fused tensor
        #             Dn -> effective shape of block tn in fused tensor
        #             sln -> slice specifying the location of serialized tn block in 1d data of fused tensor  
        #
        # meta_mrg -> t1 is effective charge of source block after fusion. I.e. t1==tn, means, that 
        #             this source block will belong to destination block tn
        #          -> gr: tuple holding description of source data
        #                 slo -> specifies the location of source block in 1d data
        #                 Do  -> shape of the source block
        #                 Dscl-> list of slice data which specifies the location of the "transformed"
        #                        source block in the destination block tn
        #                 Drsh-> the shape of the "transformed" source block in the destination block tn
        # 
        for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
            assert tn == t1
            temp = torch.zeros(Dn, dtype=data.dtype, device=data.device)
            for (_, slo, Do, Dslc, Drsh) in gr:
                temp[tuple(slice(*x) for x in Dslc)] = data[slice(*slo)].reshape(Do).permute(order).reshape(Drsh)
            newdata[slice(*sln)] = temp.ravel()
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        order= ctx.order
        meta_new= ctx.meta_new
        meta_mrg= ctx.meta_mrg
        D_source= ctx.D_source

        inv_order= tuple(np.argsort(order))

        newdata_b = torch.zeros((D_source,), dtype=data_b.dtype, device=data_b.device)
        for (tn, D_source, slice_source), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
            tmp_b = data_b[slice(*slice_source)].view(D_source)
            for (_, slice_destination, D_destination, slice_source_block, D_source_block) in gr:
                slc = tuple(slice(*x) for x in slice_source_block)
                Drsh = tuple(D_destination[n] for n in order)
                newdata_b[slice(*slice_destination)] = tmp_b[slc].reshape(Drsh).permute(inv_order).ravel()
        return newdata_b, None, None, None, None


def merge_to_dense(data, Dtot, meta):
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


def unmerge_from_1d(data, meta, new_sl, Dsize):
    return kernel_unmerge_from_1d.apply(data, meta, new_sl, Dsize)

class kernel_unmerge_from_1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, meta, new_sl, Dsize):
        ctx.meta= meta
        ctx.new_sl= new_sl
        ctx.fwd_data_size= data.size()

        # slo -> slice in source tensor, specifying location of t_effective(fused) block
        # Do  -> shape of the fused block with t_eff
        newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
        for (slo, Do, sub_slc), snew in zip(meta, new_sl):
            #                                                     take a "subblock" of t_eff block
            #                                                     specified by a list of slices sub_slc
            newdata[slice(*snew)] = data[slice(*slo)].reshape(Do)[tuple(slice(*x) for x in sub_slc)].ravel()
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        meta= ctx.meta
        new_sl= ctx.new_sl
        fwd_data_size= ctx.fwd_data_size

        newdata_b = torch.zeros(fwd_data_size, dtype=data_b.dtype, device=data_b.device)
        for (slice_destination, D_eff_block, sub_slc), slice_source in zip(meta, new_sl):
            slc= tuple(slice(*x) for x in sub_slc)
            Drsh= tuple(x[1]-x[0] for x in sub_slc)
            newdata_b[slice(*slice_destination)].view(D_eff_block)[slc]=data_b[slice(*slice_source)].view(Drsh)
        return newdata_b, None, None, None

#############
#   tests   #
#############

def is_independent(x, y):
    """
    check if two arrays are identical, or share the same view.
    """
    return not ((x is y) or (x.numel() > 0 and x.storage().data_ptr() == y.storage().data_ptr()))
