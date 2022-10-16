"""Support of torch as a data structure used by yast."""
from itertools import chain, groupby
import numpy as np
import torch

def _torch_version_check(version):
    # for version="X.Y.Z" checks if current version is higher or equal to X.Y
    assert version.count('.')==2 and version.replace('.','').isdigit(),"Invalid version string"
    try:
        import pkg_resources
        return pkg_resources.parse_version(torch.__version__) >= pkg_resources.parse_version(version)
    except ModuleNotFoundError:  # pragma: no cover
        try:
            from packaging import version
            return version.parse(torch.__version__) >= version.parse(version)
        except ModuleNotFoundError:
            tokens= torch.__version__.split('.')
            tokens_v= version.split('.')
            return int(tokens[0]) > int(tokens_v[0]) or \
                (int(tokens[0])==int(tokens_v[0]) and int(tokens[1]) >= int(tokens_v[1])) 


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


def rand(D, dtype='float64', device='cpu'):
    ds = 1 if dtype=='float64' else 1 + 1j
    return 2 * torch.rand(D, dtype=DTYPE[dtype], device=device) - ds


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

if _torch_version_check("1.11.0"):
    def conj(data):
        return data.conj()
elif _torch_version_check("1.10.0"):
    def conj(data):
        return data.conj_physical() 
        # return data.conj()
else:
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
        ctx.axes = axes
        ctx.meta_transpose = meta_transpose
        
        newdata = torch.zeros_like(data)
        for sln, Dn, slo, Do in meta_transpose:
            newdata[slice(*sln)].view(Dn)[:] = data[slice(*slo)].view(Do).permute(axes)
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        axes = ctx.axes
        inv_axes = tuple(np.argsort(axes))
        meta_transpose = ctx.meta_transpose

        newdata_b = torch.zeros_like(data_b)
        for sln, Dn, slo, Do in meta_transpose:
            newdata_b[slice(*slo)].view(Do)[:] = data_b[slice(*sln)].view(Dn).permute(inv_axes)
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


def bitwise_not(data):
    return torch.bitwise_not(data)


def svd_lowrank(data, meta, sizes, n_iter=60, k_fac=6, **kwargs):
    # torch.svd_lowrank decomposes A = USV^T and return U,S,V
    # complex A is not supported 
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    Udata = torch.zeros((sizes[0],), dtype=data.dtype, device=data.device)
    Sdata = torch.zeros((sizes[1],), dtype=real_dtype, device=data.device)
    Vdata = torch.zeros((sizes[2],), dtype=data.dtype, device=data.device)
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        q = slS[1] - slS[0]
        U, S, V = torch.svd_lowrank(data[slice(*sl)].view(D), q=q, niter=n_iter)
        Udata[slice(*slU)].reshape(DU)[:] = U
        Sdata[slice(*slS)] = S
        Vdata[slice(*slV)].reshape(DV)[:] = V.t().conj()
    return Udata, Sdata, Vdata


def svd(data, meta, sizes, fullrank_uv=False, ad_decomp_reg=1.0e-12,\
    diagnostics=None, **kwargs):
    # SVDGESDD decomposes A = USV^\dag and return U,S,V^\dag
    #
    # NOTE: switch device to cpu as svd on cuda seems to be very slow.
    # device = data.device
    # data = data.to(device='cpu')
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    Udata = torch.empty((sizes[0],), dtype=data.dtype, device=data.device)
    Sdata = torch.empty((sizes[1],), dtype=real_dtype, device=data.device)
    Vhdata = torch.empty((sizes[2],), dtype=data.dtype, device=data.device)
    reg = torch.as_tensor(ad_decomp_reg, dtype=real_dtype, device=data.device)
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        # is_zero_block = torch.linalg.vector_norm(data[slice(*sl)]) == 0. if _torch_version_check("1.7.0") \
        #     else data[slice(*sl)].norm() == 0.
        # if is_zero_block: continue
        U, S, Vh = SVDGESDD.apply(data[slice(*sl)].view(D), reg, fullrank_uv, diagnostics)
        Udata[slice(*slU)].reshape(DU)[:] = U
        Sdata[slice(*slS)] = S
        Vhdata[slice(*slV)].reshape(DV)[:] = Vh
    #
    # Udata.to(device=device), Sdata.to(device=device), Vhdata.to(device=device)
    return Udata, Sdata, Vhdata



def eigh(data, meta=None, sizes=(1, 1), order_by_magnitude=False, ad_decomp_reg=1.0e-12):
    real_dtype= data.real.dtype if data.is_complex() else data.dtype
    Sdata = torch.zeros((sizes[0],), dtype=real_dtype, device=data.device)
    Udata = torch.zeros((sizes[1],), dtype=data.dtype, device=data.device)
    if meta is not None:
        if order_by_magnitude:
            reg = torch.as_tensor(ad_decomp_reg, dtype=real_dtype, device=data.device)
            f = lambda x: SYMEIG.apply(x, reg)
        else:
            f = lambda x: torch.linalg.eigh(x)
        for (sl, D, slU, DU, slS) in meta:
            S, U = f(data[slice(*sl)].view(D))
            Sdata[slice(*slS)] = S
            Udata[slice(*slU)].view(DU)[:] = U
        return Sdata, Udata
    return torch.linalg.eigh(data)  # S, U


def qr(data, meta, sizes):
    Qdata = torch.zeros((sizes[0],), dtype=data.dtype, device=data.device)
    Rdata = torch.zeros((sizes[1],), dtype=data.dtype, device=data.device)
    for (sl, D, slQ, DQ, slR, DR) in meta:
        Q, R = torch.linalg.qr(data[slice(*sl)].view(D))
        sR = torch.sign(real(R.diag()))
        sR[sR == 0] = 1
        Qdata[slice(*slQ)].view(DQ)[:] = Q * sR  # positive diag of R
        Rdata[slice(*slR)].view(DR)[:] = sR.reshape([-1, 1]) * R
    return Qdata, Rdata


@torch.no_grad()
def argsort(data):
    return torch.argsort(data)


@torch.no_grad()
def eigs_which(val, which):
    if which == 'LM':
        return (-abs(val)).argsort()
    if which == 'SM':
        return abs(val).argsort()
    if which == 'LR':
        return (-real(val)).argsort()
    #if which == 'SR':
    return (real(val)).argsort()


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


def apply_slice(data, slcn, slco):
    Dsize = slcn[-1][1] if len(slcn) > 0 else 0
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for sn, so in zip(slcn, slco):
        newdata[slice(*sn)] = data[slice(*so)]
    return newdata


def vdot(Adata, Bdata):
    dtype = _common_type((Adata, Bdata))
    if dtype != Adata.dtype:
        Adata = Adata.to(dtype=dtype)
    if dtype != Bdata.dtype:
        Bdata = Bdata.to(dtype=dtype)
    return Adata @ Bdata


def diag_1dto2d(data, meta, Dsize):
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for sln, slo in meta:
        newdata[slice(*sln)] = torch.diag(data[slice(*slo)]).ravel()
    return newdata


def diag_2dto1d(data, meta, Dsize):
    newdata = torch.zeros((Dsize,), dtype=data.dtype, device=data.device)
    for sln, slo, Do in meta:
        #newdata[slice(*sln)] = torch.diag(data[slice(*slo)].reshape(Do))
        torch.diag(data[slice(*slo)].reshape(Do), out=newdata[slice(*sln)])
    return newdata


def dot(Adata, Bdata, meta_dot, Dsize):
    dtype = _common_type((Adata, Bdata))
    if dtype != Adata.dtype:
        Adata = Adata.to(dtype=dtype)
    if dtype != Bdata.dtype:
        Bdata = Bdata.to(dtype=dtype)
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    for (slc, Dc, sla, Da, slb, Db, ia, ib) in meta_dot:
        newdata[slice(*slc)].view(Dc)[:] = Adata[slice(*sla)].view(Da) @ Bdata[slice(*slb)].view(Db)
    return newdata


def dot_with_mask(Adata, Bdata, meta_dot, Dsize, msk_a, msk_b):
    dtype = _common_type((Adata, Bdata))
    if dtype != Adata.dtype:
        Adata = Adata.to(dtype=dtype)
    if dtype != Bdata.dtype:
        Bdata = Bdata.to(dtype=dtype)
    newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
    for (slc, Dc, sla, Da, slb, Db, ia, ib) in meta_dot:
        newdata[slice(*slc)].view(Dc)[:] = Adata[slice(*sla)].view(Da)[:, msk_a[ia]] @ Bdata[slice(*slb)].view(Db)[msk_b[ib], :]
    return newdata


def dot_diag(Adata, Bdata, meta, Dsize, axis, a_ndim):
    dim = [1] * a_ndim
    dim[axis] = -1
    dtype = _common_type((Adata, Bdata))
    newdata = torch.empty((Dsize,), dtype=dtype, device=Adata.device)
    for sln, slb, Db, sla in meta:
        newdata[slice(*sln)].reshape(Db)[:] = Adata[slice(*sla)].reshape(dim) * Bdata[slice(*slb)].reshape(Db)
    return newdata


def mask_diag(Adata, Bdata, meta, Dsize, axis, a_ndim):
    slc1 = (slice(None),) * axis
    slc2 = (slice(None),) * (a_ndim - (axis + 1))
    newdata = torch.zeros((Dsize,), dtype=Adata.dtype, device=Adata.device)
    for sln, sla, Da, slb in meta:
        cut = (Bdata[slice(*slb)].nonzero(),)
        newdata[slice(*sln)] = Adata[slice(*sla)].reshape(Da)[slc1 + cut + slc2].ravel()
    return newdata


# dot_dict = {(0, 0): lambda x, y: x @ y,
#             (0, 1): lambda x, y: x @ y.conj(),
#             (1, 0): lambda x, y: x.conj() @ y,
#             (1, 1): lambda x, y: x.conj() @ y.conj()}


# def dot_nomerge(Adata, Bdata, cc, oA, oB, meta, Dsize):
#     f = dot_dict[cc]  # proper conjugations
#     dtype = _common_type((Adata, Bdata))
#     newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
#     for (sln, sla, Dao, Dan, slb, Dbo, Dbn) in meta:
#         newdata[slice(*sln)] += f(Adata[slice(*sla)].reshape(Dao).permute(oA).reshape(Dan), \
#                                   Bdata[slice(*slb)].reshape(Dbo).permute(oB).reshape(Dbn)).ravel()
#     return newdata


# def dot_nomerge_masks(Adata, Bdata, cc, oA, oB, meta, Dsize, tcon, ma, mb):
#     f = dot_dict[cc]  # proper conjugations
#     dtype = _common_type((Adata, Bdata))
#     newdata = torch.zeros((Dsize,), dtype=dtype, device=Adata.device)
#     for (sln, sla, Dao, Dan, slb, Dbo, Dbn), tt in zip(meta, tcon):
#         newdata[slice(*sln)] += f(Adata[slice(*sla)].reshape(Dao).permute(oA).reshape(Dan)[:, ma[tt]], \
#                                   Bdata[slice(*slb)].reshape(Dbo).permute(oB).reshape(Dbn)[mb[tt], :]).ravel()
#     return newdata

#####################################################
#     block merging, truncations and un-merging     #
#####################################################


def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge.apply(data, order, meta_new, meta_mrg, Dsize)


class kernel_transpose_and_merge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, order, meta_new, meta_mrg, Dsize):
        ctx.order = order
        ctx.meta_new = meta_new
        ctx.meta_mrg = meta_mrg
        ctx.D_source = data.numel()

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
            temp = newdata[slice(*sln)].reshape(Dn)
            for (_, slo, Do, Dslc, Drsh) in gr:
                slcs = tuple(slice(*x) for x in Dslc)
                temp[slcs] = data[slice(*slo)].reshape(Do).permute(order).reshape(Drsh)
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        order = ctx.order
        inv_order= tuple(np.argsort(order))
        meta_new = ctx.meta_new
        meta_mrg = ctx.meta_mrg
        D_source = ctx.D_source

        newdata_b = torch.zeros((D_source,), dtype=data_b.dtype, device=data_b.device)
        for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
            assert tn == t1
            tmp_b = data_b[slice(*sln)].view(Dn)
            for (_, slo, Do, Dslc, _) in gr:
                slcs = tuple(slice(*x) for x in Dslc)
                inv_Do = tuple(Do[n] for n in order)
                newdata_b[slice(*slo)].reshape(Do)[:] = tmp_b[slcs].reshape(inv_Do).permute(inv_order)
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
        for (_, slo, Do, pos, Dslc) in gr:
            slcs = tuple(slice(*x) for x in Dslc)
            newdata[slice(*sln)].reshape(Dn)[slcs] = pos_tens[pos]._data[slice(*slo)].reshape(Do)
    return newdata

def unmerge(data, meta):
    return kernel_unmerge.apply(data, meta)

class kernel_unmerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, meta):
        Dsize = data.size()
        ctx.meta = meta
        ctx.fwd_data_size = Dsize
        # slo -> slice in source tensor, specifying location of t_effective(fused) block
        # Do  -> shape of the fused block with t_eff
        # no zero blocks should be introduces here
        newdata = torch.empty(Dsize, dtype=data.dtype, device=data.device)
        for sln, Dn, slo, Do, sub_slc in meta:
            #                                                     take a "subblock" of t_eff block
            #                                                     specified by a list of slices sub_slc
            slcs = tuple(slice(*x) for x in sub_slc)
            newdata[slice(*sln)].view(Dn)[:] = data[slice(*slo)].view(Do)[slcs]
        return newdata

    @staticmethod
    def backward(ctx, data_b):
        meta = ctx.meta
        fwd_data_size = ctx.fwd_data_size
        # no zero blocks should be introduces here
        newdata_b = torch.empty(fwd_data_size, dtype=data_b.dtype, device=data_b.device)
        for sln, Dn, slo, Do, sub_slc in meta:
            slcs = tuple(slice(*x) for x in sub_slc)
            newdata_b[slice(*slo)].view(Do)[slcs] = data_b[slice(*sln)].view(Dn)
        return newdata_b, None, None, None

#############
#   tests   #
#############

def is_independent(x, y):
    """
    check if two arrays are identical, or share the same view.
    """
    return not ((x is y) or (x.numel() > 0 and x.storage().data_ptr() == y.storage().data_ptr()))
