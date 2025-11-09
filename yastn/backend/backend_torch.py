# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Support of torch as a data structure used by yastn."""
from itertools import groupby
from functools import reduce
import torch
from torch.utils.checkpoint import checkpoint as _checkpoint
from .linalg.torch_eig_sym import SYMEIG
from ._backend_torch_backwards import kernel_svd, kernel_svds_scipy
from ._backend_torch_backwards import kernel_dot, kernel_transpose_dot_sum, kernel_negate_blocks
from ._backend_torch_backwards import kernel_apply_mask, kernel_embed_mask
from ._backend_torch_backwards import kernel_transpose, kernel_transpose_and_merge, kernel_unmerge


__all__= [
    'DTYPE', 'cuda_is_available',
    'get_dtype', 'is_complex', 'get_device', 'random_seed', 'grad',
    'detach', 'detach_', 'clone', 'copy',
    'to_numpy', 'get_shape', 'get_size', 'diag_create', 'diag_get', 'real',
    'imag', 'max_abs', 'maximum', 'norm_matrix', 'delete', 'insert',
    'expm', 'first_element', 'item', 'sum_elements', 'norm', 'entropy',
    'zeros', 'ones', 'rand', 'to_tensor', 'to_mask', 'square_matrix_from_dict',
    'requires_grad_', 'requires_grad', 'move_to', 'conj',
    'trace', 'rsqrt', 'reciprocal', 'exp', 'sqrt', 'absolute',
    'fix_svd_signs', 'svdvals', 'svd_lowrank', 'svd', 'svd_randomized', 'svds_scipy',
    'eigh', 'qr', 'pinv',
    'argsort', 'eigs_which', 'allclose',
    'add', 'sub', 'apply_mask', 'vdot', 'diag_1dto2d', 'diag_2dto1d',
    'dot', 'dot_diag', 'transpose_dot_sum',
    'merge_to_dense', 'merge_super_blocks', 'is_independent',
    'apply_mask', 'embed_mask', 
    'transpose', 'transpose_and_merge', 'unmerge']


torch.random.seed()
BACKEND_ID = "torch"
DTYPE = {'float32': torch.float32,
         'float64': torch.float64,
         'complex64': torch.complex64,
         'complex128': torch.complex128}


def cuda_is_available():
    return torch.cuda.is_available()


def get_dtype(t):
    return t.dtype


def is_complex(x):
    return x.is_complex()


def get_device(x):
    return str(x.device)


def random_seed(seed):
    torch.random.manual_seed(seed)


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
    return x.resolve_conj().detach().cpu().numpy()


def get_shape(x):
    return x.size()


def get_size(x):
    return x.numel()


def diag_create(x, p=0):
    return torch.diag(x, diagonal=p)


def diag_get(x):
    return torch.diag(x)


def real(x):
    return torch.real(x)


def imag(x):
    return torch.imag(x) if torch.is_complex(x) else 0 * x


def max_abs(x):
    return x.abs().max()

def maximum(input, output):
    return torch.maximum(input, output)

def norm_matrix(x):
    return torch.linalg.norm(x)


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
    return data.abs().max() if len(data) > 0 else torch.tensor(0.)


def entropy(data, alpha, tol):
    """ von Neuman or Renyi entropy from svd's"""
    Snorm = torch.sum(data) if len(data) > 0 else 0.
    if Snorm > 0:
        data = data / Snorm
        data = data[data > tol]
        if alpha == 1:
            return -1 * torch.sum(data * torch.log2(data))
        return torch.log2(torch.sum(data ** alpha)) / (1 - alpha)
    return torch.tensor(0.)


##########################
#     setting values     #
##########################


def zeros(D, dtype='float64', device='cpu'):
    return torch.zeros(D, dtype=DTYPE[dtype], device=device)


def ones(D, dtype='float64', device='cpu'):
    return torch.ones(D, dtype=DTYPE[dtype], device=device)


def rand(D, dtype='float64', distribution=(0, 1), device='cpu'):
    if distribution == 'normal':
        return torch.randn(D, dtype=DTYPE[dtype], device=device)
    ds = 1 if dtype=='float64' else 1 + 1j
    out = torch.rand(D, dtype=DTYPE[dtype], device=device)
    return out if distribution == (0, 1) else (distribution[1] - distribution[0]) * out + distribution[0] * ds


def randint(low, high):
    return torch.randint(low, high, (1,))[0]


def to_tensor(val, Ds=None, dtype='float64', device='cpu'):
    T = torch.as_tensor(val, dtype=DTYPE[dtype], device=device)
    return T if Ds is None else T.reshape(Ds).contiguous()


@torch.no_grad()
def to_mask(val):
    return torch.as_tensor(val).nonzero().ravel()


def square_matrix_from_dict(H, D=None, **kwargs):
    dtype = reduce(torch.promote_types, (a.dtype for a in H.values()))
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
    dtype = kwargs.pop("dtype", None)
    if dtype is not None:
        kwargs["dtype"] = DTYPE[dtype]
    return data.to(*args, **kwargs)


def conj(data):
    return data.conj()


def trace(data, order, meta, Dsize):
    newdata = torch.zeros(Dsize, dtype=data.dtype, device=data.device)
    for (sln, list_sln) in meta:
        tmp = newdata[slice(*sln)]
        for slo, Do, Drsh in list_sln:
            tmp_sln = data[slice(*slo)].reshape(Do).permute(order).reshape(Drsh)
            tmp += torch.sum(torch.diagonal(tmp_sln, dim1=0, dim2=1), dim=-1)
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


def bitwise_not(data):
    return torch.bitwise_not(data)


def svd_lowrank(data, meta, sizes, **kwargs):
    return svds_scipy(data, meta, sizes, solver='arpack', **kwargs)


def svd(data, meta, sizes, fullrank_uv=False, ad_decomp_reg=1.0e-12, diagnostics=None, **kwargs):
    return kernel_svd.apply(data, meta, sizes, fullrank_uv, ad_decomp_reg, diagnostics)


def svdvals(data, meta, sizeS, **kwargss):
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    Sdata = torch.zeros((sizeS,), dtype=real_dtype, device=data.device)
    for (sl, D, _, _, slS, _, _) in meta:
        Sdata[slice(*slS)] = torch.linalg.svdvals(data[slice(*sl)].view(D))
    return Sdata


def svd_randomized(data, meta, sizes, q=None, niter=3, **kwargs):
    """
    Computes the SVD of block-sparse matrix using randomized SVD within each block.
    See torch.svd_lowrank for details.
    """
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    Udata = torch.zeros((sizes[0],), dtype=data.dtype, device=data.device)
    Sdata = torch.zeros((sizes[1],), dtype=real_dtype, device=data.device)
    Vdata = torch.zeros((sizes[2],), dtype=data.dtype, device=data.device)
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        k = slS[1] - slS[0]
        q=max(k,min([2*k]+list(D)))
        U, S, V = torch.svd_lowrank(data[slice(*sl)].view(D), q=q, niter=niter)
        Udata[slice(*slU)].reshape(DU)[:] = U[:,:k]
        Sdata[slice(*slS)] = S[:k]
        Vdata[slice(*slV)].reshape(DV)[:] = V[:,:k].t().conj()
    return Udata, Sdata, Vdata


def svds_scipy(data, meta, sizes, thresh=0.1, solver='arpack', **kwargs):
    return kernel_svds_scipy.apply(data,meta,sizes, thresh, solver, **kwargs)


def fix_svd_signs(Udata, Vhdata, meta):
    Ud = torch.empty_like(Udata)
    Vhd = torch.empty_like(Vhdata)
    Uamp = (abs(Udata) * (2**40)).to(dtype=torch.int64)
    for (_, _, slU, DU, _, slV, DV) in meta:
        Utemp = Udata[slice(*slU)].reshape(DU)
        Vtemp = Vhdata[slice(*slV)].reshape(DV)
        Utemp_amp = Uamp[slice(*slU)].reshape(DU)
        ii = torch.argmax(Utemp_amp, dim=0, keepdims=True)
        phase = torch.take_along_dim(Utemp, ii, dim=0)
        phase = phase / abs(phase)
        Ud[slice(*slU)].reshape(DU)[:] = Utemp * phase.conj().reshape(1, -1)
        Vhd[slice(*slV)].reshape(DV)[:] = Vtemp * phase.reshape(-1, 1)
    return Ud, Vhd


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


def eig(data, meta=None, sizes=(1, 1), **kwargs):
    if meta is None:
        return torch.linalg.eig(data)  # S, U
    # NOTE torch.linalg.eig returns right eigenvectors U only, i.e. M U = diag(S) U
    #
    # Assume worst case ?
    Udata = torch.empty((sizes[0],), dtype=DTYPE['complex128'], device=data.device)
    Sdata = torch.empty((sizes[1],), dtype=DTYPE['complex128'], device=data.device)
    Vdata = torch.empty((sizes[2],), dtype=DTYPE['complex128'], device=data.device)
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        S, U = torch.linalg.eig(data[slice(*sl)].reshape(D))
        #
        # in general diag(U.H @ U) = 1 but not U.H @ U = I, i.e. right eigenvectors are not orthogonal
        #
        # The solutions satisfy
        # M @ U / U = S (as cols)
        # V.H @ M / V.H = S (as rows)
        #
        # Search for left eigenvectors V (rows) via biorthogonality condition V.H @ U = I
        try:
            V= torch.linalg.solve(U.conj().T, torch.eye(len(S), dtype=U.dtype, device=data.device), left=True, out=None)
            V= V.conj().T
        except Exception as e:
            raise ValueError("Biorthonormalization of left/right eigenvector pairs failed.") from e

        tol= 1.0e-12 if data.is_complex() else 1.0e-14
        if any( torch.abs(torch.sum(V.T * U, axis=0) - 1) > tol ):
            raise ValueError("Biorthonormalization of left/right eigenvector pairs failed.")

        s_order= eigs_which(S, which=kwargs.get('which', 'LM'))
        Udata[slice(*slU)].reshape(DU)[:] = U[:,s_order]
        Sdata[slice(*slS)] = S[s_order]
        Vdata[slice(*slV)].reshape(DV)[:] = V[s_order,:]
    return Udata, Sdata, Vdata


def eigvals(data, meta, sizeS, **kwargs):
    Sdata = torch.empty((sizeS,), dtype=DTYPE['complex128'], device=data.device)
    for (sl, D, _, _, slS, _, _) in meta:
        S = torch.linalg.eigvals(data[slice(*sl)].reshape(D))
        Sdata[slice(*slS)]= S[eigs_which(S, which=kwargs.get('which', 'LM'))]
    return Sdata


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


def pinv(A, rcond=None, hermitian=False, out=None, atol=None, rtol=None):
    return torch.linalg.pinv(A, atol=atol, rtol=rtol if not rtol is None else rcond, hermitian=hermitian, out=out)


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


################################
#     two dicts operations     #
################################


def allclose(Adata, Bdata, rtol, atol):
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    return torch.allclose(Adata.to(dtype=dtype), Bdata.to(dtype=dtype), rtol=rtol, atol=atol)


def add(datas, metas, Dsize):
    dtype = reduce(torch.promote_types, (data.dtype for data in datas))
    newdata = torch.zeros(Dsize, dtype=dtype, device=datas[0].device)
    for data, meta in zip(datas, metas):
        for sl_c, sl_a in meta:
            newdata[slice(*sl_c)] += data[slice(*sl_a)]
    return newdata


def sub(Adata, Bdata, metas, Dsize):
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    newdata = torch.zeros(Dsize, dtype=dtype, device=Adata.device)
    for sl_c, sl_a in metas[0]:
        newdata[slice(*sl_c)] += Adata[slice(*sl_a)]
    for sl_c, sl_b in metas[1]:
        newdata[slice(*sl_c)] -= Bdata[slice(*sl_b)]
    return newdata


def vdot(Adata, Bdata, meta):
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    if dtype != Adata.dtype:
        Adata = Adata.to(dtype=dtype)
    if dtype != Bdata.dtype:
        Bdata = Bdata.to(dtype=dtype)
    tmp = torch.empty(len(meta), dtype=dtype, device=Adata.device)
    for ii, (sla, slb) in enumerate(meta):
        tmp[ii] = torch.dot(Adata[slice(*sla)], Bdata[slice(*slb)])
    return torch.sum(tmp)


def dot(Adata, Bdata, meta_dot, Dsize):
    return kernel_dot.apply(Adata, Bdata, meta_dot, Dsize)


def transpose_dot_sum(Adata, Bdata, meta_dot, Areshape, Breshape, Aorder, Border, Dsize):
    return kernel_transpose_dot_sum.apply(Adata, Bdata, meta_dot, Areshape, Breshape, Aorder, Border, Dsize)


def dot_diag(Adata, Bdata, meta, Dsize, axis, a_ndim):
    dim = [1] * a_ndim
    dim[axis] = -1
    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    newdata = torch.empty(Dsize, dtype=dtype, device=Adata.device)
    for sln, slb, Db, sla in meta:
        newdata[slice(*sln)].reshape(Db)[:] = Adata[slice(*sla)].reshape(dim) * Bdata[slice(*slb)].reshape(Db)
    return newdata


def negate_blocks(Adata, slices):
    return kernel_negate_blocks.apply(Adata, slices)


#####################################################
#     block merging, truncations and un-merging     #
#####################################################


def apply_mask(Adata, mask, meta, Dsize, axis, a_ndim):
    return kernel_apply_mask.apply(Adata, mask, meta, Dsize, axis, a_ndim)


def embed_mask(Adata, mask, meta, Dsize, axis, a_ndim):
    return kernel_embed_mask.apply(Adata, mask, meta, Dsize, axis, a_ndim)


def transpose(data, axes, meta_transpose):
    return kernel_transpose.apply(data, axes, meta_transpose)


def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    return kernel_transpose_and_merge.apply(data, order, meta_new, meta_mrg, Dsize)


def unmerge(data, meta):
    return kernel_unmerge.apply(data, meta)


def merge_to_dense(data, Dtot, meta):
    newdata = torch.zeros(Dtot, dtype=data.dtype, device=data.device)
    for (sl, Dss) in meta:
        newdata[tuple(slice(*Ds) for Ds in Dss)] = data[sl].reshape(tuple(Ds[1] - Ds[0] for Ds in Dss))
    return newdata.ravel()


def merge_super_blocks(pos_tens, meta_new, meta_block, Dsize):
    dtype = reduce(torch.promote_types, (a._data.dtype for a in pos_tens.values()))
    device = next(iter(pos_tens.values()))._data.device
    newdata = torch.zeros(Dsize, dtype=dtype, device=device)
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_block, key=lambda x: x[0])):
        assert tn == t1
        for (_, slo, Do, pos, Dslc) in gr:
            slcs = tuple(slice(*x) for x in Dslc)
            newdata[slice(*sln)].reshape(Dn)[slcs] = pos_tens[pos]._data[slice(*slo)].reshape(Do)
    return newdata


def diag_1dto2d(data, meta, Dsize):
    newdata = torch.zeros(Dsize, dtype=data.dtype, device=data.device)
    for sln, slo in meta:
        newdata[slice(*sln)] = torch.diag(data[slice(*slo)]).ravel()
    return newdata


def diag_2dto1d(data, meta, Dsize):
    newdata = torch.zeros(Dsize, dtype=data.dtype, device=data.device)
    for sln, slo, Do in meta:
        torch.diag(data[slice(*slo)].reshape(Do), out=newdata[slice(*sln)])
    return newdata


# functionals
def checkpoint(f, *args, **kwargs):
    # context_fn=kwargs.pop('context_fn',None)
    # torch.utils.checkpoint.checkpoint
    return _checkpoint(f, *args,
                       use_reentrant=kwargs.pop('use_reentrant',None),
                       determinism_check=kwargs.pop('determinism_check', 'default'),
                       debug=kwargs.pop('debug', False),
                       **kwargs)


#############
#   tests   #
#############


def is_independent(x, y):
    """
    check if two arrays are identical, or share the same view.
    """
    return (x is not y) and (x.numel() == 0 or x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr())
