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
""" Linalg methods for yastn.Tensor. """
from __future__ import annotations

import logging
import sys
from numbers import Number
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from ._auxiliary import _struct, _clear_axes, _unpack_axes, get_blocks, find_index, argsort_t
from ._auxiliary import convert_to_tuples_and_slices, find_matching_indices, get_trimmed_struct
from ._legbasic import LegBasic
from ._merging import _Fusion, _fuse_blocks, _unfuse_blocks
from ._tests import YastnError, _test_axes_all
from ._single import remove_zero_blocks

if TYPE_CHECKING:
    from . import Tensor

__all__ = ['qr', 'norm', 'entropy', 'truncation_mask', 'truncation_mask_multiplets',
           'svd', 'svd_with_truncation', 'eig', 'eigh', 'eigh_with_truncation']

logger = logging.getLogger(__name__)


def norm(a, p='fro') -> Number:
    r"""
    Return the norm of the tensor.

    Parameters
    ----------
    p: str
        ``'fro'`` for Frobenius norm;  ``'inf'`` for :math:`l^\infty` (or supremum) norm.
    """
    if p not in ('fro', 'inf'):
        raise YastnError("p should be 'fro', or 'inf'.")
    return a.config.backend.norm(a._data, p)


def svd_with_truncation(a, axes=(0, 1),
                        sU=1,
                        nU=True,
                        Uaxis=-1,
                        Vaxis=0,
                        policy='fullrank',
                        fix_signs=False,
                        svd_on_cpu=False,
                        tol=float('-inf'),
                        tol_block=float('-inf'),
                        D_total=float('inf'),
                        D_block=float('inf'),
                        largest_gap=False,
                        eps_multiplet=None,
                        hermitian=False,
                        mask_f=None,
                        **kwargs) -> tuple['Tensor', 'Tensor', 'Tensor']:
    r"""
    Split a tensor using an exact singular value decomposition (SVD) into :math:`a = U S V`,
    where the columns of `U` and the rows of `V` form orthonormal bases
    and `S` is a positive diagonal matrix.

    The function allows optional truncation based on relative tolerance, block-wise bond dimension,
    and total bond dimension across all blocks (whichever gives the smaller total dimension).

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform SVD, as well as
        their final order.

    sU: int
        Signature of the new leg in `U`; equal to `1` or `-1`. The default is `1`.
        `V` has the opposite signature on the connecting leg.

    nU: bool
        Whether or not to attach the charge of  ``a`` to `U`.
        If ``False``, it is attached to `V`. The default is ``True``.

    Uaxis, Vaxis: int
        Specify which legs of `U` and `V` connect to `S`. By default,
        these are the last leg of `U` and the first leg of `V`.

    policy: str
        ``"fullrank"`` or ``"lowrank"`` are allowed. For ``"fullrank"`` use a standard full (but reduced) SVD,
        while ``"lowrank"`` uses a randomized or truncated SVD and requires ``D_block`` or ``k_block`` in ``kwargs``.

    tol: float
        Relative tolerance with respect to the largest absolute value element of ``S``.

    tol_block: float
        Relative tolerance per block.

    D_total: int
        Maximum number of elements kept across all blocks.

    D_block: int | dict
        Maximum number of elements kept per block.
        It is also possible to provide a dictionary mapping charges to maximal number of elements in the charge sector.

    largest_gap: bool
        If ``True``, enlarge the truncation range specified by other arguments by shifting
        the cut to the largest gap between to-be-truncated singular values across all blocks.
        It provides a heuristic mechanism to avoid truncating part of a multiplet.
        If ``True``, ``tol_block`` and ``D_block`` are ignored, as ``largest_gap`` is a global condition.
        The default is ``False``.

    eps_multiplet: float
        Relative tolerance on multiplet splitting. If relative difference between
        two consecutive elements of ``S`` is larger than ``eps_multiplet``, these
        elements are not considered as part of the same multiplet.
        Partially truncated multiplets are truncated down.
        The default is None, when this scheme is not used.
        If ``True``, ``tol_block`` and ``D_block`` are ignored, as ``eps_multiplet`` is a global condition.
        Cannot be used together with largest_gap scheme.

    hermitian: bool
        If True, blocks related by hermitian conjugation are truncated equally, truncating down to the intersecting part.
        The default is False.

    mask_f: None | function[yastn.Tensor] -> yastn.Tensor
        It is possible to provide a custom mask function, which provides a mechanism to pass such a function
        to many tensor network algorithms where the function truncation_mask is being called.
        If provided, it overrides the default function, and all other parameters are ignored.
        The default is None.

    Returns
    -------
    `U`, `S`, `V`
    """
    verbosity = kwargs.get('verbosity', 0)
    U, S, V = svd(a, axes=axes, sU=sU, nU=nU, policy=policy, D_block=D_block,
                  fix_signs=fix_signs, svd_on_cpu=svd_on_cpu, **kwargs)
    Smask = truncation_mask(S, tol=tol, tol_block=tol_block,
                            D_block=D_block, D_total=D_total,
                            largest_gap=largest_gap,
                            eps_multiplet=eps_multiplet,
                            hermitian=hermitian,
                            mask_f=mask_f,
                            verbosity=verbosity)

    U, S, V = Smask.apply_mask(U, S, V, axes=(-1, 0, 0))
    U = U.moveaxis(source=-1, destination=Uaxis)
    V = V.moveaxis(source=0, destination=Vaxis)
    return U, S, V


def svd(a, axes=(0, 1), sU=1, nU=True, compute_uv=True,
        Uaxis=-1, Vaxis=0, policy='fullrank',
        fix_signs=False, svd_on_cpu=False, thresh=0.1, **kwargs) -> tuple['Tensor', 'Tensor', 'Tensor'] | 'Tensor':
    r"""
    Split a tensor into :math:`a = U S V` using an exact singular value decomposition (SVD),
    where the columns of `U` and the rows of `V` form orthonormal bases
    and `S` is a positive diagonal matrix.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform SVD, as well as
        their final order.

    sU: int
        Signature of the new leg in `U`; equal to 1 or -1. The default is 1.
        `V` is going to have the opposite signature on the connecting leg.

    nU: bool
        Whether or not to attach the charge of ``a`` to `U`.
        If ``False``, it is attached to `V`. The default is ``True``.

    compute_uv: bool
        If ``True``, compute and return `U`, `S`, `V`.  If ``False``, compute and return only `S`.
        The default is ``True``.

    Uaxis, Vaxis: int
        Specify which leg of `U` and `V` tensors are connecting with `S`. By default,
        it is the last leg of `U` and the first of `V`, in which case ``a = U @ S @ V``.

    policy: str
        Strategy for computing the SVD or a partial SVD.

            * (default) ``"fullrank"`` compute full SVD then truncate.
            * ``"lowrank"`` default policy for partial SVD.
                On NumPy backend uses ``block_arnoldi``. On torch backend uses ``block_arnoldi``.
            * ``"randomized"`` randomized SVD up to desired size in each block. Requires providing ``k_block`` in ``kwargs``.
                Requires torch backend and uses ``torch.svd_lowrank``.
            * ``"block_arnoldi"`` partial SVD using scipy's svds arnoldi method. Requires providing ``k_block`` in ``kwargs``.
            * ``"block_propack"`` partial SVD using scipy's svds propack method. Requires providing ``k_block`` in ``kwargs``.

        kwargs will be passed to those functions for non-default settings.

    thresh: float
        For ``policy='block_arnoldi'`` or ``policy='block_propack'``, this sets the
        threshold on the minimal block size for applying a partial SVD solver instead of a full SVD.
        The default is ``thresh=0.1``. If for a matrix of size :math:`N \times N`
        ``N * thresh`` is smaller than the requested number of singular triples, a full SVD is applied.

    fix_signs: bool
        Whether or not to fix phases in `U` and `V`,
        so that the largest element in each column of `U` is positive.
        Provide uniqueness of decomposition for non-degenerate cases.
        The default is ``False``.

    svd_on_cpu: bool
        GPU tends to be very slow when executing SVD.
        If ``True``, the data will be copied to CPU for SVD,
        and the results will be copied back to the device.
        Nothing is done for data already residing on CPU.
        The default is ``False``.

    k_block: None (default) | int | dict
        When ``policy='lowrank'``, number of singular values to compute in each block.
        If ``D_block`` is provided, it is used instead to determine number of singular values to compute.

    Returns
    -------
    `U`, `S`, `V` (when ``compute_uv=True``) or `S` (when ``compute_uv=False``)
    """
    sym = a.config.sym
    POLICIES = ['fullrank', 'lowrank', 'randomized', 'block_arnoldi', 'block_propack', 'krylov']
    #
    # 1. validation
    if policy not in POLICIES:
       raise YastnError(f"Invalid SVD solver/policy {policy}. Choose one of {POLICIES}.")
    _test_axes_all(a, axes)
    #
    #  non-default D_block provides defaults for k_block
    if 'D_block' in kwargs and kwargs['D_block'] not in [None, float('inf')] and \
        ('k_block' not in kwargs or kwargs['k_block'] in [None,]):
        kwargs['k_block'] = kwargs['D_block']

    # 2. Global solvers
    verbosity = kwargs.get('verbosity', 0)
    if policy == "krylov":
        from ..krylov._krylov import svds
        if 'k_block' not in kwargs:
            raise YastnError(policy + " policy in svd requires passing argument k_block.")
        # Sparse global SVD cannot initialize an all-zero sector. Its partial
        # result is allowed to omit such sectors.
        a = remove_zero_blocks(a)
        # WIP: BUG for SVDS
        k_block = min(kwargs['k_block'], min(a.get_shape(axes=0), a.get_shape(axes=1)))
        U, S, Vh = svds(a, axes=axes, sU=sU, nU=nU, k=k_block, ncv=None, tol=0, which='LM', solver='arpack')
        return U, S, Vh

    # 3. Continue with block-wise SVD
    out_ml, out_mr = _clear_axes(*axes)
    #
    # unpack meta-fusion and apply transpose
    out_hl, out_hr = _unpack_axes(a.mfs, out_ml, out_mr)
    out_hl = tuple(a.trans[ax] for ax in out_hl)
    out_hr = tuple(a.trans[ax] for ax in out_hr)
    #
    data, struct_am, hfsm = _fuse_blocks(a.config, a._data, a.struct, (out_hl, out_hr))
    #
    if svd_on_cpu:
        device = a.config.backend.get_device(data)
        data = a.config.backend.move_to(data, device='cpu')
    #
    k_block = None
    if policy in ['lowrank', 'randomized', 'block_arnoldi', 'block_propack']:
        if 'k_block' not in kwargs:
            raise YastnError(policy + " policy in svd requires passing argument D_block or k_block.")
        k_block = kwargs['k_block']
        # A partial solver must not receive an exactly-zero effective block.
        # Keep this after fusion: fusion can introduce structural zero blocks.
        am = remove_zero_blocks(a._replace(struct=struct_am, data=data))
        data, struct_am = am._data, am.struct

    if verbosity > 2:
        fname = sys._getframe().f_code.co_name
        logger.info(f"{fname} {policy} struct {struct_am}")
        logger.info(f"{fname} D_block {kwargs.get('D_block', 'NA')}")
        logger.info(f"{fname} k_block {k_block}")

    meta, sizes, struct_Um, struct_S, struct_Vm = _meta_svd(sym, struct_am, sU, nU, k_block)

    if compute_uv and policy == 'fullrank':
        Udata, Sdata, Vdata = a.config.backend.svd(data, meta, sizes, diagnostics=kwargs.get('diagnostics', None))
    elif not compute_uv and policy == 'fullrank':
        Sdata = a.config.backend.svdvals(data, meta, sizes[1])
    elif compute_uv and policy == 'lowrank':
        Udata, Sdata, Vdata = a.config.backend.svd_lowrank(data, meta, sizes)
    elif policy == 'randomized': # always computes partial U and V
        Udata, Sdata, Vdata = a.config.backend.svd_randomized(data, meta, sizes, **kwargs)
    elif compute_uv and policy in ['block_arnoldi', 'block_propack']:
        thresh = kwargs.get('svds_thresh', 0.1)
        if policy == 'block_arnoldi':
            solver = 'arpack'
        elif policy == 'block_propack':
            solver = 'propack'
        Udata, Sdata, Vdata = a.config.backend.svds_scipy(data, meta, sizes, thresh, solver)
    else:
        raise YastnError("compute_uv == False is supported only for policy='fullrank'")

    # 4. post-processing
    if svd_on_cpu:
        Sdata = a.config.backend.move_to(Sdata, device=device)
        if compute_uv:
            Udata = a.config.backend.move_to(Udata, device=device)
            Vdata = a.config.backend.move_to(Vdata, device=device)

    if compute_uv and fix_signs:
        Udata, Vdata = a.config.backend.fix_svd_signs(Udata, Vdata, meta)

    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=struct_S, data=Sdata, mfs=Smfs, hfs=Shfs, trans=None)

    if not compute_uv:
        return S

    hfsUm = (hfsm[0], _Fusion(s=(sU,)))
    Udata, struct_U = _unfuse_blocks(a.config, Udata, struct_Um, (0,), hfsUm)
    Umfs = tuple(a.mfs[ii] for ii in out_ml) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in out_hl) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=struct_U, data=Udata, mfs=Umfs, hfs=Uhfs, trans=None)

    hfsVm = (_Fusion(s=(-sU,)), hfsm[1])
    Vdata, struct_V = _unfuse_blocks(a.config, Vdata, struct_Vm, (1,), hfsVm)
    Vmfs = ((1,),) + tuple(a.mfs[ii] for ii in out_mr)
    Vhfs = (_Fusion(s=(-sU,)),) + tuple(a.hfs[ii] for ii in out_hr)
    V = a._replace(struct=struct_V, data=Vdata, mfs=Vmfs, hfs=Vhfs, trans=None)

    U = U.moveaxis(source=-1, destination=Uaxis)
    V = V.moveaxis(source=0, destination=Vaxis)
    return U, S, V


def _meta_svd(sym, struct, sU, nU, k_block):
    """
    Return metadata and structures for an SVD.

    `U` has signature ``(legs[0].s, sU)``, `S` has signature ``(-sU, sU)``,
    and `V` has signature ``(-sU, legs[1].s)``. If ``nU`` is true, `U` carries
    the tensor charge; otherwise `V` does.

    Returns
    -------
        tuple[tuple[slice, shape, slice in U, shape in U, slice in S, slice in V, shape in V ]]
    """
    bl_a = get_blocks(sym, struct)

    ax0 = 1 if nU else 0
    minD = {tuple(tt): min(DD) for tt, DD in zip(bl_a.t[:, ax0, :].tolist(), bl_a.D)}
    if k_block is not None:
        if isinstance(k_block, dict):
            sector_minD = min(k_block.values())  # TODO: control default for sectors not present in k_block
            minD = {t: min(k_block.get(t, sector_minD), d) for t, d in minD.items()}
        else:
            minD = {t: min(k_block, d) for t, d in minD.items()}

    ts = tuple(sorted(t for t, d in minD.items() if d > 0))
    Ds = tuple(minD[tt] for tt in ts)
    ss = struct.legs[1].s if nU else -struct.legs[0].s
    legU = LegBasic(s=ss, t=ts, D=Ds)
    if sU != legU.s:
        legU = legU.conj_charges(sym)

    n0 = sym.zero()
    struct_U = _struct(legs=(struct.legs[0], legU), n=struct.n if nU else n0, isdiag=False)
    struct_S = _struct(legs=(legU.conj(), legU), n=n0, isdiag=True)
    struct_V = _struct(legs=(legU.conj(), struct.legs[1]), n=n0 if nU else struct.n, isdiag=False)

    struct_U = get_trimmed_struct(sym, struct_U)
    struct_S = get_trimmed_struct(sym, struct_S)
    struct_V = get_trimmed_struct(sym, struct_V)

    bl_U = get_blocks(sym, struct_U)
    bl_S = get_blocks(sym, struct_S)
    bl_V = get_blocks(sym, struct_V)

    inds = argsort_t(bl_U.t[:, 1, :])
    ind_a = find_matching_indices(bl_a.t[:, 0, :], bl_U.t[:, 0, :], both=False)
    ind_a = ind_a[inds]  # in case some blocks are eliminated by zero dimension in minD

    meta_dt = np.dtype([
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (2,)),
        ('slU', np.int64, (2,)),
        ('DU',  np.int64, (2,)),
        ('slS', np.int64, (2,)),
        ('slV', np.int64, (2,)),
        ('DV',  np.int64, (2,))])
    meta = np.hstack([bl_a.slc[ind_a], bl_a.D[ind_a], bl_U.slc[inds], bl_U.D[inds], bl_S.slc, bl_V.slc, bl_V.D]).astype(np.int64, copy=False)
    meta = meta.view(meta_dt).reshape(-1)
    meta = convert_to_tuples_and_slices(meta)
    sizes = (bl_U.size, bl_S.size, bl_V.size)
    return meta, sizes, struct_U, struct_S, struct_V


def eig(a, axes=(0, 1), sU=1, nU=True, compute_uv=True,
        Uaxis=-1, Vaxis=0, policy='fullrank', which='LM', **kwargs) -> tuple['Tensor', 'Tensor', 'Tensor'] | 'Tensor':
    r"""
    Split a tensor into :math:`a = U S V` using an exact eigenvalue decomposition (ED),
    where the columns of `U` and the rows of `V` satisfy biorthogonality, i.e. `V @ U = I`,
    and `S` is a diagonal matrix. Unlike in the symmetric/Hermitian case, `U` and `V`
    are not necessarily related and need not form orthonormal bases.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform ED, as well as
        their final order.

    sU: int
        Signature of the new leg in `U`; equal to 1 or -1. The default is 1.
        `V` is going to have the opposite signature on the connecting leg.

    nU: bool
        Whether or not to attach the charge of ``a`` to `U`.
        If ``False``, it is attached to `V`. The default is ``True``.

    compute_uv: bool
        If ``True``, compute and return `U`, `S`, `V`.  If ``False``, compute and return only `S`.
        The default is ``True``.

    Uaxis, Vaxis: int
        Specify which leg of `U` and `V` tensors are connecting with `S`. By default,
        it is the last leg of `U` and the first of `V`, in which case ``a = U @ S @ V``.

    policy: str
        ``"fullrank"`` or ``"lowrank"`` are allowed. Use standard ED for ``"fullrank"``.
        For ``"lowrank"``, uses dominant ED methods and requires providing ``D_block`` in ``kwargs``.
        This employs ``scipy.sparse.linalg.eigs`` for numpy backend.
        kwargs will be passed to those functions for non-default settings.

    which: str
        One of [``'SR'``, ``'LR'`, ``'SM'``, ``'LM'``] specifying how to order S:
        ``'LM'`` : (default) sort by absolute value, largest first,
        ``'SM'`` : sort by absolute value, smallest first,
        ``'SR'`` : sort by real part, smallest first,
        ``'LR'`` : sort by real part, largest first.

    Returns
    -------
    `U`, `S`, `V` (when ``compute_uv=True``) or `S` (when ``compute_uv=False``)
    """
    sym = a.config.sym
    _test_axes_all(a, axes)
    out_ml, out_mr = _clear_axes(*axes)
    #
    # unpack meta-fusion and apply transpose
    out_hl, out_hr = _unpack_axes(a.mfs, out_ml, out_mr)
    out_hl = tuple(a.trans[ax] for ax in out_hl)
    out_hr = tuple(a.trans[ax] for ax in out_hr)
    #
    data, struct_am, hfsm = _fuse_blocks(a.config, a._data, a.struct, (out_hl, out_hr))
    #
    if hfsm[0] != hfsm[1].conj() or struct_am.legs[0] != struct_am.legs[1].conj():
        raise YastnError("Legs of effective square blocks do not match.")

    k_block = None
    meta, sizes, struct_Um, struct_S, struct_Vm = _meta_svd(sym, struct_am, sU, nU, k_block)

    if compute_uv and policy == 'fullrank':
        Udata, Sdata, Vdata = a.config.backend.eig(data, meta, sizes, which=which, diagnostics=kwargs.get('diagnostics', None))
    elif not compute_uv and policy == 'fullrank':
        Sdata = a.config.backend.eigvals(data, meta, sizes[1], which=which)
    else:
        raise YastnError('eig() policy should in (``fullrank`). compute_uv == False only works with `fullrank`')

    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=struct_S, data=Sdata, mfs=Smfs, hfs=Shfs, trans=None)

    if not compute_uv:
        return S

    hfsUm = (hfsm[0], _Fusion(s=(sU,)))
    Udata, struct_U = _unfuse_blocks(a.config, Udata, struct_Um, (0,), hfsUm)
    Umfs = tuple(a.mfs[ii] for ii in out_ml) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in out_hl) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=struct_U, data=Udata, mfs=Umfs, hfs=Uhfs, trans=None)

    hfsVm = (_Fusion(s=(-sU,)), hfsm[1])
    Vdata, struct_V = _unfuse_blocks(a.config, Vdata, struct_Vm, (1,), hfsVm)
    Vmfs = ((1,),) + tuple(a.mfs[ii] for ii in out_mr)
    Vhfs = (_Fusion(s=(-sU,)),) + tuple(a.hfs[ii] for ii in out_hr)
    V = a._replace(struct=struct_V, data=Vdata, mfs=Vmfs, hfs=Vhfs, trans=None)

    U = U.moveaxis(source=-1, destination=Uaxis)
    V = V.moveaxis(source=0, destination=Vaxis)
    return U, S, V


def truncation_mask_multiplets(S, tol=0, D_total=float('inf'),
                               eps_multiplet=1e-13, hermitian=False, **kwargs) -> 'Tensor[bool]':
    """
    Generate a mask tensor from a real positive spectrum ``S``, while preserving
    degenerate multiplets by truncating at the boundary between multiplets.

    !!! This method is deprecated and can be removed at some point; !!!
    Use linalg.truncation_mask() that now include those truncation schemes.

    Parameters
    ----------
    S: yastn.Tensor
        Diagonal tensor with spectrum.

    tol: float
        relative tolerance.

    D_total: int
        maximum number of elements kept in the result.

    eps_multiplet: float
        relative tolerance on multiplet splitting. If relative difference between
        two consecutive elements of ``S`` is larger than ``eps_multiplet``, these
        elements are not considered as part of the same multiplet.

    hermitian: bool = False
        If true, blocks related by hermitian conjugation are truncated equally.
    """
    warn('This method is deprecated; use truncation_mask() instead.', DeprecationWarning, stacklevel=2)
    return truncation_mask(S, which='LR',
                           tol=tol, D_total=D_total,
                           eps_multiplet=eps_multiplet,
                           hermitian=hermitian)


def truncation_mask(S, which='LR',
                    tol=float('-inf'),
                    tol_block=float('-inf'),
                    D_total=float('inf'),
                    D_block=float('inf'),
                    largest_gap=False,
                    eps_multiplet=None,
                    hermitian=False,
                    mask_f=None,
                    **kwargs) -> 'Tensor[bool]':
    """
    Generate a mask tensor from a diagonal tensor ``S``.
    The mask can then be used for truncation.

    Parameters
    ----------
    S: yastn.Tensor
        Diagonal tensor with spectrum.

    which: str
        Which values to keep from [``'LM'``, ``'LR'``, ``'SR'``, ``'SM'``]:
        ``'LR'`` : largest real part (the default),
        ``'LM'`` : largest magnitude,
        ``'SR'`` : smallest real part,
        ``'SM'`` : smallest magnitude.

    tol: float
        Relative tolerance with respect to the largest absolut value element of ``S``.

    tol_block: float
        Relative tolerance per block.

    D_total: int
        Maximum number of elements kept across all blocks.

    D_block: int | dict
        Maximum number of elements kept per block.
        It is also possible to provide a dictionary mapping charges to maximal number of elements in the charge sector.

    largest_gap: bool
        If ``True``, enlarge the truncation range specified by other arguments by shifting
        the cut to the largest gap between to-be-truncated singular values across all blocks.
        It provides a heuristic mechanism to avoid truncating part of a multiplet.
        If ``True``, ``tol_block`` and ``D_block`` are ignored, as ``largest_gap`` is a global condition.
        The default is ``False``.

    eps_multiplet: float
        Relative tolerance on multiplet splitting. If relative difference between
        two consecutive elements of ``S`` is larger than ``eps_multiplet``, these
        elements are not considered as part of the same multiplet.
        Partially truncated multiplets are truncated down.
        The default is None, when this scheme is not used.
        If ``True``, ``tol_block`` and ``D_block`` are ignored, as ``eps_multiplet`` is a global condition.
        Cannot be used together with largest_gap scheme.

    hermitian: bool
        If True, blocks related by hermitian conjugation are truncated equally, truncating down to the intersecting part.
        The default is False.

    mask_f: None | function[yastn.Tensor] -> yastn.Tensor
        It is possible to provide a custom mask function, which provides a mechanism to pass such a function
        to many tensor network algorithms where the function truncation_mask is being called.
        If provided, it overrides the default function, and all other parameters are ignored.
        The default is None.
    """
    if mask_f is not None:
        return mask_f(S)

    if not S.isdiag:
        raise YastnError("truncation_mask() requires S to be diagonal.")

    verbosity = kwargs.get('verbosity', 0)
    if verbosity > 2:
        fname = sys._getframe().f_code.co_name
        logger.info(f"{fname} {tol=} {tol_block=} {D_total=} {D_block=}")
        logger.info(f"{fname} {largest_gap=} {eps_multiplet=} {hermitian=}")

    if which in ["SR", "SM"] and (tol != -float('inf') or tol_block != -float('inf')):
        raise YastnError("Truncation by tolerance with which='SR' or 'SM' is not supported."
            + "Set tol and tol_block to -inf or use mask_f for custom truncation mask if needed.")

    if (largest_gap or eps_multiplet) and  (tol_block != float('-inf') or D_block != float('inf')):
        raise YastnError("Truncation by block cannot be used when multiplet-related schmes are invoked."
            + "Set D_block to the default float('inf') and tol_block to the default float('-inf').")

    if (largest_gap or eps_multiplet) and which not in ['LM', 'LR']:
        raise YastnError("Only which = 'LM' or 'LR' are supported when multiplet-related schmes are invoked.")

    if largest_gap and eps_multiplet:
        raise YastnError("Truncation multiplets cannot perform both schemes largest_gap and eps_multiplets simultaneously."
                         + "Switch one off by providing the default value.")

    backend = S.config.backend
    nsym = S.config.sym.NSYM
    f_which = {'LR': backend.real, 'LM': backend.absolute}
    ff = f_which.get(which, None)

    # makes a copy for partial truncations; also detaches from autograd computation graph
    Smask = abs(S.detach()) > float('-inf')  # all True

    if tol_block != float('-inf') or D_block != float('inf'):
        tol_null = float('inf') if isinstance(tol_block, dict) else tol_block
        D_null = 0 if isinstance(D_block, dict) else D_block

        start = 0
        for tt, DD in zip(S.struct.legs[0].t, S.struct.legs[0].D):
            finish = start + DD
            slc = slice(start, finish)
            D_bl = D_block[tt] if (isinstance(D_block, dict) and tt in D_block) else D_null
            if which in ['LR', 'LM']:
                tol_rel = tol_block[tt] if (isinstance(tol_block, dict) and tt in tol_block) else tol_null
                above_tol = ff(S.data[slc]) > tol_rel * backend.max_abs(S.data[slc])
                D_tol = backend.sum_elements(above_tol).item()
                D_bl = min(D_bl, D_tol)

            if 0 < D_bl < DD:  # block truncation
                inds = backend.argsort_which(S.data[slc], which)
                Smask._data[slc][inds[D_bl:]] = False
            elif D_bl == 0:
                Smask._data[slc] = False
            start = finish
    #
    D_total = min(D_total, len(S.data))
    if which in ['LR', 'LM']:
        above_tol = ff(S.data) > tol * backend.max_abs(S.data)
        D_total = min(D_total, backend.sum_elements(above_tol).item())
    #
    inds = backend.argsort_which(S.data, which)
    #
    if largest_gap and D_total < len(S.data):
        s = ff(S._data[inds[D_total - 1:]])
        gaps = abs(s[:-1] - s[1:]) * (s[0] * s[:-1] > 0)  # (s[0] * ...) does not allow sign change
        D_total += backend.argmax(gaps).item()
    #
    if eps_multiplet is not None and 0 < D_total < len(S.data):
        s = ff(S._data[inds[:D_total + 1]])
        maxgap = backend.maximum(abs(s[:-1]), abs(s[1:])) + 1.0e-16
        normalized_gaps = abs(s[:-1] - s[1:]) * (s[0] * s[:-1] > 0) / maxgap  # (s[0] * ...) does not allow sign change
        relevant_gaps = (normalized_gaps > eps_multiplet) * 1  # * 1 to change dtype
        D_total -= backend.argmax(backend.flip(relevant_gaps)).item() # + (D_total == len(S.data) and any(relevant_gaps)) # if D_total == len(S.data)
    #
    Smask._data[inds[D_total:]] = False
    #
    # check blocks related by Hermitian symmetry and truncate to equal length
    if hermitian:
        considered_t = []
        bl = get_blocks(S.config.sym, Smask.struct)
        for tt, DD, sl in zip(bl.t, bl.D, bl.slc):
            tt = tuple(tt[0].tolist())
            tc = S.config.sym.conj_charge(tt)
            #
            if tt == tc or tt in considered_t:
                continue
            #
            slc_t = slice(*sl)
            try:
                itc = find_index(bl.t, np.array(tc + tc, dtype=np.int64), sorted=True)
            except ValueError:  # conjugated sector not in S
                Smask.data[slc_t] = False
                continue
            slc_tc = slice(*bl.slc[itc])
            #
            considered_t.append(tt)
            considered_t.append(tc)
            lt, ltc = DD[0], bl.D[itc, 0]
            common_size = min(lt, ltc)
            inds_t = backend.argsort_which(S.data[slc_t], which)
            inds_tc = backend.argsort_which(S.data[slc_tc], which)
            St = Smask.data[slc_t]
            Stc = Smask.data[slc_tc]
            #
            # if related blocks do not have equal length
            if common_size < lt:
                St[inds_t[common_size:]] = False
            if common_size < ltc:
                Stc[inds_tc[common_size:]] = False
            #
            St[inds_t[:common_size]] = Stc[inds_tc[:common_size]] = St[inds_t[:common_size]] & Stc[inds_tc[:common_size]]

    return Smask


def qr(a, axes=(0, 1), sQ=1, Qaxis=-1, Raxis=0) -> tuple['Tensor', 'Tensor']:
    r"""
    Split a tensor using a reduced QR decomposition such that :math:`a = Q R`,
    with :math:`Q Q^\dagger = I`. The charge of `R` is zero, and the charge of ``a`` is carried by `Q`.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform QR, as well as their final order.

    sQ: int
        signature of connecting leg in `Q`; equal 1 or -1. The default is 1.
        `R` is going to have opposite signature on connecting leg.

    Qaxis, Raxis: int
        specify which leg of `Q` and `R` tensors are connecting to the other tensor.
        By default, it is the last leg of `Q` and the first leg of `R`.

    Returns
    -------
    `Q`, `R`
    """
    sym = a.config.sym
    _test_axes_all(a, axes)
    out_ml, out_mr = _clear_axes(*axes)
    #
    # unpack meta-fusion and apply transpose
    out_hl, out_hr = _unpack_axes(a.mfs, out_ml, out_mr)
    out_hl = tuple(a.trans[ax] for ax in out_hl)
    out_hr = tuple(a.trans[ax] for ax in out_hr)

    data, struct_am, hfsm = _fuse_blocks(a.config, a._data, a.struct, (out_hl, out_hr))

    meta, sizes, struct_Qm, struct_Rm = _meta_qr(a.config.sym, struct_am, sQ)
    Qdata, Rdata = a.config.backend.qr(data, meta, sizes)

    hfsQm = (hfsm[0], _Fusion(s=(sQ,)))
    Qdata, struct_Q = _unfuse_blocks(a.config, Qdata, struct_Qm, (0,), hfsQm)
    Qmfs = tuple(a.mfs[ii] for ii in out_ml) + ((1,),)
    Qhfs = tuple(a.hfs[ii] for ii in out_hl) + (_Fusion(s=(sQ,)),)
    Q = a._replace(struct=struct_Q, data=Qdata, mfs=Qmfs, hfs=Qhfs, trans=None)

    hfsRm = (_Fusion(s=(-sQ,)), hfsm[1])
    Rdata, struct_R = _unfuse_blocks(a.config, Rdata, struct_Rm, (1,), hfsRm)
    Rmfs = ((1,),) + tuple(a.mfs[ii] for ii in out_mr)
    Rhfs = (_Fusion(s=(-sQ,)),) + tuple(a.hfs[ii] for ii in out_hr)
    R = a._replace(struct=struct_R, data=Rdata, mfs=Rmfs, hfs=Rhfs, trans=None)

    Q = Q.moveaxis(source=-1, destination=Qaxis)
    R = R.moveaxis(source=0, destination=Raxis)
    return Q, R


def _meta_qr(sym, struct, sQ):
    """
    Return metadata and structures for QR.

    `Q` has signature ``(legs[0].s, sQ)`` and `R` has signature ``(-sQ, legs[1].s)``.
    """
    bl_a = get_blocks(sym, struct)
    minD = {tuple(tt): min(DD) for tt, DD in zip(bl_a.t[:, 1, :].tolist(), bl_a.D)}
    ts = tuple(sorted(minD.keys()))
    Ds = tuple(minD[tt] for tt in ts)
    legQ = LegBasic(s=struct.legs[1].s, t=ts, D=Ds)
    if sQ != legQ.s:
        legQ = legQ.conj_charges(sym)

    struct_Q = _struct(legs=(struct.legs[0], legQ), n=struct.n, isdiag=False)
    struct_R = _struct(legs=(legQ.conj(), struct.legs[1]), n=sym.zero(), isdiag=False)
    bl_Q = get_blocks(sym, struct_Q)
    bl_R = get_blocks(sym, struct_R)
    inds = argsort_t(bl_Q.t[:, 1, :])

    meta_dt = np.dtype([
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (2,)),
        ('slQ', np.int64, (2,)),
        ('DQ',  np.int64, (2,)),
        ('slR', np.int64, (2,)),
        ('DR',  np.int64, (2,))])
    meta = np.hstack([bl_a.slc[inds], bl_a.D[inds], bl_Q.slc[inds], bl_Q.D[inds], bl_R.slc, bl_R.D]).astype(np.int64, copy=False)
    meta = meta.view(meta_dt).reshape(-1)
    meta = convert_to_tuples_and_slices(meta)
    sizes = (bl_Q.size, bl_R.size)
    return meta, sizes, struct_Q, struct_R


def eigh(a, axes, sU=1, Uaxis=-1, which='LR', policy='fullrank', **kwargs) -> tuple['Tensor', 'Tensor']:
    r"""
    Split a symmetric tensor using an exact eigenvalue decomposition, :math:`a = U S U^\dagger`.

    The tensor is expected to be symmetric (Hermitian) with total charge `0`.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform eigh, as well as their final order.

    sU: int
        signature of connecting leg in `U` equal 1 or -1. The default is 1.

    Uaxis: int
        specify which leg of `U` is the new connecting leg. By default, it is the last leg.

    which: str
        One of [``'SR'``, ``'LR'``, ``'SM'``, ``'LM'``] specifying how to order S:
        ``'LM'`` : sort by absolute value, largest first,
        ``'SM'`` : sort by absolute value, smallest first,
        ``'SR'`` : (default) sort by real part, smallest first,
        ``'LR'`` : sort by real part, largest first.

    policy: str
        ``'fullrank'`` : (default) use standard full eigenvalue decomposition.
        ``'block_lanczos'`` : use partial eigenvalue decomposition via ``scipy.sparse.linalg.eigsh``.
        Requires ``D_block`` or ``k_block`` in ``kwargs`` specifying the number of eigenvalues per block.

    k_block: None (default) | int | dict
        When ``policy='block_lanczos'``, number of eigenvalues to compute in each block.
        If ``D_block`` is provided, it is used instead to determine number of eigenvalues to compute.

    Returns
    -------
    `S`, `U`
    """
    sym = a.config.sym
    POLICIES = ['fullrank', 'block_lanczos',]
    verbosity = kwargs.get('verbosity', 0)

    # 1. validation
    if policy not in POLICIES:
       raise YastnError(f"Invalid EIGH solver/policy {policy}. Choose one of {POLICIES}.")

    # 1.1 non-default D_block provides defaults for k_block
    if 'D_block' in kwargs and kwargs['D_block'] not in [None, float('inf')] and \
        ('k_block' not in kwargs or kwargs['k_block'] in [None,]):
        kwargs['k_block'] = kwargs['D_block']

    _test_axes_all(a, axes)
    out_ml, out_mr = _clear_axes(*axes)
    #
    # unpack meta-fusion and apply transpose
    out_hl, out_hr = _unpack_axes(a.mfs, out_ml, out_mr)
    out_hl = tuple(a.trans[ax] for ax in out_hl)
    out_hr = tuple(a.trans[ax] for ax in out_hr)
    #
    if not a.n == sym.zero():
        raise YastnError('eigh requires tensor charge to be zero.')
    #
    # 2. merge to block, square matrix
    data, struct_am, hfsm = _fuse_blocks(a.config, a._data, a.struct, (out_hl, out_hr))
    #
    # 3.1 Set minimal number of eigenpairs to solve for in each block.
    #     Used by block-wise sparse solvers and ignored by 'fullrank' policy.
    k_block = None
    if policy in ['block_lanczos',]:
        if 'k_block' not in kwargs:
            raise YastnError(policy + " policy in eighs requires passing argument D_block.")
        k_block = kwargs['k_block']
        # A partial solver must not receive an exactly-zero effective block.
        # Keep this after fusion: fusion can introduce structural zero blocks.
        am = remove_zero_blocks(a._replace(struct=struct_am, data=data))
        data, struct_am = am._data, am.struct

    if verbosity > 2:
        fname = sys._getframe().f_code.co_name
        logger.info(f"{fname} {policy} struct {struct_am}")
        logger.info(f"{fname} D_block {kwargs.get('D_block', 'NA')}")
        logger.info(f"{fname} k_block {k_block}")

    if hfsm[0] != hfsm[1].conj() or struct_am.legs[0] != struct_am.legs[1].conj():
        raise YastnError("Tensor likely is not hermitian. Legs of effective square blocks do not match.")

    meta, sizes, struct_Um, struct_S = _meta_eigh(sym, struct_am, sU, k_block)

    if policy == 'fullrank':
        Sdata, Udata = a.config.backend.eigh(data, meta, sizes)
    elif policy == 'block_lanczos':
        Sdata, Udata = a.config.backend.eigh_lowrank(data, meta, sizes, thresh=None, which=which, **kwargs)
        # _real_dtype = {'complex128': 'float64', 'complex64': 'float32'}.get(a.yastn_dtype, a.yastn_dtype)
        # Sdata = a.config.backend.to_tensor(Sdata_np, dtype=_real_dtype, device=a.device)
        # Udata = a.config.backend.to_tensor(Udata_np, dtype=a.yastn_dtype, device=a.device)
    else:
        raise YastnError("eigh() policy should be 'fullrank' or 'block_lanczos'.")

    hfsUm = (hfsm[0], _Fusion(s=(sU,)))
    Udata, struct_U = _unfuse_blocks(a.config, Udata, struct_Um, (0,), hfsUm)
    Umfs = tuple(a.mfs[ii] for ii in out_ml) + ((1,),)
    Uhfs = tuple(a.hfs[ii] for ii in out_hl) + (_Fusion(s=(sU,)),)
    U = a._replace(struct=struct_U, data=Udata, mfs=Umfs, hfs=Uhfs, trans=None)

    Smfs = ((1,), (1,))
    Shfs = (_Fusion(s=(-sU,)), _Fusion(s=(sU,)))
    S = a._replace(struct=struct_S, data=Sdata, mfs=Smfs, hfs=Shfs, trans=None)

    # sort in case of non-default order
    if policy in ['fullrank'] and which != 'SR':
        nsym = sym.NSYM
        blocks_U = U.get_blocks_charge()
        for b in S.get_blocks_charge():
            arg_b = a.config.backend.argsort_which(S[b], which)
            S[b] = S[b][arg_b]
            slice_U = tuple([slice(None),] * (U.ndim_n - 1) + [arg_b,])
            for b_U in blocks_U: # suboptimal since U may have more blocks
                if b_U[-nsym:] == b[:nsym]:
                    # blocks_U.remove(b_U)
                    U[b_U] = U[b_U][slice_U]

    U = U.moveaxis(source=-1, destination=Uaxis)
    return S, U


def _meta_eigh(sym, struct, sU, k_block):
    """
    Return metadata and structures for an eigendecomposition.

    `U` has signature ``(legs[0].s, sU)`` and `S` has signature ``(-sU, sU)``.
    """
    bl_a = get_blocks(sym, struct)

    n0 = sym.zero()
    minD = {tuple(tt): min(DD) for tt, DD in zip(bl_a.t[:, 1, :].tolist(), bl_a.D)}
    if k_block is not None:
        if isinstance(k_block, dict):
            sector_minD = min(k_block.values())  # TODO: control default for sectors not present in k_block
            minD = {t: min(k_block.get(t, sector_minD), d) for t, d in minD.items()}
        else:
            minD = {t: min(k_block, d) for t, d in minD.items()}

    ts = tuple(sorted(t for t, d in minD.items() if d > 0))
    Ds = tuple(minD[tt] for tt in ts)
    legU = LegBasic(s=struct.legs[1].s, t=ts, D=Ds)
    if sU != legU.s:
        legU = legU.conj_charges(sym)
    #
    struct_U = _struct(legs=(struct.legs[0], legU), n=n0, isdiag=False)
    struct_S = _struct(legs=(legU.conj(), legU), n=n0, isdiag=True)

    struct_U = get_trimmed_struct(sym, struct_U)
    struct_S = get_trimmed_struct(sym, struct_S)

    bl_U = get_blocks(sym, struct_U)
    bl_S = get_blocks(sym, struct_S)
    inds = argsort_t(bl_U.t[:, 1, :])

    inds_a = find_matching_indices(bl_a.t[:, 0, :], bl_U.t[:, 0, :], both=False)
    inds_a = inds_a[inds]  # in case some blocks in a are eliminated by zero dimenion in minD

    meta_dt = np.dtype([
        ('slo', np.int64, (2,)),
        ('Do',  np.int64, (2,)),
        ('slU', np.int64, (2,)),
        ('DU',  np.int64, (2,)),
        ('slS', np.int64, (2,))])
    meta = np.hstack([bl_a.slc[inds_a], bl_a.D[inds_a], bl_U.slc[inds], bl_U.D[inds], bl_S.slc])
    meta = meta.view(meta_dt).reshape(-1)
    meta = convert_to_tuples_and_slices(meta)
    sizes = (bl_S.size, bl_U.size)
    return meta, sizes, struct_U, struct_S


def eigh_with_truncation(a, axes, sU=1, Uaxis=-1, which='LR', policy='fullrank',
                         tol=0, tol_block=0, D_block=float('inf'), D_total=float('inf'),
                         largest_gap=False, mask_f=None, **kwargs) -> tuple['Tensor', 'Tensor']:
    r"""
    Split a symmetric tensor using an exact eigenvalue decomposition, :math:`a = U S U^\dagger`.
    Optionally truncate the resulting decomposition.

    The tensor is expected to be symmetric (Hermitian) with total charge `0`.
    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).
    Truncate based on tolerance only if some eigenvalues are positive -- then all negative ones are discarded.

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform eigh, as well as their final order.

    sU: int
        signature of connecting leg in `U` equal 1 or -1. The default is 1.

    Uaxis: int
        specify which leg of `U` is the new connecting leg. By default, it is the last leg.

    which: str
        One of [``'SR'``, ``'LR'`, ``'SM'``, ``'LM'``] specifying how to order S:
        ``'LM'`` : sort by absolute value, largest first,
        ``'SM'`` : sort by absolute value, smallest first,
        ``'SR'`` : (default) sort by real part, smallest first,
        ``'LR'`` : sort by real part, largest first.

    policy: str
        ``"fullrank"`` : Use standard full ED for ``"fullrank"`` and then truncate.
        kwargs will be passed to those functions for non-default settings.

    tol: float
        relative tolerance of eigen-values below which to truncate across all blocks.

    tol_block: float
        relative tolerance of eigen-values below which to truncate within individual blocks.

    D_block: int
        largest number of eigen-values to keep in a single block.

    D_total: int
        largest total number of eigen-values to keep.

    mask_f: function[yastn.Tensor] -> yastn.Tensor
        custom truncation-mask function.
        If provided, it overrides all other truncation-related arguments.

    Returns
    -------
    `S`, `U`
    """
    S, U = eigh(a, axes=axes, sU=sU, Uaxis=Uaxis, which=which, policy=policy)

    Smask = truncation_mask(S, which=which, tol=tol, tol_block=tol_block,
                        D_block=D_block, D_total=D_total,
                        largest_gap=largest_gap, mask_f=mask_f)
    S, U = Smask.apply_mask(S, U, axes=(0, Uaxis))
    return S, U


def entropy(a, alpha=1, tol=1e-12) -> Number:
    r"""
    Compute the entropy from probabilities encoded in the diagonal tensor ``a``.

    The sum of ``a`` is normalized to ``1``, but its correctness is not checked otherwise.
    Base-2 logarithms are used. For an empty or zero tensor, ``0`` is returned.

    Parameters
    ----------
    alpha: float
        Order of Renyi entropy.
        ``alpha=1`` (the default) is von Neumann entropy: :math:`-{\rm Tr}(a \cdot {\rm log2}(a))`
        otherwise: :math:`\frac{1}{1-alpha} {\rm log2}({\rm Tr}(a^{alpha}))`

    tol: float
        Discard all probabilities smaller than ``tol`` during calculation.
    """
    if not a.isdiag:
        raise YastnError("yastn.linalg.entropy requires diagonal tensor.")
    if not alpha > 0:
        raise YastnError("yastn.linalg.entropy requires positive order alpha.")
    return a.config.backend.entropy(a._data, alpha=alpha, tol=tol)
