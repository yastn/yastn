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
"""
Methods creating new YASTN tensors from scratch and
importing tensors from different formats such as 1D + metadata or dictionary representation
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter

import numpy as np

from .tensor import Tensor, YastnError, ncon
from .tensor._auxiliary import _clear_axes, _unpack_legs, LegBasic, get_blocks, find_indices, update_old_struct
from .tensor._legs import Leg, LegMeta, legs_union, _legs_mask_needed
from .tensor._merging import _embed_tensor, _combine_hfs_sum
from .tensor._tests import _test_can_be_combined

__all__ = ['rand', 'rand_like', 'randR', 'randC', 'zeros', 'ones', 'eye', 'block', 'Method']


def _fill(config=None, legs=(), n=None, isdiag=False, val='rand', **kwargs):
    if 's' in kwargs or 't' in kwargs or 'D' in kwargs:
        s = kwargs.pop('s') if 's' in kwargs else ()
        t = kwargs.pop('t') if 't' in kwargs else ()
        D = kwargs.pop('D') if 'D' in kwargs else ()
        mfs, hfs = None, None
    else:  # use legs for initialization
        if isinstance(legs, (Leg, LegMeta)):
            legs = (legs,)
        if isdiag and len(legs) == 1:
            legs = (legs[0], legs[0].conj())
        ulegs, mfs = _unpack_legs(legs)
        s = tuple(leg.s for leg in ulegs)
        t = tuple(leg.t for leg in ulegs)
        D = tuple(leg.D for leg in ulegs)
        hfs = tuple(leg.hf for leg in ulegs)

        if any(config.sym.SYM_ID != leg.sym.SYM_ID for leg in ulegs):
            raise YastnError('Different symmetry of initialized tensor and some of the legs.')
        if isdiag and any(mf != (1,) for mf in mfs):
            raise YastnError('Diagonal tensor cannot be initialized with fused legs.')
        if isdiag and any(hf.tree != (1,) for hf in hfs):
            raise YastnError('Diagonal tensor cannot be initialized with fused legs.')

    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, mfs=mfs, hfs=hfs, **kwargs)
    a._fill_tensor(t=t, D=D, val=val)
    return a


def rand(config=None, distribution=(-1, 1), legs=(), n=None, isdiag=False, **kwargs) -> Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with random numbers.

    Draws from a uniform distribution in the range specified by ``lim``,
    or ``lim`` in real and imaginary part, depending on desired ``dtype``.
    ``distribution='normal'`` invokes normal distribution with zero mean and standard deviation one.
    The default is ``distribution=(-1, 1)``.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    distribution: tuple[int, int] | str
        The range from which the numbers are drawn from a uniform distribution.
        For ``distribution='normal'`` normal distribution is used.
    legs: Sequence[yastn.Leg]
        Specify legs of the tensor passing a list of :class:`yastn.Leg`.
    n: int | Sequence[int]
        Total charge of the tensor.
    isdiag: bool
        whether or not to make tensor diagonal.
    dtype: str
        Desired datatype, overrides :code:`default_dtype` specified in configuration.
    device: str
        Device on which the tensor should be initialized. Overrides attribute :code:`default_device`
        specified in configuration.
    s: Optional[Sequence[int]]
        (alternative) Tensor signature. Also determines the number of legs. The default is s=().
    t: Optional[Sequence[Sequence[int | Sequence[int]]]]
        (alternative) List of charges for each leg. The default is t=().
    D: Optional[Sequence[Sequence[int]]]
        (alternative) List of corresponding bond dimensions. The default is D=().

    Note
    ----
    If any of :code:`s`, :code:`t`, or :code:`D` are specified,
    :code:`legs` are overridden and only :code:`t`, :code:`D`, and :code:`s` are used.
    """
    val = distribution if distribution == 'normal' else ('rand', distribution)
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val=val, **kwargs)


def rand_like(T: Tensor, distribution=(-1, 1), **kwargs) -> Tensor:
    r"""
    Initialize tensor with same structure as ``T`` filled with random numbers.

    Draws from a uniform distribution in the range specified by ``lim``,
    or ``lim`` in real and imaginary part, depending on desired ``dtype``.
    ``distribution='normal'`` invokes normal distribution. The default is ``distribution=(-1, 1)``.
    """
    return rand(config=T.config, legs=T.get_legs(), n=T.n, isdiag=T.isdiag, distribution=distribution, **kwargs)


def randR(config=None, distribution=(-1, 1), legs=(), n=None, isdiag=False, **kwargs) -> Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with real random numbers,
    see :meth:`yastn.rand`.
    """
    if 'dtype' not in kwargs or kwargs['dtype'] == 'complex128':
        kwargs['dtype'] = 'float64'
    if kwargs['dtype'] == 'complex64':
        kwargs['dtype'] = 'float32'
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', distribution=distribution, **kwargs)


def randC(config=None, distribution=(-1, 1), legs=(), n=None, isdiag=False, **kwargs) -> Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with complex random numbers,
    see :meth:`yastn.rand`.
    """
    if 'dtype' not in kwargs or kwargs['dtype'] == 'float64':
        kwargs['dtype'] = 'complex128'
    if kwargs['dtype'] == 'float32':
        kwargs['dtype'] = 'complex64'
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', distribution=distribution, **kwargs)


def zeros(config=None, legs=(), n=None, isdiag=False, **kwargs) -> Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with zeros.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    legs: Sequence[yastn.Leg]
        Specify legs of the tensor passing a list of :class:`yastn.Leg`.
    n: int | Sequence[int]
        total charge of the tensor.
    isdiag: bool
        whether or not to make tensor diagonal
    dtype: str
        Desired datatype, overrides :code:`default_dtype` specified in configuration.
    device: str
        Device on which the tensor should be initialized. Overrides attribute :code:`default_device`
        specified in configuration.
    s: Optional[Sequence[int]]
        (alternative) Tensor signature. Also determines the number of legs. The default is s=().
    t: Optional[Sequence[Sequence[int | Sequence[int]]]]
        (alternative) List of charges for each leg. The default is t=().
    D: Optional[Sequence[Sequence[int]]]
        (alternative) List of corresponding bond dimensions. The default is D=().

    Note
    ----
    If any of :code:`s`, :code:`t`, or :code:`D` are specified,
    :code:`legs` are overridden and only :code:`t`, :code:`D`, and :code:`s` are used.
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='zeros', **kwargs)


def ones(config=None, legs=(), n=None, isdiag=False, **kwargs) -> Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with ones.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    legs: Sequence[yastn.Leg]
        Specify legs of the tensor passing a list of :class:`yastn.Leg`.
    n: int | Sequence[int]
        total charge of the tensor.
    isdiag: bool
        whether or not to make tensor diagonal.
    dtype: str
        Desired datatype, overrides :code:`default_dtype` specified in configuration.
    device: str
        Device on which the tensor should be initialized. Overrides attribute :code:`default_device`
        specified in configuration.
    s: Optional[Sequence[int]]
        (alternative) Tensor signature. Also determines the number of legs. The default is s=().
    t: Optional[Sequence[Sequence[int | Sequence[int]]]]
        (alternative) List of charges for each leg. The default is t=().
    D: Optional[Sequence[Sequence[int]]]
        (alternative) List of corresponding bond dimensions. The default is D=().

    Note
    ----
    If any of :code:`s`, :code:`t`, or :code:`D` are specified,
    :code:`legs` are overridden and only :code:`t`, :code:`D`, and :code:`s` are used.
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='ones', **kwargs)


def eye(config=None, legs=(), isdiag=True, **kwargs) -> Tensor:
    r"""
    Initialize diagonal tensor of identity matrix.
    In presence of symmetries, such matrix is block-diagonal with all allowed blocks filled with identity matrices.

    .. note::
        Currently supports either one or two legs as input. In case of a single leg,
        an identity matrix with Leg and its conjugate :meth:`yastn.Leg.conj()` is returned.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    legs: Sequence[yastn.Leg]
        Specify legs of the tensor passing a list of :class:`yastn.Leg`.
    isdiag: bool
        Specify by bool whether to return explicitly diagonal tensor.
        If :code:`True`, the signatures of the legs have to be opposite, and fused legs are not supported.
        If :code:`False`, it supports having fused legs and any combination of signatures.
    device: str
        Device on which the tensor should be initialized. Overrides attribute :code:`default_device`
        specified in configuration.
    s: Optional[Sequence[int]]
        (alternative) Tensor signature; should be (1, -1) or (-1, 1). The default is s=(1, -1).
    t: Optional[Sequence[Sequence[int | Sequence[int]]]]
        (alternative) List of charges for each leg. The default is t=().
    D: Optional[list]
        (alternative) List of corresponding bond dimensions. The default is D=().

    Note
    ----
    If any of :code:`s`, :code:`t`, or :code:`D` are specified,
    :code:`legs` are overridden and only :code:`t`, :code:`D`, and :code:`s` are used.
    """
    if isdiag:
        return _fill(config=config, legs=legs, isdiag=True, val='ones', **kwargs)
    if isinstance(legs, (Leg, LegMeta)):
        legs = (legs,)
    if len(legs) == 1:
        legs = (legs[0], legs[0].conj())
    legs = legs[:2]  # in case more than 2 legs are provided
    if any(isinstance(leg, LegMeta) for leg in legs):
        raise YastnError("eye() does not support 'meta'-fused legs")

    if legs[0].is_fused():
        ulegs0 = legs[0].unfuse_leg()
        ulegs1 = legs[1].unfuse_leg()
        tens = [eye(config=config, legs=(l0, l1), isdiag=False, **kwargs)
                    for l0, l1 in zip(ulegs0, ulegs1)]
        lt = len(tens)
        inds = [[-2 * i for i in range(lt)],
                [-2 * i - 1 for i in range(lt)]]
        tmp = ncon(tens, inds)
        axes = (tuple(range(lt)), tuple(range(lt, 2 * lt)))
        return tmp.fuse_legs(axes=axes)
    else:
        tmp = _fill(config=config, legs=legs, val='zeros', **kwargs)
        for t, D in zip(tmp.struct.t, tmp.struct.D):
            blk = tmp[t]
            for i in range(min(D)):
                blk[i, i] = 1
        return tmp


# def load_from_hdf5(config, file, path) -> Tensor:
#     """
#     Create tensor from hdf5 file.

#     Parameters
#     ----------
#     config: module | _config(NamedTuple)
#         :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
#     file:
#         pointer to opened HDF5 file.
#     path:
#         path inside the file which contains the state.
#     """
#     g = file.get(path)
#     c_isdiag = bool(g.get('isdiag')[:][0])
#     c_n = tuple(g.get('n')[:].tolist())
#     c_s = tuple(g.get('s')[:].tolist())
#     c_t = tuple(tuple(x) for x in g.get('ts')[:].tolist())
#     c_D = tuple(tuple(x) for x in g.get('Ds')[:].tolist())
#     c_Dp = [x[0] for x in c_D] if c_isdiag else np.prod(c_D, axis=1, dtype=np.int64).tolist()
#     slices = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(c_Dp), c_Dp, c_D))
#     struct = _struct(s=c_s, n=c_n, diag=c_isdiag, t=c_t, D=c_D, size=sum(c_Dp))
#     legs = legs_from_struct(struct)
#     struct = _struct(s=c_s, n=c_n, diag=c_isdiag, t=c_t, D=c_D, size=sum(c_Dp), legs=legs)

#     mfs = literal_eval(tuple(file.get(path+'/mfs').keys())[0])
#     hfs = tuple(_Fusion(*hf) if isinstance(hf, tuple) else _Fusion(**hf) \
#                 for hf in literal_eval(tuple(g.get('hfs').keys())[0]))
#     c = Tensor(config=config, struct=struct, slices=slices, mfs=mfs, hfs=hfs)

#     vmat = g.get('matrix')[:]
#     c._data = c.config.backend.to_tensor(vmat, dtype=vmat.dtype.name, device=c.device)
#     c.is_consistent()
#     return c


def block(tensors, common_legs=None) -> Tensor:
    """
    Assemble new tensor by blocking a group of tensors.

    History of blocking is stored together with history of hard-fusions.
    Subsequent blocking in a few steps and its equivalent single step blocking give the same tensor.
    Applying block on tensors turns all previous meta-fused legs into hard-fused ones.

    Parameters
    ----------
    tensors: dict[Sequence[int], Tensor]
        dictionary of tensors {(x,y,...): tensor at position x,y,.. in the new, blocked super-tensor}.
        Length of tuple should be equal to :code:`tensor.ndim - len(common_legs)`.

    common_legs: Sequence[int]
        Legs that are not blocked.
        This is equivalently to all tensors having the same position
        (not specified explicitly) in the super-tensor on that leg.
    """
    #
    # merge_super_blocks do not perform transpose, so we do it here
    tensors = {k: v.consume_transpose() for k, v in tensors.items()}
    tn0 = next(iter(tensors.values()))  # first tensor; used to initialize new objects and retrieve common values
    sym = tn0.config.sym
    out_s, = ((),) if common_legs is None else _clear_axes(common_legs)
    out_b = tuple(ii for ii in range(tn0.ndim) if ii not in out_s)

    pos = list(_clear_axes(*tensors))
    lind = tn0.ndim - len(out_s)
    if any(len(ind) != lind for ind in pos):
        raise YastnError('Wrong number of coordinates encoded in tensors.keys()')

    posa = np.zeros((len(pos), tn0.ndim), dtype=np.int64)
    posa[:, out_b] = np.array(pos, dtype=np.int64).reshape(len(pos), len(out_b)).tolist()
    posa = tuple(tuple(x) for x in posa)

    # perform hard fusion of meta-fused legs before blocking
    tensors = {pa: a.fuse_meta_to_hard() for pa, a in zip(posa, tensors.values())}
    tn0 = next(iter(tensors.values()))  # first tensor; used to initialize new objects and retrieve common values

    for tn in tensors.values():
        _test_can_be_combined(tn, tn0)
        if tn.s_n != tn0.s_n:
            raise YastnError('Signatures of blocked tensors are inconsistent.')
        if tn.n != tn0.n:
            raise YastnError('Tensor charges of blocked tensors are inconsistent.')
        if tn.isdiag:
            raise YastnError('Block does not support diagonal tensors. Use .diag() first.')

    legs_tn = {pa: a.get_legs() for pa, a in tensors.items()}
    ulegs, legs, hfs, ltDtot, ltDslc, legs_new = [], [], [], [], [], []
    for n in range(tn0.ndim_n):
        legs_n = {}
        for pa, ll in legs_tn.items():
            if pa[n] not in legs_n:
                legs_n[pa[n]] = []
            legs_n[pa[n]].append(ll[n])
        legs.append(legs_n)
        legs_n = {p: legs_union(*plegs) for p, plegs in legs_n.items()}
        ulegs.append(legs_n)
        pn = sorted(legs_n.keys())
        hfs.append(_sum_legs_hfs([legs_n[p] for p in pn]))

        tpD = sorted((t, p, D) for p, leg in legs_n.items() for t, D in zip(leg.t, leg.D))
        ltDtot_leg, ltDslc_leg = {}, {}
        for t, gr in groupby(tpD, key=itemgetter(0)):
            Dlow, tpDslc = 0, {}
            for _, p, D in gr:
                Dhigh = Dlow + D
                tpDslc[p] = (Dlow, Dhigh)
                Dlow = Dhigh
            ltDtot_leg[t] = Dhigh
            ltDslc_leg[t] = tpDslc
        ltDtot.append(ltDtot_leg)
        ltDslc.append(ltDslc_leg)

        ttn = tuple(sorted(ltDtot_leg.keys()))
        DDn = tuple(ltDtot_leg[t] for t in ttn)
        legs_new.append(LegBasic(s=tn0.legs[n].s, t=ttn, D=DDn))

    legs_new = tuple(legs_new)
    bl_new = get_blocks(sym, legs_new, tn0.n, tn0.isdiag)

    for pa in tensors.keys():
        if any(_legs_mask_needed(ulegs[n][pa[n]], leg) for n, leg in enumerate(legs_tn[pa])):
            legs_embed = {n: legs[pa[n]] for n, legs in enumerate(ulegs)}
            tensors[pa] = _embed_tensor(tensors[pa], legs_tn[pa], legs_embed)

    meta = []
    for pa, a in tensors.items():
        bl_a = get_blocks(sym, a.legs, a.n, a.isdiag)
        indc, inda = find_indices(bl_new.t, bl_a.t)
        for sln, Dn, ta, sla, Da in zip(bl_new.slc[indc], bl_new.D[indc], bl_a.t[inda], bl_a.slc[inda], bl_a.D[inda]):
            Dslcs = tuple(tDslc[tuple(ta[n].tolist())][pa[n]] for n, tDslc in enumerate(ltDslc))
            meta.append((sln, Dn, pa, sla, Da, Dslcs))

    data = tn0.config.backend.merge_super_blocks(tensors, meta, bl_new.size)
    struct, slices = update_old_struct(tn0.struct, bl_new)
    out = tn0._replace(struct=struct, slices=slices, data=data, hfs=tuple(hfs))
    return out


def _sum_legs_hfs(legs):
    """ sum hfs based on info in legs"""
    hfs = [leg.hf for leg in legs]
    t_in = [leg.t for leg in legs]
    D_in = [leg.D for leg in legs]
    s_out = legs[0].s
    return _combine_hfs_sum(hfs, t_in, D_in, s_out)


@dataclass
class Method():
    """
    Auxiliary mutable method class.
    It introduces the mechanism to change the method used in :meth:`yastn.tn.mps.dmrg_`, :meth:`yastn.tn.mps.tdvp_`,
    and other generator functions in between consecutive sweeps.
    Updating the value in place will inject the new value back into the generator.
    """
    string: str = ''

    def __eq__(self, string):
        return string == self.string

    def __str__(self):
        return self.string

    def update_(self, string):
        """ Update the method name in place. """
        self.string = str(string)