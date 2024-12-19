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
from ast import literal_eval
from itertools import groupby, accumulate
from operator import itemgetter
import numpy as np
from .tensor import Tensor, YastnError
from .tensor._auxliary import _struct, _config, _slc, _clear_axes, _unpack_legs
from .tensor._merging import _Fusion, _embed_tensor, _sum_hfs
from .tensor._legs import Leg, leg_union, _leg_fusions_need_mask
from .tensor._tests import _test_can_be_combined
from .backend import backend_np
from .sym import sym_none, sym_U1, sym_Z2, sym_Z3, sym_U1xU1, sym_U1xU1xZ2


_syms = {"dense": sym_none,
         "none": sym_none,
         "U1": sym_U1,
         "Z2": sym_Z2,
         "Z3": sym_Z3,
         "U1xU1": sym_U1xU1,
         "U1xU1xZ2": sym_U1xU1xZ2}

__all__ = ['rand', 'randR', 'randC', 'zeros', 'ones', 'eye', 'block',
           'make_config', 'load_from_dict', 'load_from_hdf5', 'decompress_from_1d']


# def make_config(backend=backend_np, sym=sym_none, default_device='cpu',
#                 default_dtype='float64', fermionic=False,
#                 default_fusion='meta', force_fusion=None, **kwargs):
def make_config(**kwargs) -> NamedTuple:
    r"""
    Create structure with YASTN configuration

    Parameters
    ----------
    backend : backend module or str
        Specify ``backend`` providing linear algebra and base dense tensors.
        Currently supported backends are

            * NumPy as ``yastn.backend.backend_np``
            * PyTorch as ``yastn.backend.backend_torch``

        The above backends can be specified as strings: "np", "torch".
        Defaults to NumPy backend.

    sym : symmetry module or compatible object or str
        Specify abelian symmetry. To see how YASTN defines symmetries,
        see :class:`yastn.sym.sym_abelian`.
        Defaults to ``yastn.sym.sym_none``, effectively a dense tensor.
        For predefined symmetries, takes string input from
        'none' (or 'dense'), 'Z2', 'Z3', 'U1', 'U1xU1', 'U1xU1xZ2'.

    default_device : str
        Tensors can be stored on various devices as supported by ``backend``

            * NumPy supports only ``'cpu'`` device
            * PyTorch supports multiple devices, see
              https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

        If not specified, the default device is ``'cpu'``.

    default_dtype: str
        Default data type (dtype) of YASTN tensors. Supported options are: ``'float64'``,
        ``'complex128'``. If not specified, the default dtype is ``'float64'``.
    fermionic : bool or tuple[bool,...]
        Specify behavior of :meth:`yastn.swap_gate` function, allowing to introduce fermionic statistics.
        Allowed values: ``False``, ``True``, or a tuple ``(True, False, ...)`` with one bool for each component
        charge vector, i.e., of length sym.NSYM. The default is ``False``.
    default_fusion: str
        Specify default strategy to handle leg fusion: ``'hard'`` or ``'meta'``. See :meth:`yastn.Tensor.fuse_legs`
        for details. The default is ``'hard'``.
    force_fusion : str
        Overrides fusion strategy provided in :meth:`yastn.Tensor.fuse_legs`. The default is ``None``.
    tensordot_policy: str
        Contraction approach used by :meth:`yastn.tensordot`

            * ``'fuse_to_matrix'`` Tensordot involves suitable permutation of each tensor while performing a fusion of each tensor into a sequence of matrices and calling matrix-matrix multiplication. Postprocessing includes unfusioning the remaining legs in the result, which often copy data adding extra overhead.
            * ``'fuse_contracted'`` Tensordot involves suitable permutation of each tensor while performing a fusion of to-be-contracted legs of each tensor and calling multiplication. It involves a larger number of multiplication calls for smaller objects, but unfusing the legs of the result is not needed.
            * ``'no_fusion'`` Tensordot involves suitable permutation of tensor blocks and calling matrix-matrix multiplication for a potentially large number of small objects. Resulting contributions to new blocks get added. However, overheads of initial fusion (copying data) can sometimes be avoided in this approach.

    Example
    -------

    ::

        config = yastn.make_config(backend='np', sym='U1')
    """
    if "backend" not in kwargs or kwargs["backend"] == 'np':
        kwargs["backend"] = backend_np
    elif kwargs["backend"] == 'torch':
        from .backend import backend_torch
        kwargs["backend"] = backend_torch
    elif isinstance(kwargs["backend"], str):
        raise YastnError("backend encoded as string only supports: 'np', 'torch'")

    if "sym" not in kwargs:
        kwargs["sym"] = sym_none
    elif isinstance(kwargs["sym"], str):
        try:
            kwargs["sym"] = _syms[kwargs["sym"]]
        except KeyError:
            raise YastnError("sym encoded as string only supports: 'dense', 'Z2', 'Z3', 'U1', 'U1xU1', 'U1xU1xZ2'.")

    return _config(**{a: kwargs[a] for a in _config._fields if a in kwargs})


def _fill(config=None, legs=(), n=None, isdiag=False, val='rand', **kwargs):
    if 's' in kwargs or 't' in kwargs or 'D' in kwargs:
        s = kwargs.pop('s') if 's' in kwargs else ()
        t = kwargs.pop('t') if 't' in kwargs else ()
        D = kwargs.pop('D') if 'D' in kwargs else ()
        mfs, hfs = None, None
    else:  # use legs for initialization
        if isinstance(legs, Leg):
            legs = (legs,)
        if isdiag and len(legs) == 1:
            legs = (legs[0], legs[0].conj())
        ulegs, mfs = _unpack_legs(legs)
        s = tuple(leg.s for leg in ulegs)
        t = tuple(leg.t for leg in ulegs)
        D = tuple(leg.D for leg in ulegs)
        hfs = tuple(leg.legs[0] for leg in ulegs)

        if any(config.sym.SYM_ID != leg.sym.SYM_ID for leg in ulegs):
            raise YastnError('Different symmetry of initialized tensor and some of the legs.')
        if isdiag and any(mf != (1,) for mf in mfs):
            raise YastnError('Diagonal tensor cannot be initialized with fused legs.')
        if isdiag and any(hf.tree != (1,) for hf in hfs):
            raise YastnError('Diagonal tensor cannot be initialized with fused legs.')

    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, mfs=mfs, hfs=hfs, **kwargs)
    a._fill_tensor(t=t, D=D, val=val)
    return a


def rand(config=None, legs=(), n=None, isdiag=False, **kwargs) -> yastn.Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with random numbers.

    Draws from a uniform distribution in [-1, 1] or [-1, 1] + 1j * [-1, 1],
    depending on desired ``dtype``.

    Parameters
    ----------
    config : module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    legs : Sequence[yastn.Leg]
        Specify legs of the tensor passing a list of :class:`yastn.Leg`.
    n : int | Sequence[int]
        Total charge of the tensor.
    isdiag : bool
        whether or not to make tensor diagonal.
    dtype : str
        Desired datatype, overrides :code:`default_dtype` specified in configuration.
    device : str
        Device on which the tensor should be initialized. Overrides attribute :code:`default_device`
        specified in configuration.
    s : Optional[Sequence[int]]
        (alternative) Tensor signature. Also determines the number of legs. The default is s=().
    t : Optional[Sequence[Sequence[int | Sequence[int]]]]
        (alternative) List of charges for each leg. The default is t=().
    D : Optional[Sequence[Sequence[int]]]
        (alternative) List of corresponding bond dimensions. The default is D=().

    Note
    ----
    If any of :code:`s`, :code:`t`, or :code:`D` are specified,
    :code:`legs` are overriden and only :code:`t`, :code:`D`, and :code:`s` are used.
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', **kwargs)


def randR(config=None, legs=(), n=None, isdiag=False, **kwargs) -> yastn.Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with real random numbers,
    see :meth:`yastn.rand`.
    """
    kwargs['dtype'] = 'float64'
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', **kwargs)


def randC(config=None, legs=(), n=None, isdiag=False, **kwargs) -> yastn.Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with complex random numbers,
    see :meth:`yastn.rand`.
    """
    kwargs['dtype'] = 'complex128'
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='rand', **kwargs)


def zeros(config=None, legs=(), n=None, isdiag=False, **kwargs) -> yastn.Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with zeros.

    Parameters
    ----------
    config : module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    legs : Sequence[yastn.Leg]
        Specify legs of the tensor passing a list of :class:`yastn.Leg`.
    n : int | Sequence[int]
        total charge of the tensor.
    isdiag : bool
        whether or not to make tensor diagonal
    dtype : str
        Desired datatype, overrides :code:`default_dtype` specified in configuration.
    device : str
        Device on which the tensor should be initialized. Overrides attribute :code:`default_device`
        specified in configuration.
    s : Optional[Sequence[int]]
        (alternative) Tensor signature. Also determines the number of legs. The default is s=().
    t : Optional[Sequence[Sequence[int | Sequence[int]]]]
        (alternative) List of charges for each leg. The default is t=().
    D : Optional[Sequence[Sequence[int]]]
        (alternative) List of corresponding bond dimensions. The default is D=().

    Note
    ----
    If any of :code:`s`, :code:`t`, or :code:`D` are specified,
    :code:`legs` are overriden and only :code:`t`, :code:`D`, and :code:`s` are used.
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='zeros', **kwargs)


def ones(config=None, legs=(), n=None, isdiag=False, **kwargs) -> yastn.Tensor:
    r"""
    Initialize tensor with all allowed blocks filled with ones.

    Parameters
    ----------
    config : module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    legs : Sequence[yastn.Leg]
        Specify legs of the tensor passing a list of :class:`yastn.Leg`.
    n : int | Sequence[int]
        total charge of the tensor.
    isdiag : bool
        whether or not to make tensor diagonal.
    dtype : str
        Desired datatype, overrides :code:`default_dtype` specified in configuration.
    device : str
        Device on which the tensor should be initialized. Overrides attribute :code:`default_device`
        specified in configuration.
    s : Optional[Sequence[int]]
        (alternative) Tensor signature. Also determines the number of legs. The default is s=().
    t : Optional[Sequence[Sequence[int | Sequence[int]]]]
        (alternative) List of charges for each leg. The default is t=().
    D : Optional[Sequence[Sequence[int]]]
        (alternative) List of corresponding bond dimensions. The default is D=().

    Note
    ----
    If any of :code:`s`, :code:`t`, or :code:`D` are specified,
    :code:`legs` are overriden and only :code:`t`, :code:`D`, and :code:`s` are used.
    """
    return _fill(config=config, legs=legs, n=n, isdiag=isdiag, val='ones', **kwargs)


def eye(config=None, legs=(), isdiag=True, **kwargs) -> yastn.Tensor:
    r"""
    Initialize diagonal tensor of identity matrix.
    In presence of symmetries, such matrix is block-diagonal with all allowed blocks filled with identity matrices.

    .. note::
        Currently supports either one or two legs as input. In case of a single leg,
        an identity matrix with Leg and its conjugate :meth:`yastn.Leg.conj()` is returned.

    Parameters
    ----------
    config : module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    legs : Sequence[yastn.Leg]
        Specify legs of the tensor passing a list of :class:`yastn.Leg`.
    isdiag : bool
        Specify by bool whether to return explicitly diagonal tensor.
        If :code:`True`, the signatures of the legs have to be opposite, and fused legs are not supported.
        If :code:`False`, it supports having fused legs and any combination of signatures.
    device : str
        Device on which the tensor should be initialized. Overrides attribute :code:`default_device`
        specified in configuration.
    s : Optional[Sequence[int]]
        (alternative) Tensor signature; should be (1, -1) or (-1, 1). The default is s=(1, -1).
    t : Optional[Sequence[Sequence[int | Sequence[int]]]]
        (alternative) List of charges for each leg. The default is t=().
    D : Optional[list]
        (alternative) List of corresponding bond dimensions. The default is D=().

    Note
    ----
    If any of :code:`s`, :code:`t`, or :code:`D` are specified,
    :code:`legs` are overriden and only :code:`t`, :code:`D`, and :code:`s` are used.
    """
    if isdiag:
        return _fill(config=config, legs=legs, isdiag=True, val='ones', **kwargs)
    if isinstance(legs, Leg):
        legs = (legs,)
    if len(legs) == 1:
        legs = (legs[0], legs[0].conj())
    legs = legs[:2]  # in case more then 2 legs are provided
    if any(leg.fusion != 'hard' for leg in legs):
        raise YastnError("eye() does not support 'meta'-fused legs")
    tmp = _fill(config=config, legs=legs, val='zeros', **kwargs)
    for t, D in zip(tmp.struct.t, tmp.struct.D):
        blk = tmp[t]
        for i in range(min(D)):
            blk[i, i] = 1
    return tmp

def load_from_dict(config=None, d=None) -> yastn.Tensor:
    """
    Create tensor from the dictionary :code:`d`.

    Parameters
    ----------
    config : module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn  configuration>`
    d : dict
        Tensor stored in form of a dictionary. Typically provided by an output
        of :meth:`yastn.Tensor.save_to_dict`.
    """
    if d is not None:
        c_isdiag = bool(d['isdiag'])
        c_Dp = [x[0] for x in d['D']] if c_isdiag else np.prod(d['D'], axis=1, dtype=np.int64).tolist()
        slices = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(c_Dp), c_Dp, d['D']))
        struct = _struct(s=d['s'], n=d['n'], diag=c_isdiag, t=d['t'], D=d['D'], size=sum(c_Dp))
        hfs = tuple(_Fusion(**hf) for hf in d['hfs'])
        c = Tensor(config=config, struct=struct, slices=slices, hfs=hfs, mfs=d['mfs'])
        if 'SYM_ID' in d and c.config.sym.SYM_ID != d['SYM_ID'].replace('(','').replace(')',''):  # for backward compatibility matching U1 and U(1)
            raise YastnError("Symmetry rule in config do not match loaded one.")
        if 'fermionic' in d and c.config.fermionic != d['fermionic']:
            raise YastnError("Fermionic statistics in config do not match loaded one.")
        c._data = c.config.backend.to_tensor(d['_d'], dtype=d['_d'].dtype.name, device=c.device)
        c.is_consistent()
        return c
    raise YastnError("Dictionary d is required.")


def load_from_hdf5(config, file, path) -> yastn.Tensor:
    """
    Create tensor from hdf5 file.

    Parameters
    ----------
    config : module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    file:
        pointer to opened HDF5 file.
    path:
        path inside the file which contains the state.
    """
    g = file.get(path)
    c_isdiag = bool(g.get('isdiag')[:][0])
    c_n = tuple(g.get('n')[:].tolist())
    c_s = tuple(g.get('s')[:].tolist())
    c_t = tuple(tuple(x) for x in g.get('ts')[:].tolist())
    c_D = tuple(tuple(x) for x in g.get('Ds')[:].tolist())
    c_Dp = [x[0] for x in c_D] if c_isdiag else np.prod(c_D, axis=1, dtype=np.int64).tolist()
    slices = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(c_Dp), c_Dp, c_D))
    struct = _struct(s=c_s, n=c_n, diag=c_isdiag, t=c_t, D=c_D, size=sum(c_Dp))

    mfs = literal_eval(tuple(file.get(path+'/mfs').keys())[0])
    hfs = tuple(_Fusion(*hf) if isinstance(hf, tuple) else _Fusion(**hf) \
                for hf in literal_eval(tuple(g.get('hfs').keys())[0]))
    c = Tensor(config=config, struct=struct, slices=slices, mfs=mfs, hfs=hfs)

    vmat = g.get('matrix')[:]
    c._data = c.config.backend.to_tensor(vmat, dtype=vmat.dtype.name, device=c.device)
    c.is_consistent()
    return c


def decompress_from_1d(r1d, meta) -> yastn.Tensor:
    """
    Generate tensor from dictionary :code:`meta` describing the structure of the tensor,
    charges and dimensions of its non-zero blocks, and 1-D array :code:`r1d` containing
    serialized data of non-zero blocks.

    Typically, the pair :code:`r1d` and :code:`meta` is obtained from :meth:`yastn.Tensor.compress_to_1d`.

    Parameters
    ----------
    r1d : rank-1 tensor
        1-D array (of backend type) holding serialized blocks.

    meta : dict
        structure of symmetric tensor. Non-zero blocks are indexed by associated charges.
        Each such entry contains block's dimensions and the location of its data
        in rank-1 tensor :code:`r1d`.
    """
    hfs = tuple(leg.legs[0] for leg in meta['legs'])
    a = Tensor(config=meta['config'], hfs=hfs, mfs=meta['mfs'], struct=meta['struct'], slices=meta['slices'])
    a._data = r1d
    return a


def block(tensors, common_legs=None) -> yastn.Tensor:
    """
    Assemble new tensor by blocking a group of tensors.

    History of blocking is stored together with history of hard-fusions.
    Subsequent blocking in a few steps and its equivalent single step blocking give the same tensor.
    Applying block on tensors turns all previous meta-fused legs into hard-fused ones.

    Parameters
    ----------
    tensors : dict[Sequence[int], yastn.Tensor]
        dictionary of tensors {(x,y,...): tensor at position x,y,.. in the new, blocked super-tensor}.
        Length of tuple should be equall to :code:`tensor.ndim - len(common_legs)`.

    common_legs : Sequence[int]
        Legs that are not blocked.
        This is equivalently to all tensors having the same position
        (not specified explicitly) in the super-tensor on that leg.
    """
    tn0 = next(iter(tensors.values()))  # first tensor; used to initialize new objects and retrive common values
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
    tn0 = next(iter(tensors.values()))  # first tensor; used to initialize new objects and retrive common values

    for tn in tensors.values():
        _test_can_be_combined(tn, tn0)
        if tn.struct.s != tn0.struct.s:
            raise YastnError('Signatues of blocked tensors are inconsistent.')
        if tn.struct.n != tn0.struct.n:
            raise YastnError('Tensor charges of blocked tensors are inconsistent.')
        if tn.isdiag:
            raise YastnError('Block does not support diagonal tensors. Use .diag() first.')

    legs_tn = {pa: a.get_legs() for pa, a in tensors.items()}
    ulegs, legs, hfs, ltDtot, ltDslc = [], [], [], [], []
    for n in range(tn0.ndim_n):
        legs_n = {}
        for pa, ll in legs_tn.items():
            if pa[n] not in legs_n:
                legs_n[pa[n]] = [ll[n]]
            else:
                legs_n[pa[n]].append(ll[n])
        legs.append(legs_n)
        legs_n = {p: leg_union(*plegs) for p, plegs in legs_n.items()}
        ulegs.append(legs_n)
        pn = sorted(legs_n.keys())
        hfs.append(_sum_legs_hfs([legs_n[p] for p in pn]))

        tpD = sorted((t, p, D) for p, leg in legs_n.items() for t, D in zip(leg.t, leg.D))
        ltDtot.append({})
        ltDslc.append({})
        for t, gr in groupby(tpD, key=itemgetter(0)):
            Dlow, tpDslc = 0, {}
            for _, p, D in gr:
                Dhigh = Dlow + D
                tpDslc[p] = (Dlow, Dhigh)
                Dlow = Dhigh
            ltDtot[-1][t] = Dhigh
            ltDslc[-1][t] = tpDslc


    for pa in tensors.keys():
        if any(_leg_fusions_need_mask(ulegs[n][pa[n]], leg) for n, leg in enumerate(legs_tn[pa])):
            legs_new = {n: legs[pa[n]] for n, legs in enumerate(ulegs)}
            tensors[pa] = _embed_tensor(tensors[pa], legs_tn[pa], legs_new)

    # all unique blocks
    # meta_new = {tind: Dtot};  #meta_block = [(tind, pos, Dslc)]
    meta_new, meta_block = {}, []
    nsym = tn0.config.sym.NSYM
    for pa, a in tensors.items():
        for tind, slind, Dind in zip(a.struct.t, a.slices, a.struct.D):
            Dslcs = tuple(tDslc[tind[n * nsym : n * nsym + nsym]][pa[n]] for n, tDslc in enumerate(ltDslc))
            meta_block.append((tind, slind.slcs[0], Dind, pa, Dslcs))
            if tind not in meta_new:
                meta_new[tind] = tuple(tDtot[tind[n * nsym : n * nsym + nsym]] for n, tDtot in enumerate(ltDtot))

    meta_block = tuple(sorted(meta_block, key=itemgetter(0)))
    meta_new = tuple(sorted(meta_new.items()))
    c_t = tuple(t for t, _ in meta_new)
    c_D = tuple(D for _, D in meta_new)
    c_Dp = np.prod(c_D, axis=1, dtype=np.int64).tolist() if len(c_D) > 0 else ()
    c_slices = tuple(_slc(((stop - dp, stop),), ds, dp) for stop, dp, ds in zip(accumulate(c_Dp), c_Dp, c_D))
    c_struct = _struct(n=a.struct.n, s=a.struct.s, t=c_t, D=c_D, size=sum(c_Dp))
    meta_new = tuple((x, y, z.slcs[0]) for x, y, z in zip(c_t, c_D, c_slices))
    data = tn0.config.backend.merge_super_blocks(tensors, meta_new, meta_block, c_struct.size)
    return tn0._replace(struct=c_struct, slices=c_slices, data=data, hfs=tuple(hfs))


def _sum_legs_hfs(legs):
    """ sum hfs based on info in legs"""
    hfs = [leg.legs[0] for leg in legs]
    t_in = [leg.t for leg in legs]
    D_in = [leg.D for leg in legs]
    s_out = legs[0].s
    return _sum_hfs(hfs, t_in, D_in, s_out)
