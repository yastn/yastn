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
""" Methods outputting data from yastn.Tensor. """
from __future__ import annotations
from functools import reduce
from numbers import Number
from operator import mul
from typing import Sequence

import numpy as np

from ._auxiliary import _clear_axes, _unpack_axes, _struct, _flatten, get_blocks, find_index
from ._legbasic import legs_from_dict_v2
from ._legs import Leg, LegMeta, legs_union, _legs_mask_needed
from ._merging import _embed_tensor
from ._tests import YastnError
from ..sym import sym_none
from .._split_combine_dict import combine_data_and_meta

__all__ = ['requires_grad']


def to_dict(a, level=2, meta=None, resolve_ops=False) -> dict:
    r"""
    Serialize YASTN tensor to a dictionary containing all the information needed to recreate the tensor.
    Complementary function is :meth:`yastn.Tensor.from_dict` or a general :meth:`yastn.from_dict`.

    Using argument `level == 2` allows robust saving with numpy.save method.

    Parameters
    ----------
    a: yastn.Tensor
        tensor to serialize.
    level: int
        Controls how much internal Tensor data and meta sub-classes are turned into basic python data structures.
        For level == 0, nothing is converted.
        For level >= 1, converts information about config and tensor block structure into python dictionaries.
        For level >= 2, turns tensor data into numpy array.
        Level == 2 (or == 1 for numpy backend) allows saving with numpy.save, which is the default.

    meta: dict
        There is an option to provide meta-data obtained from earlier application of :meth:`yastn.Tensor.to_dict`.
        Extra zero blocks (missing in tensor) are then included in the returned 1D array
        to make it consistent with structure given in ``meta``.
        Raises error if tensor has some blocks which are not included in ``meta`` or otherwise
        :code:`meta` does not match the tensor.

    .. note::
        :meth:`yastn.Tensor.to_dict` and :meth:`yastn.Tensor.from_dict`, together with
        :meth:`yastn.split_data_and_meta` and :meth:`yastn.combine_data_and_meta` provide mechanism
        that allows using external matrix-free methods, such as :func:`eigs` implemented in SciPy.
        See example at :ref:`examples/tensor/decomposition:combining with scipy.sparse.linalg.eigs`.
    """
    if resolve_ops:
        return a.consume_transpose().to_dict(level=level, meta=meta, resolve_ops=False)

    if level >= 1:
        config = a.config._asdict()
        config['sym'] = config['sym'].SYM_ID
        config['backend'] = config['backend'].BACKEND_ID
        hfs = tuple(hf._asdict() for hf in a.hfs)
        struct = a.struct._asdict()
        struct['legs'] = tuple(leg._asdict() for leg in struct['legs'])
    else:
        config = a.config
        hfs = a.hfs
        struct = a.struct

    data = a.data if level < 2 else a.config.backend.to_numpy(a.data)

    d = {'type': type(a).__name__,
         'dict_ver': 3,
         'level': level,
         'config': config,
         'data': data,
         'struct': struct,
         'trans': a.trans,
         'hfs': hfs,
         'mfs': a.mfs,
         'size': a.size}

    if meta is not None:
        if not all(meta[k] == d[k] for k in ['type', 'dict_ver', 'config', 'struct', 'trans', 'hfs', 'mfs']):
            tmp = a.config.backend.zeros(meta['size'], dtype=a.yastn_dtype, device=a.device)
            ap = type(a).from_dict(combine_data_and_meta(tmp, meta))
            try:
                a = a + ap  # fill-in zero blocks
            except YastnError as e:
                raise YastnError("Tensor is inconsistent with meta: " + str(e))
            d = a.to_dict(level=level)
            if not all(meta[k] == d[k] for k in ['type', 'dict_ver', 'config', 'struct', 'trans', 'hfs', 'mfs']):
                raise YastnError("Tensor is inconsistent with meta.")
    return d


# def save_to_hdf5(a, file, path) -> None:
#     """
#     Export tensor into hdf5 type file.

#     Complementary function is :meth:`yastn.load_from_hdf5`.

#     Parameters
#     ----------
#     a : yastn.Tensor
#         tensor to export.
#     """
#     a = a.consume_transpose()
#     _d = a.config.backend.to_numpy(a._data)
#     hfs = tuple(tuple(hf) for hf in a.hfs)
#     file.create_dataset(path+'/isdiag', data=[int(a.isdiag)])
#     file.create_group(path+'/mfs/'+str(a.mfs))
#     file.create_group(path+'/hfs/'+str(hfs))
#     file.create_dataset(path+'/n', data=a.struct.n)
#     file.create_dataset(path+'/s', data=a.struct.s)
#     file.create_dataset(path+'/ts', data=a.struct.t)
#     file.create_dataset(path+'/Ds', data=a.struct.D)
#     file.create_dataset(path+'/matrix', data=_d)


############################
#    output information    #
############################


def print_properties(a, file=None) -> Never:
    """
    Print a number of properties of the tensor:

        * symmetry,
        * signature,
        * total charge,
        * whether it is a diagonal tensor,
        * meta/logical rank - treating meta-fused legs as a single logical leg,
        * native rank,
        * total dimension of all existing charge sectors for each leg, treating meta-fused legs as a single leg,
        * total dimension of all existing charge sectors for native leg,
        * number of non-empty blocks
        * total number of elements across all non-empty blocks,
        * fusion tree for each leg,
        * fusion history with ``'o'`` indicating original legs, ``'m'`` meta-fusion,
          ``'p'`` hard-fusion (product), ``'s'`` blocking (sum).
    """
    print("symmetry     :", a.config.sym.SYM_ID, file=file)
    print("signature    :", a.s, file=file)  # signature
    print("charge       :", a.n, file=file)  # total charge of tensor
    print("isdiag       :", a.isdiag, file=file)
    print("dim meta     :", a.ndim, file=file)  # number of meta legs
    print("dim native   :", a.ndim_n, file=file)  # number of native legs
    print("shape meta   :", a.get_shape(native=False), file=file)
    print("shape native :", a.get_shape(native=True), file=file)
    print("no. blocks   :", a.num_blocks, file=file)  # number of blocks
    print("size         :", a.size, file=file)  # total number of elements in all blocks
    st = {i: leg.history() for i, leg in enumerate(a.get_legs())}
    print("legs fusions :", st, "\n", file=file)


def __str__(a) -> str:
    legs = a.get_legs()
    ts = tuple(leg.t for leg in legs)
    Ds = tuple(leg.D for leg in legs)
    s = f"{a.config.sym.SYM_ID} s= {a.s} n= {a.n}\n"
    s += f"leg charges : {ts}\n"
    s += f"dimensions  : {Ds}"
    return s


def __repr__(a) -> str:
    """
    Return string representation of the tensor.
    """
    return __str__(a)


def requires_grad(a) -> bool:
    """
    Return ``True`` if tensor data have autograd enabled.
    """
    return a.config.backend.requires_grad(a._data)


def print_blocks_shape(a, file=None) -> str:
    """
    Print shapes of blocks as a sequence of block's charge followed by its shape.
    """
    for t, D in zip(a.get_blocks_charge(), a.get_blocks_shape()):
        print(f"{t} {D}", file=file)


def is_complex(a) -> bool:
    """
    Return ``True`` if tensor data are complex.
    """
    return a.config.backend.is_complex(a._data)


def get_tensor_charge(a) -> Sequence[int]:
    """
    Return :attr:`yastn.Tensor.n`.
    """
    return a.struct.n


def get_signature(a, native=False) -> Sequence[int]:
    """
    Return tensor signature, equivalent to :attr:`yastn.Tensor.s`.

    If ``native=True``, ignore fusion with ``mode=meta`` and return the signature of tensors's native legs, see :attr:`yastn.Tensor.s_n`.
    """
    return a.s_n if native else a.s


def get_rank(a, native=False) -> int:
    """
    Return tensor rank equivalent to :attr:`yastn.Tensor.ndim`.

    If ``native=True``, ignore fusion with ``mode=meta`` and count native legs, see :attr:`yastn.Tensor.ndim_n`.
    """
    return a.ndim_n if native else a.ndim


def get_blocks_charge(a) -> Sequence[Sequence[int]]:
    """
    Return charges of all native blocks.

    In case of product of abelian symmetries, for each block the individual symmetry
    charges are flattened into a single tuple.
    """
    bl = get_blocks(a.config.sym, a.struct.legs, a.struct.n, a.struct.isdiag)
    tset = bl.t[:, a.trans, :].reshape(bl.nblocks, a.ndim_n * a.config.sym.NSYM)
    return tuple(map(tuple, tset.tolist()))


def get_blocks_shape(a) -> Sequence[Sequence[int]]:
    """
    Shapes of all native blocks.
    """
    bl = get_blocks(a.config.sym, a.struct.legs, a.struct.n, a.struct.isdiag)
    Dset = bl.D[:, a.trans]
    return tuple(map(tuple, Dset.tolist()))


def get_shape(a, axes=None, native=False) ->  int | Sequence[int]:
    r"""
    Return effective bond dimensions as sum of dimensions along sectors for each leg.

    Parameters
    ----------
    axes : int | Sequence[int]
        indices of legs; If ``axes=None`` returns shape for all legs. The default is ``axes=None``.
    """
    if axes is None:
        axes = tuple(range(a.ndim_n if native else a.ndim))
    return_int = False
    if isinstance(axes, int):
        axes = [axes]
        return_int = True

    legs_n = []  # native legs
    legs_m = []  # number of legs forming LegMeta
    for leg in a.get_legs(axes, native=native):
        if isinstance(leg, LegMeta):
            legs_n.extend(leg.legs)
            legs_m.append(len(leg.legs))
        else:
            legs_n.append(leg)
            legs_m.append(1)
    Dtot_n = tuple(sum(leg.D) for leg in legs_n)
    Dtot_a, i = [], 0
    for dd in legs_m:
        Dtot_a.append(reduce(mul, Dtot_n[i: i+dd], 1))
        i += dd
    Dtot_a = tuple(Dtot_a)
    return Dtot_a[0] if return_int else Dtot_a


def get_dtype(a) -> numpy.dtype | torch.dtype:
    """
    ``dtype`` of tensor data used by the backend.
    """
    return a.config.backend.get_dtype(a._data)


def __getitem__(a, key) -> numpy.ndarray | torch.tensor:
    """
    Block corresponding to a given charge combination.

    The type of the returned tensor corresponds to specified backend, e.g.,
    :class:`numpy.ndarray` or :class:`torch.Tensor` for *NumPy* and *PyTorch* respectively.
    In case of diagonal tensor, the output is a 1D array.

    Parameters
    ----------
    key : Sequence[int] | Sequence[Sequence[int]]
        charges of the block.
    """
    try:
        key = np.array(key, dtype=np.int64).reshape(a.ndim_n, a.config.sym.NSYM)
        reverse_trans = np.argsort(a.trans)
        ukey = key[reverse_trans, :].ravel()
        bl = get_blocks(a.config.sym, a.struct.legs, a.struct.n, a.isdiag)
        ind = find_index(bl.t, ukey, sorted=True)
    except ValueError as exc:
        raise YastnError('Tensor does not have the block specified by key.') from exc
    x = a._data[slice(*bl.slc[ind])]
    return x if a.isdiag else a.config.backend.permute_dims(x, bl.D[ind], a.trans)


def __contains__(a, key) -> bool:
    key = tuple(_flatten(key)) if (hasattr(key,'__iter__') or hasattr(key,'__next__')) else (key,)
    if a.isdiag and len(key) == a.config.sym.NSYM:
        key = key + key
    key = np.array(key, dtype=np.int64)
    try:
        bl = get_blocks(a.config.sym, a.struct.legs, a.struct.n, a.isdiag)
        _ = find_index(bl.t, key, sorted=True)
        return True
    except ValueError:
        return False


##################################################
#    output tensors info - advanced structure    #
##################################################


def get_legs(a, axes=None, native=False) -> yastn.Leg | Sequence[yastn.Leg]:
    r"""
    Return a leg or a set of legs of the tensor ``a``.

    Parameters
    ----------
    axes : int | Sequence[int] | None
        Indices of legs to retrieve. If ``None`` returns list with all legs.

    native : bool
        If ``True``, ignore fusion with ``mode=meta`` and return native legs.
        Otherwise returns meta-fused legs (if such leg fusion was performed).
        The default is ``False``.
    """
    legs = []
    if axes is None:
        axes = tuple(range(a.ndim if not native else a.ndim_n))
    multiple_legs = hasattr(axes, '__iter__')
    axes, = _clear_axes(axes)
    for ax in axes:
        nax = (ax,)
        if not native:
            nax, = _unpack_axes(a.mfs, (ax,))

        nax = tuple(a.trans[ax] for ax in nax)

        legs_ax = []
        for i in nax:
            leg = Leg(a.config, s=a.struct.legs[i].s, t=a.struct.legs[i].t, D=a.struct.legs[i].D, hf=a.hfs[i])
            legs_ax.append(leg)

        if not native and a.mfs[ax][0] > 1:
            bl = get_blocks(a.config.sym, a.struct.legs, a.n, a.isdiag)

            tseta = bl.t[:, nax, :].reshape(bl.nblocks, len(nax) * a.config.sym.NSYM).tolist()
            Dseta = np.prod(bl.D[:, nax], axis=1, dtype=np.int64).tolist()
            tDn = {tuple(tn): Dn for tn, Dn in zip(tseta, Dseta)}
            tDn = dict(sorted(tDn.items()))
            t, D = tuple(tDn.keys()), tuple(tDn.values())
            leg = LegMeta(a.config.sym, s=legs_ax[0].s, t=t, D=D, mf=a.mfs[ax], legs=tuple(legs_ax))
            legs.append(leg)
        else:
            legs.append(legs_ax.pop())

    return tuple(legs) if multiple_legs else legs.pop()


############################
#   Down-casting tensors   #
############################

def to_dense(a, legs=None, native=False, reverse=False) -> numpy.ndarray | torch.tensor:
    r"""
    Create dense tensor corresponding to the symmetric tensor.

    The type of the returned tensor depends on the backend, i.e. ``numpy.ndarray`` or ``torch.tensor``.
    Blocks are ordered according to increasing charges on each leg.
    It is possible to supply a list of additional charge sectors to be included by
    explicitly specifying ``legs``.
    Specified ``legs`` should be consistent with current structure of the tensor.
    This allows to fill in extra zero blocks.

    Parameters
    ----------
    legs : dict[int, yastn.Leg]
        specify extra charge sectors on the legs by adding desired :class:`yastn.Leg`
        under legs's index into dictionary.

    native: bool
        output native tensor (ignoring meta-fusion of legs).

    reverse: bool
        reverse the order in which blocks are sorted. Default order is ascending in
        values of block's charges.
    """
    c = a.to_nonsymmetric(legs, native, reverse)
    x = c.config.backend.clone(c._data)
    D = tuple(leg.D[0] for leg in c.legs)
    x = c.config.backend.diag_create(x) if c.isdiag else x.reshape(D)
    return x


def to_numpy(a, legs=None, native=False, reverse=False) -> numpy.ndarray:
    r"""
    Create dense :class:`numpy.ndarray`` corresponding to the symmetric tensor.
    See :func:`yastn.to_dense`.
    """
    return a.config.backend.to_numpy(a.to_dense(legs, native, reverse))


def to_raw_tensor(a) -> numpy.ndarray | torch.tensor:
    """
    If the symmetric tensor has just a single non-empty block, return raw tensor representing
    that block.

    The type of the returned tensor depends on the backend, i.e. ``numpy.ndarray`` or ``torch.tensor``.
    """
    bl = get_blocks(a.config.sym, a.struct.legs, a.struct.n, a.struct.isdiag)
    if len(bl.D) == 1:
        Dblock = tuple(bl.D[0])
        return a._data.reshape(Dblock)
    raise YastnError('Only tensor with a single block can be converted to raw tensor.')


def to_nonsymmetric(a, legs=None, native=False, reverse=False) -> 'Tensor':
    r"""
    Create equivalent :class:`yastn.Tensor` with no explicit symmetry. All blocks of the original
    tensor are accumulated into a single block.

    Blocks are ordered according to increasing charges on each leg.
    It is possible to supply a list of additional charge sectors to be included by explicitly
    specifying ``legs``. These legs should be consistent with current structure of the tensor.
    This allows to fill in extra zero blocks.

    .. note::
        YASTN structure is redundant since resulting tensor is effectively just
        a single dense block. To obtain this single dense block directly, use :meth:`yastn.Tensor.to_dense`.

    Parameters
    ----------
    legs : dict[int, yastn.Leg]
        specify extra charge sectors on the legs by adding desired :class:`yastn.Leg`
        under legs's index into dictionary.

    native: bool
        output native tensor (ignoring meta-fusion of legs).

    reverse: bool
        reverse the order in which blocks are sorted. Default order is ascending in
        values of block's charges.
    """
    config_dense = a.config._replace(sym=sym_none)
    a = a.consume_transpose()
    #
    legs_a = list(a.get_legs(native=native))
    ndim_a = len(legs_a)  # ndim_n if native else ndim

    if legs is not None:
        if any((n < 0) or (n >= ndim_a) for n in legs.keys()):
            raise YastnError('Specified leg out of ndim')
        legs_new = {n: legs_union(legs_a[n], leg) for n, leg in legs.items()}
        if any(_legs_mask_needed(leg, legs_a[n]) for n, leg in legs_new.items()):
            a = _embed_tensor(a, legs_a, legs_new)  # mask needed
        for n, leg in legs_new.items():
            legs_a[n] = leg

    legs_n = []  # native legs
    legs_m = []  # number of legs forming LegMeta
    for leg in legs_a:
        if isinstance(leg, LegMeta):
            legs_n.extend(leg.legs)
            legs_m.append(len(leg.legs))
        else:
            legs_n.append(leg)
            legs_m.append(1)

    if ndim_a == 0:  # scalar
        meta = [((None,), ())]
    else:
        step = -1 if reverse else 1
        tD = []
        for leg in legs_n:
            Dlow, tDn = 0, {}
            for tn, Dn in zip(leg.t[::step], leg.D[::step]):
                Dhigh = Dlow + Dn
                tDn[tn] = (Dlow, Dhigh)
                Dlow = Dhigh
            tD.append(tDn)

        bl = get_blocks(a.config.sym, a.legs, a.n, a.isdiag)
        tset_ax = list(zip(*[bl.t[:, ax, :].reshape(bl.nblocks, a.config.sym.NSYM).tolist() for ax in range(a.ndim_n)]))
        meta = [(sl, tuple(tDn[tuple(tt)] for tDn, tt in zip(tD, t_ax))) for sl, t_ax in zip(bl.slc, tset_ax)]

    Dtot_n = tuple(sum(leg.D) for leg in legs_n)
    Dtot_a, i = [], 0
    for dd in legs_m:
        Dtot_a.append(reduce(mul, Dtot_n[i: i+dd], 1))
        i += dd
    Dtot_a = tuple(Dtot_a)

    Dtot = Dtot_n if native else Dtot_a
    c_s = a.get_signature(native)
    c_t = ((),)
    c_D = (Dtot,)

    if a.isdiag:
        Dtot = Dtot[:1]
        Dtot_n = Dtot_n[:-1]
        meta = [(sl, D[:1]) for sl, D in meta]

    c_legs = legs_from_dict_v2({"s": c_s, "n": (), "t": c_t, "D": c_D})
    c_struct = _struct(legs=c_legs, n=(), isdiag=a.isdiag)
    data = a.config.backend.merge_to_dense(a._data, Dtot_n, meta)
    return a._replace(config=config_dense, struct=c_struct, data=data, mfs=None, hfs=None, trans=None)


def zero_of_dtype(a):
    """ Return zero scalar of the instance specified by backend and dtype. """
    return a.config.backend.zeros((), dtype=a.yastn_dtype, device=a.device)


def to_number(a) -> Number:
    r"""
    Assuming the symmetric tensor has just a single non-empty block of total dimension one,
    return this element as a scalar.

    The type of the scalar is given by the backend.
    For empty tensor returns `0`.

    .. note::
        This operation preserves autograd.
    """
    size = a.size
    if size == 1:
        x = a.config.backend.first_element(a._data)
    elif size == 0:
        x = a.zero_of_dtype()
    else:
        raise YastnError('Only single-element (symmetric) Tensor can be converted to scalar')
    return x


def item(a) -> float:
    """
    Assuming the symmetric tensor has just a single non-empty block of total dimension one,
    return this element as standard Python scalar.

    For empty tensor returns :math:`0`.
    """
    size = a.config.backend.get_size(a._data)
    if size == 1:
        return a.config.backend.item(a._data)
    if size == 0:
        return 0
    raise YastnError("Only single-element (symmetric) Tensor can be converted to scalar")
