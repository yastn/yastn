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
r"""
yastn.Tensor

This class defines generic arbitrary-rank tensor supporting abelian symmetries.
In principle, any number of symmetries can be used, including dense tensor with no symmetries.

An instance of a Tensor is specified by a list of blocks (dense tensors) labeled by symmetries' charges on each leg.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

from .._split_combine_dict import *
from ._algebra import *
from ._auxiliary import *
from ._contractions import *
from ._control_lru import *
from ._einsum import *
from ._initialize import *
from ._legbasic import *
from ._legs import *
from ._merging import *
from ._output import *
from ._single import *
from ._tests import *
from .linalg import *
from . import _algebra
from . import _contractions
from . import _control_lru
from . import _einsum
from . import _initialize
from . import _legs
from . import _merging
from . import _output
from . import _single
from . import _tests
from . import linalg

__all__ = ['Tensor', 'linalg', 'YastnError']
__all__.extend(_algebra.__all__)
__all__.extend(_contractions.__all__)
__all__.extend(_control_lru.__all__)
__all__.extend(_einsum.__all__)
__all__.extend(_initialize.__all__)
__all__.extend(_legbasic.__all__)
__all__.extend(_legs.__all__)
__all__.extend(_merging.__all__)
__all__.extend(_output.__all__)
__all__.extend(_single.__all__)
__all__.extend(_tests.__all__)
__all__.extend(linalg.__all__)


class Tensor:
    # Define a tensor with abelian symmetries and operations on such tensor(s).

    def __init__(self, config=None, s=(), n=None, isdiag=False, **kwargs):
        r"""
        Initialize empty (without any blocks allocated) YASTN tensor.

        Parameters
        ----------
            config : module | _config(NamedTuple)
                :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
            s : Sequence[int]
                a signature of tensor. Also determines the number of legs.
            n : int | Sequence[int]
                total charge of the tensor. In case of direct product of several
                abelian symmetries, `n` is a tuple with total charge for each individual
                symmetry.
            isdiag : bool
                distinguish diagonal tensor as a special case of a tensor.
        """
        self.config = config if isinstance(config, _config) else _config(**{a: getattr(config, a) for a in _config._fields if hasattr(config, a)})

        if 'data' in kwargs:
            assert (kwargs['data'] is None or kwargs['data'].ndim == 1), "Tensor data should be None or a 1D array."
            self._data = kwargs['data']  # 1d container for tensor data
        else:
            dev = kwargs.get('device', self.config.default_device)
            dty = kwargs.get('dtype', self.config.default_dtype)
            self._data = self.config.backend.zeros((1,), dtype=dty, device=dev)
        #
        try:
            self.struct = kwargs['struct']
        except KeyError:
            try:
                s = tuple(s)
            except TypeError:
                s = (s,)
            try:
                n = tuple(n)
            except TypeError:
                n = self.config.sym.zero() if n is None else (n,)
            if len(n) != self.config.sym.NSYM:
                raise YastnError("n does not match the number of symmetry sectors.")
            if isdiag:
                if len(s) == 0:
                    s = (1, -1)  # default
                if s not in ((-1, 1), (1, -1)):
                    raise YastnError("Diagonal tensor should have s equal (1, -1) or (-1, 1).")
                if any(x != 0 for x in n):
                    raise YastnError("Tensor charge of a diagonal tensor should be 0.")
            legs = tuple(LegBasic(s=s_l, t=(), D=()) for s_l in s)
            self.struct = _struct(legs=legs, n=n, isdiag=bool(isdiag))
        #
        # self.mfs and self.trans describe logical/meta transformation of legs
        # 1) at the highest level, self.mfs is a logical fusion of legs,
        # where a group of consecutive legs is treated as a single leg
        # 2) Next, self.trans is a transpose/permutation of leg indices,
        # mapping from logical legs (unpacked from mfs) to native legs in data that are described by self.struct
        # 3) At the lowest level, self.hfs contains information about hard-fusion of native legs
        #
        self._trans = kwargs.get('trans', None)
        try:
            self._trans = tuple(self._trans)
        except TypeError:
            self._trans = tuple(range(self.ndim_n))
        #
        # fusion tree for each leg: encodes number of fused legs e.g. 5 2 1 1 3 1 2 1 1 = ((1, 1), (1, (1, 1)))
        self.mfs = kwargs.get('mfs', None)
        try:
            self.mfs = tuple(self.mfs)
        except TypeError:
            self.mfs = ((1,),) * self.ndim_n
        #
        self.hfs = kwargs.get('hfs', None)
        try:
            self.hfs = tuple(self.hfs)
        except TypeError:
            self.hfs = tuple(_Fusion() for _ in self.s_n)

    # pylint: disable=C0415
    from ._initialize import set_block, _fill_tensor, __setitem__
    from .linalg import norm, svd, svd_with_truncation, eig, eigh, eigh_with_truncation, qr, truncation_mask
    from ._contractions import tensordot, __matmul__, vdot, trace, swap_gate, broadcast, apply_mask
    from ._algebra import __add__, __sub__, __mul__, __rmul__, __array_ufunc__, __neg__, add
    from ._algebra import __lt__, __gt__, __le__, __ge__, __truediv__, __pow__, allclose
    from ._algebra import __abs__, real, imag, sqrt, rsqrt, reciprocal, exp, bitwise_not
    from ._single import conj, conj_blocks, flip_signature, flip_charges, switch_signature, transpose, moveaxis, move_leg, diag
    from ._single import grad, requires_grad_, add_leg, remove_leg, drop_leg_history
    from ._single import copy, shallow_copy, clone, detach, detach_, to, consume_transpose
    from ._single import remove_random_blocks, remove_zero_blocks
    from ._output import print_properties, __str__, __repr__, print_blocks_shape, is_complex
    from ._output import get_blocks_charge, get_blocks_shape, get_legs
    from ._output import zero_of_dtype, item, __getitem__, __contains__
    from ._output import get_shape, get_signature, get_dtype
    from ._output import get_tensor_charge, get_rank
    from ._output import to_number, to_dense, to_numpy, to_raw_tensor, to_nonsymmetric
    from ._output import to_dict
    from ._tests import is_consistent, are_independent
    from ._merging import fuse_legs, unfuse_legs, fuse_meta_to_hard
    from ._krylov import expand_krylov_space

    __iter__ = None  # ensure that the Tensor is not iterable

    def _replace(self, **kwargs) -> Tensor:
        """Create a shallow copy with the specified fields replaced."""
        for arg in ('config', 'struct', 'mfs', 'hfs', 'data', 'trans'):
            if arg not in kwargs:
                kwargs[arg] = getattr(self, arg)
        return Tensor(**kwargs)

    @classmethod
    def from_dict(cls, d: dict, config:None | _config=None) -> Tensor:
        """
        Deserialize a tensor from the dictionary ``d``.

        Parameters
        ----------
        d : dict
            Tensor stored in form of a dictionary. Typically provided by an output
            of :meth:`yastn.Tensor.to_dict`.

        config : Optional[module | _config(NamedTuple)]
            :ref:`YASTN configuration <tensor/configuration:yastn  configuration>`
            If provided, overrides configuration stored in `d`.
        """
        #
        if 'dict_ver' not in d:  # old save_to_dict()
            d = {'type': 'Tensor',
                 'dict_ver': 1,
                 'data': d['_d'],
                 'hfs': d['hfs'],
                 'mfs': d['mfs'],
                 'config': {'sym': d['SYM_ID'],
                            'fermionic': d['fermionic']},
                 'struct': {'s': d['s'],
                            'n': d['n'],
                            't': d['t'],
                            'D': d['D'],
                            'diag': d['isdiag']}
                }

        if d['dict_ver'] in [1, 2]:  # d from method to_dict (single version as of now)

            if 'trans' not in d:  # to handle dict_ver==1 with no trans
                d['trans'] = None

            if d['type'] != 'Tensor':
                raise YastnError(f"{cls.__name__} does not match d['type'] == {d['type']}")

            d = d.copy()

            if config is not None:
                if (d['config']['sym'] if isinstance(d['config'], dict) else d['config'].sym.SYM_ID)  != config.sym.SYM_ID:
                    raise YastnError("Symmetry rule in config does not match the one in stored in d.")
                if (d['config']['fermionic'] if isinstance(d['config'], dict) else d['config'].fermionic) != config.fermionic:
                    raise YastnError("Fermionic statistics in config does not match the one in stored in d.")
                d['config'] = config

            for k in ['struct', 'slices', 'hfs', 'mfs']:
                if k in d:
                    d[k] = _convert_lists_to_tuples(d[k])
            if not isinstance(d['config'], _config):
                d['config'] = make_config(**d['config'])
            d['hfs'] = tuple(_Fusion.from_dict(hf) for hf in d['hfs'])
            old_struct = d['struct']
            legs = legs_from_dict_v2(old_struct)
            d['struct'] = _struct(legs=legs, n=d['struct']['n'], isdiag=d['struct']['diag'])
            dtype = d['config'].default_dtype
            if hasattr(d['data'], 'dtype'):
                if 'complex128' in str(d['data'].dtype):
                    dtype = 'complex128'
                if 'float64' in str(d['data'].dtype):
                    dtype = 'float64'
            data = d['config'].backend.to_tensor(d['data'], dtype=dtype, device=d['config'].default_device)
            bl_new = get_blocks(d['config'].sym, d['struct'])
            t_old = np.array(old_struct['t'], dtype=np.int64)
            t_old = t_old.reshape(len(t_old), len(d['struct'].legs), len(d['struct'].n))
            if 'slices' in d:
                slc_old = np.array([x[0][0] for x in d['slices']], dtype=np.int64)
            else:
                D_old = np.array(old_struct['D'], dtype=np.int64)
                if d['struct'].isdiag:
                    Dp = D_old[:, 0]
                else:
                    Dp = np.prod(D_old, axis=1, dtype=np.int64)
                slc_old = np.zeros((len(Dp), 2), dtype=np.int64)
                slc_old[:, 1] = np.cumsum(Dp)
                slc_old[1:, 0] = slc_old[:-1, 1]

            ind1, ind2 = find_matching_indices(bl_new.t, t_old)
            meta = _compress_slices(np.column_stack([bl_new.slc[ind1], slc_old[ind2]]))
            meta_dt = np.dtype([
                ('sln', np.int64, (2,)),
                ('slo', np.int64, (2,))])
            meta = meta.view(meta_dt).reshape(-1)
            meta = convert_to_tuples_and_slices(meta)
            newdata = d['config'].backend.embed_slices(data, meta, bl_new.size)
            d['data'] = newdata
            return cls(**d)

        if d['dict_ver'] == 3:  # d from method to_dict (single version as of now)

            if d['type'] != 'Tensor':
                raise YastnError(f"{cls.__name__} does not match d['type'] == {d['type']}")

            if d['level'] >= 1 or config is not None:
                d = d.copy()

            if config is not None:
                if (d['config']['sym'] if isinstance(d['config'], dict) else d['config'].sym.SYM_ID) != config.sym.SYM_ID:
                    raise YastnError("Symmetry rule in config does not match the one in stored in d.")
                if (d['config']['fermionic'] if isinstance(d['config'], dict) else d['config'].fermionic) != config.fermionic:
                    raise YastnError("Fermionic statistics in config does not match the one in stored in d.")
                d['config'] = config

            if d['level'] >= 1:
                for k in ['struct', 'hfs', 'mfs']:
                    d[k] = _convert_lists_to_tuples(d[k])
                if not isinstance(d['config'], _config):
                    d['config'] = make_config(**d['config'])
                d['hfs'] = tuple(_Fusion.from_dict(hf) for hf in d['hfs'])
                d['struct'] = _struct.from_dict(d['struct'])

            if d['level'] >= 2 or config is not None:
                dtype = d['config'].default_dtype
                if hasattr(d['data'], 'dtype'):
                    if 'complex128' in str(d['data'].dtype):
                        dtype = 'complex128'
                    if 'float64' in str(d['data'].dtype):
                        dtype = 'float64'
                d['data'] = d['config'].backend.to_tensor(d['data'], dtype=dtype, device=d['config'].default_device)
            assert d['config'].backend.get_size(d['data']) == d['size'], "Sanity check. Stored sizes does not match data size."

            return cls(**d)
        raise YastnError(f"Tensor.to_dict with dict_ver = {d['dict_ver']} not supported")

    @property
    def trans(self) -> Sequence[int]:
        r"""Return the transpose mapping between logical legs and data-space legs."""
        return self._trans

    @property
    def s(self) -> Sequence[int]:
        r"""
        Return the signature of the tensor's effective legs.

        Legs (spaces) fused together by :meth:`yastn.Tensor.fuse` are treated as a single leg.
        The signature of each fused leg is given by the first native leg in the fused space.
        """
        inds, n = [], 0
        for mf in self.mfs:
            inds.append(self.trans[n])
            n += mf[0]
        return tuple(self.struct.legs[ind].s for ind in inds)

    @property
    def s_n(self) -> Sequence[int]:
        r"""
        Return the signature of the tensor's native legs.

        This includes legs (spaces) which have been fused together
        by :meth:`yastn.fuse_legs` using ``mode='meta'``.
        """
        return tuple(self.struct.legs[ind].s for ind in self.trans)

    @property
    def n(self) -> Sequence[int]:
        r"""
        Return the total charge of the tensor.

        In case of direct product of abelian symmetries,
        total charge for each symmetry, accumulated in a tuple.
        """
        return self.get_tensor_charge()

    @property
    def ndim(self) -> int:
        r"""
        Return the effective rank of the tensor.

        Legs (spaces) fused together by :meth:`yastn.fuse_legs` are treated as single leg.
        """
        return len(self.mfs)

    @property
    def ndim_n(self) -> int:
        r"""
        Return the native rank of the tensor.

        It distinguishes legs (spaces) which were fused
        by :meth:`yastn.fuse_legs` using ``mode='meta'``.
        """
        return len(self.struct.legs)

    @property
    def isdiag(self) -> bool:
        """Return ``True`` if the tensor is diagonal."""
        return self.struct.isdiag

    @property
    def requires_grad(self) -> bool:
        """Return ``True`` if the tensor data have autograd enabled."""
        return requires_grad(self)

    @property
    def size(self) -> int:
        """Return the total number of elements in all non-empty blocks of the tensor."""
        return self.config.backend.get_size(self._data)

    @property
    def device(self) -> str:
        """Return the name of the device on which the data resides."""
        return self.config.backend.get_device(self._data)

    @property
    def dtype(self) -> 'numpy.dtype' | 'torch.dtype':
        """Return the data type used by the backend for the tensor data."""
        return self.config.backend.get_dtype(self._data)

    @property
    def yastn_dtype(self) -> str:
        """
        Return string representing data dtype, e.g., ``'complex128'``, ``'float64'``, ``'complex64'``, ``'float32'``. ``'bool'``.
        """
        return self.config.backend.get_yastn_dtype(self._data)

    @property
    def data(self) -> 'numpy.array' | 'torch.tensor':
        """Return the underlying 1D array storing the tensor elements."""
        return self._data

    @property
    def T(self) -> Tensor:
        r""" Same as :meth:`self.transpose()<yastn.transpose>`. """
        return self.transpose()

    @property
    def H(self) -> Tensor:
        r""" Same as :meth:`self.T.conj()`, i.e., transpose and conjugate. """
        return self.transpose().conj()

    @property
    def shape(self) -> tuple[int]:
        return self.get_shape()

    @property
    def nblocks(self) -> int:
        bl = get_blocks(self.config.sym, self.struct)
        return bl.nblocks

def _convert_lists_to_tuples(nested_iterable):
    if isinstance(nested_iterable, list):
        return tuple( _convert_lists_to_tuples(v) if isinstance(v, (list, tuple, set, dict)) else v for v in nested_iterable)
    elif isinstance(nested_iterable, dict):
        return {k: (_convert_lists_to_tuples(v) if isinstance(v, (list, tuple, set, dict)) else v) for k, v in nested_iterable.items()}
    elif isinstance(nested_iterable, (tuple, set)):
        return type(nested_iterable)(_convert_lists_to_tuples(v) if isinstance(v, (list, tuple, set, dict)) else v for v in nested_iterable)
    else:
        return nested_iterable
