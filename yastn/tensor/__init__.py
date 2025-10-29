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
from ._auxliary import _struct, _config
from ._merging import _Fusion, _slc
from ._tests import YastnError
from ._tests import *
from ._control_lru import *
from ._contractions import *
from ._output import *
from ._single import *
from ._algebra import *
from ._merging import *
from .linalg import *
from ._legs import *
from ._initialize import *
from . import _tests
from . import _control_lru
from . import _contractions
from . import _output
from . import _single
from . import _algebra
from . import linalg
from . import _merging
from . import _legs
from . import _initialize
__all__ = ['Tensor', 'linalg', 'YastnError']
__all__.extend(linalg.__all__)
__all__.extend(_tests.__all__)
__all__.extend(_control_lru.__all__)
__all__.extend(_contractions.__all__)
__all__.extend(_single.__all__)
__all__.extend(_algebra.__all__)
__all__.extend(_output.__all__)
__all__.extend(_merging.__all__)
__all__.extend(_legs.__all__)
__all__.extend(_initialize.__all__)


class Tensor:
    # Class defining a tensor with abelian symmetries, and operations on such tensor(s).

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
            self._data = kwargs['data']  # 1d container for tensor data
        else:
            dev = kwargs['device'] if 'device' in kwargs else self.config.default_device
            dty = kwargs['dtype'] if 'dtype' in kwargs else self.config.default_dtype
            self._data = self.config.backend.zeros((0,), dtype=dty, device=dev)

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
            self.struct = _struct(s=s, n=n, diag=bool(isdiag))

        self.slices = kwargs['slices'] if 'slices' in kwargs else ()

        # fusion tree for each leg: encodes number of fused legs e.g. 5 2 1 1 3 1 2 1 1 = [[1, 1], [1, [1, 1]]]
        try:
            self.mfs = tuple(kwargs['mfs'])
        except (KeyError, TypeError):
            self.mfs = ((1,),) * len(self.struct.s)
        try:
            self.hfs = tuple(kwargs['hfs'])
        except (KeyError, TypeError):
            self.hfs = tuple(_Fusion(s=(x,)) for x in self.struct.s)

    # pylint: disable=C0415
    from ._initialize import set_block, _fill_tensor, __setitem__
    from .linalg import norm, svd, svd_with_truncation, eig, eigh, eigh_with_truncation, qr, truncation_mask
    from ._contractions import tensordot, __matmul__, vdot, trace, swap_gate, broadcast, apply_mask
    from ._algebra import __add__, __sub__, __mul__, __rmul__, __array_ufunc__, __neg__, add
    from ._algebra import __lt__, __gt__, __le__, __ge__, __truediv__, __pow__, allclose
    from ._algebra import __abs__, real, imag, sqrt, rsqrt, reciprocal, exp, bitwise_not
    from ._single import conj, conj_blocks, flip_signature, flip_charges, switch_signature, transpose, moveaxis, move_leg, diag, grad
    from ._single import copy, clone, detach, detach_, to, requires_grad_, remove_zero_blocks, add_leg, remove_leg, drop_leg_history
    from ._output import print_properties, __str__, __repr__, print_blocks_shape, is_complex
    from ._output import get_blocks_charge, get_blocks_shape, get_legs
    from ._output import zero_of_dtype, item, __getitem__, __contains__
    from ._output import get_shape, get_signature, get_dtype
    from ._output import get_tensor_charge, get_rank
    from ._output import to_number, to_dense, to_numpy, to_raw_tensor, to_nonsymmetric
    from ._output import save_to_hdf5, save_to_dict, compress_to_1d, to_dict
    from ._tests import is_consistent, are_independent
    from ._merging import fuse_legs, unfuse_legs, fuse_meta_to_hard
    from ._krylov import expand_krylov_space

    def _replace(self, **kwargs) -> Tensor:
        """ Creates a shallow copy replacing fields specified in kwargs. """
        for arg in ('config', 'struct', 'mfs', 'hfs', 'data', 'slices'):
            if arg not in kwargs:
                kwargs[arg] = getattr(self, arg)
        return Tensor(**kwargs)

    @classmethod
    def from_dict(cls, d: dict, config:_config | None=None):
        #
        d = d.copy()
        if d['type'] != 'Tensor':
            raise YastnError(f"{cls.__name__} does not match d['type'] == {d['type']}")

        if 'level' not in d:
            d['level'] = 2

        if d['level'] >= 1:
            for k in ['struct', 'slices', 'hfs', 'mfs']:
                d[k] = _convert_lists_to_tuples(d[k])

            d['config'] = make_config(**d['config'])
            d['hfs'] = tuple(_Fusion(**hf) for hf in d['hfs'])
            d['struct'] = _struct(**d['struct'])
            d['slices'] = tuple(_slc(*x) for x in d['slices'])

        if config is not None:
            if d['config'].sym.SYM_ID != config.sym.SYM_ID:
                raise YastnError("Symmetry rule in config does not match the one in stored in d.")
            if d['config'].fermionic != d['config'].fermionic:
                raise YastnError("Fermionic statistics in config does not match the one in stored in d.")
            d['config'] = config

        if d['level'] >= 2:
            dtype = d['data'].dtype.name if hasattr(d['data'], 'dtype') else d['config'].default_dtype
            d['data'] = d['config'].backend.to_tensor(d['data'], dtype=dtype, device=d['config'].default_device)

        return cls(**d)


    @property
    def s(self) -> Sequence[int]:
        r"""
        Signature of tensor's effective legs.

        Legs (spaces) fused together by :meth:`yastn.Tensor.fuse` are treated as a single leg.
        The signature of each fused leg is given by the first native leg in the fused space.
        """
        inds, n = [], 0
        for mf in self.mfs:
            inds.append(n)
            n += mf[0]
        return tuple(self.struct.s[ind] for ind in inds)

    @property
    def s_n(self) -> Sequence[int]:
        r"""
        Signature of tensor's native legs.

        This includes legs (spaces) which have been fused together
        by :meth:`yastn.fuse_legs` using ``mode='meta'``.
        """
        return self.struct.s

    @property
    def n(self) -> Sequence[int]:
        r"""
        Total charge of the tensor.

        In case of direct product of abelian symmetries,
        total charge for each symmetry, accummulated in a tuple.
        """
        return self.struct.n

    @property
    def ndim(self) -> int:
        r"""
        Effective rank of the tensor.

        Legs (spaces) fused together by :meth:`yastn.fuse_legs` are treated as single leg.
        """
        return len(self.mfs)

    @property
    def ndim_n(self) -> int:
        r"""
        Native rank of the tensor.

        It distinguishes legs (spaces) which were fused
        by :meth:`yastn.fuse_legs` using ``mode='meta'``.
        """
        return len(self.struct.s)

    @property
    def isdiag(self) -> bool:
        """ Return ``True`` if the tensor is diagonal. """
        return self.struct.diag

    @property
    def requires_grad(self) -> bool:
        """ Return ``True`` if tensor data have autograd enabled. """
        return requires_grad(self)

    @property
    def size(self) -> int:
        """ Total number of elements in all non-empty blocks of the tensor. """
        return self.struct.size

    @property
    def device(self) -> str:
        """ Name of device on which the data resides. """
        return self.config.backend.get_device(self._data)

    @property
    def dtype(self) -> 'numpy.dtype' | 'torch.dtype':
        """ Datatype ``dtype`` of tensor data used by the ``backend``. """
        return self.config.backend.get_dtype(self._data)

    @property
    def yastn_dtype(self) -> str:
        """ Return :code:`'complex128'` if tensor data are complex else :code:`'float64'`. """
        return 'complex128' if self.config.backend.is_complex(self._data) else 'float64'

    @property
    def data(self) -> 'numpy.array' | 'torch.tensor':
        """ Return underlying 1D-array storing the elements of the tensor. """
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

def _convert_lists_to_tuples(nested_iterable):
    if isinstance(nested_iterable, list):
        return tuple( _convert_lists_to_tuples(v) if isinstance(v, (list, tuple, set, dict)) else v for v in nested_iterable)
    elif isinstance(nested_iterable, dict):
        return {k: (_convert_lists_to_tuples(v) if isinstance(v, (list, tuple, set, dict)) else v) for k, v in nested_iterable.items()}
    elif isinstance(nested_iterable, (tuple, set)):
        return type(nested_iterable)(_convert_lists_to_tuples(v) if isinstance(v, (list, tuple, set, dict)) else v for v in nested_iterable)
    else:
        return nested_iterable
