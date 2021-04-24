r"""
Yet another symmetric tensor

This class defines generic arbitrary-rank tensor supporting abelian symmetries.
In principle, any number of symmetries can be used (including no symmetries).

An instance of a Tensor is specified by a list of blocks (dense tensors) labeled by symmetries' charges on each leg.
"""

import numpy as np
from ._auxliary import _struct, _config, YastError
from ._auxliary import *
from ._contractions import *
from ._initialize import *
from ._output import *
from ._single import *
from ._krylov import *
from .linalg import *
from . import _auxliary
from . import _contractions
from . import _initialize
from . import _output
from . import _single
from . import _krylov
from . import linalg
__all__ = ['Tensor', 'linalg', 'YastError']
__all__.extend(_initialize.__all__)
__all__.extend(_krylov.__all__)
__all__.extend(linalg.__all__)
__all__.extend(_auxliary.__all__)
__all__.extend(_contractions.__all__)
__all__.extend(_single.__all__)
__all__.extend(_output.__all__)


class Tensor:
    """ Class defining a tensor with abelian symmetries, and operations on such tensor(s). """

    def __init__(self, config=None, s=(), n=None, isdiag=False, **kwargs):
        self.config = config if isinstance(config, _config) else \
                      _config(**{a: getattr(config, a) for a in _config._fields if hasattr(config, a)})
        self.isdiag = isdiag
        try:
            self.nlegs = len(s)  # number of native legs
        except TypeError:
            self.nlegs = 1
        self.s = np.array(s, dtype=int).reshape(self.nlegs)
        self.n = np.zeros(self.config.sym.NSYM, dtype=int) if n is None else \
            np.array(n, dtype=int).reshape(self.config.sym.NSYM)
        if self.isdiag:
            if len(self.s) == 0:
                self.s = np.array([1, -1], dtype=int)
                self.nlegs = 2
            if np.sum(self.s) != 0:
                raise YastError("Signature should be (-1, 1) or (1, -1) in diagonal tensor")
            if np.sum(np.abs(self.n)) != 0:
                raise YastError("Tensor charge should be 0 in diagonal tensor")
            if self.nlegs != 2:
                raise YastError("Diagonal tensor should have ndim == 2")
        self.A = {}  # dictionary of blocks
        # immutable fusion tree for each leg: encodes number of fused legs e.g. 5 2 1 1 3 1 2 1 1 -- 5 legs fused and history
        try:
            self.meta_fusion = tuple(kwargs['meta_fusion'])
        except (KeyError, TypeError):
            self.meta_fusion = ((1,),) * self.nlegs
        self.mlegs = len(self.meta_fusion)  # number of meta legs
        self.struct = _struct((), (), tuple(self.s), tuple(self.n))

    from ._initialize import set_block, fill_tensor, copy_empty
    from .linalg import norm, norm_diff, svd, svd_lowrank, eigh, qr, eigs_anm, eigh_anm, eig_anm, expm_anm
    from ._contractions import tensordot, vdot, trace, swap_gate
    from ._single import conj, conj_blocks, flip_signature, transpose, moveaxis, diag, abs, sqrt, rsqrt, reciprocal, exp
    from ._single import __add__, __sub__, __mul__, __rmul__, apxb, __truediv__, __pow__
    from ._single import copy, clone, detach, to, real, imag, fuse_legs, unfuse_legs
    from ._output import export_to_dict, compress_to_1d, show_properties, __str__, print_blocks, is_complex
    from ._output import get_blocks_charges, get_leg_charges_and_dims, zero_of_dtype, item, __getitem__
    from ._output import get_blocks_shapes, get_leg_fusion, get_leg_structure, get_ndim, get_shape, get_signature
    from ._output import get_size, get_tensor_charge, to_dense, to_nonsymmetric, to_number, to_numpy, to_raw_tensor
    from ._auxliary import update_struct, is_consistent, are_independent
