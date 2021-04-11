r"""
Yet another symmetric tensor

This class defines generic arbitrary-rank tensor supporting abelian symmetries.
In principle, any number of symmetries can be used (including no symmetries).

An instance of a Tensor is specified by a list of blocks (dense tensors) labeled by symmetries' charges on each leg.
"""

import numpy as np
from ._auxliary import _struct, _config
from ._initialize import *
from .krylov import *
from .linalg import *
from ._testing import *
from ._contractions import *
from ._single import *
from ._output import *

from . import _initialize
from . import krylov
from . import linalg
from . import _testing
from . import _contractions
from . import _single
from . import _output
__all__ = ['Tensor', 'linalg']
__all__.extend(_initialize.__all__)
__all__.extend(krylov.__all__)
__all__.extend(linalg.__all__)
__all__.extend(_testing.__all__)
__all__.extend(_contractions.__all__)
__all__.extend(_single.__all__)
__all__.extend(_output.__all__)

class Tensor:
    """ Class defining a tensor with abelian symmetries, and operations on such tensor(s). """

    def __init__(self, config=None, s=(), n=None, isdiag=False, **kwargs):
        self.config = config if isinstance(config, _config) else _config(**{a: getattr(config, a) for a in _config._fields if hasattr(config, a)})
        self.isdiag = isdiag
        self.nlegs = 1 if isinstance(s, int) else len(s)  # number of native legs
        self.s = np.array(s, dtype=int).reshape(self.nlegs)
        self.n = np.zeros(self.config.sym.nsym, dtype=int) if n is None else np.array(n, dtype=int).reshape(self.config.sym.nsym)
        if self.isdiag:
            if len(self.s) == 0:
                self.s = np.array([1, -1], dtype=int)
                self.nlegs = 2
            if not np.sum(self.s) == 0:
                raise YastError("Signature should be (-1, 1) or (1, -1) in diagonal tensor")
            if not np.sum(np.abs(self.n)) == 0:
                raise YastError("Tensor charge should be 0 in diagonal tensor")
            if not self.nlegs == 2:
                raise YastError("Diagonal tensor should have ndim == 2")
        self.A = {}  # dictionary of blocks
        # (meta) fusion tree for each leg: (encodes number of fused legs e.g. 5 2 1 1 3 1 2 1 1 -- 5 legs fused and history); is immutable
        self.meta_fusion = tuple(kwargs['meta_fusion']) if ('meta_fusion' in kwargs and kwargs['meta_fusion'] is not None) else ((1,),) * self.nlegs
        self.mlegs = len(self.meta_fusion)  # number of meta legs
        self.struct = _struct((), (), tuple(self.s), tuple(self.n))

    from ._initialize import set_block, fill_tensor, copy_empty
    from .linalg import norm, norm_diff, svd, svd_lowrank, eigh, qr
    from ._contractions import tensordot, vdot, trace, swap_gate
    from ._single import conj, conj_blocks, flip_signature, transpose, moveaxis, diag, abs, sqrt, rsqrt, reciprocal, exp, __add__, __sub__, __mul__, __rmul__, apxb, __truediv__, __pow__
    from ._single import copy, clone, detach, to, real, imag, fuse_legs, unfuse_legs
    from ._output import export_to_dict, compress_to_1d, show_properties, __str__, print_blocks, is_complex, get_blocks_charges, get_leg_charges_and_dims, zero_of_dtype, item, __getitem__
    from ._output import get_blocks_shapes, get_leg_fusion, get_leg_structure, get_ndim, get_shape, get_signature, get_size, get_tensor_charge, to_dense, to_nonsymmetric, to_number, to_numpy, to_raw_tensor
    from ._testing import is_consistent, are_independent
    from ._auxliary import update_struct
