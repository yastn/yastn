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
from .linalg import *
from . import _auxliary
from . import _contractions
from . import _initialize
from . import _output
from . import _single
from . import linalg
__all__ = ['Tensor', 'linalg', 'YastError']
__all__.extend(_initialize.__all__)
__all__.extend(linalg.__all__)
__all__.extend(_auxliary.__all__)
__all__.extend(_contractions.__all__)
__all__.extend(_single.__all__)
__all__.extend(_output.__all__)


class Tensor:
    """
    Class defining a tensor with abelian symmetries, and operations on such tensor(s).

    Parameters
    ----------
        config : module
            imported module containing configuration
        s : tuple
            a signature of tensor. Also determines the number of legs
        n : int
            total charge of the tensor
    """
    def __init__(self, config=None, s=(), n=None, isdiag=False, **kwargs):
        if isinstance(config, _config):
            self.config = config
        else:
            temp_config = {a: getattr(config, a) for a in _config._fields if hasattr(config, a)}
            if 'device' not in temp_config:
                temp_config['device'] = config.default_device
            self.config = _config(**temp_config)
        if 'device' in kwargs and kwargs['device'] != self.config.device:
            self.config._replace(device= kwargs['device'])

        self.isdiag = isdiag
        self.A = {}  # dictionary of blocks

        try:
            self.struct = kwargs['struct']
            self.nlegs = len(self.struct.s)  # number of native legs
        except KeyError:
            try:
                self.nlegs = len(s)  # number of native legs
                s = tuple(s)
            except TypeError:
                self.nlegs = 1
                s = (s,)
            try:
                n = tuple(n)
            except TypeError:
                n = (0,) * self.config.sym.NSYM if n is None else (n,)

            if len(n) != self.config.sym.NSYM:
                raise YastError('n does not match the number of symmetries')

            if self.isdiag:
                if len(s) == 0:
                    s = (1, -1)
                    self.nlegs = 2
                if sum(s) != 0:
                    raise YastError("Signature should be (-1, 1) or (1, -1) in diagonal tensor")
                if sum(abs(x) for x in n) != 0:
                    raise YastError("Tensor charge should be 0 in diagonal tensor")
                if self.nlegs != 2:
                    raise YastError("Diagonal tensor should have ndim == 2")
            self.struct = _struct(t=(), D=(), s=s, n=n)

        # immutable fusion tree for each leg: encodes number of fused legs e.g. 5 2 1 1 3 1 2 1 1 -- 5 legs fused and history
        try:
            self.meta_fusion = tuple(kwargs['meta_fusion'])
        except (KeyError, TypeError):
            self.meta_fusion = ((1,),) * self.nlegs
        self.mlegs = len(self.meta_fusion)  # number of meta legs

    from ._initialize import set_block, fill_tensor, copy_empty
    from .linalg import norm, norm_diff, svd, svd_lowrank, eigh, qr
    from ._contractions import tensordot, vdot, trace, swap_gate
    from ._single import conj, conj_blocks, flip_signature, transpose, moveaxis, diag, absolute, sqrt, rsqrt, reciprocal, exp
    from ._single import __add__, __sub__, __mul__, __rmul__, apxb, __truediv__, __pow__, remove_zero_blocks, add_leg
    from ._single import copy, clone, detach, to, requires_grad_, real, imag, fuse_legs, unfuse_legs
    from ._output import export_to_dict, compress_to_1d, show_properties, __str__, print_blocks, is_complex
    from ._output import get_blocks_charges, get_leg_charges_and_dims, zero_of_dtype, item, __getitem__
    from ._output import get_blocks_shapes, get_leg_fusion, get_leg_structure, get_ndim, get_shape, get_signature, unique_dtype
    from ._output import get_size, get_tensor_charge, to_dense, to_nonsymmetric, to_number, to_numpy, to_raw_tensor
    from ._auxliary import update_struct, is_consistent, are_independent
    abs = absolute

    @property
    def s(self):
        """ Return signature of the tensor as a tuple. """
        return self.struct.s

    @property
    def n(self):
        """ Return total charge of the tensor as a tuple. """
        return self.struct.n

    @property
    def requires_grad(self):
        return requires_grad(self)
