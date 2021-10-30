r"""
Yet another symmetric tensor

This class defines generic arbitrary-rank tensor supporting abelian symmetries.
In principle, any number of symmetries can be used, including no symmetries.

An instance of a Tensor is specified by a list of blocks (dense tensors) labeled by symmetries' charges on each leg.
"""
import numpy as np
from ._auxliary import _struct, _config
from ._merging import _Fusion
from ._tests import YastError
from ._tests import *
from ._control_lru import *
from ._contractions import *
from ._initialize import *
from ._output import *
from ._single import *
from ._merging import *
from .linalg import *
from . import _tests
from . import _control_lru
from . import _contractions
from . import _initialize
from . import _output
from . import _single
from . import linalg
from . import _merging
__all__ = ['Tensor', 'linalg', 'YastError']
__all__.extend(_initialize.__all__)
__all__.extend(linalg.__all__)
__all__.extend(_tests.__all__)
__all__.extend(_control_lru.__all__)
__all__.extend(_contractions.__all__)
__all__.extend(_single.__all__)
__all__.extend(_output.__all__)
__all__.extend(_merging.__all__)


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
        """ init new tensor """
        if isinstance(config, _config):
            self.config = config
        else:
            temp_config = {a: getattr(config, a) for a in _config._fields if hasattr(config, a)}
            if 'device' not in temp_config:
                temp_config['device'] = config.default_device
            self.config = _config(**temp_config)
        if 'device' in kwargs and kwargs['device'] != self.config.device:
            self.config._replace(device=kwargs['device'])

        self._isdiag = isdiag
        self.A = {}  # dictionary of blocks

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
                n = (0,) * self.config.sym.NSYM if n is None else (n,)
            if len(n) != self.config.sym.NSYM:
                raise YastError('n does not match the number of symmetries')
            if self.isdiag:
                if len(s) == 0:
                    s = (1, -1)  # default
                if any(x != 0 for x in n):
                    raise YastError("Tensor charge of a diagonal tensor should be 0")
                if len(s) != 2:
                    raise YastError("Diagonal tensor should have ndim == 2")
            self.struct = _struct(t=(), D=(), s=s, n=n)

        # fusion tree for each leg: encodes number of fused legs e.g. 5 2 1 1 3 1 2 1 1 = [[1, 1], [1, [1, 1]]]
        try:
            self.meta_fusion = tuple(kwargs['meta_fusion'])
        except (KeyError, TypeError):
            self.meta_fusion = ((1,),) * len(self.struct.s)
        try:
            self.hard_fusion = tuple(kwargs['hard_fusion'])
        except (KeyError, TypeError):
            self.hard_fusion = tuple(_Fusion(s=(x,), ms=(-1 * x,)) for x in self.struct.s)

    # pylint: disable=C0415
    from ._initialize import set_block, fill_tensor
    from .linalg import norm, norm_diff, svd, svd_lowrank, eigh, qr
    from ._contractions import tensordot, vdot, trace, swap_gate
    from ._single import conj, conj_blocks, flip_signature, transpose, moveaxis, diag, absolute, sqrt, rsqrt, reciprocal, exp
    from ._single import __add__, __sub__, __mul__, __rmul__, apxb, __truediv__, __pow__, remove_zero_blocks, add_leg
    from ._single import copy, clone, detach, to, requires_grad_, real, imag
    from ._output import export_to_dict, compress_to_1d, show_properties, __str__, print_blocks, is_complex
    from ._output import get_blocks_charges, get_blocks_shapes, get_leg_charges_and_dims, get_leg_structure
    from ._output import zero_of_dtype, item, __getitem__
    from ._output import get_leg_fusion, get_ndim, get_shape, get_signature, unique_dtype
    from ._output import get_size, get_tensor_charge
    from ._output import to_number, to_dense, to_numpy, to_raw_tensor, to_nonsymmetric
    from ._tests import is_consistent, are_independent
    from ._auxliary import update_struct
    from ._merging import fuse_legs, unfuse_legs, fuse_meta_to_hard

    abs = absolute  # allow yest.abs(tensor)

    @property
    def s(self):
        """ Return signature of the tensor as nparray. """
        return np.array(self.struct.s, dtype=int)

    @property
    def n(self):
        """ Return total charge of the tensor as aparray. """
        return np.array(self.struct.n, dtype=int)

    @property
    def nlegs(self):
        """ Return the number of legs (native) """
        return len(self.struct.s)

    @property
    def mlegs(self):
        """ Return the number of meta legs (meta-fusion of native legs) """
        return len(self.meta_fusion)

    @property
    def isdiag(self):
        """ Return  """
        return self._isdiag

    @property
    def requires_grad(self):
        """ Return value of requires_grad """
        return requires_grad(self)
