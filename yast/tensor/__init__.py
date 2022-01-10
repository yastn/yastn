r"""
Yet another symmetric tensor

This class defines generic arbitrary-rank tensor supporting abelian symmetries.
In principle, any number of symmetries can be used, including dense tensor with no symmetries.

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
from ._algebra import *
from ._merging import *
from .linalg import *
from . import _tests
from . import _control_lru
from . import _contractions
from . import _initialize
from . import _output
from . import _single
from . import _algebra
from . import linalg
from . import _merging
__all__ = ['Tensor', 'linalg', 'YastError']
__all__.extend(_initialize.__all__)
__all__.extend(linalg.__all__)
__all__.extend(_tests.__all__)
__all__.extend(_control_lru.__all__)
__all__.extend(_contractions.__all__)
__all__.extend(_single.__all__)
__all__.extend(_algebra.__all__)
__all__.extend(_output.__all__)
__all__.extend(_merging.__all__)


class Tensor:
    # Class defining a tensor with abelian symmetries, and operations on such tensor(s).

    def __init__(self, config=None, s=(), n=None, isdiag=False, **kwargs):
        r"""
        Initialize empty (no allocated blocks) YAST tensor

        Parameters
        ----------
            config : module
                imported module containing configuration
            s : tuple
                a signature of tensor. Also determines the number of legs
            n : int or tuple
                total charge of the tensor. In case of direct product of several
                abelian symmetries `n` is tuple with total charge for each individual
                symmetry
        """
        if isinstance(config, _config):
            self.config = config
        else:
            temp_config = {a: getattr(config, a) for a in _config._fields if hasattr(config, a)}
            if 'device' in kwargs:
                temp_config['device'] = kwargs['device']
            if 'dtype' in kwargs:
                temp_config['dtype'] = kwargs['dtype']
            if 'device' not in temp_config:
                temp_config['device'] = config.default_device
            if 'dtype' not in temp_config:
                temp_config['dtype'] = config.default_dtype
            self.config = _config(**temp_config)
        if 'device' in kwargs and kwargs['device'] != self.config.device:
            self.config._replace(device=kwargs['device'])
        if 'dtype' in kwargs and kwargs['dtype'] != self.config.dtype:
            self.config._replace(device=kwargs['dtype'])
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
                if s not in ((-1, 1), (1, -1)):
                    raise YastError("Diagonal tensor should have s = (1, -1) or (-1, 1)")
            self.struct = _struct(t=(), D=(), s=s, n=n)

        # fusion tree for each leg: encodes number of fused legs e.g. 5 2 1 1 3 1 2 1 1 = [[1, 1], [1, [1, 1]]]
        try:
            self.meta_fusion = tuple(kwargs['meta_fusion'])
        except (KeyError, TypeError):
            self.meta_fusion = ((1,),) * len(self.struct.s)
        try:
            self.hard_fusion = tuple(kwargs['hard_fusion'])
        except (KeyError, TypeError):
            self.hard_fusion = tuple(_Fusion(s=(x,)) for x in self.struct.s)

    # pylint: disable=C0415
    from ._initialize import set_block, fill_tensor
    from .linalg import norm, svd, svd_lowrank, eigh, qr
    from ._contractions import tensordot, __matmul__, vdot, trace, swap_gate, broadcast, mask
    from ._algebra import __add__, __sub__, __mul__, __rmul__, apxb, __truediv__, __pow__, __lt__, __gt__, __le__, __ge__
    from ._algebra import __abs__, real, imag, sqrt, rsqrt, reciprocal, exp
    from ._single import conj, conj_blocks, flip_signature, transpose, moveaxis, diag
    from ._single import copy, clone, detach, to, requires_grad_, remove_zero_blocks, add_axis, remove_axis
    from ._output import show_properties, __str__, print_blocks_shape, is_complex
    from ._output import get_blocks_charge, get_blocks_shape, get_leg_charges_and_dims, get_leg_structure
    from ._output import zero_of_dtype, item, __getitem__
    from ._output import get_leg_fusion, get_shape, get_signature, unique_dtype
    from ._output import get_tensor_charge, get_rank
    from ._output import to_number, to_dense, to_numpy, to_raw_tensor, to_nonsymmetric
    from ._output import save_to_hdf5, save_to_dict, compress_to_1d
    from ._tests import is_consistent, are_independent
    from ._merging import fuse_legs, unfuse_legs, fuse_meta_to_hard


    @property
    def s_n(self):
        """
        Returns
        -------
        s_n : tuple(int)
            signature of tensor's native legs. This includes legs (spaces) which have been
            fused together by :meth:`yast.Tensor.fuse`.
        """
        return self.struct.s

    @property
    def s(self):
        """
        Returns
        -------
        s : tuple(int)
            signature of tensor's effective legs. Legs (spaces) fused together
            by :meth:`yast.Tensor.fuse` are treated as single leg. The signature
            of each fused leg is given by the first native leg in the fused space.
        """
        inds, n = [], 0
        for mf in self.meta_fusion:
            inds.append(n)
            n += mf[0]
        return tuple(self.struct.s[ind] for ind in inds)

    @property
    def n(self):
        """
        Returns
        -------
        n : tuple(int)
            total charge of the tensor. In case of direct product of abelian symmetries, total
            charge for each symmetry, accummulated in a tuple.
        """
        return self.struct.n

    @property
    def ndim_n(self):
        """
        Returns
        -------
        ndim_n : int
            native rank of the tensor. This includes legs (spaces) which have been
            fused together by :meth:`yast.Tensor.fuse`.
        """
        return len(self.struct.s)

    @property
    def ndim(self):
        """
        Returns
        -------
        ndim : int
            effective rank of the tensor. Legs (spaces) fused together by :meth:`yast.Tensor.fuse`
            are treated as single leg.
        """
        return len(self.meta_fusion)

    @property
    def isdiag(self):
        """
        Returns
        -------
        isdiag : bool
            ``True`` if the tensor is diagonal.
        """
        return self._isdiag

    @property
    def requires_grad(self):
        """
        Returns
        -------
        requires_grad : bool
            ``True`` if any block of the tensor has autograd enabled
        """
        return requires_grad(self)

    @property
    def size(self):
        """
        Returns
        -------
        size : int
            total number of elements in all non-empty blocks of the tensor
        """
        Dset = np.array(self.struct.D, dtype=int).reshape((len(self.struct.D), len(self.struct.s)))
        return sum(np.prod(Dset, axis=1))
