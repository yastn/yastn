r"""
Yet another symmetric tensor

This class defines generic arbitrary-rank tensor supporting abelian symmetries.
In principle, any number of symmetries can be used, including dense tensor with no symmetries.

An instance of a Tensor is specified by a list of blocks (dense tensors) labeled by symmetries' charges on each leg.
"""

from ._auxliary import _struct, _config
from ._merging import _Fusion
from ._tests import YastError
from ._tests import *
from ._control_lru import *
from ._contractions import *
from ._output import *
from ._single import *
from ._algebra import *
from ._merging import *
from .linalg import *
from ._legs import *
from . import _tests
from . import _control_lru
from . import _contractions
from . import _output
from . import _single
from . import _algebra
from . import linalg
from . import _merging
from . import _legs
__all__ = ['Tensor', 'linalg', 'YastError']
__all__.extend(linalg.__all__)
__all__.extend(_tests.__all__)
__all__.extend(_control_lru.__all__)
__all__.extend(_contractions.__all__)
__all__.extend(_single.__all__)
__all__.extend(_algebra.__all__)
__all__.extend(_output.__all__)
__all__.extend(_merging.__all__)
__all__.extend(_legs.__all__)


class Tensor:
    # Class defining a tensor with abelian symmetries, and operations on such tensor(s).

    def __init__(self, config=None, s=(), n=None, isdiag=False, **kwargs):
        r"""
        Initialize empty (no allocated blocks) YAST tensor

        Parameters
        ----------
            config : module, types.SimpleNamespace, or typing.NamedTuple
                :ref:`YAST configuration <tensor/configuration:yast configuration>`
            s : tuple
                a signature of tensor. Also determines the number of legs
            n : int or tuple
                total charge of the tensor. In case of direct product of several
                abelian symmetries `n` is tuple with total charge for each individual
                symmetry
            isdiag : bool
                distinguish diagonal tensor as a special case of a tensor
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
                n = (0,) * self.config.sym.NSYM if n is None else (n,)
            if len(n) != self.config.sym.NSYM:
                raise YastError("n does not match the number of symmetry sectors")
            if isdiag:
                if len(s) == 0:
                    s = (1, -1)  # default
                if s not in ((-1, 1), (1, -1)):
                    raise YastError("Diagonal tensor should have s equal (1, -1) or (-1, 1)")
                if any(x != 0 for x in n):
                    raise YastError("Tensor charge of a diagonal tensor should be 0")
            self.struct = _struct(s=s, n=n, diag=isdiag)

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
    from ._initialize import set_block, fill_tensor, __setitem__
    from .linalg import norm, svd, svd_with_truncation, eigh, eigh_with_truncation, qr
    from ._contractions import tensordot, __matmul__, vdot, trace, swap_gate, broadcast, apply_mask
    from ._algebra import __add__, __sub__, __mul__, __rmul__, apxb, __truediv__, __pow__, __lt__, __gt__, __le__, __ge__
    from ._algebra import __abs__, real, imag, sqrt, rsqrt, reciprocal, exp
    from ._single import conj, conj_blocks, flip_signature, transpose, moveaxis, move_leg, diag, grad
    from ._single import copy, clone, detach, to, requires_grad_, remove_zero_blocks, add_leg, remove_leg
    from ._output import show_properties, __str__, print_blocks_shape, is_complex
    from ._output import get_blocks_charge, get_blocks_shape, get_leg_charges_and_dims, get_leg_structure, get_legs
    from ._output import zero_of_dtype, item, __getitem__
    from ._output import get_leg_fusion, get_shape, get_signature, get_dtype
    from ._output import get_tensor_charge, get_rank
    from ._output import to_number, to_dense, to_numpy, to_raw_tensor, to_nonsymmetric
    from ._output import save_to_hdf5, save_to_dict, compress_to_1d
    from ._tests import is_consistent, are_independent
    from ._merging import fuse_legs, unfuse_legs, fuse_meta_to_hard

    def _replace(self, **kwargs):
        """ Creates a shallow copy replacing fields specified in kwargs """
        for arg in ('config', 'struct', 'mfs', 'hfs', 'data'):
            if arg not in kwargs:
                kwargs[arg] = getattr(self, arg)
        return Tensor(**kwargs)

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
        for mf in self.mfs:
            inds.append(n)
            n += mf[0]
        return tuple(self.struct.s[ind] for ind in inds)

    @property
    def s_n(self):
        """
        Returns
        -------
        s_n : tuple(int)
            signature of tensor's native legs. This includes legs (spaces) which have been
            fused together by :meth:`yast.Tensor.fuse` using mode=`meta`.
        """
        return self.struct.s

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
    def ndim(self):
        """
        Returns
        -------
        ndim : int
            effective rank of the tensor. Legs (spaces) fused together by :meth:`yast.Tensor.fuse`
            are treated as single leg.
        """
        return len(self.mfs)

    @property
    def ndim_n(self):
        """
        Returns
        -------
        ndim_n : int
            native rank of the tensor. This includes legs (spaces) which have been
            fused together by :meth:`yast.Tensor.fuse` using mode=`meta`.
        """
        return len(self.struct.s)

    @property
    def isdiag(self):
        """
        Returns
        -------
        isdiag : bool
            ``True`` if the tensor is diagonal.
        """
        return self.struct.diag

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
        return self.config.backend.get_size(self._data)

    @property
    def device(self):
        """
        Returns
        -------
        device : str
            name of device on which the data reside
        """
        return self.config.backend.get_device(self._data)

    @property
    def dtype(self):
        """
        Returns
        -------
        dtype :
            dtype of tensor data used by the backend
        """
        return self.config.backend.get_dtype(self._data)

    @property
    def yast_dtype(self):
        """
        Returns
        -------
        dtype : str
            'complex128' if tensor data are complex else 'float64'
        """
        return 'complex128' if self.config.backend.is_complex(self._data) else 'float64'

    @property
    def data(self):
        """
        Returns
        -------
        data : backend tensor type
            underlying 1D-array storing the elements of the tensor 
        """
        return self._data
