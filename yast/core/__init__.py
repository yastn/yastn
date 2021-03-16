""" Consolidation of tensor functions. """
from .core import *
from . import core
from .initialize import *
from . import initialize
from .utils import *
from . import utils
from .krylov import *
from . import krylov
from . import linalg

__all__ = ['copy', 'clone', 'detach', 'export_to_dict','compress_to_1d',
           'norm', 'max_abs', 'norm_diff',
           'apxb', 'conj', 'conj_blocks', 'flip_signature',
           'transpose', 'moveaxis', 'diag',
           'exp', 'sqrt', 'rsqrt', 'reciprocal',
           'tensordot', 'vdot', 'trace',
           'fuse_legs', 'unfuse_legs']

for name in __all__:
    globals()[name] =getattr(Tensor, name)

__all__.extend(core.__all__)
__all__.extend(initialize.__all__)
__all__.extend(utils.__all__)
__all__.extend(krylov.__all__)
__all__.append('linalg')
