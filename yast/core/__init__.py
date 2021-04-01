""" Consolidation of tensor functions. """
from .core import *
from . import core
from .initialize import *
from . import initialize
from .utils import *
from . import utils
from .krylov import *
from . import krylov
from .linalg import *
from . import linalg

__all__ = ['copy', 'clone', 'detach', 'export_to_dict','compress_to_1d',
            'apxb', 'conj', 'conj_blocks', 'flip_signature',
            'transpose', 'moveaxis', 'diag', 'swap_gate',
            'exp', 'sqrt', 'rsqrt', 'reciprocal', 'abs',
            'tensordot', 'vdot', 'trace',
            'fuse_legs', 'unfuse_legs']

for name in __all__:
    globals()[name] =getattr(Tensor, name)

__all__.extend(core.__all__)
__all__.extend(initialize.__all__)
__all__.extend(utils.__all__)
__all__.extend(krylov.__all__)
__all__.append('linalg')
__all__.extend(linalg.__all__)
