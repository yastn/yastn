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

__all__ = []
__all__.extend(core.__all__)
__all__.extend(initialize.__all__)
__all__.extend(utils.__all__)
__all__.extend(krylov.__all__)
__all__.append('linalg')
__all__.extend(linalg.__all__)
