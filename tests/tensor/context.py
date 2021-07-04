try:
    import yast
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), './../..'))  # folder with yast
    import yast

from . import config_dense
from . import config_Z2
from . import config_U1
from . import config_U1_fermionic
from . import config_Z2_U1
