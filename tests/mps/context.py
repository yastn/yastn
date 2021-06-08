try:
    import yast
    import yamps
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), './../..'))  # folder with yast
    import yast
    import yamps
try:
    import config_dense
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), './../configs'))  # folder with configs for tests
    import config_dense
import config_Z2
import config_U1
import config_U1_fermionic
import config_Z2_U1
