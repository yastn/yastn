""" configuration of yast tensor """
import yast.backend.backend_np as backend  # pylint: disable=unused-import
import yast.sym.sym_Z2 as sym  # pylint: disable=unused-import
fermionic = (True)
default_dtype = 'float64'
default_device = 'cpu'
default_fusion = 'hard'