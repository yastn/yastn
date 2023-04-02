""" configuration of yastn tensor """
import yastn.backend.backend_np as backend  # pylint: disable=unused-import
import yastn.sym.sym_U1xU1xZ2 as sym  # pylint: disable=unused-import
fermionic = (False, False, True)
default_dtype = 'float64'
default_device = 'cpu'
default_fusion = 'hard'