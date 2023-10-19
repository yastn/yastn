""" configuration of yastn tensor """
import yastn.backend.backend_np as backend  # pylint: disable=unused-import
import yastn.sym.sym_Z2 as sym  # pylint: disable=unused-import
default_dtype = 'float64'
default_device = 'cpu'
default_fusion = 'hard'