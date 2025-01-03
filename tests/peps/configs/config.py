""" example of config; here used by pytest to modify and inject backend and device. """
# import yastn.backend.backend_np as backend
import yastn.backend.backend_torch as backend
import yastn.sym.sym_none as sym
default_device = 'cpu'
default_dtype = 'float64'
fermionic = False
default_fusion = 'hard'
force_fusion: str = None
