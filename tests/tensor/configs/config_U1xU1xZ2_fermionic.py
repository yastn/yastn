import yastn.backend.backend_np as backend
import yastn.sym.sym_U1xU1xZ2 as sym
fermionic = (False, False, True) # only last channel is fermionic
default_device = 'cpu'
default_dtype = 'float64'
default_fusion: str = 'hard'
force_fusion: str = None
