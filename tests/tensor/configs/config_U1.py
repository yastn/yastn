import yast.backend.backend_np as backend
import yast.sym.sym_U1 as sym

default_device: str = 'cpu'
default_dtype: str = 'float64'
fermionic: tuple = False
default_fusion: str = 'hard'
force_fusion: str = None
default_tensordot: str = 'hybrid'
force_tensordot: str = None