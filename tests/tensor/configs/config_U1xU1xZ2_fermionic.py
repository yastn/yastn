import yast.backend.backend_np as backend
from .syms import sym_U1xU1xZ2 as sym
fermionic = (False, False, True) # only last channel is fermionic
default_device = 'cpu'
default_dtype = 'float64'
