import yast.backend.backend_np as backend
# Specify backend supporting underlaying tensor data
# Currently: yast.backend.backend_np or yast.backend.backend_torch
import yast.sym.sym_U1 as sym
# Import file specifying type of abelian symmetry as sym
fermionic: tuple = False
# Specify behavior of swap_gate function, allowing to introduce fermionic symmetries
# False, True, or a tuple (True, False, ...) with one boolen for each symmetry channel.
# Default is False
default_device = 'cpu'
# Depending on the backend, differt devices can be used.
# Default is 'cpu'
default_dtype = 'float64'
# Default data type 'float64' or 'complex128' used during initialization of the Tensor
# Default is 'float64'
default_fusion: str = 'hard'
# Specify default strategy to handle leg fusion: 'hard' or 'meta'
# Default is 'hard' (see yast.tensor.fuse_legs for details)
force_fusion: str = None
# Overrides fusion strategy provided in yast.tensor.fuse_legs
# Default is None
default_tensordot: str = 'hybrid'
# Specify policy used during exacution of tensordot: 'merge', 'hybrid, 'direct'
# Default is 'hybrid'
force_tensordot: str = None
# overrides tensordot policy that can be provided in yast.tensordot
# Default is None
