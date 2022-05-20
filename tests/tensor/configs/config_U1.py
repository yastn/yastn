# Specify ``backend`` providing Linear algebra and base dense tensor
# Currently support backends 
# 
# * Numpy as yast.backend.backend_np 
# * PyTorch as yast.backend.backend_torch
import yast.backend.backend_np as backend

# Specify abelian symmetry as ``sym``. To see how YAST defines symmetries,
# see :class:`yast.sym.sym_abelian`.
import yast.sym.sym_U1 as sym

# Base tensors can be stored on various devices as supported by ``backend`` 
# 
# * NumPy supports only 'cpu' device
# * PyTorch supports multiple devices, see 
#   https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
#
# If not specified, the default device is 'cpu'
default_device: str = 'cpu'

# Default data type (dtype) of YAST tensors. Supported options are: 'float64', 'complex128'.
# If not specified, the default dtype is 'float64'
default_dtype: str = 'float64'

# Specify behavior of swap_gate function, allowing to introduce fermionic symmetries.
# Allowed values:
# ``False``, ``True``, or a tuple ``(True, False, ...)`` with one bool for each component 
# charge vector i.e. of length sym.NSYM
# 
# If note specified, the default value is False
fermionic: tuple = False

# Specify default strategy to handle leg fusion: 'hard' or 'meta'. See yast.tensor.fuse_legs 
# for details.
#
# If not specified, the default fusion is 'hard' 
default_fusion: str = 'hard'

# Overrides fusion strategy provided in yast.tensor.fuse_legs
# Default is None
force_fusion: str = None

# Specify policy used during execution of tensordot: 'merge', 'hybrid, or 'direct'.
# See yast.tensor.tensordot for details.
# 
# If not specified, the default is 'hybrid'
default_tensordot: str = 'hybrid'

# overrides tensordot policy that can be provided in yast.tensordot
# Default is None
force_tensordot: str = None
