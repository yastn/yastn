Inspecting YAST tensors
=======================

Get information about tensor's structure and properties
-------------------------------------------------------

.. autoclass:: yast.Tensor
   :exclude-members: __init__, __new__
   :members: s, s_n, n, ndim, ndim_n, isdiag, requires_grad, size,
             show_properties, print_blocks_shape, is_complex, get_rank,
             get_tensor_charge, get_signature, get_leg_structure, get_legs, get_blocks_charge,
             get_blocks_shape, get_shape, get_dtype
