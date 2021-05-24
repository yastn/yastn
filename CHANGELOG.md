4-5-2021 
- config is now assumed to contain default_dtype and default_device. Property `device` is no
  longer part of the config
- Tensor has a new property `device`. All blocks of a given tensor must reside on the same device.
  This might be changed in the future.
- plain Tensor constructor accepts `device` argument, which overrides config.default_device
- tensor creation ops zeros, rand, randR, randC, ones, eye now accepts `dtype` and `device`
  arguments which override the defaults given in config
- functions `fill_tensor`, `set_block`, and `_set_block` now accept dtype argument overriding
  default specified in config.default_dtype
- functions `set_block` and `_set_block` now accept `device` argument. The `device` argument
  must agree with the `Tensor.device` property
- new auxiliary function `unique_dtype` has been introduced. If all blocks of a given tensor
  share the same dtype it is returned by the function. Otherwise, it returns `False`.
- BUGFIX: tensordot called with diagonal tensor as one of the arguments did not update both
  `Tensor.s` and `Tensor.struct.s` signatures. 

18-5-2021
- signature `s` and tensor charge `n` are now a part fo `struct`
- T.s and T.n return `s` and `n` as property.
- struct can be provided in Tensor(), and this overrides `s` and `n` (convinient for internal operations within module)
- There is no property `Tensor.device`
- `device` is in config (because config is inherited by new tensors).
  It cen be overriden when initializing Tensor(), e.g. in initialization functions, or using `Tensor.to_device`
- `set_blocks` and `_set_blocks` do not accept argument `device`
- new function get_device (of a block) in the backend
- new test in function is_consistent, that check if all blocks are on the same device, 
  and if it matches config.device (this can be unwanted test in numpy ...)
- flipping signature of diag tensor in tensordot is commented out (moving toward general signature of diagonal tensor)
