4-5-2020 
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
