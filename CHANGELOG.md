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

24-04-2021
- change syntax of `linalg.expmv` and `linalg.eigs` with support in _krylov

27-5-2021
- `requires_grad_` function added to Tensor interface which allows to turn on operation tracking on all blocks
- property `requires_grad` added to Tensor which returns `True` if any of the blocks has operation tracking enabled
- fix passing dtype to `backend.to_tensor`. The latter always uses specified dtype (without try ... except ...)

04-06-2021
- new function `add_leg` which addes an auxliary leg with dimension 1, carrying a charge of the tensor (or part of it)

07-06-2021
- option `reverse` added to `to_dense()`, `to_numpy()` and `to_nonsymmetric()` functions allowing for reverse
  ordering of blocks. If `True`, the blocks are ordered by their charges in descending order.

08-06-2021
- refactoring of mps. Moved to folder yamps in the main catalogue.
  It is fully operational, but we still need to add some functionality.

04-07-2021
- fuse_legs has parameter mode = None, 'meta', 'hard', providing support for hard fusion
  None uses default that can be set in config.default_fusion (if not set, default_fusion='meta').
  config.force_fusion overrides mode set in the code
  hard-fusion applied on tensor that have been meta-fused first changes all (!) meta-fusions into hard fusions
- function `fuse_meta_to_hard()` changes all meta fusions into hard fusions; 
  and do nothing if there are no meta fusions
- hard fusion keeps the information about history. mismatches in hard-fusions are cought and 
  resolved in tensordot (todo: add such support for vdot, trace, norm_diff, _add__, ...)

07-07-2021
- `_leg_struct_truncation` accepts `tol_block` parameter, which truncates spectral values
  in each block individually relative to spectral value of largest magnitude within the block 

20-07-2021
- `ncon` numerates dangling legs with indices starting from 0 (or -0) towards negative.
  This allows to match python numbering conventions.
- handling hard-fusion mismatches in `__add__`, `__sub__`, `apxb`, `norm_diff`.
  In the first three also catches a possibility to introduce bond dimension mismatch in the result.

02-11-2021
- new funciton `broadcast(a, b, axis, conj)`. 
  Perform broadcasted multiplication of leg axis of tensor a by diagonal tensor b. 
  Only axis of tensor a is specified, following convention that diagonal tensor 
  has signature (1, -1) or (-1, 1). [for real b, there is no chance for uncatched error of wrong conjugation of b].
  Do not change the order of legs in tensor a (unlike tensordot).
  It is called by tensordot (together with moveaxis) for operations involving diagonal tensor.
- in `set_block` allow to have ts as a tuple of tuples.
  This is more natural syntax when number of symmetry charges is greater than one.
- in `svd` and `eigh` new option `untruncated_S` addtionally returns dict uS with a copy 
  (i.e. not a clone) of singular- or eigenvalues. uS also has field `D` with 
  truncated bond dimensions.

17-11-2021
- new function `mask(a, b, axis)` where tensor b is diagonal. 
  Applies mask based on nonzero elements of b on specified axis of tensor a. 
  Results in a new projected tensor with truncated bond dimensions (also possibly removed charges) on leg of a given by axis.
- define magic methods for <, <=, >, >= to create logical tensors -- intended as diagonal masks.
  It passes <, <=, >, >= on the level of backend block tensors.

8-12-2021
- in `import_from_dict` perform checks only after loading all the block; this speeds up loading if there are many blocks.
