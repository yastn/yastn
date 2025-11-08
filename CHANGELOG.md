24-12-2021
- is_independent for pytorch backend now compares values of pointers

21-12-2021
- `get_blocks_charges` renamed to `get_blocks_charge`, `get_blocks_shapes` renamed to `get_blocks_shape`,
  `print_blocks` renamed to `print_blocks_shape`

19-12-2021
- `move_to_device` in backend is renamed to `move_to` and now handles both device and dtype
  conversions

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
  and if it matches device (this can be unwanted test in numpy ...)
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
- in `load_from_dict` perform checks only after loading all the block; this speeds up loading if there are many blocks.

17-12-2021
- `tensordot` has a new parameter `policy` switching between dot that marges and unmerges effective 2d blocks (`merge`) and direct block by block multiplication (`direct`). Default is `hybrid` that switches between those two based on a simple heuristics. It can be fixed in config files under variable `default_tensordot`.

19-12-2021
- introduce property `tensor.ndim` replacing `tensor.mlegs` and `tensor.ndim_n` replacing `tensor.nlegs`.  Property `tensor.size` replaces `tensor.get_size()`
  property `tensor.s` returns signature of meta-fused legs (i.e. showing only part of native signeture). property `tensor.s_n` return full (native) signature.
  `tensor.s`, `tensor.s_n` and `tensor.n` return tuple. `tensor.get_rank(native)` and `tensor.get_signature(native)` are unified getters.

20-12-2021
- removed flag _check["check_signature"] and _check["check_consistency"]. Performing tests of correctness is fast, so it is not critical to switch it off.

23-12-2021
- add option `force_tensordot` to config, that enforces specific policy used by tensordot
- BUGFIX: fuse_legs(.. mode='hard') was breaking for dense tensor -- this is now fixed (even though using hard fusion for dense tensor does not give anything)
- BUGFIX: fix some issues with passing device to config during initialization

25-12-2021
- add function yastn.clear_cache() to clear caches of all lru_cache employed to speed-up precomputing.

01-01-2022
- remove function `norm_diff(a, b)`. Use `norm(a - b)` instead.
- `svd` got argument `policy` = `fullrank` (default) or `lowrank`.
  `svd_lowrank` calls `svd` with policy = `lowrank`

06-01-2022
- change function names `export_to_dict` to `save_to_dict`, `export_to_hdf5` to `save_to_hdf5`
  `import_from_dict` to `load_from_dict`; `import_from_hdf5` to `load_from_hdf5`
- save/load functions are changed to use 1d data container.
  It is now also more robust, using only native python/numpy data structures, i.e. no NamedTuples used by yastn.
  WARNING: Backward compatibility is broken!

08-01-2022
- New function `remove_leg`
- define `abs` as a magic method `__abs__`. Function `absolut` has been removed.
- define `__matmul__`, giving shorthand a @ b == yastn.tensordot(a, b, axes=(a.ndim - 1, 0))
- BUGFIX: raising exceptions when .diag() cannot be used to create diagonal tensor
- BUGFIX: ncon in some rare special case could ignore content of conjs.
- new function `einsum` (for now a place holder doing nothing).
- new function `move_leg` which is an alias to `moveaxis`

v0.9
  the last version employing dictionary of blocks before the transition to a single 1d data structure

13-03-2022
- Transition to 1d data structure
- dtype and device removed from config
- `unique_dtype` replaced by `get_dtype`
- properties `tensor.dtype`, `tensor.yastn_dtype`, `tensor.device`
- new function '__setitem__()' that gives direct access to change existing blocks
- in rsqrt(x, cutoff) cutoff is done with respect to x, not sqrt(x)
- new function `grad()` that generates gradient yastntensor

07-04-2022
- `apply_mask` replaces function `mask`.
- `apply_mask` and `broadcast` take diagonal tensor (to be use as a mask or to broadcast) as the first argument.
  Previously it was the second argument.
- `apply_mask` and `broadcast` can apply the same mask to a few tensors in a single-line execution.
- `svd` and `eigh` do not support truncation
- new function `truncation_mask` generates a mask tensor that can be used for truncation,
  but this is not the only way such mask can be obtianed.
 - `svd_with_truncation` and `eigh_with_truncation` combine pure `svd` with `truncation_mask` and `apply_mask`
 - `broadcast` does not take conj argument anymore
 - `svd_lowrank` is depreciated. Call it though `policy='lowrank'` in `svd`

01-05-2022
- svd_with_truncation can take mask-generating funcion as an argument
- function `truncation_mask_multiplets` to retain multiplets in generating mask for truncation

30-05-2022
- function `Leg` generates `_Leg` object describing linear space of a single tensor leg.
  It takes, signature `s`, list of charges `t` and corresponding dimensions `D`, as well as symmetry (or config).
- function `yastn.legs_union(*legs)` creates a leg that describes a space being an union of provided spaces/legs.
- tensor has method `.get_legs(axes=None)` outputing `_Leg` (or list of specified legs, of list of all tensor legs for axis=None). It also carries information about fusions.
- Initializing tensor with `yastn.ones`, `yastn.zeors`, `yastn.rand`, `yastn.eye` takes list of a list of legs,(argument `legs`). Old input with `s`, `t`, `D` is supported and providing any of those parameters overrides argument `legs`
- methods `to_numpy`, `to_dense`, `to_nonsymmetric` take dict `legs` specifying how to fill-in zeros to make output consistent with provided `legs` (for now does not work for mismatch in hard fusions). This replaces old argument `leg_structures`.
- `truncation_mask` does not have arguments keep_multiplets and multiplets_eps, as there is specialised function `truncation_mask_multiplets`

17-6-2022
- a new version of `yastn.block` that supports tracking history of blocking; to resolve possible merge conflicts
- `.drop_leg_history(axis=None)` gives a shallow copy of the tensor, where information about fusion/blocking history on some legs (of all for axis=None) is dropped.
- `Leg` got method `.history()` that returns a string representation of the fusion history, with 'o' marking original legs, `s` is for sum, `p` is for product, 'm' is for meta fusion.
- simplify syntax of `yastn.decompose_from_1d(r1d, meta)`. It no longer takes config, that is stored in meta.

20-07-2022
- a new function `yastn.gaussian_leg` that allows that randomly distributes bond dimensions
  according to Gaussian distribution cenered at provided mean charge.

23-07-2022
- In yastn.operators predefine classes that generate sets of a few standard local operators.

16-10-2022
- new function `yastn.bitwise_not()`
- new function `yamps.is_canonical()`
- change order of tenor legs in MPO  (left virtual, ket, right virtual, bra)

01-11-2022
- move mps routines to yastn.tn.mps

28-11-2022
- renamed functions `mps.dmrg_`, `mps.variational_`, `mps.tdvp_`, `psi.canonize_`, `psi.truncate_`.
  `_` is used to indicate that they change mps (first argument) in place.
- `mps.tdvp_` is a genetator iterating over specified time snapshots.
- `mps.dmrg_`, `mps.variational_` can be made into generator by providing `iterator_step` : int, that gives the number of forth-and-back sweeps after which generator yields snapshot.

13-03-2023
- raname `mps.variational_` to `mps.compression_`. New more flexible syntax specifying target state.
- change axis to axes in parameters of all functions that can take multiple axes.

11-2023
- Tensor.transpose() and Tensor.T reverse th order of axes
- make_config  accepts str for predefined symmetries

06-2024
- tag v1.0

07-2024  (v1.01)
- fix engine type consistency for numpy2.0
- consistently use 'U1' for U(1)-symmetry
- allow using **config._asdict() while initializing operator classes
