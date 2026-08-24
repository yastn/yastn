# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Methods creating a new yastn.Tensor """
import os
import warnings
import numbers
from functools import reduce
from operator import mul

import numpy as np

from ._auxiliary import _config, get_blocks, get_trimmed_struct, find_index, find_matching_indices
from ._auxiliary import convert_to_tuples_and_slices, _compress_slices
from ._tests import YastnError
from ..backend import backend_np, import_backend
from ..sym import sym_none, sym_U1, sym_Z2, sym_Z3, sym_U1xU1, sym_U1xU1xZ2

__all__ = ['make_config']


_syms = {"dense": sym_none,
         "none": sym_none,
         "U1": sym_U1,
         "Z2": sym_Z2,
         "Z3": sym_Z3,
         "U1xU1": sym_U1xU1,
         "U1xU1xZ2": sym_U1xU1xZ2}


# def make_config(backend=backend_np, sym=sym_none, default_device='cpu',
#                 default_dtype='float64', fermionic=False,
#                 default_fusion='hard', force_fusion=None, tensordot_policy='fuse_contracted', **kwargs):
def make_config(**kwargs) -> _config:
    r"""
    Create a YASTN configuration object.

    Parameters
    ----------
    backend : backend module or str
        Specify ``backend`` providing linear algebra and base dense tensors.
        Currently supported backends are

            * NumPy as ``yastn.backend.backend_np``
            * PyTorch as ``yastn.backend.backend_torch``

        The above backends can be specified as strings: "np", "torch".
        Defaults to NumPy backend.

    sym : symmetry module or compatible object or str
        Specify abelian symmetry. To see how YASTN defines symmetries,
        see :class:`yastn.sym.sym_abelian`.
        Defaults to ``yastn.sym.sym_none``, effectively a dense tensor.
        For predefined symmetries, takes string input from
        'none' (or 'dense'), 'Z2', 'Z3', 'U1', 'U1xU1', 'U1xU1xZ2'.

    default_device : str
        Tensors can be stored on various devices as supported by ``backend``

            * NumPy supports only ``'cpu'`` device
            * PyTorch supports multiple devices, see
              https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

        If not specified, the default device is ``'cpu'``.

    default_dtype: str
        Default data type (dtype) of YASTN tensors. Supported options are: ``'float64'``,
        ``'complex128'``. If not specified, the default dtype is ``'float64'``.
    fermionic : bool or tuple[bool,...]
        Specify behavior of :meth:`yastn.swap_gate` function, allowing to introduce fermionic statistics.
        Allowed values: ``False``, ``True``, or a tuple ``(True, False, ...)`` with one bool for each component
        charge vector, i.e., of length sym.NSYM. The default is ``False``.
    default_fusion: str
        Specify default strategy to handle leg fusion: ``'hard'`` or ``'meta'``. See :meth:`yastn.Tensor.fuse_legs`
        for details. The default is ``'hard'``.
    force_fusion : str
        Overrides fusion strategy provided in :meth:`yastn.Tensor.fuse_legs`. The default is ``None``.
    tensordot_policy: str
        Contraction approach used by :meth:`yastn.tensordot`

            * ``'fuse_to_matrix'`` Tensordot involves suitable permutation of each tensor while performing a fusion of each tensor into a sequence of matrices and calling matrix-matrix multiplication. Postprocessing includes unfusing the remaining legs in the result, which often copy data adding extra overhead.
            * ``'fuse_contracted'`` Tensordot involves suitable permutation of each tensor while performing a fusion of to-be-contracted legs of each tensor and calling multiplication. It involves a larger number of multiplication calls for smaller objects, but unfusing the legs of the result is not needed.
            * ``'no_fusion'`` Tensordot involves suitable permutation of tensor blocks and calling matrix-matrix multiplication for a potentially large number of small objects. Resulting contributions to new blocks get added. However, overheads of initial fusion (copying data) can sometimes be avoided in this approach.

    lazy_threshold: float = 0 if backend is cuTensor, else 0.5
        Not all symmetry-allowed blocks need to be present in "lazy" tensor. Hence, when computing a contractions with "lazy" tensors,
        not all blocks allowed by the symmetry need to exist in the resulting tensor.
        If the fraction (retained blocks / all allowed blocks) < ``lazy_threshold``, then blocks are initialized lazily,
        i.e., only when they are needed. On ``cuTensor`` backend, defaults to 0, otherwise 0.5
        Impact:
            Decreases memory usage and flop count in contractions. The block-sparsity algebra is more expensive.
        Revelant scenarious:
            Outer-product-like contractions, where number of legs of resulting tensor is larger than the number of legs of the input tensors.
            In such cases, the number of allowed blocks can be much larger than the number of retained blocks.

    meta_tensordot_policy: str = "cpu"|"gpu"|"auto"
        Block-sparsity algorithm used by :meth:`yastn.tensordot`. The default is ``'auto'``,
        which uses the optimized GPU algorithm if available, otherwise the CPU algorithm.
        When "auto" can be also overriden by setting the environment variable ``YASTN_META_CUTENSOR`` to ``"GPU"`` or ``"CPU"``.

    Example
    -------

    ::

        config = yastn.make_config(backend='np', sym='U1')
    """
    if "backend" not in kwargs:
        kwargs["backend"] = backend_np
    elif isinstance(kwargs["backend"], str):
        try:
            kwargs["backend"] = import_backend(kwargs["backend"])
        except ValueError:
            raise YastnError("backend encoded as string only supports: 'np', 'torch'")

    if "sym" not in kwargs:
        kwargs["sym"] = sym_none
    elif isinstance(kwargs["sym"], str):
        try:
            kwargs["sym"] = _syms[kwargs["sym"]]
        except KeyError:
            raise YastnError("sym encoded as string only supports: 'dense', 'Z2', 'Z3', 'U1', 'U1xU1', 'U1xU1xZ2'.")

    if kwargs.get("lazy_threshold", None) is None:
        if kwargs["backend"].BACKEND_ID in ["torch_cutensor",]:
            kwargs["lazy_threshold"] = 0
        else:
            kwargs["lazy_threshold"] = 0.5

    return _config(**{a: kwargs[a] for a in _config._fields if a in kwargs})


def __setitem__(a, key, newvalue):
    """
    Update data of the selected block.

    The data (its shape) should be consistent with
    the dimensions of the charge sectors where the block belongs.

    Parameters
    ----------
    key : Sequence[int] | Sequence[Sequence[int]]
        charges of the block
    """
    try:
        key = np.array(key, dtype=np.int64).reshape(a.ndim_n, a.config.sym.NSYM)
        reverse_trans = np.argsort(a.trans)
        ukey = key[reverse_trans, :].ravel()
        bl = get_blocks(a.config.sym, a.struct)
        ind = find_index(bl.t, ukey, sorted=True)
    except ValueError as exc:
        raise YastnError('Tensor does not have the block specified by key.') from exc
    slc = slice(*bl.slc[ind])
    Dt = bl.D[ind]
    Dr = tuple(Dt[ax] for ax in a.trans)
    if not a.isdiag:
        newvalue = a.config.backend.permute_dims(newvalue, Dr, reverse_trans)
    a._data[slc] = newvalue.reshape(-1)


def _fill_tensor(a, t=(), D=(), val='rand'):  # dtype = None
    r"""
    Create all allowed blocks based on signature ``s``, total charge ``n``,
    and a set of charge sectors ``t`` for each leg of the tensor.

    First, all allowed blocks are identified by checking the
    :ref:`selection rule<symmetry selection rule>`.
    Then each allowed block is created as a tensor with
    sizes specified in ``D`` and filled with value ``val``.

    .. note::
        This operation overwrites the data of the tensor.

    Parameters
    ----------
    a : yastn.Tensor

    t : Sequence[Sequence[int]] or Sequence[Sequence[Sequence[int]]]
        list of charge sectors for each leg of the tensor, see examples.
        In case of tensor without symmetry this argument is ignored.

    D : Sequence[int] or Sequence[Sequence[int]]
        list of sector sizes for each leg of the tensor, see examples.

    val : str
        ``'rand'``, ``'ones'``, or  ``'zeros'``
    """
    try:
        D = tuple(D)
    except TypeError:
        D = (D,)
    try:
        t = tuple(t)
    except TypeError:
        t = (t,)

    if a.config.sym.NSYM == 0:
        if a.isdiag and len(D) == 1:
            D = D + D
        D = tuple(x if x else (0,) for x in D)  # replace () with (0,)
        D = tuple(x if isinstance(x, tuple) else (x,) for x in D)
        if len(D) != a.ndim_n:
            raise YastnError("Number of elements in D does not match tensor rank.")
        t = (((),),) * a.ndim_n
    else:  # a.config.sym.NSYM >= 1
        D = (D,) if (a.ndim_n == 1 or a.isdiag) and isinstance(D[0], numbers.Number) else D
        t = (t,) if (a.ndim_n == 1 or a.isdiag) and isinstance(t[0], numbers.Number) else t
        D = D + D if a.isdiag and len(D) == 1 else D
        t = t + t if a.isdiag and len(t) == 1 else t

        D = list((x,) if isinstance(x, numbers.Number) else x for x in D)
        t = list((x,) if isinstance(x, numbers.Number) else x for x in t)

        if len(D) != a.ndim_n:
            raise YastnError("Number of elements in D does not match tensor rank.")
        if len(t) != a.ndim_n:
            raise YastnError("Number of elements in t does not match tensor rank.")
        for x, y in zip(D, t):
            if len(x) != len(y):
                raise YastnError("Elements of t and D do not match")

    legs = []
    for leg, tt, DD in zip(a.struct.legs, t, D):
        tt = map(tuple, np.array(tt, dtype=np.int64).reshape(len(tt), a.config.sym.NSYM).tolist())
        DD = np.array(DD, dtype=np.int64).reshape(len(DD)).tolist()
        tD = dict(sorted(zip(tt, DD)))
        legs.append(leg._replace(t=tuple(tD.keys()), D=tuple(tD.values())))

    struct = a.struct.replace(legs=legs, mask=None)
    struct = get_trimmed_struct(a.config.sym, struct)
    bl = get_blocks(a.config.sym, struct)

    if a.isdiag and struct.legs[0] != struct.legs[1].conj():
        raise YastnError("Diagonal tensor requires the same bond dimensions on both legs.")

    a.struct = struct
    a._data = _init_block(a.config, bl.size, val, dtype=a.yastn_dtype, device=a.device)
    a.is_consistent()


def set_block(a, ts=(), Ds=None, val='zeros'):
    """
    Add new block to tensor or change the existing one.

    This is the intended way to add new blocks by hand.
    Checks if bond dimensions of the new block are consistent with the existing ones
    and updates the legs of the tensors accordingly.

    Parameters
    ----------
    ts : Sequence[int] | Sequence[Sequence[int]]
        Charges identifying the block. Ignored if tensor has no symmetry.

    Ds : Sequence[int]
        Dimensions of the block. If ``None``, tries to infer
        dimensions from legs of the tensor.

    val : tensor-like | str | tuple[str, tuple]
        recognized string values are ``'ones'``, ``'zeros'``, ``'normal'``, ``'rand'``,
        or a tuple ``('rand', distribution)``, for uniform distribution in range given by tuple lim.
        Otherwise any tensor-like format such as nested list, numpy.ndarray, etc.,
        can be used provided it is supported by :doc:`tensor's backend </tensor/configuration>`.
    """
    if a.trans != tuple(range(a.ndim_n)):
        raise YastnError("Setting block of transpoded tensor is not supported.")
    ts = np.array(ts, dtype=np.int64).ravel()
    nsym = a.config.sym.NSYM
    if a.isdiag and len(ts) == nsym:
        ts = np.hstack([ts, ts])
    if len(ts) != a.ndim_n * nsym:
        raise YastnError('Size of ts is not consistent with tensor rank and the number of symmetry sectors.')

    ats = ts.reshape((1, a.ndim_n, nsym))
    if not np.all(a.config.sym.fuse(ats, a.s_n, 1) == a.n):
        raise YastnError('Charges ts are not consistent with the symmetry rules: f(t @ s) == n')
    ats = ats[0]

    tss = tuple(map(tuple, ats.tolist()))

    if Ds is None:  # attempt to read Ds from existing blocks.
        try:
            Ds = tuple(leg[tt] for leg, tt in zip(a.struct.legs, tss))
        except ValueError as err:
            raise YastnError('Provided Ds. Cannot infer all bond dimensions from existing blocks.') from err
    else:  # Ds was provided
        Ds = np.array(Ds, dtype=np.int64).ravel()
        if a.isdiag and len(Ds) == 1:
            Ds = np.hstack([Ds, Ds])
        Ds = tuple(Ds.tolist())
    if len(Ds) != a.ndim_n:
        raise YastnError('Size of Ds is not consistent with tensor rank.')

    if a.isdiag and Ds[0] != Ds[1]:
        raise YastnError("Diagonal tensor requires the same bond dimensions on both legs.")

    if any(tt in leg and leg[tt] != DD for leg, tt, DD in zip(a.struct.legs, tss, Ds)):
        raise YastnError("Provided Ds is not consistent with dimensions of existing legs.")
    #
    # embed old data in new data; makes a data copy in the process
    #
    bl_old = get_blocks(a.config.sym, a.struct)
    legs_new = tuple(leg.add_charge(tt, DD) for leg, tt, DD in zip(a.struct.legs, tss, Ds))
    struct_new = a.struct.replace(legs=legs_new, mask=None)
    bl_new = get_blocks(a.config.sym, struct_new)
    #
    # prepare mask for lazy initialization
    #
    inds = find_matching_indices(bl_new.t, bl_old.t, both=False)
    mask = np.zeros(bl_new.nblocks, dtype=bool)
    mask[inds] = True
    ind = find_index(bl_new.t, ats)
    mask[ind] = True
    struct_new = struct_new.replace(mask=mask)
    bl_new = get_blocks(a.config.sym, struct_new)
    #
    ind1, ind2 = find_matching_indices(bl_new.t, bl_old.t)
    meta = _compress_slices(np.column_stack([bl_new.slc[ind1], bl_old.slc[ind2]]))
    meta_dt = np.dtype([
        ('sln', np.int64, (2,)),
        ('slo', np.int64, (2,))])
    meta = meta.view(meta_dt).reshape(-1)
    meta = convert_to_tuples_and_slices(meta)
    newdata = a.config.backend.embed_slices(a.data, meta, bl_new.size)
    #
    a.struct = struct_new
    a._data = newdata
    #
    Dsize = Ds[0] if a.isdiag else reduce(mul, Ds, 1)
    new_block = _init_block(a.config, Dsize, val, dtype=a.yastn_dtype, device=a.device)
    ind = find_index(bl_new.t, ats)
    slc = bl_new.slc[ind]
    a._data[slice(*slc)] = new_block


def _init_block(config, Dsize, val, dtype, device):
    if isinstance(val, tuple) and val[0] == 'rand':
        return config.backend.rand((Dsize,), distribution=val[1], dtype=dtype, device=device)
    if isinstance(val, str):
        if val == 'zeros':
            return config.backend.zeros((Dsize,), dtype=dtype, device=device)
        if val == 'rand':
            return config.backend.rand((Dsize,), dtype=dtype, device=device)
        if val == 'normal':
            return config.backend.rand((Dsize,), distribution='normal', dtype=dtype, device=device)
        if val == 'ones':
            return config.backend.ones((Dsize,), dtype=dtype, device=device)
        raise YastnError('val should be in ("zeros", "ones", "rand") or an array of the correct size')
    x = config.backend.to_tensor(val, Ds=Dsize, dtype=dtype, device=device)
    if config.backend.get_size(x) == Dsize ** 2:
        x = config.backend.diag_get(x.reshape(Dsize, Dsize))
    return x
